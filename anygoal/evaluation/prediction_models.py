"""Thin model adapters for the shared prediction evaluator."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def _processor_parameters(processor) -> tuple[np.ndarray, np.ndarray]:
    minimum = getattr(processor, "minimum", None)
    maximum = getattr(processor, "maximum", None)
    if minimum is not None and maximum is not None:
        minimum = np.asarray(minimum, dtype=np.float32)
        maximum = np.asarray(maximum, dtype=np.float32)
        return minimum, maximum - minimum
    mean = getattr(processor, "mean_", getattr(processor, "mean", None))
    scale = getattr(
        processor,
        "scale_",
        getattr(processor, "std", getattr(processor, "scale", None)),
    )
    if mean is None or scale is None:
        raise TypeError(f"Unsupported processor type: {type(processor)}")
    return np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)


def standardize_tensor(values: torch.Tensor, processor) -> torch.Tensor:
    mean, scale = _processor_parameters(processor)
    mean = torch.as_tensor(mean, device=values.device, dtype=torch.float32)
    scale = torch.as_tensor(scale, device=values.device, dtype=torch.float32)
    return (values.float() - mean) / scale


def standardize_action_blocks(values: torch.Tensor, processor) -> torch.Tensor:
    mean, _ = _processor_parameters(processor)
    raw_action_dim = int(mean.size)
    if values.shape[-1] % raw_action_dim:
        raise ValueError(
            f"Blocked action dimension {values.shape[-1]} is not divisible by "
            f"the processor dimension {raw_action_dim}."
        )
    shape = values.shape
    normalized = standardize_tensor(values.reshape(-1, raw_action_dim), processor)
    return normalized.reshape(shape)


def imagenet_preprocess(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
    """Vectorized equivalent of the LeWM MPC image preprocessing."""
    pixels = pixels.to(dtype=torch.float32)
    if pixels.detach().amax() > 1.0:
        pixels = pixels / 255.0
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    leading = pixels.shape[:-3]
    pixels = pixels.reshape(-1, *pixels.shape[-3:])
    pixels = (pixels - mean) / std
    if tuple(pixels.shape[-2:]) != (image_size, image_size):
        pixels = F.interpolate(
            pixels,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return pixels.reshape(*leading, 3, image_size, image_size)


class LatentARPredictionAdapter:
    """One adapter for LeWM and sensorimotor AR latent predictors."""

    error_space = "latent_mse"
    observation_keys = ("pixels",)

    def __init__(
        self,
        *,
        model,
        model_name: str,
        action_processor,
        device: torch.device,
        pixel_preprocessor: Callable[[torch.Tensor], torch.Tensor],
        history_length: int = 3,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.action_processor = action_processor
        self.device = device
        self.pixel_preprocessor = pixel_preprocessor
        self.history_length = int(history_length)

    def rollout_errors(
        self,
        observations: dict[str, torch.Tensor],
        action_blocks: torch.Tensor,
        *,
        rollout_seed: int,
    ) -> torch.Tensor:
        del rollout_seed
        pixels = self.pixel_preprocessor(observations["pixels"].to(self.device))
        embeddings = self.model.encode({"pixels": pixels})["emb"]
        actions = standardize_action_blocks(
            action_blocks.to(self.device), self.action_processor
        )
        history = embeddings[:, : self.history_length]
        errors = []
        prediction_steps = embeddings.shape[1] - self.history_length
        for step in range(prediction_steps):
            action_context = actions[:, step : step + self.history_length]
            action_embeddings = self.model.action_encoder(action_context)
            prediction = self.model.predict(history, action_embeddings)[:, -1]
            target = embeddings[:, self.history_length + step]
            errors.append((prediction - target).square().mean(dim=-1))
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return torch.stack(errors, dim=1)


class PreJEPAPredictionAdapter:
    """Autoregressive PreJEPA evaluation excluding known action latents."""

    error_space = "latent_mse_non_action_modalities"

    def __init__(
        self,
        *,
        model,
        action_processor,
        processors: Mapping[str, object],
        device: torch.device,
        image_size: int,
        history_length: int = 3,
    ) -> None:
        self.model = model
        self.model_name = "prejepa"
        self.action_processor = action_processor
        self.processors = dict(processors)
        self.device = device
        self.image_size = int(image_size)
        self.history_length = int(history_length)
        self.extra_keys = tuple(model.extra_encoders)
        self.predicted_keys = tuple(key for key in self.extra_keys if key != "action")
        self.observation_keys = ("pixels", *self.predicted_keys)

    def rollout_errors(
        self,
        observations: dict[str, torch.Tensor],
        action_blocks: torch.Tensor,
        *,
        rollout_seed: int,
    ) -> torch.Tensor:
        del rollout_seed
        pixels = imagenet_preprocess(
            observations["pixels"].to(self.device), self.image_size
        )
        actions = standardize_action_blocks(
            action_blocks.to(self.device), self.action_processor
        )
        padded_actions = torch.cat(
            (actions, actions.new_zeros(actions.shape[0], 1, actions.shape[-1])),
            dim=1,
        )
        info = {"pixels": pixels, "action": padded_actions}
        for key in self.predicted_keys:
            if key not in self.processors:
                raise KeyError(f"No MPC processor was fitted for PreJEPA key {key!r}.")
            info[key] = standardize_tensor(
                observations[key].to(self.device), self.processors[key]
            )
        encoded = self.model.encode(info)
        full_embeddings = encoded["emb"]
        history = full_embeddings[:, : self.history_length]
        pixel_dim = encoded["pixels_emb"].shape[-1]
        extra_slices = {}
        feature_start = pixel_dim
        for key in self.extra_keys:
            feature_end = feature_start + encoded[f"{key}_emb"].shape[-1]
            extra_slices[key] = slice(feature_start, feature_end)
            feature_start = feature_end

        errors = []
        prediction_steps = full_embeddings.shape[1] - self.history_length
        for step in range(prediction_steps):
            prediction = self.model.predict(history)[:, -1:]
            target_index = self.history_length + step
            components = [
                (
                    prediction[:, 0, :, :pixel_dim]
                    - encoded["pixels_emb"][:, target_index]
                )
                .square()
                .mean(dim=(-2, -1))
            ]
            for key in self.predicted_keys:
                components.append(
                    (
                        prediction[:, 0, :, extra_slices[key]]
                        - encoded[f"{key}_emb"][:, target_index, None]
                    )
                    .square()
                    .mean(dim=(-2, -1))
                )
            errors.append(torch.stack(components, dim=1).sum(dim=1))

            if step + 1 < prediction_steps:
                next_action = actions[
                    :, self.history_length + step : self.history_length + step + 1
                ]
                prediction = self.model.replace_action_in_embedding(
                    prediction[:, None], next_action[:, None]
                )[:, 0]
            history = torch.cat((history[:, 1:], prediction), dim=1)
        return torch.stack(errors, dim=1)


class TrueStatePredictionAdapter:
    """Native normalized-state evaluator for both toy predictor variants."""

    error_space = "normalized_state_mse"

    def __init__(
        self,
        *,
        model,
        model_name: str,
        state_adapter,
        state_processor,
        action_processor,
        device: torch.device,
        history_length: int = 3,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.state_adapter = state_adapter
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.device = device
        self.history_length = int(history_length)
        self.observation_keys = tuple(state_adapter.state_keys)

    def _encode_states(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.observation_keys == ("state",):
            state_input = observations["state"]
        else:
            state_input = {
                key: observations[key] for key in self.observation_keys
            }
        states = self.state_adapter.encode_state(state_input).to(self.device)
        return standardize_tensor(states, self.state_processor)

    def rollout_errors(
        self,
        observations: dict[str, torch.Tensor],
        action_blocks: torch.Tensor,
        *,
        rollout_seed: int,
    ) -> torch.Tensor:
        states = self._encode_states(observations)
        actions = standardize_action_blocks(
            action_blocks.to(self.device), self.action_processor
        )
        history = states[:, : self.history_length]
        generator = torch.Generator(device=self.device).manual_seed(rollout_seed)
        errors = []
        prediction_steps = states.shape[1] - self.history_length
        for step in range(prediction_steps):
            action_context = actions[:, step : step + self.history_length]
            if hasattr(self.model, "predict_next_state"):
                prediction = self.model.predict_next_state(
                    history,
                    action_context,
                    generator=generator,
                )
            else:
                prediction = self.model(history, action_context)
            if prediction.ndim == 3:
                prediction = prediction[..., 0]
            target = states[:, self.history_length + step]
            errors.append((prediction - target).square().mean(dim=-1))
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return torch.stack(errors, dim=1)


class DriftWorldPredictionAdapter:
    """Pixel-space autoregressive evaluator for DriftWorld."""

    error_space = "normalized_pixel_mse"

    def __init__(
        self,
        *,
        model,
        action_processor,
        device: torch.device,
        history_length: int = 3,
    ) -> None:
        self.model = model
        self.model_name = "driftworld"
        self.action_processor = action_processor
        self.device = device
        self.history_length = int(history_length)
        self.observation_keys = (
            ("pixels", "proprio")
            if model.action_mode == "absolute_target"
            else ("pixels",)
        )
        if model.num_history_frames != self.history_length:
            raise ValueError(
                f"DriftWorld checkpoint expects {model.num_history_frames} history "
                f"frames, not {self.history_length}."
            )

    def rollout_errors(
        self,
        observations: dict[str, torch.Tensor],
        action_blocks: torch.Tensor,
        *,
        rollout_seed: int,
    ) -> torch.Tensor:
        pixels = self.model.preprocess_pixels(observations["pixels"].to(self.device))
        actions = standardize_action_blocks(
            action_blocks.to(self.device), self.action_processor
        )
        history = pixels[:, : self.history_length]
        errors = []
        prediction_steps = pixels.shape[1] - self.history_length
        for step in range(prediction_steps):
            candidate = actions[:, None, step + self.history_length - 1 : step + self.history_length]
            info = {}
            if "proprio" in observations:
                proprio_index = step + self.history_length - 1
                info["proprio"] = observations["proprio"][
                    :, proprio_index
                ].to(self.device)[:, None]
            model_action = self.model.prepare_model_actions(info, candidate)
            prediction = self.model._rollout_once(
                history,
                model_action,
                int(rollout_seed) + step,
            )[:, 0].clamp(-1.0, 1.0)
            target = pixels[:, self.history_length + step]
            errors.append(
                (prediction - target).square().mean(dim=(-3, -2, -1))
            )
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return torch.stack(errors, dim=1)


class IRASimPredictionAdapter:
    """Pixel-space autoregressive evaluator for IRASim."""

    error_space = "normalized_pixel_mse"
    observation_keys = ("pixels",)

    def __init__(
        self,
        *,
        model,
        action_processor,
        device: torch.device,
        history_length: int = 3,
    ) -> None:
        self.model = model
        self.model_name = "irasim"
        self.action_processor = action_processor
        self.device = device
        self.history_length = int(history_length)
        if model.num_history_frames != self.history_length:
            raise ValueError(
                f"IRASim checkpoint expects {model.num_history_frames} history "
                f"frames, not {self.history_length}."
            )

    def rollout_errors(
        self,
        observations: dict[str, torch.Tensor],
        action_blocks: torch.Tensor,
        *,
        rollout_seed: int,
    ) -> torch.Tensor:
        pixels = self.model.preprocess_pixels(observations["pixels"].to(self.device))
        actions = standardize_action_blocks(
            action_blocks.to(self.device), self.action_processor
        )
        initial_pixels = pixels[:, : self.history_length]
        encoded_history = self.model._encode(
            initial_pixels.flatten(0, 1), int(rollout_seed)
        )
        history = encoded_history.reshape(
            initial_pixels.shape[0],
            self.history_length,
            *encoded_history.shape[1:],
        )

        errors = []
        prediction_steps = pixels.shape[1] - self.history_length
        for step in range(prediction_steps):
            history_action_blocks = actions[
                :, step : step + self.history_length - 1
            ]
            future_action_block = actions[
                :, step + self.history_length - 1 : step + self.history_length
            ]
            history_actions = self.model.prepare_model_actions(
                history_action_blocks
            )[:, None]
            future_actions = self.model.prepare_model_actions(
                future_action_block
            )[:, None]
            prediction = self.model._rollout_once(
                history,
                history_actions,
                future_actions,
                int(rollout_seed) + step,
            )[:, 0]
            prediction_pixels = self.model._decode(prediction).clamp(-1.0, 1.0)
            target = pixels[:, self.history_length + step]
            errors.append(
                (prediction_pixels - target).square().mean(dim=(-3, -2, -1))
            )
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return torch.stack(errors, dim=1)


__all__ = [
    "DriftWorldPredictionAdapter",
    "IRASimPredictionAdapter",
    "LatentARPredictionAdapter",
    "PreJEPAPredictionAdapter",
    "TrueStatePredictionAdapter",
    "imagenet_preprocess",
    "standardize_action_blocks",
    "standardize_tensor",
]
