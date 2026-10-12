"""StableWM-compatible IRASim model wrapper."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .actions import IRASimActionProcessor
from .upstream import get_irasim_pipeline_class


class IRASimWorldModel(nn.Module):
    """IRASim video diffusion model implementing StableWM's cost contract."""

    action_dim = 2

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        scheduler,
        action_dim: int = 2,
        action_frameskip: int = 1,
        video_size=(224, 224),
        num_frames: int = 16,
        num_history_frames: int = 1,
        inference_steps: int = 50,
        guidance_scale: float = 1.0,
        rollout_batch_size: int = 1,
        num_rollout_samples: int = 1,
        rollout_seed: int = 0,
        action_scale=(20.0, 20.0),
        action_processor: IRASimActionProcessor | None = None,
        cost_type: str = "pixel_mse",
        pipeline_factory=None,
    ) -> None:
        super().__init__()
        if (
            min(
                action_frameskip,
                num_history_frames,
                num_frames - num_history_frames,
                inference_steps,
                rollout_batch_size,
                num_rollout_samples,
            )
            < 1
        ):
            raise ValueError(
                "IRASim frame, sampling, and rollout sizes must be positive."
            )
        if guidance_scale != 1.0:
            raise ValueError(
                "The vendored IRASim pipeline does not duplicate actions for CFG; "
                "guidance_scale must remain 1.0."
            )
        if cost_type not in {"pixel_mse", "latent_mse"}:
            raise ValueError("cost_type must be pixel_mse or latent_mse.")
        self.action_dim = int(action_dim)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive.")
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.action_frameskip = int(action_frameskip)
        self.model_action_dim = self.action_dim * self.action_frameskip
        self.video_size = tuple(int(value) for value in video_size)
        self.num_frames = int(num_frames)
        self.num_history_frames = int(num_history_frames)
        self.num_future_frames = self.num_frames - self.num_history_frames
        self.inference_steps = int(inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.rollout_batch_size = int(rollout_batch_size)
        self.num_rollout_samples = int(num_rollout_samples)
        self.rollout_seed = int(rollout_seed)
        self.cost_type = str(cost_type)
        self._action_processor = action_processor or IRASimActionProcessor()
        action_scale = np.asarray(action_scale, dtype=np.float32)
        if action_scale.shape != (self.action_dim,):
            raise ValueError(f"action_scale must have shape ({self.action_dim},).")
        self.register_buffer(
            "action_scale",
            torch.from_numpy(action_scale),
            persistent=True,
        )
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    @property
    def action_processor(self) -> IRASimActionProcessor:
        return self._action_processor

    def prepare_model_actions(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"Expected blocked action dim {self.model_action_dim}, got "
                f"{candidates.shape[-1]}."
            )
        blocked = candidates.reshape(
            *candidates.shape[:-1], self.action_frameskip, self.action_dim
        )
        raw = self.action_processor.inverse_transform_tensor(blocked)
        scaled = raw * self.action_scale.to(raw)
        return scaled.flatten(-2)

    def preprocess_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.shape[-1] == 3 and pixels.shape[-3] != 3:
            pixels = pixels.movedim(-1, -3)
        if pixels.shape[-3] != 3:
            raise ValueError(f"Expected RGB pixels, got {tuple(pixels.shape)}.")
        was_uint8 = pixels.dtype == torch.uint8
        pixels = pixels.float()
        if was_uint8 or pixels.detach().amax() > 1.0:
            pixels = pixels.div(255.0)
        leading = pixels.shape[:-3]
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
        if tuple(pixels.shape[-2:]) != self.video_size:
            pixels = F.interpolate(
                pixels,
                size=self.video_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pixels = pixels.mul(2.0).sub(1.0)
        return pixels.reshape(*leading, 3, *self.video_size)

    def _encode(self, images: torch.Tensor, seed: int) -> torch.Tensor:
        cuda_devices = (
            [images.device.index]
            if images.device.type == "cuda" and images.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            latent = self.vae.encode(images).latent_dist.sample()
        return latent * self.vae.config.scaling_factor

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = []
        scale = self.vae.config.scaling_factor
        for start in range(0, len(latents), self.rollout_batch_size):
            decoded.append(
                self.vae.decode(
                    latents[start : start + self.rollout_batch_size] / scale
                ).sample
            )
        return torch.cat(decoded)

    def _get_pipeline(self):
        if self._pipeline is None:
            factory = self.pipeline_factory or get_irasim_pipeline_class()
            self._pipeline = factory(
                vae=self.vae,
                scheduler=self.scheduler,
                transformer=self.transformer,
            )
        return self._pipeline

    def _rollout_once(
        self,
        context_latents: torch.Tensor,
        history_actions: torch.Tensor,
        actions: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        batch_size, num_candidates, horizon = actions.shape[:3]
        flat_actions = actions.reshape(-1, horizon, self.model_action_dim)
        flat_history_actions = (
            history_actions.reshape(
                -1, self.num_history_frames - 1, self.model_action_dim
            )
            if self.num_history_frames > 1
            else flat_actions.new_empty(len(flat_actions), 0, self.model_action_dim)
        )
        env_indices = torch.arange(batch_size, device=actions.device).repeat_interleave(
            num_candidates
        )
        current_context = (
            context_latents[:, None]
            .expand(batch_size, num_candidates, *context_latents.shape[1:])
            .reshape(-1, *context_latents.shape[1:])
        )
        current_history_actions = flat_history_actions
        pipeline = self._get_pipeline()
        generator = torch.Generator(device=actions.device)
        for segment_start in range(0, horizon, self.num_future_frames):
            segment_end = min(segment_start + self.num_future_frames, horizon)
            segment_length = segment_end - segment_start
            generator.manual_seed(seed + segment_start)
            shared_noise = torch.randn(
                batch_size,
                segment_length,
                *current_context.shape[2:],
                generator=generator,
                device=current_context.device,
                dtype=current_context.dtype,
            )
            segment_outputs = []
            for start in range(0, len(current_context), self.rollout_batch_size):
                end = min(start + self.rollout_batch_size, len(current_context))
                mask_x = current_context[start:end]
                initial_noise = torch.cat(
                    [mask_x, shared_noise[env_indices[start:end]]], dim=1
                )
                segment_actions = torch.cat(
                    [
                        current_history_actions[start:end],
                        flat_actions[start:end, segment_start:segment_end],
                    ],
                    dim=1,
                )
                _, latent_video = pipeline(
                    segment_actions,
                    mask_x=mask_x,
                    video_length=self.num_history_frames + segment_length,
                    height=self.video_size[0],
                    width=self.video_size[1],
                    num_inference_steps=self.inference_steps,
                    guidance_scale=self.guidance_scale,
                    device=actions.device,
                    latents=initial_noise,
                    output_type="latent",
                    return_dict=False,
                )
                segment_outputs.append(latent_video[:, -segment_length:])
            predicted = torch.cat(segment_outputs)
            current_context = torch.cat([current_context, predicted], dim=1)[
                :, -self.num_history_frames :
            ]
            if self.num_history_frames > 1:
                current_history_actions = torch.cat(
                    [
                        current_history_actions,
                        flat_actions[:, segment_start:segment_end],
                    ],
                    dim=1,
                )[:, -(self.num_history_frames - 1) :]
        terminal = current_context[:, -1]
        return terminal.reshape(batch_size, num_candidates, *terminal.shape[1:])

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict[str, torch.Tensor],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if "pixels" not in info_dict or "goal" not in info_dict:
            raise KeyError("IRASim planning requires pixels and goal.")
        if action_candidates.ndim != 4:
            raise ValueError("action_candidates must have shape (B,S,H,D).")
        batch_size, num_candidates = action_candidates.shape[:2]
        pixels = info_dict["pixels"]
        goals = info_dict["goal"]
        if pixels.ndim != 6 or goals.ndim != 6:
            raise ValueError("Expanded pixels/goal must have shape (B,S,T,C,H,W).")
        if pixels.shape[:2] != (batch_size, num_candidates):
            raise ValueError("Info and action candidate axes do not match.")

        if pixels.shape[2] < self.num_history_frames:
            raise ValueError(
                f"IRASim requires {self.num_history_frames} context frames, got "
                f"{pixels.shape[2]}."
            )
        current_pixels = self.preprocess_pixels(
            pixels[:, 0, -self.num_history_frames :]
        )
        goal_pixels = self.preprocess_pixels(goals[:, 0, -1])
        flat_context_latents = self._encode(
            current_pixels.flatten(0, 1), self.rollout_seed
        )
        context_latents = flat_context_latents.reshape(
            batch_size,
            self.num_history_frames,
            *flat_context_latents.shape[1:],
        )
        goal_latent = (
            self._encode(goal_pixels, self.rollout_seed + 1)
            if self.cost_type == "latent_mse"
            else None
        )
        model_actions = self.prepare_model_actions(action_candidates)
        if self.num_history_frames > 1:
            if "history_actions" not in info_dict:
                raise KeyError(
                    "IRASim planning requires frame-aligned history_actions."
                )
            history_actions = info_dict["history_actions"]
            expected_shape = (
                batch_size,
                num_candidates,
                self.num_history_frames - 1,
                self.model_action_dim,
            )
            if tuple(history_actions.shape) != expected_shape:
                raise ValueError(
                    f"Expected history_actions {expected_shape}, got "
                    f"{tuple(history_actions.shape)}."
                )
            history_actions = self.prepare_model_actions(history_actions)
        else:
            history_actions = model_actions.new_empty(
                batch_size, num_candidates, 0, self.model_action_dim
            )
        costs = []
        for rollout_index in range(self.num_rollout_samples):
            terminal = self._rollout_once(
                context_latents,
                history_actions,
                model_actions,
                self.rollout_seed + 10_000 * rollout_index,
            )
            if self.cost_type == "latent_mse":
                target = goal_latent[:, None]
            else:
                flat_terminal = terminal.flatten(0, 1)
                decoded = self._decode(flat_terminal).reshape(
                    batch_size, num_candidates, 3, *self.video_size
                )
                terminal = decoded.clamp(-1.0, 1.0)
                target = goal_pixels[:, None]
            costs.append((terminal - target).pow(2).mean(dim=(-3, -2, -1)))
        return torch.stack(costs).mean(dim=0)
