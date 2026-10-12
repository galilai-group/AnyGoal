"""StableWM-compatible DriftWorld model wrappers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .actions import DriftWorldActionProcessor
from .upstream import create_driftworld_denoiser, create_robomimic_driftworld_denoiser


class DriftWorldWorldModel(nn.Module):
    """Upstream DriftWorld denoiser plus StableWM's terminal-cost contract."""

    action_dim = 2
    image_size = 96

    def __init__(
        self,
        *,
        denoiser=None,
        unet_name: str = "UNet_PushT",
        temp_list=(0.02, 0.05, 0.2),
        n_neg: int = 16,
        num_future_frames: int = 4,
        num_history_frames: int = 4,
        decay: float = 0.999,
        action_frameskip: int = 1,
        action_mode: str = "relative",
        rollout_batch_size: int = 16,
        num_rollout_samples: int = 1,
        rollout_seed: int = 0,
        action_scale: float = 100.0,
        absolute_action_max: float = 511.0,
        action_processor: DriftWorldActionProcessor | None = None,
    ) -> None:
        super().__init__()
        if action_frameskip < 1 or rollout_batch_size < 1 or num_rollout_samples < 1:
            raise ValueError(
                "frameskip, rollout_batch_size, and samples must be positive."
            )
        if action_mode not in {"relative", "absolute_target"}:
            raise ValueError("action_mode must be 'relative' or 'absolute_target'.")
        self.action_frameskip = int(action_frameskip)
        self.model_action_dim = self.action_dim * self.action_frameskip
        self.num_future_frames = int(num_future_frames)
        self.num_history_frames = int(num_history_frames)
        self.action_mode = str(action_mode)
        self.rollout_batch_size = int(rollout_batch_size)
        self.num_rollout_samples = int(num_rollout_samples)
        self.rollout_seed = int(rollout_seed)
        self.action_scale = float(action_scale)
        self.absolute_action_max = float(absolute_action_max)
        self._action_processor = action_processor or DriftWorldActionProcessor()
        if self._action_processor.mean.shape != (self.action_dim,):
            raise ValueError(
                f"Expected {self.action_dim} action statistics, got "
                f"{self._action_processor.mean.shape}."
            )
        self.denoiser = denoiser or create_driftworld_denoiser(
            unet_name=unet_name,
            temp_list=temp_list,
            n_neg=n_neg,
            num_future_frames=num_future_frames,
            num_history_frames=num_history_frames,
            decay=decay,
            action_frameskip=action_frameskip,
        )

    @property
    def action_processor(self) -> DriftWorldActionProcessor:
        return self._action_processor

    def load_native_checkpoint(self, checkpoint: str | Path) -> int | None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        self.denoiser.load_state_dict(state_dict, strict=True)
        return state.get("step") if isinstance(state, dict) else None

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
        if tuple(pixels.shape[-2:]) != (self.image_size, self.image_size):
            pixels = F.interpolate(
                pixels,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pixels = pixels.mul(2.0).sub(1.0)
        return pixels.reshape(*leading, 3, self.image_size, self.image_size)

    def _absolute_targets(
        self, info_dict: dict[str, torch.Tensor], relative_actions: torch.Tensor
    ) -> torch.Tensor:
        if "proprio" not in info_dict:
            raise KeyError(
                "The bundled absolute-action DriftWorld checkpoint requires proprio."
            )
        proprio = info_dict["proprio"].float()
        if proprio.ndim == 4:
            proprio = proprio[:, :, -1]
        if proprio.ndim != 3 or proprio.shape[-1] < 2:
            raise ValueError(f"Expected expanded proprio (B,S,D), got {proprio.shape}.")
        position = proprio[..., :2]
        velocity = (
            proprio[..., -2:] if proprio.shape[-1] >= 4 else torch.zeros_like(position)
        )
        flat_actions = relative_actions.flatten(2, 3)
        normalized_targets = []
        for action in flat_actions.unbind(dim=2):
            target = position + action * self.action_scale
            normalized_targets.append(target / self.absolute_action_max * 2.0 - 1.0)
            # Approximate PushT's free pusher dynamics so later relative commands
            # are converted around an evolving position. Contacts are unavailable
            # before image generation and are therefore intentionally omitted.
            for _ in range(10):
                acceleration = 100.0 * (target - position) - 20.0 * velocity
                velocity = velocity + acceleration * 0.01
                position = position + velocity * 0.01
        targets = torch.stack(normalized_targets, dim=2)
        return targets.reshape(*relative_actions.shape[:-2], self.model_action_dim)

    def prepare_model_actions(
        self, info_dict: dict[str, torch.Tensor], candidates: torch.Tensor
    ) -> torch.Tensor:
        if candidates.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"Expected blocked action dim {self.model_action_dim}, got "
                f"{candidates.shape[-1]}."
            )
        standardized = candidates.reshape(
            *candidates.shape[:-1], self.action_frameskip, self.action_dim
        )
        relative = self.action_processor.inverse_transform_tensor(standardized)
        if self.action_mode == "relative":
            return relative.flatten(-2)
        return self._absolute_targets(info_dict, relative)

    def _rollout_once(
        self,
        history: torch.Tensor,
        model_actions: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        batch_size, num_samples, horizon = model_actions.shape[:3]
        flat_count = batch_size * num_samples
        current_history = (
            history[:, None]
            .expand(batch_size, num_samples, *history.shape[1:])
            .reshape(flat_count, *history.shape[1:])
        )
        flat_actions = model_actions.reshape(flat_count, horizon, self.model_action_dim)
        terminal = current_history[:, -1]
        generator = torch.Generator(device=history.device).manual_seed(seed)

        for start in range(0, horizon, self.num_future_frames):
            end = min(start + self.num_future_frames, horizon)
            chunk_actions = flat_actions[:, start:end]
            future_frames = end - start
            base_noise = torch.randn(
                batch_size,
                1,
                3,
                future_frames,
                self.image_size,
                self.image_size,
                generator=generator,
                device=history.device,
                dtype=history.dtype,
            )
            noise = base_noise.expand(
                batch_size, num_samples, *base_noise.shape[2:]
            ).reshape(flat_count, *base_noise.shape[2:])
            generated_parts = []
            for batch_start in range(0, flat_count, self.rollout_batch_size):
                batch_end = min(batch_start + self.rollout_batch_size, flat_count)
                generated = self.denoiser.ema_model(
                    noise[batch_start:batch_end],
                    current_history[batch_start:batch_end].permute(0, 2, 1, 3, 4),
                    chunk_actions[batch_start:batch_end],
                )
                generated_parts.append(generated.permute(0, 2, 1, 3, 4))
            generated = torch.cat(generated_parts, dim=0)
            current_history = torch.cat([current_history, generated], dim=1)[
                :, -self.num_history_frames :
            ]
            terminal = generated[:, -1]
        return terminal.reshape(
            batch_size, num_samples, 3, self.image_size, self.image_size
        )

    @torch.inference_mode()
    def get_cost(
        self, info_dict: dict[str, torch.Tensor], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        if "pixels" not in info_dict or "goal" not in info_dict:
            raise KeyError("DriftWorld planning requires pixels and goal.")
        if action_candidates.ndim != 4:
            raise ValueError("action_candidates must have shape (B,S,H,D).")
        batch_size, num_samples = action_candidates.shape[:2]
        pixels = info_dict["pixels"]
        goal = info_dict["goal"]
        if pixels.ndim != 6 or goal.ndim != 6:
            raise ValueError("Expanded pixels/goal must have shape (B,S,T,C,H,W).")
        if pixels.shape[:2] != (batch_size, num_samples):
            raise ValueError("Info and action candidate axes do not match.")

        history = self.preprocess_pixels(pixels[:, 0])
        if history.shape[1] != self.num_history_frames:
            raise ValueError(
                f"DriftWorld requires {self.num_history_frames} history frames, "
                f"got {history.shape[1]}."
            )
        goal_pixels = self.preprocess_pixels(goal[:, 0, -1])
        model_actions = self.prepare_model_actions(info_dict, action_candidates)
        costs = []
        for sample_index in range(self.num_rollout_samples):
            terminal = self._rollout_once(
                history,
                model_actions,
                self.rollout_seed + sample_index,
            ).clamp(-1.0, 1.0)
            costs.append(
                (terminal - goal_pixels[:, None]).pow(2).mean(dim=(-3, -2, -1))
            )
        return torch.stack(costs).mean(dim=0)


class DriftWorldCubeWorldModel(DriftWorldWorldModel):
    """Robomimic-backbone DriftWorld adapter for OGBench Cube's 5-D actions."""

    action_dim = 5

    def __init__(
        self,
        *,
        denoiser=None,
        unet_type: str = "multi",
        temp_list=(0.02, 0.05, 0.2),
        n_neg: int = 32,
        num_future_frames: int = 1,
        num_history_frames: int = 3,
        decay: float = 0.999,
        action_frameskip: int = 5,
        action_mode: str = "relative",
        rollout_batch_size: int = 16,
        num_rollout_samples: int = 1,
        rollout_seed: int = 0,
        action_processor: DriftWorldActionProcessor | None = None,
    ) -> None:
        if action_mode != "relative":
            raise ValueError("OGBench Cube DriftWorld requires relative actions.")
        denoiser = denoiser or create_robomimic_driftworld_denoiser(
            unet_type=unet_type,
            temp_list=temp_list,
            n_neg=n_neg,
            num_future_frames=num_future_frames,
            num_history_frames=num_history_frames,
            decay=decay,
            action_dim=self.action_dim,
            action_frameskip=action_frameskip,
        )
        super().__init__(
            denoiser=denoiser,
            temp_list=temp_list,
            n_neg=n_neg,
            num_future_frames=num_future_frames,
            num_history_frames=num_history_frames,
            decay=decay,
            action_frameskip=action_frameskip,
            action_mode=action_mode,
            rollout_batch_size=rollout_batch_size,
            num_rollout_samples=num_rollout_samples,
            rollout_seed=rollout_seed,
            action_processor=action_processor,
        )
