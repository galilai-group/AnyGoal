"""Read-only DriftWorld adapters for Stable WorldModel training and planning."""

from __future__ import annotations

import copy
import importlib
import importlib.machinery
import os
import sys
import types
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from stable_worldmodel.policy import WorldModelPolicy
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms


def driftworld_source_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "worldmodels" / "driftworld"
    root = Path(os.environ.get("DRIFTWORLD_SOURCE_ROOT", default)).expanduser()
    root = root.resolve() / "pusht_driftworld"
    if not (root / "drifting_denoiser_multi.py").is_file():
        raise FileNotFoundError(
            "DriftWorld Push-T source was not found. Set DRIFTWORLD_SOURCE_ROOT "
            f"to the vendored driftworld directory: {root.parent}"
        )
    return root


def robomimic_driftworld_source_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "worldmodels" / "driftworld"
    root = Path(os.environ.get("DRIFTWORLD_SOURCE_ROOT", default)).expanduser()
    root = root.resolve() / "robomimic_driftworld"
    if not (root / "drifting_denoiser_multi.py").is_file():
        raise FileNotFoundError(
            "DriftWorld Robomimic source was not found. Set DRIFTWORLD_SOURCE_ROOT "
            f"to the vendored driftworld directory: {root.parent}"
        )
    return root


def _module_is_below(module, root: Path) -> bool:
    path = getattr(module, "__file__", None)
    return path is not None and Path(path).resolve().is_relative_to(root)


@lru_cache(maxsize=1)
def get_driftworld_modules():
    """Load the flat-import upstream modules without editing their source tree."""
    root = driftworld_source_root()
    for name in ("data", "unet_multi", "drifting"):
        existing = sys.modules.get(name)
        if existing is not None and not _module_is_below(existing, root):
            raise ImportError(
                f"Top-level module {name!r} is already loaded from another package; "
                "start a clean process before loading DriftWorld."
            )

    if "data" not in sys.modules:
        # The denoiser imports Batch only as a type annotation. Loading the
        # upstream data package would eagerly import its unused Zarr reader,
        # so expose the minimal package contract needed by the model.
        data_package = types.ModuleType("data")
        data_package.__file__ = str(root / "data" / "__init__.py")
        data_package.__path__ = [str(root / "data")]
        data_package.__package__ = "data"
        data_package.__spec__ = importlib.machinery.ModuleSpec(
            "data", loader=None, is_package=True
        )
        data_package.Batch = dict
        sys.modules["data"] = data_package

    sys.path.insert(0, str(root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        denoiser_module = importlib.import_module("drifting_denoiser_multi")
        unet_configs = importlib.import_module("unet_multi.unet_configs")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(root))
    return denoiser_module, unet_configs


@lru_cache(maxsize=1)
def get_robomimic_driftworld_modules():
    """Load the one-view Robomimic model without editing vendored sources."""
    root = robomimic_driftworld_source_root()
    for name in ("unet", "drifting", "drifting_denoiser_multi"):
        existing = sys.modules.get(name)
        if existing is not None and not _module_is_below(existing, root):
            raise ImportError(
                f"Top-level module {name!r} is already loaded from another package; "
                "start a clean process before loading Robomimic DriftWorld."
            )

    sys.path.insert(0, str(root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        denoiser_module = importlib.import_module("drifting_denoiser_multi")
        unet_configs = importlib.import_module("unet.unet_configs")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(root))
    return denoiser_module, unet_configs


def create_driftworld_denoiser(
    *,
    unet_name: str,
    temp_list,
    n_neg: int,
    num_future_frames: int,
    num_history_frames: int,
    decay: float,
    action_frameskip: int = 1,
    device: torch.device | str | None = None,
):
    """Construct the upstream denoiser, widening only its action input for skip > 1.

    ``action_frameskip=1`` calls the original constructor with the original U-Net
    registry entry. For larger skips, the original ``UNet_PushT`` builder is used
    and only its first action projection is replaced to accept concatenated 2-D
    actions, matching StableWM/LeWM action blocking.
    """
    if action_frameskip < 1:
        raise ValueError("action_frameskip must be positive.")
    denoiser_module, unet_configs = get_driftworld_modules()
    if unet_name != "UNet_PushT":
        raise ValueError(f"Only the upstream Push-T U-Net is supported, got {unet_name!r}.")

    if action_frameskip == 1:
        denoiser = denoiser_module.Denoiser(
            unet_name=unet_name,
            temp_list=temp_list,
            n_neg=n_neg,
            num_future_frames=num_future_frames,
            num_history_frames=num_history_frames,
            decay=decay,
        )
    else:
        registry_key = f"_UNet_PushT_frameskip_{action_frameskip}"

        def build_frameskip_unet(num_history=4):
            model = unet_configs.UNet_PushT(num_history=num_history)
            original_projection = model.action_embed[0]
            model.action_in = 2 * action_frameskip
            model.action_embed[0] = nn.Linear(
                model.action_in,
                original_projection.out_features,
                bias=original_projection.bias is not None,
            )
            return model

        unet_configs.UNet_model_dict[registry_key] = build_frameskip_unet
        try:
            denoiser = denoiser_module.Denoiser(
                unet_name=registry_key,
                temp_list=temp_list,
                n_neg=n_neg,
                num_future_frames=num_future_frames,
                num_history_frames=num_history_frames,
                decay=decay,
            )
        finally:
            unet_configs.UNet_model_dict.pop(registry_key, None)

    return denoiser.to(device) if device is not None else denoiser


def create_robomimic_driftworld_denoiser(
    *,
    unet_type: str,
    temp_list,
    n_neg: int,
    num_future_frames: int,
    num_history_frames: int,
    decay: float,
    action_dim: int = 5,
    action_frameskip: int = 5,
    device: torch.device | str | None = None,
):
    """Construct the upstream one-view Robomimic U-Net for blocked Cube actions."""
    if unet_type != "multi":
        raise ValueError("OGBench Cube currently supports Robomimic's one-view U-Net only.")
    if action_dim < 1 or action_frameskip < 1:
        raise ValueError("action_dim and action_frameskip must be positive.")
    denoiser_module, _ = get_robomimic_driftworld_modules()
    denoiser = denoiser_module.Denoiser(
        unet_type=unet_type,
        temp_list=temp_list,
        n_neg=n_neg,
        num_future_frames=num_future_frames,
        num_history_frames=num_history_frames,
        decay=decay,
    )
    model_action_dim = int(action_dim) * int(action_frameskip)
    original_projection = denoiser.inner_model.action_embed[0]
    denoiser.inner_model.action_in = model_action_dim
    denoiser.inner_model.action_embed[0] = nn.Linear(
        model_action_dim,
        original_projection.out_features,
        bias=original_projection.bias is not None,
    )
    denoiser.ema_model = copy.deepcopy(denoiser.inner_model)
    denoiser.ema_model.requires_grad_(False)
    return denoiser.to(device) if device is not None else denoiser


class DriftWorldActionProcessor:
    """Standardize CEM actions while keeping DriftWorld and the env in raw units."""

    def __init__(self, mean=None, scale=None) -> None:
        mean = np.zeros(2, dtype=np.float32) if mean is None else mean
        scale = np.ones(2, dtype=np.float32) if scale is None else scale
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != self.scale.shape or self.mean.ndim != 1:
            raise ValueError("Action mean and scale must be same-sized 1-D arrays.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Action mean and scale must be finite.")
        if np.any(self.scale <= 0):
            raise ValueError("Action scale must be positive.")

    @classmethod
    def fit(cls, values) -> "DriftWorldActionProcessor":
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("Action statistics require a 2-D array.")
        valid = values[~np.isnan(values).any(axis=1)]
        if not len(valid):
            raise ValueError("Cannot fit action statistics without non-NaN rows.")
        scaler = preprocessing.StandardScaler().fit(valid)
        return cls(mean=scaler.mean_, scale=scaler.scale_)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / self.scale

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return values * self.scale + self.mean

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
        return values * scale + mean


class DriftWorldHDF5Dataset(torch.utils.data.Dataset):
    """Expose a StableWM RGB/2-D-action H5 file in the PushT batch format."""

    def __init__(
        self,
        *,
        name: str,
        cache_dir: str | Path,
        pred_horizon: int,
        frameskip: int,
        resize_scale: int = 96,
        num_demos: int | None = None,
        max_samples: int | None = None,
        action_dim: int = 2,
        normalize_img: bool = False,
    ) -> None:
        if pred_horizon < 1 or frameskip < 1:
            raise ValueError("pred_horizon and frameskip must be positive.")
        self.dataset = swm.data.HDF5Dataset(
            name,
            cache_dir=Path(cache_dir).expanduser(),
            frameskip=int(frameskip),
            num_steps=int(pred_horizon),
            keys_to_load=["pixels", "action"],
            keys_to_cache=["action"],
        )
        if self.dataset.get_dim("action") != int(action_dim):
            raise ValueError(
                f"This DriftWorld adapter requires a raw {int(action_dim)}-D action "
                f"column, got {self.dataset.get_dim('action')}."
            )

        episode_count = len(self.dataset.lengths)
        self.num_demos = episode_count if num_demos is None else min(
            int(num_demos), episode_count
        )
        span = int(pred_horizon) * int(frameskip)
        sample_count = sum(
            max(0, int(length) - span + 1)
            for length in self.dataset.lengths[: self.num_demos]
        )
        if max_samples is not None:
            if int(max_samples) < 1:
                raise ValueError("data.max_samples must be positive when set.")
            sample_count = min(sample_count, int(max_samples))
        self.sample_count = sample_count
        self.resize_scale = int(resize_scale)
        self.normalize_img = bool(normalize_img)
        self.image_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.Resize(self.resize_scale, antialias=True),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample = self.dataset[index]
        images = torch.stack([self.image_transform(frame) for frame in sample["pixels"]])
        if self.normalize_img:
            images = images.mul(2.0).sub(1.0)
        episode_id, _ = self.dataset.clip_indices[index]
        return {
            "image": images,
            "action": sample["action"].float(),
            "id": int(episode_id),
        }


def get_driftworld_hdf5_loader(cfg, rank: int = 0, world_size: int = 1):
    """Drop-in replacement for upstream ``get_pushT_loader``."""
    max_samples = cfg.data.get("max_samples")
    dataset = DriftWorldHDF5Dataset(
        name=str(cfg.data.dataset_name),
        cache_dir=cfg.data.dataset_root,
        pred_horizon=int(cfg.data.pred_horizon),
        frameskip=int(cfg.data.frameskip),
        resize_scale=int(cfg.data.resize_scale),
        num_demos=int(cfg.data.num_train_demos),
        max_samples=None if max_samples is None else int(max_samples),
    )
    workers = int(cfg.dataloader.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.data.batch_size),
        "num_workers": workers,
        "pin_memory": bool(cfg.dataloader.pin_memory),
    }
    if workers:
        kwargs["prefetch_factor"] = int(cfg.dataloader.prefetch_factor)
        kwargs["persistent_workers"] = True

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        return torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
    return torch.utils.data.DataLoader(dataset, shuffle=True, **kwargs)


class DriftWorldCubeHDF5Dataset(DriftWorldHDF5Dataset):
    """Expose StableWM Cube clips in Robomimic DriftWorld's batch format."""

    def __init__(self, **kwargs) -> None:
        super().__init__(action_dim=5, normalize_img=True, **kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        return {
            "obs": {"agentview_image": sample["image"]},
            "actions": sample["action"],
            "id": sample["id"],
        }


def get_driftworld_cube_hdf5_loader(cfg, rank: int = 0, world_size: int = 1):
    """Drop-in replacement for Robomimic DriftWorld's HDF5 loader."""
    max_samples = cfg.data.get("max_samples")
    dataset = DriftWorldCubeHDF5Dataset(
        name=str(cfg.data.dataset_name),
        cache_dir=cfg.data.dataset_root,
        pred_horizon=int(cfg.data.pred_horizon),
        frameskip=int(cfg.data.frameskip),
        resize_scale=int(cfg.data.resize_scale),
        num_demos=int(cfg.data.num_train_demos),
        max_samples=None if max_samples is None else int(max_samples),
    )
    workers = int(cfg.dataloader.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.data.batch_size),
        "num_workers": workers,
        "pin_memory": bool(cfg.dataloader.pin_memory),
        "drop_last": True,
    }
    if workers:
        kwargs["prefetch_factor"] = int(cfg.dataloader.prefetch_factor)
        kwargs["persistent_workers"] = True
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
    return torch.utils.data.DataLoader(dataset, shuffle=True, **kwargs)


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
            raise ValueError("frameskip, rollout_batch_size, and samples must be positive.")
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
            proprio[..., -2:]
            if proprio.shape[-1] >= 4
            else torch.zeros_like(position)
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
        current_history = history[:, None].expand(
            batch_size, num_samples, *history.shape[1:]
        ).reshape(flat_count, *history.shape[1:])
        flat_actions = model_actions.reshape(
            flat_count, horizon, self.model_action_dim
        )
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


class DriftWorldWorldModelPolicy(WorldModelPolicy):
    """Accumulate frameskip-aligned visual history for DriftWorld planning."""

    def __init__(
        self,
        *,
        action_processor: DriftWorldActionProcessor | None = None,
        **kwargs,
    ) -> None:
        self.action_processor = action_processor or DriftWorldActionProcessor()
        process = dict(kwargs.pop("process", {}) or {})
        process["action"] = self.action_processor
        super().__init__(process=process, **kwargs)
        model = self.solver.model
        self.action_frameskip = int(model.action_frameskip)
        self.num_history_frames = int(model.num_history_frames)
        if self.cfg.action_block != self.action_frameskip:
            raise ValueError(
                "plan_config.action_block must match DriftWorld's training "
                f"frameskip ({self.action_frameskip})."
            )
        if self.cfg.history_len != self.num_history_frames:
            raise ValueError(
                "plan_config.history_len must match DriftWorld's fixed visual "
                f"history ({self.num_history_frames})."
            )
        self._pixel_history: list[deque[torch.Tensor]] | None = None
        self._pending_actions: list[deque[torch.Tensor]] | None = None

    def set_env(self, env) -> None:
        super().set_env(env)
        self._pixel_history = [
            deque(maxlen=self.num_history_frames) for _ in range(env.num_envs)
        ]
        self._pending_actions = [
            deque(maxlen=self.action_frameskip) for _ in range(env.num_envs)
        ]

    @staticmethod
    def _latest(values: torch.Tensor) -> torch.Tensor:
        return values[:, -1] if values.ndim >= 3 else values

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        if self._pixel_history is None or self._pending_actions is None:
            raise RuntimeError("Set the policy environment before requesting actions.")
        prepared = super()._prepare_info(info_dict)
        if "pixels" not in prepared or "goal" not in prepared:
            raise KeyError("DriftWorld planning requires pixels and goal observations.")
        if "action" not in prepared:
            raise KeyError("DriftWorld history alignment requires the last action.")

        current_pixels = self._latest(prepared["pixels"])
        current_actions = torch.nan_to_num(
            self._latest(prepared["action"]).float(), nan=0.0
        )
        needs_flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(len(current_pixels), dtype=bool))
        ).reshape(len(current_pixels), -1)[:, 0]

        pixel_batches = []
        for index, (pixel, action) in enumerate(zip(current_pixels, current_actions)):
            if needs_flush[index]:
                self._pixel_history[index].clear()
                self._pending_actions[index].clear()
            if not self._pixel_history[index]:
                self._pixel_history[index].append(pixel)
            else:
                pending = self._pending_actions[index]
                pending.append(action)
                if len(pending) == self.action_frameskip:
                    self._pixel_history[index].append(pixel)
                    pending.clear()
            pixels = list(self._pixel_history[index])
            pixels = [pixels[0]] * (self.num_history_frames - len(pixels)) + pixels
            pixel_batches.append(torch.stack(pixels))

        output = {
            "pixels": torch.stack(pixel_batches),
            "goal": prepared["goal"][:, -1:],
        }
        if "proprio" in prepared:
            output["proprio"] = self._latest(prepared["proprio"])
        if "_needs_flush" in info_dict:
            output["_needs_flush"] = torch.as_tensor(needs_flush)
        if "terminated" in prepared:
            output["terminated"] = prepared["terminated"]
        return output


def driftworld_image_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((image_size, image_size), antialias=True),
        ]
    )


__all__ = [
    "DriftWorldActionProcessor",
    "DriftWorldCubeHDF5Dataset",
    "DriftWorldCubeWorldModel",
    "DriftWorldHDF5Dataset",
    "DriftWorldWorldModel",
    "DriftWorldWorldModelPolicy",
    "create_driftworld_denoiser",
    "create_robomimic_driftworld_denoiser",
    "driftworld_image_transform",
    "driftworld_source_root",
    "get_driftworld_hdf5_loader",
    "get_driftworld_cube_hdf5_loader",
    "robomimic_driftworld_source_root",
]
