"""StableWM HDF5 bridges for DriftWorld training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import stable_worldmodel as swm
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms


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
        dataset_path = Path(name).expanduser()
        location = (
            {"path": dataset_path}
            if dataset_path.is_absolute() or dataset_path.suffix in {".h5", ".hdf5"}
            else {"name": name, "cache_dir": Path(cache_dir).expanduser()}
        )
        self.dataset = swm.data.HDF5Dataset(
            **location,
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
        self.num_demos = (
            episode_count if num_demos is None else min(int(num_demos), episode_count)
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
        images = torch.stack(
            [self.image_transform(frame) for frame in sample["pixels"]]
        )
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


def driftworld_image_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((image_size, image_size), antialias=True),
        ]
    )
