"""StableWM HDF5 bridges for IRASim training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


class IRASimHDF5Dataset(torch.utils.data.Dataset):
    """Expose StableWM H5 clips in the original IRASim batch format."""

    action_dim = 2

    def __init__(
        self,
        *,
        name: str,
        cache_dir: str | Path,
        num_frames: int,
        frameskip: int,
        video_size: tuple[int, int] | list[int],
        action_scale=(20.0, 20.0),
        action_dim: int = 2,
        episode_indices: np.ndarray | list[int] | None = None,
        max_samples: int | None = None,
    ) -> None:
        if num_frames < 2 or frameskip < 1:
            raise ValueError("num_frames must be at least 2 and frameskip positive.")
        dataset_path = Path(name).expanduser()
        location = (
            {"path": dataset_path}
            if dataset_path.is_absolute() or dataset_path.suffix in {".h5", ".hdf5"}
            else {"name": name, "cache_dir": Path(cache_dir).expanduser()}
        )
        self.dataset = swm.data.HDF5Dataset(
            **location,
            frameskip=int(frameskip),
            num_steps=int(num_frames),
            keys_to_load=["pixels", "action"],
            keys_to_cache=["action"],
        )
        self.action_dim = int(action_dim)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive.")
        if self.dataset.get_dim("action") != self.action_dim:
            raise ValueError(
                f"IRASim expected {self.action_dim}-D actions, got "
                f"{self.dataset.get_dim('action')}."
            )
        self.num_frames = int(num_frames)
        self.frameskip = int(frameskip)
        self.video_size = tuple(int(value) for value in video_size)
        if len(self.video_size) != 2 or any(value < 8 for value in self.video_size):
            raise ValueError("video_size must contain two dimensions of at least 8.")
        action_scale = np.asarray(action_scale, dtype=np.float32)
        if action_scale.shape != (self.action_dim,):
            raise ValueError(f"action_scale must have shape ({self.action_dim},).")
        self.action_scale = torch.from_numpy(np.tile(action_scale, self.frameskip))

        allowed_episodes = (
            None
            if episode_indices is None
            else set(np.asarray(episode_indices, dtype=np.int64).tolist())
        )
        self.sample_indices = [
            index
            for index, (episode, _) in enumerate(self.dataset.clip_indices)
            if allowed_episodes is None or episode in allowed_episodes
        ]
        if max_samples is not None:
            if int(max_samples) < 1:
                raise ValueError("max_samples must be positive when set.")
            self.sample_indices = self.sample_indices[: int(max_samples)]
        selected_episodes = (
            range(len(self.dataset.lengths))
            if allowed_episodes is None
            else sorted(allowed_episodes)
        )
        self.ann_files = [f"h5_episode_{episode}" for episode in selected_episodes]
        self.samples = self.sample_indices
        self.training = False

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(
        self,
        index: int,
        cam_id: int | None = None,
        return_video: bool = False,
    ) -> dict[str, Any]:
        del cam_id, return_video
        source_index = self.sample_indices[index]
        sample = self.dataset[source_index]
        video = sample["pixels"]
        if video.dtype != torch.uint8:
            raise TypeError(f"IRASim expects uint8 H5 pixels, got {video.dtype}.")
        video = F.interpolate(
            video.float().div(255.0),
            size=self.video_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        video = video.sub(0.5).div(0.5)
        # HDF5Dataset groups dense actions into one flat block per strided
        # observation. The last observation has no outgoing transition.
        actions = sample["action"][:-1].float() * self.action_scale
        episode, start = self.dataset.clip_indices[source_index]
        return {
            "video": video,
            "action": actions,
            "video_name": {
                "episode_id": str(episode),
                "start_frame_id": str(start),
                "cam_id": "0",
            },
        }


def get_irasim_hdf5_datasets(args):
    """Drop-in replacement for the upstream IRASim ``get_dataset``."""
    if bool(args.pre_encode):
        raise ValueError(
            "StableWM H5 stores RGB frames, so IRASim H5 training requires "
            "pre_encode=false and uses the original online VAE path."
        )
    validation_fraction = float(args.get("validation_fraction", 0.05))
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    probe_name = Path(str(args.h5_dataset_name)).expanduser()
    probe_location = (
        {"path": probe_name}
        if probe_name.is_absolute() or probe_name.suffix in {".h5", ".hdf5"}
        else {
            "name": str(args.h5_dataset_name),
            "cache_dir": Path(args.h5_dataset_root).expanduser(),
        }
    )
    probe = swm.data.HDF5Dataset(
        **probe_location,
        keys_to_load=["action"],
        keys_to_cache=["action"],
    )
    episode_count = len(probe.lengths)
    generator = np.random.default_rng(int(args.global_seed))
    shuffled = generator.permutation(episode_count)
    validation_count = max(1, round(episode_count * validation_fraction))
    validation_episodes = np.sort(shuffled[:validation_count])
    training_episodes = np.sort(shuffled[validation_count:])
    if not len(training_episodes):
        raise ValueError("The H5 file needs at least two episodes for a split.")

    common = {
        "name": str(args.h5_dataset_name),
        "cache_dir": args.h5_dataset_root,
        "num_frames": int(args.num_frames),
        "frameskip": int(args.frameskip),
        "video_size": tuple(args.video_size),
        "action_scale": tuple(args.action_scale),
        "action_dim": int(args.get("action_dim", 2)),
    }
    max_samples = args.get("max_samples")
    max_validation_samples = args.get("max_validation_samples", max_samples)
    train = IRASimHDF5Dataset(
        **common,
        episode_indices=training_episodes,
        max_samples=None if max_samples is None else int(max_samples),
    )
    validation = IRASimHDF5Dataset(
        **common,
        episode_indices=validation_episodes,
        max_samples=(
            None if max_validation_samples is None else int(max_validation_samples)
        ),
    )
    return train, validation


def irasim_image_transform(video_size=(224, 224)):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(tuple(video_size), antialias=True),
        ]
    )
