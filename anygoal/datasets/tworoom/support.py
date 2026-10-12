"""HDF5 schema and writer helpers for generated TwoRoom variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np


@dataclass
class TwoRoomEpisode:
    columns: dict[str, np.ndarray]
    route_id: int
    condition: str
    sampled_magnitude: float

    @property
    def length(self) -> int:
        return int(self.columns["action"].shape[0])


class TwoRoomHDF5Writer:
    """Append variable-length TwoRoom episodes to the canonical schema."""

    PER_STEP_SPECS: ClassVar = {
        "pixels": ((224, 224, 3), np.uint8),
        "action": ((2,), np.float32),
        "proprio": ((2,), np.float32),
        "next_proprio": ((2,), np.float32),
        "observation": ((10,), np.float32),
        "next_observation": ((10,), np.float32),
        "target": ((2,), np.float32),
        "waypoint": ((2,), np.float32),
        "action_magnitude": ((), np.float32),
        "route_id": ((), np.int16),
        "condition_id": ((), np.int8),
        "ep_idx": ((), np.int32),
        "waypoint_index": ((), np.int8),
        "step_idx": ((), np.int32),
        "terminated": ((), np.bool_),
    }

    def __init__(self, path: Path) -> None:
        self.file = h5py.File(path, "w")
        self.datasets = {}
        for key, (tail_shape, dtype) in self.PER_STEP_SPECS.items():
            kwargs = (
                {"compression": "gzip", "compression_opts": 1}
                if key == "pixels"
                else {}
            )
            self.datasets[key] = self.file.create_dataset(
                key,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                chunks=(1, *tail_shape) if key == "pixels" else True,
                dtype=dtype,
                **kwargs,
            )
        self.ep_len = self.file.create_dataset(
            "ep_len", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self.ep_offset = self.file.create_dataset(
            "ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self.num_steps = 0
        self.num_episodes = 0

    def write(self, episode: TwoRoomEpisode) -> None:
        length = episode.length
        for key, dataset in self.datasets.items():
            values = (
                np.full(length, self.num_episodes, dtype=np.int32)
                if key == "ep_idx"
                else episode.columns[key]
            )
            dataset.resize(self.num_steps + length, axis=0)
            dataset[self.num_steps : self.num_steps + length] = values
        self.ep_len.resize(self.num_episodes + 1, axis=0)
        self.ep_offset.resize(self.num_episodes + 1, axis=0)
        self.ep_len[self.num_episodes] = length
        self.ep_offset[self.num_episodes] = self.num_steps
        self.num_steps += length
        self.num_episodes += 1

    def close(self, metadata: dict) -> None:
        for key, value in metadata.items():
            self.file.attrs[key] = value
        self.file.close()


def make_episode(
    columns: dict[str, list],
    route_id: int,
    condition: str,
    sampled_magnitude: float,
) -> TwoRoomEpisode:
    arrays = {
        key: np.asarray(values, dtype=TwoRoomHDF5Writer.PER_STEP_SPECS[key][1])
        for key, values in columns.items()
    }
    return TwoRoomEpisode(arrays, route_id, condition, sampled_magnitude)


def validate_dataset(path: Path, expected_episodes: int) -> None:
    """Validate TwoRoom schema and episode indexing after an atomic write."""
    with h5py.File(path, "r") as dataset:
        expected_keys = set(TwoRoomHDF5Writer.PER_STEP_SPECS) | {
            "ep_len",
            "ep_offset",
        }
        if set(dataset) != expected_keys:
            raise ValueError(
                f"Output schema differs from TwoRoom: "
                f"{sorted(set(dataset) ^ expected_keys)}"
            )
        lengths = dataset["ep_len"][:].astype(np.int64)
        offsets = dataset["ep_offset"][:].astype(np.int64)
        if len(lengths) != expected_episodes:
            raise ValueError(
                f"Expected {expected_episodes} episodes, found {len(lengths)}"
            )
        expected_offsets = np.r_[0, np.cumsum(lengths)[:-1]]
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("TwoRoom episode offsets are incorrect")
        if int(lengths.sum()) != len(dataset["action"]):
            raise ValueError("TwoRoom episode lengths do not cover all action rows")
        expected_indices = np.repeat(
            np.arange(expected_episodes, dtype=np.int32), lengths
        )
        if not np.array_equal(dataset["ep_idx"][:], expected_indices):
            raise ValueError("TwoRoom per-step episode indices are incorrect")
        terminal = offsets + lengths - 1
        if not dataset["terminated"][terminal].all():
            raise ValueError("Every TwoRoom episode must end in termination")


__all__ = [
    "TwoRoomEpisode",
    "TwoRoomHDF5Writer",
    "make_episode",
    "validate_dataset",
]
