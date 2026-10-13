"""Tiny canonical HDF5 fixtures used by fast contract tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_hdf5(
    path: Path,
    *,
    environment: str | None = None,
    dataset_id: str | None = None,
    length: int = 30,
) -> Path:
    """Write the common fields needed to validate and resolve a dataset."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((length, 2), dtype=np.float32))
        handle.create_dataset(
            "pixels", data=np.zeros((length, 4, 4, 3), dtype=np.uint8)
        )
        handle.create_dataset("ep_len", data=np.array([length], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        if environment is not None:
            handle.attrs["environment"] = environment
        if dataset_id is not None:
            handle.attrs["anygoal_dataset_id"] = dataset_id
    return path
