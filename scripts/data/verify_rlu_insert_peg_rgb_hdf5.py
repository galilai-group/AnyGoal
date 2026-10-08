"""Verify the final filtered all-frame RGB Insert-Peg HDF5."""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from stable_worldmodel.data import HDF5Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    path = args.dataset.expanduser()
    with h5py.File(path, "r") as dataset:
        lengths = dataset["ep_len"][:].astype(np.int64)
        offsets = dataset["ep_offset"][:].astype(np.int64)
        total = int(lengths.sum())
        if not dataset.attrs.get("conversion_complete", False):
            raise ValueError("Dataset is marked incomplete")
        if len(lengths) != 1461 or np.any(lengths < 50):
            raise ValueError("Minimum-length episode filter is incorrect")
        np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])
        if dataset["pixels"].shape != (total, 224, 224, 3):
            raise ValueError("Unexpected pixels shape")
        if dataset["pixels"].dtype != np.uint8:
            raise ValueError("Unexpected pixels dtype")
        sample_indices = np.unique(np.linspace(0, total - 1, 100, dtype=np.int64))
        samples = dataset["pixels"][sample_indices]
        if samples.min() == samples.max():
            raise ValueError("Rendered pixel samples are constant")
        for key in dataset:
            if key not in ("ep_len", "ep_offset") and len(dataset[key]) != total:
                raise ValueError(f"Per-step column has wrong length: {key}")
    reader = HDF5Dataset(
        path=path,
        num_steps=2,
        keys_to_load=["pixels", "action", "proprio", "privilege", "ep_idx"],
    )
    first, last = reader.load_episode(0), reader.load_episode(len(lengths) - 1)
    if first["pixels"].shape[1:] != (3, 224, 224):
        raise ValueError("Stable-worldmodel did not load RGB as NCHW")
    if last["action"].shape[-1] != 5:
        raise ValueError("Action dimension changed")
    print(json.dumps({
        "path": str(path.resolve()),
        "episodes": int(len(lengths)),
        "frames": total,
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_median": float(np.median(lengths)),
        "pixels_shape": [total, 224, 224, 3],
        "stable_worldmodel_loader_verified": True,
        "file_bytes": path.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
