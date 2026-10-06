"""Estimate 224x224 RGB storage and rendering time for the trimmed dataset."""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


REFERENCES = [
    Path.home() / ".stable_worldmodel/datasets/cube_hop_test.h5",
    Path.home() / ".stable_worldmodel/datasets/pusht_mini_expert_train.h5",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.dataset.expanduser(), "r") as dataset:
        frames = int(dataset["ep_len"][:].sum())
    bytes_per_frame = 224 * 224 * 3
    raw_bytes = frames * bytes_per_frame
    references = []
    ratios = []
    for path in REFERENCES:
        with h5py.File(path, "r") as dataset:
            pixels = dataset["pixels"]
            logical_bytes = int(np.prod(pixels.shape, dtype=np.int64) * pixels.dtype.itemsize)
            storage_bytes = int(pixels.id.get_storage_size())
            ratio = logical_bytes / storage_bytes
            ratios.append(ratio)
            references.append({"path": str(path), "shape": list(pixels.shape), "logical_bytes": logical_bytes, "storage_bytes": storage_bytes, "compression_ratio": ratio})
    # Job 3143081 produced 1,000 RGB renders plus 10,000 physics steps in 38 s.
    conservative_seconds_per_frame = 38.0 / 1000.0
    render_seconds = frames * conservative_seconds_per_frame
    report = {
        "frames": frames,
        "rgb_shape": [224, 224, 3],
        "bytes_per_uncompressed_frame": bytes_per_frame,
        "uncompressed_bytes": raw_bytes,
        "uncompressed_gib": raw_bytes / 2**30,
        "reference_hdf5_pixel_datasets": references,
        "estimated_compressed_gib_range": [raw_bytes / max(ratios) / 2**30, raw_bytes / min(ratios) / 2**30],
        "timing_basis": "Slurm job 3143081: 1,000 renders plus 10,000 physics steps in 38 seconds using headless EGL",
        "estimated_single_process_render_hours": render_seconds / 3600,
        "estimated_single_process_render_days": render_seconds / 86400,
        "timing_caveat": "Does not include full HDF5 compression/write overhead; exact original RGB requires restorable simulator states or original images.",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
