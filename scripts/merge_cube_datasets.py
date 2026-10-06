"""Merge complete Cube datasets while preserving the reference HDF5 schema."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  Register the Blosc filter.
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets.cube_collection import (
    CUBE_H5_COLUMNS,
    CubeHDF5Writer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )

    episode_index = 0
    with CubeHDF5Writer(temporary_output) as writer:
        for input_path in args.inputs:
            with h5py.File(input_path, "r") as source:
                for offset, length in zip(
                    source["ep_offset"][:], source["ep_len"][:]
                ):
                    start, stop = int(offset), int(offset + length)
                    episode = {
                        column: source[column][start:stop]
                        for column in CUBE_H5_COLUMNS
                    }
                    episode["ep_idx"] = np.full(
                        int(length), episode_index, dtype=np.int32
                    )
                    writer.write_episode(episode)
                    episode_index += 1
    with h5py.File(temporary_output, "r") as merged:
        lengths = merged["ep_len"][:]
        offsets = merged["ep_offset"][:]
        terminals = offsets + lengths - 1
        if len(lengths) != episode_index or not np.all(lengths == 201):
            raise ValueError("Merged episode count or lengths are invalid")
        if not np.isnan(merged["action"][terminals]).all():
            raise ValueError("Merged terminal actions are not NaN")
        if not merged["truncated"][terminals].all():
            raise ValueError("Merged terminal rows are not truncated")
    os.replace(temporary_output, args.output)
    print(f"Merged {episode_index} episodes into {args.output}", flush=True)


if __name__ == "__main__":
    main()
