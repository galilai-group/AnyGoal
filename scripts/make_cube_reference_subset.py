"""Copy complete leading episodes from a Cube HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  Register the source dataset's Blosc filter.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.input, "r") as source:
        if args.episodes <= 0 or args.episodes > len(source["ep_len"]):
            raise ValueError("episodes is outside the source episode range")
        row_count = int(source["ep_offset"][args.episodes - 1]) + int(
            source["ep_len"][args.episodes - 1]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as target:
            for key, dataset in source.items():
                if key in {"ep_len", "ep_offset"}:
                    values = dataset[: args.episodes]
                else:
                    values = dataset[:row_count]
                target.create_dataset(key, data=values, compression="gzip")
            for key, value in source.attrs.items():
                target.attrs[key] = value
    print(f"Copied {args.episodes} episodes ({row_count} rows) to {args.output}")


if __name__ == "__main__":
    main()
