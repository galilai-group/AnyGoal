"""Merge four balanced PushT-Antigoal-v3 corner datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets import PushTTrainSchemaWriter
from anygoal.envs.push_t_antigoal import ANTIGOAL_V3_POSITION_ARRAY
from anygoal.paths import data_root


DEFAULT_INPUTS = tuple(
    data_root() / f"push_t_antigoal_v3_{index}.h5"
    for index in range(4)
)
DEFAULT_OUTPUT = data_root() / "pusht_antigoal_test.h5"
COLUMNS = ("action", "pixels", "proprio", "state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) != 4:
        raise ValueError("Exactly four corner datasets are required")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )
    corner_counts = []
    resolution = None
    try:
        for input_path in args.inputs:
            with h5py.File(input_path, "r") as dataset:
                input_resolution = int(dataset["pixels"].shape[1])
                if resolution is None:
                    resolution = input_resolution
                elif input_resolution != resolution:
                    raise ValueError("Input pixel resolutions do not match")
                corner_counts.append(len(dataset["ep_len"]))
        if len(set(corner_counts)) != 1:
            raise ValueError(f"Corner episode counts are unbalanced: {corner_counts}")

        assert resolution is not None
        writer = PushTTrainSchemaWriter(temporary_output, resolution)
        with writer:
            for input_path in args.inputs:
                with h5py.File(input_path, "r") as dataset:
                    for offset, length in zip(
                        dataset["ep_offset"][:], dataset["ep_len"][:]
                    ):
                        episode_slice = slice(int(offset), int(offset + length))
                        episode = {
                            column: dataset[column][episode_slice]
                            for column in COLUMNS
                        }
                        writer.write_episode(episode)

        with h5py.File(temporary_output, "a") as dataset:
            dataset.attrs["environment"] = "anygoal/PushTAntigoal-v3"
            dataset.attrs["corner_balance"] = "balanced"
            dataset.attrs["corner_quota_per_goal"] = corner_counts[0]
            dataset.attrs["antigoal_positions"] = ANTIGOAL_V3_POSITION_ARRAY
            dataset.attrs["merged_inputs"] = [str(path.resolve()) for path in args.inputs]
        os.replace(temporary_output, args.output)
    except BaseException:
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    print(f"Merged {sum(corner_counts)} episodes: {args.output}")


if __name__ == "__main__":
    main()
