"""Copy an RLU Manipulator HDF5, trimming rows after first exact success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DERIVED_KEYS = {"ep_len", "ep_offset", "ep_idx", "step_idx", "fragment_start", "fragment_end"}
ALIASES = {"privilege": "observation", "next_privilege": "next_observation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    source_path, output_path = args.input.expanduser(), args.output.expanduser()
    if output_path.exists():
        raise FileExistsError(output_path)
    with h5py.File(source_path, "r") as source:
        original_lengths = source["ep_len"][:].astype(np.int64)
        original_offsets = source["ep_offset"][:].astype(np.int64)
        lengths = original_lengths.copy()
        success_steps = np.full(len(lengths), -1, dtype=np.int32)
        for episode, (offset, length) in enumerate(zip(original_offsets, original_lengths)):
            successful = np.flatnonzero(source["reward"][offset : offset + length] == np.float32(1.0))
            if len(successful):
                success_steps[episode] = int(successful[0])
                lengths[episode] = int(successful[0]) + 1
        offsets = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
        total = int(lengths.sum())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "x", libver="latest") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            output.attrs["conversion_complete"] = False
            output.attrs["trimmed_after_success"] = True
            output.attrs["success_definition"] = "first reward == float32(1.0); success transition retained"
            output.attrs["trim_source"] = str(source_path.resolve())
            output.attrs["original_transition_count"] = int(original_lengths.sum())
            output.attrs["trimmed_transition_count"] = total
            output.attrs["successful_episode_count"] = int((success_steps >= 0).sum())
            output.create_dataset("ep_len", data=lengths.astype(np.int32))
            output.create_dataset("ep_offset", data=offsets)
            for key, dataset in source.items():
                if key in DERIVED_KEYS or key in ALIASES:
                    continue
                output.create_dataset(
                    key,
                    shape=(total, *dataset.shape[1:]),
                    dtype=dataset.dtype,
                    chunks=(min(4096, total), *dataset.shape[1:]),
                    compression="lzf",
                    shuffle=True,
                )
            for key, target in ALIASES.items():
                output[key] = output[target]
            output.create_dataset("ep_idx", data=np.repeat(np.arange(len(lengths), dtype=np.int32), lengths))
            step_idx = np.arange(total, dtype=np.int64) - offsets[np.repeat(np.arange(len(lengths)), lengths)]
            output.create_dataset("step_idx", data=step_idx.astype(np.int32))
            output.create_dataset("fragment_start", data=step_idx == 0)
            output.create_dataset("fragment_end", data=step_idx == np.repeat(lengths - 1, lengths))
            for episode, (source_offset, target_offset, length) in enumerate(zip(original_offsets, offsets, lengths)):
                source_slice = slice(int(source_offset), int(source_offset + length))
                target_slice = slice(int(target_offset), int(target_offset + length))
                for key in output:
                    if key in DERIVED_KEYS or key in ALIASES:
                        continue
                    output[key][target_slice] = source[key][source_slice]
                if episode % 100 == 0:
                    print(f"copied episode {episode}/{len(lengths)}", flush=True)
            output.attrs["conversion_complete"] = True
            output.flush()
    report = {
        "input": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "episode_count": int(len(lengths)),
        "successful_episodes": int((success_steps >= 0).sum()),
        "unsuccessful_episodes": int((success_steps < 0).sum()),
        "transitions_before": int(original_lengths.sum()),
        "transitions_after": total,
        "transitions_removed": int(original_lengths.sum() - total),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_mean": float(lengths.mean()),
        "length_median": float(np.median(lengths)),
        "length_std": float(lengths.std()),
        "length_q25": float(np.quantile(lengths, 0.25)),
        "length_q75": float(np.quantile(lengths, 0.75)),
    }
    with h5py.File(output_path, "r") as output:
        if int(output["ep_len"][:].sum()) != len(output["action"]):
            raise ValueError("Output episode lengths do not cover all rows")
        if len(output["ep_len"]) != 1500:
            raise ValueError("Episode count changed")
        for episode, (offset, length, success_step) in enumerate(zip(output["ep_offset"][:], output["ep_len"][:], success_steps)):
            rewards = output["reward"][offset : offset + length]
            if success_step >= 0 and (rewards[-1] != np.float32(1.0) or np.any(rewards[:-1] == np.float32(1.0))):
                raise ValueError(f"Incorrect success trim in episode {episode}")
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
