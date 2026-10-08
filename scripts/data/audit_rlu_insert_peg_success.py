"""Audit first exact-success transition in each converted Insert-Peg sequence."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.input.expanduser(), "r") as dataset:
        lengths = dataset["ep_len"][:].astype(np.int64)
        offsets = dataset["ep_offset"][:].astype(np.int64)
        trimmed_lengths = lengths.copy()
        successful = np.zeros(len(lengths), dtype=bool)
        maximum_rewards = np.empty(len(lengths), dtype=np.float32)
        for episode, (offset, length) in enumerate(zip(offsets, lengths)):
            rewards = dataset["reward"][offset : offset + length]
            maximum_rewards[episode] = rewards.max()
            success_steps = np.flatnonzero(rewards == np.float32(1.0))
            if len(success_steps):
                successful[episode] = True
                trimmed_lengths[episode] = int(success_steps[0]) + 1
    report = {
        "success_definition": "first transition with reward == float32(1.0), included",
        "episode_count": int(len(lengths)),
        "successful_episodes": int(successful.sum()),
        "unsuccessful_episodes": int((~successful).sum()),
        "maximum_reward_min": float(maximum_rewards.min()),
        "maximum_reward_median": float(np.median(maximum_rewards)),
        "maximum_reward_max": float(maximum_rewards.max()),
        "trimmed_transition_count": int(trimmed_lengths.sum()),
        "removed_transition_count": int(lengths.sum() - trimmed_lengths.sum()),
        "length_min": int(trimmed_lengths.min()),
        "length_max": int(trimmed_lengths.max()),
        "length_mean": float(trimmed_lengths.mean()),
        "length_median": float(np.median(trimmed_lengths)),
        "length_std": float(trimmed_lengths.std()),
        "length_q25": float(np.quantile(trimmed_lengths, 0.25)),
        "length_q75": float(np.quantile(trimmed_lengths, 0.75)),
        "successful_length_min": int(trimmed_lengths[successful].min()),
        "successful_length_max": int(trimmed_lengths[successful].max()),
        "successful_length_mean": float(trimmed_lengths[successful].mean()),
        "successful_length_median": float(np.median(trimmed_lengths[successful])),
        "successful_length_q25": float(np.quantile(trimmed_lengths[successful], 0.25)),
        "successful_length_q75": float(np.quantile(trimmed_lengths[successful], 0.75)),
        "minimum_length": 50,
        "episodes_retained_at_minimum_length": int((trimmed_lengths >= 50).sum()),
        "episodes_dropped_below_minimum_length": int((trimmed_lengths < 50).sum()),
        "retained_transition_count": int(trimmed_lengths[trimmed_lengths >= 50].sum()),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
