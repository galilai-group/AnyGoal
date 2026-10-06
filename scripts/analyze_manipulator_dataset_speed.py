"""Compare per-frame manipulator motion and action magnitudes across datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(path: Path) -> dict:
    with h5py.File(path, "r") as dataset:
        hand = np.asarray(dataset["proprio_hand_pos"][:, :2], dtype=np.float64)
        arm = np.asarray(dataset["proprio_arm_pos"], dtype=np.float64)
        action = np.asarray(dataset["action"], dtype=np.float64)
        starts = np.asarray(dataset["fragment_start"][:], dtype=bool)
        lengths = np.asarray(dataset["ep_len"][:], dtype=np.int64)
    valid = ~starts[1:]
    hand_delta = np.linalg.norm(np.diff(hand, axis=0)[valid], axis=1)
    angles = np.arctan2(arm[:, 1::2], arm[:, 0::2])
    angle_delta = (np.diff(angles, axis=0)[valid] + np.pi) % (2.0 * np.pi) - np.pi
    joint_delta = np.linalg.norm(angle_delta, axis=1)
    arm_joint_delta = np.linalg.norm(angle_delta[:, :4], axis=1)
    action_norm = np.linalg.norm(action, axis=1)

    def stats(values):
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
        }

    return {
        "path": str(path.resolve()),
        "episodes": int(len(lengths)),
        "frames": int(len(action)),
        "episode_length_mean": float(lengths.mean()),
        "hand_displacement_per_frame": stats(hand_delta),
        "all_joint_angle_delta_l2_per_frame": stats(joint_delta),
        "arm_joint_angle_delta_l2_per_frame": stats(arm_joint_delta),
        "action_l2": stats(action_norm),
    }


def main():
    args = parse_args()
    result = {path.stem: summarize(path) for path in args.paths}
    reference = next(iter(result.values()))
    for item in result.values():
        item["relative_to_first"] = {
            key: {
                statistic: item[key][statistic] / reference[key][statistic]
                if reference[key][statistic] else None
                for statistic in ("mean", "median")
            }
            for key in (
                "hand_displacement_per_frame",
                "all_joint_angle_delta_l2_per_frame",
                "arm_joint_angle_delta_l2_per_frame",
                "action_l2",
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
