"""Merge completed IRASim prediction shards with saved base progress."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval import write_results


def sampling_digest(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def merge(root: Path) -> None:
    base_progress = json.loads((root / "progress.json").read_text())
    shard_results = [
        json.loads(path.read_text())
        for path in sorted((root / "shards").glob("shard_*/metrics.json"))
    ]
    if not shard_results:
        raise FileNotFoundError(f"No completed shards under {root / 'shards'}")

    episodes = list(base_progress["episodes"])
    for result in sorted(
        shard_results, key=lambda value: value["sampling"]["episode_start_position"]
    ):
        episodes.extend(result["episodes"])
    if len(episodes) != 100:
        raise ValueError(f"Expected 100 merged episodes under {root}; found {len(episodes)}")
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"Duplicate episodes while merging {root}")

    result = shard_results[0]
    step_means = np.asarray([episode["step_errors"] for episode in episodes])
    counts = np.asarray([episode["num_starting_points"] for episode in episodes])
    macro = step_means.mean(axis=0)
    micro = (step_means * counts[:, None]).sum(axis=0) / counts.sum()
    frame_skip = int(result["protocol"]["frame_skip"])
    horizons = result["protocol"]["horizons"]
    result["short_horizon_error"] = float(macro[0])
    result["long_horizon_error"] = {
        str(horizon): float(macro[: horizon // frame_skip].sum())
        for horizon in horizons
    }
    result["metrics"]["short_horizon"] = {
        "horizon_environment_steps": frame_skip,
        "prediction_steps": 1,
        "error": float(macro[0]),
        "episode_std": float(step_means[:, 0].std(ddof=1)),
        "start_weighted_error": float(micro[0]),
    }
    result["metrics"]["autoregressive_step_errors"] = macro.tolist()
    result["metrics"]["long_horizon"] = {}
    for horizon in horizons:
        steps = horizon // frame_skip
        result["metrics"]["long_horizon"][str(horizon)] = {
            "prediction_steps": steps,
            "cumulative_error": float(macro[:steps].sum()),
            "mean_step_error": float(macro[:steps].mean()),
            "terminal_error": float(macro[steps - 1]),
            "start_weighted_cumulative_error": float(micro[:steps].sum()),
        }
    result["episodes"] = episodes
    result["sampling"].update({
        "selected_episode_indices": episode_indices,
        "selected_episode_indices_sha256": sampling_digest(episode_indices),
        "num_starting_points": int(counts.sum()),
        "episode_start_position": 0,
        "episode_end_position": 100,
    })
    write_results(result, root, overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        merge(root)


if __name__ == "__main__":
    main()
