"""Summarize every completed result in the TwoRoom DriftWorld sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "outputs/driftworld/tworoom_checkpoint_sweep_20260826"
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=RESULT_ROOT.with_name(f"{RESULT_ROOT.name}_completed"),
    )
    args = parser.parse_args()
    selected_steps = set(args.steps) if args.steps else None
    raw_output = Path(f"{args.output_prefix}_raw.csv")
    average_output = Path(f"{args.output_prefix}_seed_average.csv")

    raw_rows = []
    for path in sorted(RESULT_ROOT.glob("*/lr*/step*/seed_*/metrics.json")):
        payload = json.loads(path.read_text())
        config = payload["config"]
        metrics = payload["metrics"]
        successes = metrics["episode_successes"]
        relative = path.relative_to(RESULT_ROOT)
        dataset, learning_rate_dir, step_dir, seed_dir, _ = relative.parts
        checkpoint_step = int(step_dir.removeprefix("step"))
        if selected_steps is not None and checkpoint_step not in selected_steps:
            continue
        scaler = payload["action_scaler"]
        raw_rows.append(
            {
                "learning_rate": learning_rate_dir.removeprefix("lr"),
                "checkpoint_step": checkpoint_step,
                "evaluation_dataset": dataset,
                "environment": config["world"]["env_name"],
                "seed": int(seed_dir.removeprefix("seed_")),
                "task_seed": int(payload["task_seed"]),
                "policy_seed": int(payload["policy_seed"]),
                "rollout_seed": int(config["model"]["rollout_seed"]),
                "num_eval": len(successes),
                "success_count": int(sum(successes)),
                "success_rate": float(metrics["success_rate"]),
                "action_mean": ";".join(str(value) for value in scaler["mean"]),
                "action_scale": ";".join(str(value) for value in scaler["scale"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
                "checkpoint": payload["checkpoint"],
                "result_path": str(path.relative_to(ROOT)),
            }
        )

    raw_rows.sort(
        key=lambda row: (
            row["learning_rate"],
            row["checkpoint_step"],
            row["evaluation_dataset"],
            row["seed"],
        )
    )
    groups = defaultdict(list)
    for row in raw_rows:
        groups[
            (
                row["learning_rate"],
                row["checkpoint_step"],
                row["evaluation_dataset"],
                row["environment"],
                row["checkpoint"],
            )
        ].append(row)

    average_rows = []
    for key, rows in sorted(groups.items()):
        learning_rate, step, dataset, environment, checkpoint = key
        seeds = sorted(row["seed"] for row in rows)
        rates = [row["success_rate"] for row in rows]
        elapsed = [row["elapsed_seconds"] for row in rows]
        average_rows.append(
            {
                "learning_rate": learning_rate,
                "checkpoint_step": step,
                "evaluation_dataset": dataset,
                "environment": environment,
                "checkpoint": checkpoint,
                "num_completed_seeds": len(seeds),
                "completed_seeds": ";".join(str(seed) for seed in seeds),
                "complete_seed_set": seeds == [0, 1, 2],
                "total_evaluations": sum(row["num_eval"] for row in rows),
                "success_count_total": sum(row["success_count"] for row in rows),
                "success_rate_mean": statistics.fmean(rates),
                "success_rate_std": statistics.stdev(rates) if len(rates) > 1 else "",
                "elapsed_seconds_mean": statistics.fmean(elapsed),
                "elapsed_seconds_std": (
                    statistics.stdev(elapsed) if len(elapsed) > 1 else ""
                ),
            }
        )

    write_csv(raw_output, raw_rows)
    write_csv(average_output, average_rows)
    print(f"wrote {len(raw_rows)} raw rows: {raw_output}")
    print(f"wrote {len(average_rows)} grouped rows: {average_output}")


if __name__ == "__main__":
    main()
