"""Write raw and seed-averaged CSVs for IRASim Push-T cross-task planning."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(statistics.mean(values)), (
        float(statistics.stdev(values)) if len(values) > 1 else 0.0
    )


def parse_row(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    config = result["config"]
    metrics = result["metrics"]
    successes = metrics["episode_successes"]
    return {
        "task": path.parents[1].name,
        "environment": config["world"]["env_name"],
        "evaluation_dataset": config["eval"]["dataset_name"],
        "seed": int(config["seed"]["task"]),
        "num_eval": len(successes),
        "success_count": int(sum(successes)),
        "success_rate": float(metrics["success_rate"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "checkpoint": result["checkpoint"],
        "output_directory": config["output"]["directory"],
        "metrics_path": str(path),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["task"])].append(row)
    averages = []
    for task, group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in group]
        if seeds != [0, 1, 2]:
            raise ValueError(f"Expected seeds 0,1,2 for {task}, found {seeds}")
        success_rates = [float(row["success_rate"]) for row in group]
        elapsed = [float(row["elapsed_seconds"]) for row in group]
        success_rate_mean, success_rate_std = mean_std(success_rates)
        elapsed_mean, elapsed_std = mean_std(elapsed)
        averages.append(
            {
                "task": task,
                "environment": group[0]["environment"],
                "evaluation_dataset": group[0]["evaluation_dataset"],
                "num_seeds": len(group),
                "seeds": ";".join(map(str, seeds)),
                "num_eval_per_seed": group[0]["num_eval"],
                "total_evaluations": sum(int(row["num_eval"]) for row in group),
                "success_count_total": sum(int(row["success_count"]) for row in group),
                "success_rate_mean": success_rate_mean,
                "success_rate_std": success_rate_std,
                "elapsed_seconds_mean": elapsed_mean,
                "elapsed_seconds_std": elapsed_std,
                "checkpoint": group[0]["checkpoint"],
            }
        )
    return averages


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    metric_paths = sorted(args.root.glob("*/seed*/metrics.json"))
    if len(metric_paths) != 9:
        raise ValueError(f"Expected 9 metrics files below {args.root}, found {len(metric_paths)}")
    rows = [parse_row(path) for path in metric_paths]
    rows.sort(key=lambda row: (str(row["task"]), int(row["seed"])))
    average_rows = summarize(rows)
    if len(average_rows) != 3:
        raise ValueError(f"Expected 3 seed-average rows, found {len(average_rows)}")
    raw_path = args.root / "irasim_pusht_cross_task_planning_raw.csv"
    average_path = args.root / "irasim_pusht_cross_task_planning_seed_average.csv"
    write_csv(raw_path, rows)
    write_csv(average_path, average_rows)
    print(f"wrote {len(rows)} raw rows to {raw_path}")
    print(f"wrote {len(average_rows)} seed-average rows to {average_path}")


if __name__ == "__main__":
    main()
