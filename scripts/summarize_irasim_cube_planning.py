"""Write raw and seed-averaged CSVs for IRASim Cube planning outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ERRORS = {
    "final_ee_position_error": "ee_position",
    "final_ee_yaw_error": "ee_yaw",
    "final_cube_position_error": "cube_position",
}
TOLERANCES = {
    "ee_position_tolerance": "ee_position",
    "ee_yaw_tolerance": "ee_yaw",
    "cube_position_tolerance": "cube_position",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(statistics.mean(values)), (
        float(statistics.stdev(values)) if len(values) > 1 else 0.0
    )


def rate_and_count(metrics: dict) -> tuple[int, float]:
    successes = metrics["episode_successes"]
    return int(sum(successes)), float(metrics["success_rate"])


def parse_row(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    config = result["config"]
    metrics = result["metrics"]
    output = config["output"]["directory"]
    seed = int(config["seed"]["task"])
    scenario = path.parents[1].name
    lr_label = path.parents[2].name.removeprefix("lr")
    row: dict[str, object] = {
        "lr_label": lr_label,
        "learning_rate": float(lr_label.replace("em", "e-")),
        "scenario": scenario,
        "environment": config["world"]["env_name"],
        "evaluation_dataset": config["eval"]["dataset_name"],
        "seed": seed,
        "num_eval": len(result["episodes"]),
        "checkpoint": result["checkpoint"],
        "output_directory": output,
        "metrics_path": str(path),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }
    for output_name, metric_name in (
        ("original_cube", "original_cube_criterion"),
        ("criterion_3", "cube_hop_criterion_3"),
    ):
        count, rate = rate_and_count(metrics[metric_name])
        row[f"{output_name}_success_count"] = count
        row[f"{output_name}_success_rate"] = rate

    errors = metrics.get("cube_final_errors")
    for output_name, metric_name in ERRORS.items():
        if errors is None:
            row[f"{output_name}_mean"] = ""
            row[f"{output_name}_std"] = ""
        else:
            row[f"{output_name}_mean"], row[f"{output_name}_std"] = mean_std(
                [float(value) for value in errors[metric_name]]
            )

    tolerances = metrics.get("cube_evaluation_tolerances")
    for output_name, metric_name in TOLERANCES.items():
        row[output_name] = "" if tolerances is None else float(tolerances[metric_name])
    return row


def average_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["lr_label"]), str(row["scenario"]))].append(row)

    rows = []
    for (_, _), group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in group]
        if seeds != [0, 1, 2]:
            raise ValueError(f"Expected seeds 0,1,2, found {seeds} for {group[0]['scenario']}")
        row: dict[str, object] = {
            "lr_label": group[0]["lr_label"],
            "learning_rate": group[0]["learning_rate"],
            "scenario": group[0]["scenario"],
            "environment": group[0]["environment"],
            "evaluation_dataset": group[0]["evaluation_dataset"],
            "num_seeds": len(group),
            "seeds": ";".join(map(str, seeds)),
            "num_eval_per_seed": group[0]["num_eval"],
            "total_evaluations": sum(int(item["num_eval"]) for item in group),
        }
        for field in (
            "original_cube_success_rate",
            "criterion_3_success_rate",
            *(f"{name}_mean" for name in ERRORS),
            "elapsed_seconds",
        ):
            values = [item[field] for item in group]
            if any(value == "" for value in values):
                row[f"{field}_mean"] = ""
                row[f"{field}_std"] = ""
            else:
                row[f"{field}_mean"], row[f"{field}_std"] = mean_std(
                    [float(value) for value in values]
                )
        for field in (
            "original_cube_success_count",
            "criterion_3_success_count",
        ):
            values = [item[field] for item in group]
            row[f"{field}_total"] = "" if any(value == "" for value in values) else sum(values)
        for output_name in TOLERANCES:
            row[output_name] = group[0][output_name]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("lr*/**/seed*/metrics.json"))
    if len(paths) != 36:
        raise ValueError(f"Expected 36 metrics files below {args.root}, found {len(paths)}")
    raw_rows = [parse_row(path) for path in paths]
    raw_rows.sort(key=lambda row: (str(row["lr_label"]), str(row["scenario"]), int(row["seed"])))
    seed_rows = average_rows(raw_rows)
    if len(seed_rows) != 12:
        raise ValueError(f"Expected 12 seed-average rows, found {len(seed_rows)}")
    raw_path = args.root / "irasim_cube_planning_raw.csv"
    average_path = args.root / "irasim_cube_planning_seed_average.csv"
    write_csv(raw_path, raw_rows)
    write_csv(average_path, seed_rows)
    print(f"wrote {len(raw_rows)} raw rows to {raw_path}")
    print(f"wrote {len(seed_rows)} seed-average rows to {average_path}")


if __name__ == "__main__":
    main()
