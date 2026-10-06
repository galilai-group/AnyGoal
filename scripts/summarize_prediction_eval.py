"""Combine completed prediction-evaluation runs into JSON and CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev


def summary_row(result: dict) -> dict:
    return {
        "model_name": result["model_name"],
        "environment": result["environment"],
        "dataset_name": result["dataset_name"],
        "base_environment": result["base_environment"],
        "variant_identifier": result["variant_identifier"],
        "is_base": result["is_base"],
        "seed": result["seed"],
        "error_space": result["error_space"],
        "num_episodes": result["protocol"]["num_episodes"],
        "num_starting_points": result["sampling"]["num_starting_points"],
        "short_horizon_error": result["short_horizon_error"],
        "long_horizon_error_h15": result["long_horizon_error"]["15"],
        "long_horizon_error_h25": result["long_horizon_error"]["25"],
        "checkpoint": result["checkpoint"],
        "metrics_path": result["metrics_path"],
    }


def average_row(rows: list[dict]) -> dict:
    first = rows[0]
    metrics = (
        "num_episodes",
        "num_starting_points",
        "short_horizon_error",
        "long_horizon_error_h15",
        "long_horizon_error_h25",
    )
    row = {
        key: first[key]
        for key in (
            "model_name",
            "environment",
            "dataset_name",
            "base_environment",
            "variant_identifier",
            "is_base",
            "error_space",
            "checkpoint",
        )
    }
    row["seed_count"] = len(rows)
    row["seeds"] = ",".join(str(item["seed"]) for item in rows)
    for metric in metrics:
        values = [
            float(item[metric])
            for item in rows
            if item[metric] is not None
        ]
        row[f"{metric}_mean"] = fmean(values) if values else None
        row[f"{metric}_std"] = pstdev(values) if values else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/prediction_eval"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metrics_paths = sorted(
        set(args.root.glob("*/seed_*/metrics.json"))
        | set(args.root.glob("*/*/seed_*/metrics.json"))
    )
    if not metrics_paths:
        raise FileNotFoundError(f"No prediction metrics found below {args.root}.")
    results = []
    for path in metrics_paths:
        result = json.loads(path.read_text())
        result["metrics_path"] = str(path)
        results.append(result)
    rows = sorted(
        (summary_row(result) for result in results),
        key=lambda row: (row["model_name"], row["environment"], row["seed"]),
    )

    grouped_rows: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped_rows.setdefault((row["model_name"], row["environment"]), []).append(row)
    averages = [
        average_row(group)
        for _, group in sorted(grouped_rows.items())
    ]

    raw_csv_path = args.root / "raw.csv"
    average_csv_path = args.root / "seed_average.csv"
    json_path = args.root / "raw.json"
    existing = [
        path for path in (raw_csv_path, average_csv_path, json_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite prediction summaries: "
            + ", ".join(str(path) for path in existing)
        )
    with raw_csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with average_csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(averages[0]))
        writer.writeheader()
        writer.writerows(averages)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"combined {len(rows)} runs")
    print(f"wrote {raw_csv_path}")
    print(f"wrote {average_csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
