"""Summarize heterogeneous MPC planning outputs into raw and seed-average CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TEXT_SUCCESS_RATE = re.compile(r"'success_rate':\s*([0-9.]+)")
TEXT_DATASET_GOAL_SUCCESS_RATE = re.compile(
    r"'dataset_goal_criterion':\s*\{.*?'success_rate':\s*([0-9.]+)",
    re.DOTALL,
)
TEXT_EVALUATION_TIME = re.compile(r"evaluation_time:\s*([0-9.]+)\s+seconds")


def parse_text_result(path: Path) -> tuple[float, float]:
    content = path.read_text()
    success_rates = TEXT_DATASET_GOAL_SUCCESS_RATE.findall(content)
    if not success_rates:
        success_rates = TEXT_SUCCESS_RATE.findall(content)
    evaluation_times = TEXT_EVALUATION_TIME.findall(content)
    if not success_rates or not evaluation_times:
        raise ValueError(f"Could not parse success rate and evaluation time: {path}")
    return float(success_rates[-1]), float(evaluation_times[-1])


def parse_json_result(path: Path) -> tuple[float, float]:
    result = json.loads(path.read_text())
    evaluation_time = result.get("evaluation_time_seconds", result.get("elapsed_seconds"))
    if evaluation_time is None:
        raise KeyError(f"Missing evaluation time in {path}")
    return float(result["metrics"]["success_rate"]), float(
        evaluation_time
    )


def collect_rows(
    root: Path,
    flat_dataset: str | None = None,
    completed_only: bool = False,
) -> list[dict[str, object]]:
    rows = []
    for model_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        direct_seeds = sorted(model_directory.glob("seed_*"))
        if direct_seeds:
            if flat_dataset is None:
                raise ValueError("--flat-dataset is required for MODEL/seed_N layouts")
            dataset_directories = [(flat_dataset, direct_seeds)]
        else:
            dataset_directories = [
                (path.name, sorted(path.glob("seed_*")))
                for path in sorted(
                    path for path in model_directory.iterdir() if path.is_dir()
                )
            ]
        for dataset, seed_directories in dataset_directories:
            for seed_directory in seed_directories:
                seed = int(seed_directory.name.removeprefix("seed_"))
                json_path = seed_directory / "metrics.json"
                if json_path.is_file():
                    success_rate, evaluation_time = parse_json_result(json_path)
                    source_path = json_path
                    source_format = "json"
                else:
                    result_files = sorted(seed_directory.glob("*results.txt"))
                    if len(result_files) != 1:
                        if completed_only:
                            continue
                        raise ValueError(
                            f"Expected one result file in {seed_directory}, found "
                            f"{len(result_files)}."
                        )
                    success_rate, evaluation_time = parse_text_result(result_files[0])
                    source_path = result_files[0]
                    source_format = "text"
                rows.append(
                    {
                        "model_name": model_directory.name,
                        "dataset": dataset,
                        "seed": seed,
                        "success_rate": success_rate,
                        "evaluation_time_seconds": evaluation_time,
                        "source_format": source_format,
                        "source_path": str(source_path),
                    }
                )
    return sorted(rows, key=lambda row: (row["model_name"], row["dataset"], row["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flat-dataset")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = collect_rows(args.root, args.flat_dataset, args.completed_only)
    if not rows:
        raise FileNotFoundError(f"No planning results found below {args.root}")
    raw_path = args.root / "planning_results_raw.csv"
    average_path = args.root / "planning_results_seed_average.csv"
    existing = [path for path in (raw_path, average_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(map(str, existing))}")

    with raw_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["dataset"])].append(row)
    average_rows = []
    for (model_name, dataset), group in sorted(grouped.items()):
        rates = [float(row["success_rate"]) for row in group]
        times = [float(row["evaluation_time_seconds"]) for row in group]
        average_rows.append(
            {
                "model_name": model_name,
                "dataset": dataset,
                "num_seeds": len(group),
                "success_rate_mean": mean(rates),
                "success_rate_std": stdev(rates) if len(rates) > 1 else 0.0,
                "evaluation_time_seconds_mean": mean(times),
                "evaluation_time_seconds_std": stdev(times) if len(times) > 1 else 0.0,
            }
        )
    with average_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(average_rows[0]))
        writer.writeheader()
        writer.writerows(average_rows)

    print(f"wrote {len(rows)} raw rows to {raw_path}")
    print(f"wrote {len(average_rows)} seed averages to {average_path}")


if __name__ == "__main__":
    main()
