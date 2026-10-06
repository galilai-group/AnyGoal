"""Summarize one complete toy-predictor planning sweep into two CSV files."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=36)
    return parser.parse_args()


def last_match(pattern: str, text: str, source: Path) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"Could not find {pattern!r} in {source}")
    return matches[-1]


def metric_success_rate(metric_name: str, text: str, source: Path) -> float:
    return float(
        last_match(
            rf"'{re.escape(metric_name)}':\s*\{{[^{{}}]*?'success_rate':\s*([0-9.eE+-]+)",
            text,
            source,
        )
    )


def parse_result(result_file: Path, results_root: Path) -> dict:
    relative = result_file.relative_to(results_root)
    if len(relative.parts) != 4:
        raise ValueError(f"Unexpected result path: {result_file}")
    predictor, dataset, seed_directory, _ = relative.parts
    if not seed_directory.startswith("seed_"):
        raise ValueError(f"Unexpected seed directory: {seed_directory}")
    text = result_file.read_text()
    if "'original_cube_criterion'" in text:
        success_criterion = "original_cube"
        original_cube_success_rate = metric_success_rate(
            "original_cube_criterion", text, result_file
        )
        cube_hop_criterion_3_success_rate = metric_success_rate(
            "cube_hop_criterion_3", text, result_file
        )
        success_rate = original_cube_success_rate
    else:
        success_criterion = "termination"
        success_rate = float(
            last_match(r"'success_rate':\s*([0-9.eE+-]+)", text, result_file)
        )
        original_cube_success_rate = None
        cube_hop_criterion_3_success_rate = None
    num_eval = int(last_match(r"^\s*num_eval:\s*(\d+)\s*$", text, result_file))
    evaluation_time = float(
        last_match(r"evaluation_time:\s*([0-9.eE+-]+)\s+seconds", text, result_file)
    )
    return {
        "predictor": predictor,
        "dataset": dataset,
        "seed": int(seed_directory.removeprefix("seed_")),
        "success_criterion": success_criterion,
        "success_rate": success_rate,
        "num_eval": num_eval,
        "num_success": int(round(success_rate * num_eval / 100.0)),
        "original_cube_success_rate": original_cube_success_rate,
        "original_cube_success_count": (
            int(round(original_cube_success_rate * num_eval / 100.0))
            if original_cube_success_rate is not None
            else None
        ),
        "cube_hop_criterion_3_success_rate": cube_hop_criterion_3_success_rate,
        "cube_hop_criterion_3_success_count": (
            int(round(cube_hop_criterion_3_success_rate * num_eval / 100.0))
            if cube_hop_criterion_3_success_rate is not None
            else None
        ),
        "evaluation_time_sec": evaluation_time,
        "result_file": str(result_file.resolve()),
    }


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    result_files = sorted(results_root.glob("*/*/seed_*/*toy_results.txt"))
    rows = [parse_result(path, results_root) for path in result_files]
    rows.sort(key=lambda row: (row["predictor"], row["dataset"], row["seed"]))
    if len(rows) != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} planning results, found {len(rows)} "
            f"under {results_root}."
        )
    identities = {
        (row["predictor"], row["dataset"], row["seed"]) for row in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError("Duplicate predictor/dataset/seed results found.")

    write_rows(
        args.output,
        [
            "predictor",
            "dataset",
            "seed",
            "success_criterion",
            "success_rate",
            "num_eval",
            "num_success",
            "original_cube_success_rate",
            "original_cube_success_count",
            "cube_hop_criterion_3_success_rate",
            "cube_hop_criterion_3_success_count",
            "evaluation_time_sec",
            "result_file",
        ],
        rows,
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["predictor"], row["dataset"])].append(row)
    summary_rows = []
    for (predictor, dataset), group in sorted(grouped.items()):
        success_criteria = {row["success_criterion"] for row in group}
        if len(success_criteria) != 1:
            raise RuntimeError(
                f"Mixed success criteria for {predictor}/{dataset}: "
                f"{sorted(success_criteria)}"
            )
        success_rates = [row["success_rate"] for row in group]
        original_cube_success_rates = [
            row["original_cube_success_rate"]
            for row in group
            if row["original_cube_success_rate"] is not None
        ]
        cube_hop_criterion_3_success_rates = [
            row["cube_hop_criterion_3_success_rate"]
            for row in group
            if row["cube_hop_criterion_3_success_rate"] is not None
        ]
        evaluation_times = [row["evaluation_time_sec"] for row in group]
        summary_rows.append(
            {
                "predictor": predictor,
                "dataset": dataset,
                "n_seeds": len(group),
                "success_criterion": success_criteria.pop(),
                "mean_success_rate": f"{statistics.mean(success_rates):.6f}",
                "std_success_rate": (
                    f"{statistics.stdev(success_rates):.6f}"
                    if len(group) > 1
                    else "0.000000"
                ),
                "total_success": sum(row["num_success"] for row in group),
                "total_eval": sum(row["num_eval"] for row in group),
                "mean_original_cube_success_rate": (
                    f"{statistics.mean(original_cube_success_rates):.6f}"
                    if original_cube_success_rates
                    else ""
                ),
                "mean_cube_hop_criterion_3_success_rate": (
                    f"{statistics.mean(cube_hop_criterion_3_success_rates):.6f}"
                    if cube_hop_criterion_3_success_rates
                    else ""
                ),
                "mean_evaluation_time_sec": (
                    f"{statistics.mean(evaluation_times):.6f}"
                ),
            }
        )
    write_rows(
        args.summary_output,
        [
            "predictor",
            "dataset",
            "n_seeds",
            "success_criterion",
            "mean_success_rate",
            "std_success_rate",
            "total_success",
            "total_eval",
            "mean_original_cube_success_rate",
            "mean_cube_hop_criterion_3_success_rate",
            "mean_evaluation_time_sec",
        ],
        summary_rows,
    )
    print(f"Wrote {len(rows)} runs to {args.output}")
    print(f"Wrote {len(summary_rows)} groups to {args.summary_output}")


if __name__ == "__main__":
    main()
