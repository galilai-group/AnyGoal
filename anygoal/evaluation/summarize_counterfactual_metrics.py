"""Collect counterfactual metric JSON files into long and wide CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPRESENTATION_ORDER = {
    **{str(index): index for index in range(6)},
    "average": 6,
    "final": 7,
}


def collect_rows(results_root: Path) -> tuple[list[dict], list[str]]:
    long_rows = []
    bias_axes = set()
    metric_paths = sorted(results_root.glob("*/*/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No metric JSON files under {results_root}")
    for metric_path in metric_paths:
        result = json.loads(metric_path.read_text())
        for protocol_name, protocol in result["protocols"].items():
            action_count = len(protocol["counterfactual_actions"])
            for representation, metrics in protocol["representations"].items():
                for bias_axis, csu in metrics["csu"].items():
                    bias_axes.add(bias_axis)
                    long_rows.append(
                        {
                            "model": result["model"],
                            "dataset": result["environment"],
                            "protocol": protocol_name,
                            "representation": representation,
                            "bias_axis": bias_axis,
                            "ras": metrics["ras"],
                            "csu": csu["value"],
                            "median_counterfactual_sensitivity": metrics[
                                "median_counterfactual_sensitivity"
                            ],
                            "factual_output_variation": metrics[
                                "factual_output_variation"
                            ],
                            "num_samples": result["num_samples"],
                            "factual_variation_samples": result[
                                "factual_variation_samples"
                            ],
                            "counterfactual_action_count": action_count,
                            "checkpoint": result["checkpoint"],
                            "metrics_json": str(metric_path.resolve()),
                        }
                    )
    long_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model"],
            row["protocol"],
            REPRESENTATION_ORDER[row["representation"]],
            row["bias_axis"],
        )
    )
    return long_rows, sorted(bias_axes)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_wide_rows(long_rows: list[dict], bias_axes: list[str]) -> list[dict]:
    wide_rows = []
    current_key = None
    current_row = None
    for row in long_rows:
        key = (
            row["model"],
            row["dataset"],
            row["protocol"],
            row["representation"],
        )
        if key != current_key:
            if current_row is not None:
                wide_rows.append(current_row)
            current_key = key
            current_row = {
                field: value
                for field, value in row.items()
                if field not in {"bias_axis", "csu"}
            }
            current_row.update({f"csu_{axis}": "" for axis in bias_axes})
        current_row[f"csu_{row['bias_axis']}"] = row["csu"]
    if current_row is not None:
        wide_rows.append(current_row)
    return wide_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_directory = (
        results_root if args.output_directory is None else args.output_directory.resolve()
    )
    long_rows, bias_axes = collect_rows(results_root)
    long_path = output_directory / "counterfactual_metrics_long.csv"
    write_csv(long_path, long_rows, list(long_rows[0]))
    wide_rows = make_wide_rows(long_rows, bias_axes)
    wide_path = output_directory / "counterfactual_metrics_wide.csv"
    write_csv(wide_path, wide_rows, list(wide_rows[0]))
    print(f"wrote {long_path} ({len(long_rows)} rows)")
    print(f"wrote {wide_path} ({len(wide_rows)} rows)")


if __name__ == "__main__":
    main()
