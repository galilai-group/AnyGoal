"""Aggregate PreJEPA seed metrics into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


VARIANTS = ("dinowm", "dinowm_noprop")
TASKS = ("pusht", "avoidt", "stackt", "push_t_antigoal")
SEEDS = (0, 1, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("outputs/prejepa"))
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def load_rows(results_root: Path) -> list[dict]:
    rows = []
    missing = []
    for variant in VARIANTS:
        for task in TASKS:
            for seed in SEEDS:
                metrics_path = results_root / variant / task / f"seed_{seed}" / "metrics.json"
                if not metrics_path.is_file():
                    missing.append(str(metrics_path))
                    continue
                payload = json.loads(metrics_path.read_text())
                rows.append(
                    {
                        "variant": variant,
                        "task": task,
                        "seed": seed,
                        "success_rate": float(payload["metrics"]["success_rate"]),
                        "evaluation_time_seconds": float(
                            payload["evaluation_time_seconds"]
                        ),
                    }
                )
    if missing:
        raise FileNotFoundError("Missing evaluation outputs:\n" + "\n".join(missing))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# PreJEPA (DINO-WM) MPC results",
        "",
        "Success rate is reported as mean ± sample standard deviation over seeds 0, 1, and 2.",
        "",
        "| Variant | Task | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        for task in TASKS:
            selected = [
                row for row in rows if row["variant"] == variant and row["task"] == task
            ]
            selected.sort(key=lambda row: row["seed"])
            rates = [row["success_rate"] for row in selected]
            mean = statistics.mean(rates)
            standard_deviation = statistics.stdev(rates)
            lines.append(
                f"| {variant} | {task} | {rates[0]:.1f}% | {rates[1]:.1f}% | "
                f"{rates[2]:.1f}% | {mean:.1f} ± {standard_deviation:.1f}% |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    csv_output = args.csv_output or args.results_root / "summary.csv"
    markdown_output = args.markdown_output or args.results_root / "RESULTS.md"
    rows = load_rows(args.results_root)
    write_csv(csv_output, rows)
    write_markdown(markdown_output, rows)
    print(f"Wrote {csv_output}")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
