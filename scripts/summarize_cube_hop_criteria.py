"""Summarize Cube planning runs with the original and CubeHop-3 criteria."""

from __future__ import annotations

import ast
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT = ROOT / "outputs/cube_hop_criteria_raw.csv"
AVERAGE_OUTPUT = ROOT / "outputs/cube_hop_criteria_seed_average.csv"

def result_path(model: str, config: str, seed: int) -> Path:
    root = "cube_hop" if config == "cube_hop" else "cube_via_cube_hop"
    model_dir = "prejepa_noprop" if model == "prejepa" else model
    directory = ROOT / "outputs" / root / model_dir / f"seed_{seed}"
    if model == "sensorimotor":
        return directory / "metrics.json"
    filename = (
        f"{config}_lewm_results.txt"
        if model == "lewm"
        else f"{config}_prejepa_noprop_results.txt"
    )
    return directory / filename


def load_result(path: Path) -> tuple[dict, float]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return payload["metrics"], float(payload["elapsed_seconds"])

    text = path.read_text()
    metric_matches = re.findall(r"^metrics: (.+)$", text, flags=re.MULTILINE)
    time_matches = re.findall(
        r"^evaluation_time: ([0-9.eE+-]+) seconds$", text, flags=re.MULTILINE
    )
    if not metric_matches or not time_matches:
        raise ValueError(f"No complete result in {path}")
    return ast.literal_eval(metric_matches[-1]), float(time_matches[-1])


def mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def build_raw_rows() -> list[dict]:
    rows = []
    job_id = 0
    for model in ("lewm", "sensorimotor", "prejepa"):
        for config in ("cube_hop", "cube"):
            for seed in (0, 1, 2):
                job_id += 1
                path = result_path(model, config, seed)
                metrics, elapsed_seconds = load_result(path)
                original = metrics["original_cube_criterion"]
                criterion_3 = metrics["cube_hop_criterion_3"]
                errors = metrics["cube_final_errors"]
                tolerances = metrics["cube_evaluation_tolerances"]
                num_eval = len(original["episode_successes"])

                row = {
                    "job_id": job_id,
                    "qos": "not-recorded",
                    "model": model,
                    "checkpoint_variant": (
                        "dinowm_noprop" if model == "prejepa" else model
                    ),
                    "evaluation_config": config,
                    "evaluation_dataset": (
                        "cube_hop_test"
                        if config == "cube_hop"
                        else "cube_single_expert"
                    ),
                    "environment": "anygoal/CubeHop-v0",
                    "seed": seed,
                    "num_eval": num_eval,
                }
                row.update(
                    {
                        "original_cube_success_count": int(
                            sum(original["episode_successes"])
                        ),
                        "original_cube_success_rate": float(original["success_rate"]),
                        "cube_hop_criterion_3_success_count": int(
                            sum(criterion_3["episode_successes"])
                        ),
                        "cube_hop_criterion_3_success_rate": float(
                            criterion_3["success_rate"]
                        ),
                        "final_ee_position_error_mean": mean(errors["ee_position"]),
                        "final_ee_yaw_error_mean": mean(errors["ee_yaw"]),
                        "final_cube_position_error_mean": mean(errors["cube_position"]),
                        "ee_position_tolerance": float(tolerances["ee_position"]),
                        "ee_yaw_tolerance": float(tolerances["ee_yaw"]),
                        "cube_position_tolerance": float(
                            tolerances["cube_position"]
                        ),
                        "elapsed_seconds": elapsed_seconds,
                        "result_path": str(path.relative_to(ROOT)),
                    }
                )
                rows.append(row)
    return rows


def build_average_rows(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for model in ("lewm", "sensorimotor", "prejepa"):
        for config in ("cube_hop", "cube"):
            group = [
                row
                for row in raw_rows
                if row["model"] == model and row["evaluation_config"] == config
            ]
            if len(group) != 3:
                raise ValueError(f"Expected 3 seeds for {model}/{config}, got {len(group)}")
            row = {
                "model": model,
                "checkpoint_variant": group[0]["checkpoint_variant"],
                "evaluation_config": config,
                "evaluation_dataset": group[0]["evaluation_dataset"],
                "environment": group[0]["environment"],
                "num_seeds": 3,
                "seeds": "0;1;2",
                "num_eval_per_seed": group[0]["num_eval"],
                "total_evaluations": sum(item["num_eval"] for item in group),
            }
            mean_fields = [
                "original_cube_success_rate",
                "cube_hop_criterion_3_success_rate",
                "final_ee_position_error_mean",
                "final_ee_yaw_error_mean",
                "final_cube_position_error_mean",
                "elapsed_seconds",
            ]
            for field in mean_fields:
                values = [float(item[field]) for item in group]
                row[f"{field}_mean"] = statistics.mean(values)
                row[f"{field}_std"] = statistics.stdev(values)
            for tolerance in (
                "ee_position_tolerance",
                "ee_yaw_tolerance",
                "cube_position_tolerance",
            ):
                row[tolerance] = group[0][tolerance]
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_rows = build_raw_rows()
    average_rows = build_average_rows(raw_rows)
    if len(raw_rows) != 18 or len(average_rows) != 6:
        raise ValueError("Unexpected summary row count")
    write_csv(RAW_OUTPUT, raw_rows)
    write_csv(AVERAGE_OUTPUT, average_rows)
    print(f"wrote {len(raw_rows)} rows: {RAW_OUTPUT}")
    print(f"wrote {len(average_rows)} rows: {AVERAGE_OUTPUT}")


if __name__ == "__main__":
    main()
