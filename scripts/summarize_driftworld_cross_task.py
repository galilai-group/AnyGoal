"""Summarize the 2026-08-23 DriftWorld cross-task planning sweep."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "outputs/driftworld/cross_task_3seed_20260823"
RAW_OUTPUT = ROOT / "outputs/driftworld/cross_task_3seed_20260823_raw.csv"
AVERAGE_OUTPUT = (
    ROOT / "outputs/driftworld/cross_task_3seed_20260823_seed_average.csv"
)

SCENARIOS = {
    "pusht500k_avoidt": ("avoidt", 3070795),
    "pusht500k_stackt_v2": ("stackt_v2", 3070796),
    "pusht500k_antigoal_v3": ("push_t_antigoal_v3", 3070797),
    "cube100k_cube_dataset_swm_env": ("cube_dataset_swm_env", 3070798),
    "cube100k_cube_dataset_anygoal_env": ("cube_dataset_anygoal_env", 3070799),
    "cube100k_cube_hop_dataset_anygoal_env": (
        "cube_hop_dataset_anygoal_env",
        3070800,
    ),
}

def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def optional_mean(values):
    return mean(values) if values else ""


def build_raw_rows() -> list[dict]:
    rows = []
    for result_directory, (scenario, job_id) in SCENARIOS.items():
        for seed in (0, 1, 2):
            matches = list((RESULT_ROOT / result_directory).glob(f"step*/seed_{seed}/metrics.json"))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one result for {result_directory}/seed_{seed}, got {matches}"
                )
            path = matches[0]
            payload = json.loads(path.read_text())
            config = payload["config"]
            metrics = payload["metrics"]
            original = metrics.get("original_cube_criterion")
            criterion_3 = metrics.get("cube_hop_criterion_3")
            primary = original if original is not None else metrics
            successes = primary["episode_successes"]
            errors = metrics.get("cube_final_errors", {})
            tolerances = metrics.get("cube_evaluation_tolerances", {})
            row = {
                "job_id": job_id,
                "scenario": scenario,
                "checkpoint": payload["checkpoint"],
                "checkpoint_step": int(payload["checkpoint_step"]),
                "evaluation_dataset": config["eval"]["dataset_name"],
                "environment": config["world"]["env_name"],
                "seed": seed,
                "task_seed": int(payload["task_seed"]),
                "policy_seed": int(payload["policy_seed"]),
                "rollout_seed": int(config["model"]["rollout_seed"]),
                "num_eval": len(successes),
                "success_count": int(sum(successes)),
                "success_rate": float(primary["success_rate"]),
            }
            row["original_cube_success_count"] = (
                int(sum(original["episode_successes"])) if original else ""
            )
            row["original_cube_success_rate"] = (
                float(original["success_rate"]) if original else ""
            )
            row["criterion_3_success_count"] = (
                int(sum(criterion_3["episode_successes"])) if criterion_3 else ""
            )
            row["criterion_3_success_rate"] = (
                float(criterion_3["success_rate"]) if criterion_3 else ""
            )
            for error_name in ("ee_position", "ee_yaw", "cube_position"):
                values = errors.get(error_name, [])
                row[f"final_{error_name}_error_mean"] = optional_mean(values)
                row[f"{error_name}_tolerance"] = tolerances.get(error_name, "")
            row["elapsed_seconds"] = float(payload["elapsed_seconds"])
            row["result_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    if len(rows) != 18:
        raise ValueError(f"Expected 18 raw rows, got {len(rows)}")
    return rows


def build_average_rows(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for scenario in (value[0] for value in SCENARIOS.values()):
        group = [row for row in raw_rows if row["scenario"] == scenario]
        if [row["seed"] for row in group] != [0, 1, 2]:
            raise ValueError(f"Expected seeds 0,1,2 for {scenario}")
        row = {
            "scenario": scenario,
            "checkpoint": group[0]["checkpoint"],
            "checkpoint_step": group[0]["checkpoint_step"],
            "evaluation_dataset": group[0]["evaluation_dataset"],
            "environment": group[0]["environment"],
            "num_seeds": 3,
            "seeds": "0;1;2",
            "num_eval_per_seed": group[0]["num_eval"],
            "total_evaluations": sum(item["num_eval"] for item in group),
            "success_count_total": sum(item["success_count"] for item in group),
        }
        fields = [
            "success_rate",
            "original_cube_success_rate",
            "criterion_3_success_rate",
            "final_ee_position_error_mean",
            "final_ee_yaw_error_mean",
            "final_cube_position_error_mean",
            "elapsed_seconds",
        ]
        for field in fields:
            values = [float(item[field]) for item in group if item[field] != ""]
            row[f"{field}_mean"] = mean(values) if values else ""
            row[f"{field}_std"] = statistics.stdev(values) if values else ""
        for tolerance in (
            "ee_position_tolerance",
            "ee_yaw_tolerance",
            "cube_position_tolerance",
        ):
            row[tolerance] = group[0][tolerance]
        rows.append(row)
    if len(rows) != 6:
        raise ValueError(f"Expected 6 average rows, got {len(rows)}")
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
    write_csv(RAW_OUTPUT, raw_rows)
    write_csv(AVERAGE_OUTPUT, average_rows)
    print(f"wrote {len(raw_rows)} rows: {RAW_OUTPUT}")
    print(f"wrote {len(average_rows)} rows: {AVERAGE_OUTPUT}")


if __name__ == "__main__":
    main()
