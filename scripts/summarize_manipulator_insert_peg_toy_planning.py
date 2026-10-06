"""Summarize Insert-Peg toy-predictor planning results."""

from __future__ import annotations

import ast
import csv
import re
import statistics
from pathlib import Path

import numpy as np
from anygoal.paths import checkpoint_root


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "outputs/planning_manipulator_insert_peg_20260911"
TRAIN_ROOT = checkpoint_root() / "20260910_manipulator_insert_peg_toy_predictors"
RAW_OUTPUT = RESULTS_ROOT / "toy_planning_raw.csv"
SUMMARY_OUTPUT = RESULTS_ROOT / "toy_planning_seed_summary.csv"

MODELS = {
    "toy_essential_adaln": (
        "essential", 33, "adaln", None,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_essential_adaln_01/weights_epoch_100.pt",
    ),
    "toy_essential_action_duplicate": (
        "essential", 33, "action_duplicate", None,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_essential_action_duplicate_01/weights_epoch_100.pt",
    ),
    "toy_essential_drift_lr1e-4": (
        "essential", 33, "drift", 1e-4,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_essential_drift_lr1e-4_01/weights_step_00411600.pt",
    ),
    "toy_essential_drift_lr1e-5": (
        "essential", 33, "drift", 1e-5,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_essential_drift_lr1e-5_01/weights_step_00411600.pt",
    ),
    "toy_extra_adaln": (
        "extra", 44, "adaln", None,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_extra_adaln/weights_epoch_100.pt",
    ),
    "toy_extra_action_duplicate": (
        "extra", 44, "action_duplicate", None,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_extra_action_duplicate/weights_epoch_100.pt",
    ),
    "toy_extra_drift_lr1e-4": (
        "extra", 44, "drift", 1e-4,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_extra_drift_lr1e-4/weights_step_00411600.pt",
    ),
    "toy_extra_drift_lr1e-5": (
        "extra", 44, "drift", 1e-5,
        TRAIN_ROOT / "20260910_manipulator_insert_peg_extra_drift_lr1e-5/weights_step_00411600.pt",
    ),
}


class ArrayCallToList(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "array":
            return self.visit(node.args[0])
        raise ValueError("Unsupported function call in result metrics.")


def parse_metrics(text: str) -> dict:
    expression = ast.parse(text.strip(), mode="eval")
    expression = ArrayCallToList().visit(expression)
    ast.fix_missing_locations(expression)
    return ast.literal_eval(expression)


def load_result(path: Path) -> tuple[dict, float]:
    blocks = path.read_text().split("==== RESULTS ====")
    for block in reversed(blocks[1:]):
        match = re.search(
            r"metrics: (.*)\nevaluation_time: ([0-9.eE+-]+) seconds",
            block,
            flags=re.DOTALL,
        )
        if match:
            return parse_metrics(match.group(1)), float(match.group(2))
    raise ValueError(f"No complete result block in {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_rows = []
    for model_id, metadata in MODELS.items():
        representation, state_dim, predictor, learning_rate, checkpoint = metadata
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for seed in range(3):
            result_path = RESULTS_ROOT / model_id / f"seed_{seed}/results.txt"
            metrics, elapsed_seconds = load_result(result_path)
            dataset_goal = metrics["dataset_goal_criterion"]
            native = metrics["native_insert_peg_criterion"]
            errors = metrics["manipulator_final_errors"]
            tolerances = metrics["manipulator_evaluation_tolerances"]
            num_eval = len(dataset_goal["episode_successes"])
            if num_eval != 50:
                raise ValueError(f"Expected 50 evaluations in {result_path}, got {num_eval}")
            row = {
                "model_id": model_id,
                "representation": representation,
                "state_dim": state_dim,
                "predictor": predictor,
                "learning_rate": "" if learning_rate is None else learning_rate,
                "seed": seed,
                "num_eval": num_eval,
                "dataset_goal_success_count": int(sum(dataset_goal["episode_successes"])),
                "dataset_goal_success_rate": float(dataset_goal["success_rate"]),
                "native_success_count": int(sum(native["episode_successes"])),
                "native_success_rate": float(native["success_rate"]),
            }
            for name in ("hand_position", "hand_angle", "object_position", "object_angle"):
                values = np.asarray(errors[name], dtype=np.float64)
                row[f"final_{name}_error_mean"] = float(values.mean())
                row[f"final_{name}_error_std"] = float(values.std())
                row[f"{name}_tolerance"] = float(tolerances[name])
            row.update(
                elapsed_seconds=elapsed_seconds,
                checkpoint=str(checkpoint),
                result_path=str(result_path),
            )
            raw_rows.append(row)

    summary_rows = []
    for model_id, metadata in MODELS.items():
        representation, state_dim, predictor, learning_rate, checkpoint = metadata
        group = [row for row in raw_rows if row["model_id"] == model_id]
        if [row["seed"] for row in group] != [0, 1, 2]:
            raise ValueError(f"Expected seeds 0,1,2 for {model_id}")
        row = {
            "model_id": model_id,
            "representation": representation,
            "state_dim": state_dim,
            "predictor": predictor,
            "learning_rate": "" if learning_rate is None else learning_rate,
            "num_seeds": 3,
            "seeds": "0;1;2",
            "num_eval_per_seed": 50,
            "total_evaluations": 150,
            "dataset_goal_success_count_total": sum(item["dataset_goal_success_count"] for item in group),
            "dataset_goal_success_rate_mean": statistics.mean(item["dataset_goal_success_rate"] for item in group),
            "dataset_goal_success_rate_std": statistics.stdev(item["dataset_goal_success_rate"] for item in group),
            "native_success_count_total": sum(item["native_success_count"] for item in group),
            "native_success_rate_mean": statistics.mean(item["native_success_rate"] for item in group),
            "native_success_rate_std": statistics.stdev(item["native_success_rate"] for item in group),
        }
        for name in ("hand_position", "hand_angle", "object_position", "object_angle"):
            seed_means = [item[f"final_{name}_error_mean"] for item in group]
            row[f"final_{name}_error_seed_mean"] = statistics.mean(seed_means)
            row[f"final_{name}_error_seed_std"] = statistics.stdev(seed_means)
            row[f"{name}_tolerance"] = group[0][f"{name}_tolerance"]
        runtimes = [item["elapsed_seconds"] for item in group]
        row["elapsed_seconds_mean"] = statistics.mean(runtimes)
        row["elapsed_seconds_std"] = statistics.stdev(runtimes)
        row["checkpoint"] = str(checkpoint)
        summary_rows.append(row)

    if len(raw_rows) != 24 or len(summary_rows) != 8:
        raise ValueError(f"Unexpected row counts: raw={len(raw_rows)}, summary={len(summary_rows)}")
    write_csv(RAW_OUTPUT, raw_rows)
    write_csv(SUMMARY_OUTPUT, summary_rows)
    print(f"Wrote {len(raw_rows)} rows to {RAW_OUTPUT}")
    print(f"Wrote {len(summary_rows)} rows to {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
