"""Summarize the 36 Cube toy-predictor planning evaluations."""

from __future__ import annotations

import ast
import csv
import re
import statistics
from pathlib import Path

import numpy as np
from anygoal.paths import checkpoint_root


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = checkpoint_root() / "toy_planning_20260813_cube_toy_predictors"
RAW_OUTPUT = ROOT / "outputs/cube_toy_predictors_eval_raw.csv"
AVERAGE_OUTPUT = ROOT / "outputs/cube_toy_predictors_eval_seed_average.csv"

SCENARIOS = {
    "cube_dataset_cube_env": ("swm/OGBCube-v0", "cube_single_expert"),
    "cube_dataset_cube_hop_env": ("anygoal/CubeHop-v0", "cube_single_expert"),
    "cube_hop_dataset_cube_hop_env": (
        "anygoal/CubeHop-v0",
        "cube_hop_test",
    ),
}
REPRESENTATIONS = {"cube_essential": 11, "cube_extra": 17}
PREDICTORS = ("adaln", "action_duplicate")
class _ArrayCallToList(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "array"
            and len(node.args) == 1
            and not node.keywords
        ):
            return self.visit(node.args[0])
        raise ValueError("Result metrics contain an unsupported function call.")


def parse_metrics(metric_text: str) -> dict:
    expression = ast.parse(metric_text.strip(), mode="eval")
    expression = _ArrayCallToList().visit(expression)
    ast.fix_missing_locations(expression)
    return ast.literal_eval(expression)


def load_result(path: Path) -> tuple[dict, float]:
    result_blocks = path.read_text().split("==== RESULTS ====")
    for block in reversed(result_blocks[1:]):
        match = re.search(
            r"metrics: (.*)\nevaluation_time: ([0-9.eE+-]+) seconds",
            block,
            flags=re.DOTALL,
        )
        if match:
            return parse_metrics(match.group(1)), float(match.group(2))
    raise ValueError(f"No complete result found in {path}")


def summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def result_path(
    scenario: str,
    representation: str,
    predictor: str,
    seed: int,
) -> Path:
    filename = {
        "cube_dataset_cube_env": "cube_toy_results.txt",
        "cube_dataset_cube_hop_env": "cube_hop_from_cube_toy_results.txt",
        "cube_hop_dataset_cube_hop_env": "cube_hop_toy_results.txt",
    }[scenario]
    return RESULTS_ROOT / scenario / representation / predictor / f"seed_{seed}" / filename


def build_raw_rows() -> list[dict]:
    rows = []
    for scenario, (environment, dataset) in SCENARIOS.items():
        for representation, state_dim in REPRESENTATIONS.items():
            for predictor in PREDICTORS:
                for seed in range(3):
                    path = result_path(scenario, representation, predictor, seed)
                    metrics, elapsed_seconds = load_result(path)
                    original = metrics["original_cube_criterion"]
                    criterion_3 = metrics["cube_hop_criterion_3"]
                    errors = metrics["cube_final_errors"]
                    tolerances = metrics["cube_evaluation_tolerances"]
                    num_eval = len(original["episode_successes"])

                    row = {
                        "scenario": scenario,
                        "environment": environment,
                        "evaluation_dataset": dataset,
                        "representation": representation,
                        "state_dim": state_dim,
                        "predictor": predictor,
                        "seed": seed,
                        "num_eval": num_eval,
                        "original_cube_success_count": int(
                            sum(original["episode_successes"])
                        ),
                        "original_cube_success_rate": float(
                            original["success_rate"]
                        ),
                        "criterion_3_success_count": int(
                            sum(criterion_3["episode_successes"])
                        ),
                        "criterion_3_success_rate": float(
                            criterion_3["success_rate"]
                        ),
                    }

                    for output_name, metric_name in (
                        ("final_ee_position_error", "ee_position"),
                        ("final_ee_yaw_error", "ee_yaw"),
                        ("final_cube_position_error", "cube_position"),
                    ):
                        values = summary(errors[metric_name])
                        row[f"{output_name}_mean"] = values["mean"]
                        row[f"{output_name}_std"] = values["std"]

                    for output_name, metric_name in (
                        ("ee_position_tolerance", "ee_position"),
                        ("ee_yaw_tolerance", "ee_yaw"),
                        ("cube_position_tolerance", "cube_position"),
                    ):
                        row[output_name] = float(tolerances[metric_name])
                    row["elapsed_seconds"] = elapsed_seconds
                    row["result_path"] = str(path)
                    rows.append(row)
    return rows


def build_average_rows(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for scenario, (environment, dataset) in SCENARIOS.items():
        for representation, state_dim in REPRESENTATIONS.items():
            for predictor in PREDICTORS:
                group = [
                    row
                    for row in raw_rows
                    if row["scenario"] == scenario
                    and row["representation"] == representation
                    and row["predictor"] == predictor
                ]
                if [row["seed"] for row in group] != [0, 1, 2]:
                    raise ValueError(
                        f"Expected seeds 0,1,2 for {scenario}/{representation}/{predictor}"
                    )
                row = {
                    "scenario": scenario,
                    "environment": environment,
                    "evaluation_dataset": dataset,
                    "representation": representation,
                    "state_dim": state_dim,
                    "predictor": predictor,
                    "num_seeds": 3,
                    "seeds": "0;1;2",
                    "num_eval_per_seed": group[0]["num_eval"],
                    "total_evaluations": sum(item["num_eval"] for item in group),
                }
                rate_fields = [
                    "original_cube_success_rate",
                    "criterion_3_success_rate",
                ]
                error_fields = [
                    "final_ee_position_error_mean",
                    "final_ee_yaw_error_mean",
                    "final_cube_position_error_mean",
                ]
                for field in rate_fields + error_fields + ["elapsed_seconds"]:
                    values = [item[field] for item in group]
                    if any(value == "" for value in values):
                        row[f"{field}_mean"] = ""
                        row[f"{field}_std"] = ""
                    else:
                        numeric = [float(value) for value in values]
                        row[f"{field}_mean"] = statistics.mean(numeric)
                        row[f"{field}_std"] = statistics.stdev(numeric)

                count_fields = [
                    "original_cube_success_count",
                    "criterion_3_success_count",
                ]
                for field in count_fields:
                    values = [item[field] for item in group]
                    row[f"{field}_total"] = (
                        "" if any(value == "" for value in values) else sum(values)
                    )
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
    expected_raw_rows = len(SCENARIOS) * len(REPRESENTATIONS) * len(PREDICTORS) * 3
    expected_average_rows = len(SCENARIOS) * len(REPRESENTATIONS) * len(PREDICTORS)
    if (
        len(raw_rows) != expected_raw_rows
        or len(average_rows) != expected_average_rows
    ):
        raise ValueError(
            f"Unexpected row counts: raw={len(raw_rows)}, average={len(average_rows)}"
        )
    if any(row["num_eval"] != 50 for row in raw_rows):
        raise ValueError("Every raw result must contain exactly 50 evaluations.")
    write_csv(RAW_OUTPUT, raw_rows)
    write_csv(AVERAGE_OUTPUT, average_rows)
    print(f"wrote {len(raw_rows)} rows: {RAW_OUTPUT}")
    print(f"wrote {len(average_rows)} rows: {AVERAGE_OUTPUT}")


if __name__ == "__main__":
    main()
