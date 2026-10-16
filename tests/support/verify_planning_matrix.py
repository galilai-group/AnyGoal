"""Run real reduced planning trials across model and task families."""

import os
from pathlib import Path
import subprocess
import sys


CASES = [
    ("lewm", "avoidt"),
    ("sensorimotor", "stackt"),
    ("prejepa", "pusht-antigoal"),
    ("driftworld", "avoidt"),
    ("toy-adaln", "cubediagonal"),
    ("toy-action-duplicate", "tworoom_route"),
    ("toy-driftworld", "avoid_peg"),
    ("irasim", "avoidt"),
]


def main():
    workdir = Path(sys.executable).parents[2]
    run_id = os.environ.get("SLURM_JOB_ID", "manual")
    failures = []
    for model, dataset in CASES:
        print(f"MATRIX_START model={model} dataset={dataset}", flush=True)
        output = (
            workdir / "outputs" / f"planning_matrix_{run_id}" / f"{model}-{dataset}"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "anygoal.cli",
                "eval",
                "planning",
                "--model",
                model,
                "--dataset",
                dataset,
                "--seed",
                "0",
                "--",
                "eval.num_eval=1",
                "eval.eval_budget=25",
                "solver.num_samples=2",
                "solver.topk=2",
                "solver.n_steps=1",
                f"output.directory={output}",
            ],
            cwd=workdir,
            check=False,
        )
        print(
            f"MATRIX_RESULT model={model} dataset={dataset} exit={result.returncode}",
            flush=True,
        )
        if result.returncode:
            failures.append((model, dataset, result.returncode))
    if failures:
        raise SystemExit(f"MATRIX_FAILURES {failures}")
    print("ALL_8_MODEL_REPRESENTATIVE_PLANNING_TRIALS_PASSED", flush=True)


if __name__ == "__main__":
    main()
