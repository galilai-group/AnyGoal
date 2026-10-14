from __future__ import annotations

import pytest

from anygoal.cli import main
from tests.support.artifacts import configured_path, evaluation_profile
from tests.support.catalog import EVALUATION_CASES, case_id


pytestmark = pytest.mark.matrix_e2e


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), EVALUATION_CASES, ids=map(case_id, EVALUATION_CASES)
)
def test_plans_every_combination_with_one_real_trial(
    monkeypatch,
    tmp_path,
    matrix_data_root,
    matrix_eval_config,
    model_id,
    dataset_id,
):
    data_home = matrix_data_root.parent
    for variable in (
        "STABLEWM_HOME",
        "DRIFTWORLD_DATA_ROOT",
        "IRASIM_DATA_ROOT",
        "SENSORIMOTOR_DATA_ROOT",
    ):
        monkeypatch.setenv(variable, str(data_home))

    profile = evaluation_profile(model_id, dataset_id, matrix_eval_config)
    checkpoint = configured_path(profile, "checkpoint")
    assert checkpoint is not None and checkpoint.is_file(), (
        f"Missing checkpoint for {model_id} x {dataset_id}: {checkpoint}"
    )
    output = tmp_path / f"{model_id}-{dataset_id}"
    overrides = [
        "eval.num_eval=1",
        "eval.eval_budget=25",
        "solver.num_samples=2",
        "solver.topk=1",
        "solver.n_steps=1",
        f"output.directory={output}",
    ]
    assert (
        main(
            [
                "eval",
                "planning",
                "--model",
                model_id,
                "--dataset",
                dataset_id,
                "--checkpoint",
                str(checkpoint),
                "--seed",
                "0",
                "--",
                *overrides,
            ]
        )
        == 0
    )
