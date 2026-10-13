from __future__ import annotations

import pytest

import anygoal
from tests.support.artifacts import data_path
from tests.support.catalog import EVALUATION_CASES, case_id


pytestmark = pytest.mark.matrix_e2e


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), EVALUATION_CASES, ids=map(case_id, EVALUATION_CASES)
)
def test_rollout_every_combination_with_one_real_start(
    tmp_path,
    matrix_data_root,
    matrix_device,
    matrix_eval_config,
    model_id,
    dataset_id,
):
    result = anygoal.evaluate_rollout(
        model_id,
        data=data_path(dataset_id, matrix_data_root),
        task=dataset_id,
        seed=0,
        history_length=3,
        frame_skip=5,
        horizons=(5,),
        num_episodes=1,
        batch_size=1,
        max_starts_per_episode=1,
        output_dir=tmp_path / f"{model_id}-{dataset_id}",
        device=matrix_device,
        config=matrix_eval_config,
    )
    assert result.model_id == model_id
    assert result.task_id == dataset_id
    assert result.metrics_path.is_file()
