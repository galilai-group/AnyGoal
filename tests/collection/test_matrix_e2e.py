from __future__ import annotations

import pytest

import anygoal
from anygoal.datasets.validation import validate_hdf5
from tests.support.artifacts import data_path
from tests.support.catalog import GENERATED_DATASET_IDS


pytestmark = pytest.mark.matrix_e2e


def _collection_options(dataset_id, output, data_root):
    base = {"output": output, "overwrite": True}
    if dataset_id == "avoidt":
        # Keep the original collector's eight-way initial-state diversity while
        # reducing each corner from 100 accepted episodes to eight.
        return {**base, "num_episodes": 32, "num_parallel": 8}
    if dataset_id == "stackt":
        return {
            **base,
            "num_episodes": 1,
            "num_parallel": 1,
            "minimum_angle_support": 0.95,
            "reference": data_path("pusht", data_root),
        }
    if dataset_id == "pusht-antigoal":
        return {
            **base,
            "num_episodes": 4,
            "num_parallel": 1,
            "skip_support_filter": True,
            "reference": data_path("pusht", data_root),
        }
    if dataset_id == "cubehop":
        return {
            **base,
            "num_episodes": 2,
            "num_parallel": 1,
            "reference": data_path("cube", data_root),
        }
    if dataset_id == "cubediagonal":
        return {
            **base,
            "num_episodes": 1,
            "num_parallel": 1,
            "require_diagonal_lift": False,
            "min_lifted_targets": 0,
            "max_abs_block_quat_xy": None,
            "reference": data_path("cube", data_root),
        }
    if dataset_id in {"avoid_peg", "extractpeg"}:
        return {
            **base,
            "num_episodes": 2,
            "reference": data_path("insertpeg", data_root),
        }
    if dataset_id == "tworoom_route":
        return {**base, "smoke": True}
    raise AssertionError(f"No reduced collection profile for {dataset_id}")


@pytest.mark.parametrize("dataset_id", GENERATED_DATASET_IDS)
def test_collects_each_generated_dataset_with_a_reduced_profile(
    tmp_path, matrix_data_root, dataset_id
):
    output = tmp_path / anygoal.get_registry().dataset(dataset_id).filename
    result = anygoal.collect(
        dataset_id,
        **_collection_options(dataset_id, output, matrix_data_root),
    )
    report = validate_hdf5(result, dataset_id=dataset_id)
    assert report.valid, [issue.message for issue in report.issues]
    assert report.episodes >= 1
