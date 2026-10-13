from __future__ import annotations

import pytest

from anygoal.sources import get_sources, source_status
from tests.support.artifacts import configured_path, data_path, evaluation_profile
from tests.support.catalog import BASE_DATASET_IDS, DATASET_IDS, MODEL_IDS


pytestmark = pytest.mark.matrix_e2e


def test_matrix_artifacts_and_sources_are_complete(
    matrix_data_root, matrix_eval_config
):
    missing = []
    for dataset_id in DATASET_IDS:
        try:
            data_path(dataset_id, matrix_data_root)
        except FileNotFoundError as error:
            missing.append(str(error))

    for model_id in MODEL_IDS:
        for dataset_id in BASE_DATASET_IDS:
            profile = evaluation_profile(model_id, dataset_id, matrix_eval_config)
            for key in ("checkpoint", "training_config", "run_directory"):
                path = configured_path(profile, key)
                if path is not None and not path.exists():
                    missing.append(
                        f"Missing {key} for {model_id} x {dataset_id}: {path}"
                    )

    source_ids = sorted({spec.id for spec in get_sources().values()})
    for source_id in source_ids:
        status = source_status(source_id)
        if status.state != "ready":
            missing.append(
                f"Model source {source_id!r} is {status.state} at {status.path}; "
                f"run `anygoal models install {source_id}`."
            )

    assert not missing, "Matrix preflight failed:\n- " + "\n- ".join(missing)
