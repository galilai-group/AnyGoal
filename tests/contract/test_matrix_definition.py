from __future__ import annotations

from pathlib import Path

import yaml

from tests.support.catalog import DATASET_IDS, MODEL_IDS


def test_public_evaluation_manifest_covers_the_declared_matrix():
    manifest = yaml.safe_load(
        Path("scripts/analysis/manifest.yaml").read_text(encoding="utf-8")
    )
    matrices = {entry["evaluation"]: entry for entry in manifest["matrices"]}
    assert set(matrices) == {"rollout", "planning"}
    for matrix in matrices.values():
        assert tuple(matrix["models"]) == MODEL_IDS
        assert tuple(matrix["datasets"]) == DATASET_IDS
