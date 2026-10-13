from __future__ import annotations

import json

from omegaconf import OmegaConf

from anygoal.registry import get_registry
from anygoal.resources import resource_path
from tests.support.catalog import BASE_DATASET_IDS, DATASET_IDS, MODEL_IDS


def test_catalog_matches_public_support_sets():
    registry = get_registry()
    assert set(registry.datasets) == set(DATASET_IDS)
    assert set(registry.models) == set(MODEL_IDS)
    assert registry.dataset("cube_diagonal").id == "cubediagonal"
    assert registry.model("adaln").id == "toy-adaln"
    for model in registry.models.values():
        assert model.capabilities == {"train", "rollout", "plan"}
        assert set(model.training_profiles) == set(BASE_DATASET_IDS)
    assert {
        item.id for item in registry.datasets.values() if item.source == "external"
    } == set(BASE_DATASET_IDS)


def test_catalog_resource_is_serializable():
    registry = get_registry()
    payload = {key: value.filename for key, value in registry.datasets.items()}
    assert json.loads(json.dumps(payload)) == {
        "pusht": "pusht_expert_train.h5",
        "avoidt": "avoidt_test.h5",
        "stackt": "stackt_test.h5",
        "pusht-antigoal": "pusht_antigoal_test.h5",
        "cube": "cube_single_expert.h5",
        "cubehop": "cube_hop_test.h5",
        "cubediagonal": "cube_diagonal_test.h5",
        "insertpeg": "manipulator_insert_peg_weak_expert_train.h5",
        "avoid_peg": "avoid_peg_test.h5",
        "extractpeg": "extract_peg_test.h5",
        "tworoom": "tworoom.h5",
        "tworoom_route": "tworoom_route_test.h5",
    }


def test_evaluation_config_resources_parse():
    paths = [resource_path("eval", "rollout.yaml")]
    paths.extend(resource_path("eval", "training").rglob("*.yaml"))
    paths.extend(resource_path("plan", "sensorimotor").glob("*.yaml"))
    for path in paths:
        OmegaConf.load(path)
