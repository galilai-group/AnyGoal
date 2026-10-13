"""Expected public support matrix, independent of the runtime registry."""

from __future__ import annotations

from itertools import product


MODEL_IDS = (
    "lewm",
    "sensorimotor",
    "prejepa",
    "driftworld",
    "irasim",
    "toy-adaln",
    "toy-action-duplicate",
    "toy-driftworld",
)

DATASET_IDS = (
    "pusht",
    "avoidt",
    "stackt",
    "pusht-antigoal",
    "cube",
    "cubehop",
    "cubediagonal",
    "insertpeg",
    "avoid_peg",
    "extractpeg",
    "tworoom",
    "tworoom_route",
)

BASE_DATASET_IDS = ("pusht", "cube", "insertpeg", "tworoom")
GENERATED_DATASET_IDS = tuple(
    dataset_id for dataset_id in DATASET_IDS if dataset_id not in BASE_DATASET_IDS
)
VARIANT_DATASET_IDS = GENERATED_DATASET_IDS

TRAINING_CASES = tuple(product(MODEL_IDS, BASE_DATASET_IDS))
UNSUPPORTED_TRAINING_CASES = tuple(product(MODEL_IDS, VARIANT_DATASET_IDS))
EVALUATION_CASES = tuple(product(MODEL_IDS, DATASET_IDS))


def case_id(case: tuple[str, str]) -> str:
    """Return a readable pytest ID for a model/dataset pair."""
    return f"{case[0]}--{case[1]}"
