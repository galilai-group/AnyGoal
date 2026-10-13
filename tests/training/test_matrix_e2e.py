from __future__ import annotations

import pytest

import anygoal
from tests.support.artifacts import data_path
from tests.support.catalog import TRAINING_CASES, case_id


pytestmark = pytest.mark.matrix_e2e


def _reduced_training_options(model_id, output_dir):
    options = {
        "batch_size": 1,
        "num_workers": 0,
        "output_dir": output_dir,
    }
    if model_id in {"driftworld", "irasim", "toy-driftworld"}:
        options["max_steps"] = 1
    else:
        options["max_epochs"] = 1
    if model_id in {"lewm", "prejepa", "toy-adaln", "toy-action-duplicate"}:
        options["+trainer__limit_train_batches"] = 1
        options["+trainer__limit_val_batches"] = 1
    if model_id == "sensorimotor":
        options["+trainer__limit_train_batches"] = 1
        options["trainer__limit_val_batches"] = 1
    if model_id in {
        "sensorimotor",
        "toy-adaln",
        "toy-action-duplicate",
        "toy-driftworld",
    }:
        options["wandb__enabled"] = False
    if model_id == "driftworld":
        options["visualization__enabled"] = False
        options["train__ckpt_every"] = 1
        options["train__permanent_ckpt_every"] = 1
    if model_id == "irasim":
        options["ckpt_every"] = 1
        options["max_samples"] = 8
        options["max_validation_samples"] = 2
    return options


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), TRAINING_CASES, ids=map(case_id, TRAINING_CASES)
)
def test_trains_saves_and_reloads_every_supported_combination(
    monkeypatch, tmp_path, matrix_data_root, matrix_device, model_id, dataset_id
):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    model, result = anygoal.train(
        model_id,
        data=data_path(dataset_id, matrix_data_root),
        task=dataset_id,
        device=matrix_device,
        **_reduced_training_options(model_id, tmp_path / f"{model_id}-{dataset_id}"),
    )
    assert model is not None
    assert result.model_id == model_id
    assert result.task_id == dataset_id
    assert result.checkpoint.is_file()
