from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import skipUnless

import torch
from omegaconf import OmegaConf


STABLE_PRETRAINING_AVAILABLE = find_spec("stable_pretraining") is not None


def _training_helpers():
    from scripts.train.train_toy_predictor import (
        _indices_digest,
        _select_validation_fraction,
        _split_primary_dataset,
        predictor_forward,
    )

    return (
        _indices_digest,
        _select_validation_fraction,
        _split_primary_dataset,
        predictor_forward,
    )


def _resume_helpers():
    from scripts.train.train_toy_predictor import (
        _resolve_resume_checkpoint,
        _resolve_run_directory,
        _wandb_logger_config,
    )

    return (
        _resolve_resume_checkpoint,
        _resolve_run_directory,
        _wandb_logger_config,
    )


@skipUnless(
    STABLE_PRETRAINING_AVAILABLE,
    "stable-pretraining is provided by the optional training dependencies",
)
def test_primary_and_fractional_validation_indices_are_deterministic():
    helpers = _training_helpers()
    indices_digest, select_fraction, split_primary, _ = helpers

    first_train, first_validation = split_primary(range(100), 0.9, 3072)
    second_train, second_validation = split_primary(range(100), 0.9, 3072)
    assert first_train.indices == second_train.indices
    assert first_validation.indices == second_validation.indices
    assert indices_digest(first_validation.indices) == indices_digest(
        second_validation.indices
    )

    first_subset, first_indices = select_fraction(range(100), 25.0, 91)
    second_subset, second_indices = select_fraction(range(100), 25.0, 91)
    assert len(first_subset) == len(second_subset) == 25
    assert first_indices == second_indices

    full_subset, full_indices = select_fraction(range(100), 100.0, 91)
    assert len(full_subset) == 100
    assert list(full_indices) == list(range(100))


@skipUnless(
    STABLE_PRETRAINING_AVAILABLE,
    "stable-pretraining is provided by the optional training dependencies",
)
def test_validation_mse_uses_dataset_metric_namespace():
    helpers = _training_helpers()
    *_, predictor_forward = helpers

    class Model:
        @staticmethod
        def loss(states, actions):
            mse = (states.square().mean() + actions.square().mean()) / 2
            return {"loss": mse, "future_loss": mse.detach()}

    class Module:
        model = Model()

        def log_dict(self, metrics, **kwargs):
            self.metrics = metrics
            self.log_kwargs = kwargs

    module = Module()
    batch = {
        "state": torch.ones(2, 4, 12),
        "action": torch.ones(2, 4, 2),
    }
    predictor_forward(module, batch, stage="validation/avoidt")

    assert set(module.metrics) == {
        "validation/avoidt/mse",
        "validation/avoidt/future_mse",
    }
    assert module.log_kwargs["add_dataloader_idx"] is False
    assert module.log_kwargs["batch_size"] == 2


@skipUnless(
    STABLE_PRETRAINING_AVAILABLE,
    "stable-pretraining is provided by the optional training dependencies",
)
def test_full_state_resume_paths_and_wandb_id_are_reused():
    resolve_checkpoint, resolve_run_directory, wandb_logger_config = (
        _resume_helpers()
    )
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        checkpoint_directory = root / "spt_run" / "checkpoints"
        checkpoint_directory.mkdir(parents=True)
        checkpoint = checkpoint_directory / "last.ckpt"
        checkpoint.touch()
        experiment_directory = root / "experiment"
        experiment_directory.mkdir()
        (experiment_directory / "wandb_resume.json").write_text(
            json.dumps(
                {
                    "id": "existing-run-id",
                    "project": "toy-project",
                    "entity": "toy-entity",
                }
            )
        )

        resolved_checkpoint = resolve_checkpoint(checkpoint_directory.parent)
        resolved_directory = resolve_run_directory(
            resolved_checkpoint,
            experiment_directory,
            root,
            "unused",
        )
        logger_config = wandb_logger_config(
            OmegaConf.create(
                {
                    "project": "toy-project",
                    "entity": "toy-entity",
                    "name": "toy-run",
                }
            ),
            resolved_directory,
            resuming=True,
        )

        assert resolved_checkpoint == checkpoint
        assert resolved_directory == experiment_directory
        assert logger_config["id"] == "existing-run-id"
        assert logger_config["resume"] == "must"
