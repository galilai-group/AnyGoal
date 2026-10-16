from __future__ import annotations

import h5py
import numpy as np
import pytest

from anygoal.cli import main
from anygoal.config import load_config
from anygoal.datasets.validation import validate_hdf5
import anygoal


def test_default_config_has_no_personal_service_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYGOAL_SITE_CONFIG", str(tmp_path / "missing.yaml"))
    config = load_config()
    assert config["wandb"]["enabled"] is False
    assert config["wandb"]["entity"] is None
    assert "slurm" not in config


def test_top_level_api_exports_all_public_workflows():
    assert callable(anygoal.collect)
    assert callable(anygoal.train)
    assert callable(anygoal.plan)
    assert callable(anygoal.evaluate_rollout)
    assert callable(anygoal.get_registry)


def test_hdf5_validation_reports_shape_and_episode_errors(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((3, 2), dtype=np.float32))
        handle.create_dataset("pixels", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("ep_len", data=np.array([3], dtype=np.int32))
    report = validate_hdf5(path, dataset_id="pusht")
    assert report.valid
    assert report.rows == 3
    assert report.episodes == 1


def test_cli_lists_registry(capsys):
    assert main(["list", "models"]) == 0
    output = capsys.readouterr().out
    assert "driftworld" in output
    assert "toy-action-duplicate" in output


def test_eval_cli_only_exposes_public_workflows(capsys):
    try:
        main(["eval", "--help"])
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out
    assert "prediction" not in output
    assert "rollout" in output


def test_reproduce_prints_commands_without_scheduler(capsys, tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """schema_version: 1
matrices:
  - evaluation: rollout
    models: [driftworld]
    datasets: [pusht]
    seeds: [2]
"""
    )

    assert main(["reproduce", str(manifest)]) == 0
    output = capsys.readouterr().out
    assert output.strip() == (
        "anygoal eval rollout --model driftworld --dataset pusht --seed 2"
    )
    assert "sbatch" not in output


def test_rollout_dry_run_preserves_underlying_evaluator(capsys):
    assert (
        main(
            [
                "eval",
                "rollout",
                "--model",
                "driftworld",
                "--dataset",
                "pusht",
                "--seed",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "-m anygoal.runners.rollout" in output
    assert "--model driftworld" in output
    assert "--environment pusht" in output


def test_planning_cli_forwards_hydra_override_after_separator(capsys):
    assert (
        main(
            [
                "eval",
                "planning",
                "--model",
                "lewm",
                "--dataset",
                "pusht",
                "--seed",
                "0",
                "--dry-run",
                "--",
                "policy=lewm",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "-m anygoal.runners.plan.planning_lewm" in output
    assert "policy=lewm" in output
    assert "seed=0" in output


def test_lewm_pusht_architecture_matches_official_weights():
    from omegaconf import OmegaConf
    from anygoal.resources import resource_path

    config = OmegaConf.load(resource_path("plan", "lewm", "pusht.yaml"))
    # The CLI resolves checkpoints from the shared artifact catalog.
    assert "checkpoint" not in config.model
    # The official PushT weights embed five 2-D actions per model step.
    assert config.model.action_dim * config.plan_config.action_block == 10


def test_eval_selects_model_policy_but_honors_explicit_random(capsys):
    command = ["eval", "planning", "--model", "lewm", "--dataset", "pusht", "--dry-run"]
    assert main(command) == 0
    assert "policy=lewm" in capsys.readouterr().out
    assert main([*command, "--", "policy=random"]) == 0
    output = capsys.readouterr().out
    assert "policy=random" in output
    assert "lewm_pusht.ckpt" not in output


@pytest.mark.parametrize("kind", ["planning", "rollout"])
@pytest.mark.parametrize(
    "model,filename",
    [
        ("lewm", "lewm_cube.ckpt"),
        ("driftworld", "driftworld_cube.pt"),
        ("sensorimotor", "sensorimotor_cube.ckpt"),
    ],
)
def test_eval_resolves_family_checkpoint_for_each_model(
    kind, model, filename, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("ANYGOAL_CHECKPOINT_ROOT", str(tmp_path))
    assert (
        main(
            [
                "eval",
                kind,
                "--model",
                model,
                "--dataset",
                "cubehop",
                "--dry-run",
            ]
        )
        == 0
    )
    assert str(tmp_path / filename) in capsys.readouterr().out


def test_explicit_checkpoint_overrides_catalog_default(tmp_path, capsys):
    custom = tmp_path / "custom.pt"
    assert (
        main(
            [
                "eval",
                "planning",
                "--model",
                "lewm",
                "--dataset",
                "pusht",
                "--checkpoint",
                str(custom),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert str(custom) in output
    assert "lewm_pusht.ckpt" not in output


def test_hydra_checkpoint_override_is_not_replaced(tmp_path, capsys):
    custom = tmp_path / "hydra.pt"
    assert (
        main(
            [
                "eval",
                "planning",
                "--model",
                "lewm",
                "--dataset",
                "pusht",
                "--dry-run",
                "--",
                f"+model.checkpoint={custom}",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f"+model.checkpoint={custom}" in output
    assert "lewm_pusht.ckpt" not in output


def test_top_level_collect_cli_forwards_native_options(monkeypatch, tmp_path):
    captured = {}

    def fake_collect(dataset, arguments):
        captured["dataset"] = dataset
        captured["arguments"] = arguments
        return tmp_path / "hug.h5"

    monkeypatch.setattr("anygoal.cli.collect_from_cli", fake_collect)
    assert main(["collect", "avoid_peg", "--num-episodes", "2"]) == 0
    assert captured == {
        "dataset": "avoid_peg",
        "arguments": ["--num-episodes", "2"],
    }


def _write_training_hdf5(path, *, environment):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((2, 2), dtype=np.float32))
        handle.create_dataset("pixels", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("ep_len", data=np.array([2], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        handle.attrs["environment"] = environment


def test_variant_training_requires_verified_profile(capsys, tmp_path):
    data = tmp_path / "avoidt.h5"
    _write_training_hdf5(data, environment="anygoal/AvoidT-v1")
    try:
        main(
            [
                "train",
                "driftworld",
                "--data",
                str(data),
                "--task",
                "avoidt",
                "--dry-run",
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    assert "no verified training profile" in capsys.readouterr().err


def test_prejepa_training_supports_pusht(capsys, tmp_path):
    data = tmp_path / "pusht.h5"
    _write_training_hdf5(data, environment="swm/PushT-v1")
    assert (
        main(
            [
                "train",
                "prejepa",
                "--data",
                str(data),
                "--task",
                "pusht",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "dataset_name=" in capsys.readouterr().out
