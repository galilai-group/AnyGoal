from __future__ import annotations

import json

import h5py
import numpy as np

from anygoal.cli import main
from anygoal.config import load_config
from anygoal.datasets.validation import validate_hdf5
from anygoal.registry import get_registry


EXPECTED_DATASETS = {
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
}
EXPECTED_MODELS = {
    "lewm",
    "sensorimotor",
    "prejepa",
    "driftworld",
    "irasim",
    "toy-adaln",
    "toy-action-duplicate",
    "toy-driftworld",
}


def test_catalog_matches_public_support_sets():
    registry = get_registry()
    assert set(registry.datasets) == EXPECTED_DATASETS
    assert set(registry.models) == EXPECTED_MODELS
    assert registry.dataset("cube_diagonal").id == "cubediagonal"
    assert registry.model("adaln").id == "toy-adaln"
    assert set(registry.model("driftworld").training_profiles) == {
        "pusht",
        "cube",
        "tworoom",
        "insertpeg",
    }
    for model_id in ("lewm", "sensorimotor", "prejepa"):
        assert set(registry.model(model_id).training_profiles) == {
            "pusht",
            "cube",
            "tworoom",
            "insertpeg",
        }
    assert {
        item.id for item in registry.datasets.values() if item.source == "external"
    } == {"pusht", "cube", "insertpeg", "tworoom"}


def test_default_config_has_no_personal_service_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYGOAL_SITE_CONFIG", str(tmp_path / "missing.yaml"))
    config = load_config()
    assert config["wandb"]["enabled"] is False
    assert config["wandb"]["entity"] is None
    assert "slurm" not in config


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


def test_counterfactual_analysis_is_not_in_public_cli(capsys):
    try:
        main(["eval", "--help"])
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out
    assert "counterfactual" not in output
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


def test_public_planning_matrix_has_a_dispatch_path(capsys):
    for model_id in EXPECTED_MODELS:
        for dataset_id in EXPECTED_DATASETS:
            assert (
                main(
                    [
                        "eval",
                        "planning",
                        "--model",
                        model_id,
                        "--dataset",
                        dataset_id,
                        "--dry-run",
                    ]
                )
                == 0
            )
    capsys.readouterr()


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
        "insertpeg": "insert_peg_weak_expert_train.h5",
        "avoid_peg": "avoid_peg_test.h5",
        "extractpeg": "extract_peg_test.h5",
        "tworoom": "tworoom.h5",
        "tworoom_route": "tworoom_route_test.h5",
    }
