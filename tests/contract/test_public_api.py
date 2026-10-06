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
    "hug_box",
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


def test_default_config_has_no_personal_cluster_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYGOAL_SITE_CONFIG", str(tmp_path / "missing.yaml"))
    config = load_config()
    assert config["wandb"]["enabled"] is False
    assert config["wandb"]["entity"] is None
    assert config["slurm"]["partition"] is None
    assert config["slurm"]["exclude"] == []


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


def test_visualize_help_describes_required_inputs(capsys):
    try:
        main(["visualize", "--help"])
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out
    assert "Required inputs:" in output
    assert "dataset HDF5 containing pixels" in output
    assert "metrics JSON containing numeric leaves" in output


def test_prediction_dry_run_preserves_underlying_evaluator(capsys):
    assert (
        main(
            [
                "eval",
                "prediction",
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
    assert "scripts/eval.py" in output
    assert "--model driftworld" in output
    assert "--environment pusht" in output


def test_manipulator_collect_smoke_uses_balanced_episode_count(capsys, tmp_path):
    assert (
        main(
            [
                "data",
                "collect",
                "hug_box",
                "--preset",
                "smoke",
                "--output",
                str(tmp_path / "hug.h5"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert "--episodes 2" in capsys.readouterr().out


def test_variant_training_requires_explicit_config(capsys):
    try:
        main(["train", "driftworld", "--dataset", "avoidt", "--dry-run"])
    except SystemExit as error:
        assert error.code == 2
    assert "requires an explicit --model-config" in capsys.readouterr().err


def test_cube_diagonal_paper_recipe_preserves_split_and_seeds(capsys, tmp_path):
    assert (
        main(
            [
                "data",
                "collect",
                "cubediagonal",
                "--output",
                str(tmp_path / "cube-diagonal.h5"),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "--episodes 50 --seed 10000" in output
    assert "--episodes 350 --seed 10065" in output
    assert "merge_cube_datasets.py" in output


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


def test_counterfactual_visualizer_forwards_declared_io(capsys, tmp_path):
    input_root = tmp_path / "results"
    output_directory = tmp_path / "figures"
    assert (
        main(
            [
                "visualize",
                "counterfactual-grid",
                "--input",
                str(input_root),
                "--output",
                str(output_directory),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "--input-root" in output and str(input_root) in output
    assert "--output-directory" in output and str(output_directory) in output


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
        "hug_box": "avoid_peg_test.h5",
        "extractpeg": "extract_peg_test.h5",
        "tworoom": "tworoom.h5",
        "tworoom_route": "tworoom_route_test.h5",
    }
