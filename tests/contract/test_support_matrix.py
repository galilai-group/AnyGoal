from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import anygoal
from anygoal.cli import main
from anygoal.training import prepare_training
from tests.support.catalog import (
    BASE_DATASET_IDS,
    DATASET_IDS,
    EVALUATION_CASES,
    GENERATED_DATASET_IDS,
    TRAINING_CASES,
    UNSUPPORTED_TRAINING_CASES,
    case_id,
)
from tests.support.hdf5 import write_hdf5


ENVIRONMENTS = {
    "pusht": "swm/PushT-v1",
    "cube": "ogbench/cube-single-play-v0",
    "insertpeg": "anygoal/InsertPeg-v0",
    "tworoom": "swm/TwoRoom-v1",
}


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), EVALUATION_CASES, ids=map(case_id, EVALUATION_CASES)
)
@pytest.mark.parametrize("evaluation", ("rollout", "planning"))
def test_cli_has_a_dispatch_path_for_every_evaluation_combination(
    capsys, model_id, dataset_id, evaluation
):
    assert (
        main(
            [
                "eval",
                evaluation,
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


@pytest.mark.parametrize(
    "model_id",
    ("toy-adaln", "toy-action-duplicate", "toy-driftworld"),
)
def test_toy_planning_cli_supplies_packaged_training_config(capsys, model_id):
    assert (
        main(
            [
                "eval",
                "planning",
                "--model",
                model_id,
                "--dataset",
                "extractpeg",
                "--checkpoint",
                "/tmp/model.pt",
                "--dry-run",
            ]
        )
        == 0
    )
    command = capsys.readouterr().out
    assert "model.config_path=" in command
    assert "/eval/training/toy/" in command
    assert "/insertpeg.yaml" in command


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), TRAINING_CASES, ids=map(case_id, TRAINING_CASES)
)
def test_api_prepares_every_supported_training_combination(
    tmp_path, model_id, dataset_id
):
    data = write_hdf5(
        tmp_path / f"{model_id}-{dataset_id}.h5",
        environment=ENVIRONMENTS[dataset_id],
    )
    plan = prepare_training(model_id, data=data, task=dataset_id)
    assert plan.model.id == model_id
    assert plan.task.id == dataset_id
    assert str(data.resolve()) in plan.shell_command()


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), TRAINING_CASES, ids=map(case_id, TRAINING_CASES)
)
def test_cli_prepares_every_supported_training_combination(
    capsys, tmp_path, model_id, dataset_id
):
    data = write_hdf5(
        tmp_path / f"cli-{model_id}-{dataset_id}.h5",
        environment=ENVIRONMENTS[dataset_id],
    )
    assert (
        main(
            [
                "train",
                model_id,
                "--data",
                str(data),
                "--task",
                dataset_id,
                "--dry-run",
            ]
        )
        == 0
    )
    assert str(data.resolve()) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("model_id", "dataset_id"),
    UNSUPPORTED_TRAINING_CASES,
    ids=map(case_id, UNSUPPORTED_TRAINING_CASES),
)
def test_api_rejects_every_unverified_variant_training_combination(
    tmp_path, model_id, dataset_id
):
    spec = anygoal.get_registry().dataset(dataset_id)
    data = write_hdf5(
        tmp_path / f"{model_id}-{dataset_id}.h5",
        environment=spec.environment,
        dataset_id=dataset_id,
    )
    with pytest.raises(ValueError, match="no verified training profile"):
        prepare_training(model_id, data=data, task=dataset_id)


@pytest.mark.parametrize(
    ("model_id", "dataset_id"),
    UNSUPPORTED_TRAINING_CASES,
    ids=map(case_id, UNSUPPORTED_TRAINING_CASES),
)
def test_cli_rejects_every_unverified_variant_training_combination(
    capsys, tmp_path, model_id, dataset_id
):
    spec = anygoal.get_registry().dataset(dataset_id)
    data = write_hdf5(
        tmp_path / f"cli-{model_id}-{dataset_id}.h5",
        environment=spec.environment,
        dataset_id=dataset_id,
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "train",
                model_id,
                "--data",
                str(data),
                "--task",
                dataset_id,
                "--dry-run",
            ]
        )
    assert error.value.code == 2
    assert "no verified training profile" in capsys.readouterr().err


@pytest.mark.parametrize("dataset_id", GENERATED_DATASET_IDS)
def test_collection_api_dispatches_every_generated_dataset(
    monkeypatch, tmp_path, dataset_id
):
    output = tmp_path / f"{dataset_id}.h5"

    def collect_dataset(*, output, seed=0):
        with h5py.File(output, "w"):
            pass
        return output

    monkeypatch.setattr(
        "anygoal.collection.import_module",
        lambda _: SimpleNamespace(collect_dataset=collect_dataset),
    )
    assert anygoal.collect(dataset_id, output=output) == output.resolve()
    with h5py.File(output, "r") as handle:
        assert handle.attrs["anygoal_dataset_id"] == dataset_id


@pytest.mark.parametrize("dataset_id", GENERATED_DATASET_IDS)
def test_collection_cli_dispatches_every_generated_dataset(
    monkeypatch, capsys, tmp_path, dataset_id
):
    output = tmp_path / f"cli-{dataset_id}.h5"

    def collect_dataset(*, output: Path, seed: int = 0):
        del seed
        with h5py.File(output, "w"):
            pass
        return output

    monkeypatch.setattr(
        "anygoal.collection.import_module",
        lambda _: SimpleNamespace(collect_dataset=collect_dataset),
    )
    assert main(["collect", dataset_id, "--output", str(output)]) == 0
    assert output.resolve() == Path(capsys.readouterr().out.strip())


@pytest.mark.parametrize("dataset_id", BASE_DATASET_IDS)
def test_collection_cli_rejects_every_external_dataset(capsys, dataset_id):
    with pytest.raises(SystemExit) as error:
        main(["collect", dataset_id])
    assert error.value.code == 2
    assert "not a collection target" in capsys.readouterr().err


class _RolloutDataset:
    lengths = np.asarray([10])

    def load_episode(self, episode_index):
        del episode_index
        return {
            "value": torch.arange(10, dtype=torch.float32)[:, None],
            "action": torch.zeros(10, 1),
        }


class _RolloutAdapter:
    model_name = "matrix-adapter"
    error_space = "state_mse"
    observation_keys = ("value",)

    def rollout_errors(self, observations, action_blocks, *, rollout_seed):
        del action_blocks, rollout_seed
        return observations["value"][:, 3:, 0].square()


class _PlanningDataset:
    lengths = np.asarray([30])

    def __init__(self, column_names):
        self.column_names = column_names

    def load_chunk(self, episodes, start, end):
        raise AssertionError("The fake world owns extraction in this contract test.")


class _PlanningWorld:
    num_envs = 1
    envs = SimpleNamespace(envs=[])

    def set_policy(self, policy):
        self.policy = policy

    def evaluate(self, **kwargs):
        self.kwargs = kwargs
        return {"success_rate": 100.0}

    def close(self):
        pass


@pytest.mark.parametrize(
    ("model_id", "dataset_id"), EVALUATION_CASES, ids=map(case_id, EVALUATION_CASES)
)
def test_rollout_api_accepts_every_registered_combination(
    monkeypatch, model_id, dataset_id
):
    def registered_adapter(requested_model, task_spec, dataset, **kwargs):
        del dataset, kwargs
        assert requested_model == model_id
        assert task_spec.id == dataset_id
        return _RolloutAdapter(), model_id, None, {}

    monkeypatch.setattr("anygoal.rollout._registered_adapter", registered_adapter)
    result = anygoal.evaluate_rollout(
        model_id,
        data=_RolloutDataset(),
        task=dataset_id,
        device="cpu",
        history_length=3,
        frame_skip=2,
        horizons=(2,),
        num_episodes=1,
        batch_size=1,
        max_starts_per_episode=1,
    )
    assert result.model_id == model_id
    assert result.task_id == dataset_id


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
def test_planning_api_accepts_every_registered_dataset(dataset_id):
    spec = anygoal.get_registry().dataset(dataset_id)
    columns = {
        "pusht": ["pixels", "action", "state"],
        "tworoom": ["pixels", "action", "proprio"],
        "insert_peg": ["pixels", "action", "observation"],
        "cube": [
            "pixels",
            "action",
            "qpos",
            "qvel",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
            "proprio_effector_pos",
            "proprio_effector_yaw",
        ],
    }[spec.family]
    world = _PlanningWorld()
    result = anygoal.plan(
        object(),
        data=_PlanningDataset(columns),
        task=dataset_id,
        num_evals=1,
        world=world,
    )
    assert result.task_id == dataset_id
    assert world.kwargs["dataset"].column_names == columns


def test_training_matrix_is_exactly_models_by_base_datasets():
    assert len(TRAINING_CASES) == 32
    assert {dataset for _, dataset in TRAINING_CASES} == set(BASE_DATASET_IDS)


def test_evaluation_matrix_is_exactly_models_by_all_datasets():
    assert len(EVALUATION_CASES) == 96
    assert {dataset for _, dataset in EVALUATION_CASES} == set(DATASET_IDS)
