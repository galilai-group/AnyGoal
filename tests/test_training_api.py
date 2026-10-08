from pathlib import Path
import warnings

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from anygoal.training import (
    _warn_for_official_checkpoint_training,
    load_trained_model,
    prepare_training,
)
from anygoal.wm.toy_predictors import AdaLNActionPredictor


def write_hdf5(path: Path, *, environment: str | None = None) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((3, 2), dtype=np.float32))
        handle.create_dataset("pixels", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("ep_len", data=np.array([3], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        if environment:
            handle.attrs["environment"] = environment
    return path


def apply_dotlist(config_path: Path, overrides: tuple[str, ...]):
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    for override in overrides:
        if override.startswith(("data=", "model=")) or override.startswith("~"):
            continue
        key, raw = override.split("=", 1)
        value = OmegaConf.from_dotlist([f"value={raw}"])["value"]
        OmegaConf.update(config, key, value, merge=True)
    return config


def test_task_is_required(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5")
    with pytest.raises(ValueError, match="task is required"):
        prepare_training("driftworld", data=data, task="")


def test_explicit_task_must_match_inferred_identity(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    with pytest.raises(ValueError, match="does not match"):
        prepare_training("driftworld", data=data, task="cube")


def test_explicit_task_may_match_inferred_identity(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    plan = prepare_training("driftworld", data=data, task="pusht")
    assert plan.task.id == "pusht"


def test_default_plan_preserves_driftworld_config_except_data_path(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    plan = prepare_training("driftworld", data=data, task="pusht")
    original = OmegaConf.load(plan.config)
    resolved = apply_dotlist(plan.config, plan.overrides)
    resolved.data.dataset_name = original.data.dataset_name
    assert OmegaConf.to_container(resolved, resolve=False) == OmegaConf.to_container(
        original, resolve=False
    )
    assert "opt.lr=1.25e-05" not in plan.overrides


def test_common_overrides_map_without_changing_unrelated_values(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    plan = prepare_training(
        "driftworld",
        data=data,
        task="pusht",
        learning_rate=2e-5,
        batch_size=4,
        seed=9,
        num_workers=3,
    )
    resolved = apply_dotlist(plan.config, plan.overrides)
    original = OmegaConf.load(plan.config)
    assert resolved.opt.lr == 2e-5
    assert resolved.data.batch_size == 4
    assert resolved.train.seed == 9
    assert resolved.dataloader.num_workers == 3
    assert resolved.model == original.model
    assert resolved.data.frameskip == original.data.frameskip


def test_model_specific_double_underscore_becomes_config_path(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    plan = prepare_training(
        "driftworld", data=data, task="pusht", visualization__enabled=False
    )
    assert "visualization.enabled=false" in plan.overrides


def test_model_specific_options_cannot_bypass_common_contract(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    with pytest.raises(ValueError, match="common learning_rate"):
        prepare_training("driftworld", data=data, task="pusht", opt__lr=2e-5)


def test_toy_profile_selects_existing_data_and_model_defaults(tmp_path):
    data = write_hdf5(tmp_path / "custom.h5", environment="swm/PushT-v1")
    plan = prepare_training("toy-adaln", data=data, task="pusht")
    assert plan.overrides[:2] == ("data=pusht", "model=adaln")
    assert not any(item.startswith("optimizer.lr=") for item in plan.overrides)


@pytest.mark.parametrize(
    ("model", "expected_prefix"),
    [
        ("toy-adaln", ("data=pusht", "model=adaln")),
        (
            "toy-action-duplicate",
            ("data=pusht", "model=action_duplicate"),
        ),
        ("toy-driftworld", ("data=pusht",)),
    ],
)
def test_toy_defaults_add_only_group_selection_and_data_path(
    tmp_path, model, expected_prefix
):
    data = write_hdf5(tmp_path / f"{model}.h5", environment="swm/PushT-v1")
    plan = prepare_training(model, data=data, task="pusht")

    assert plan.overrides[:-1] == expected_prefix
    assert plan.overrides[-1] == f"data.dataset_name={data.resolve()}"


def test_prejepa_output_directory_controls_stablewm_checkpoint_root(tmp_path):
    data = write_hdf5(
        tmp_path / "insert_peg_weak_expert_train.h5",
        environment="anygoal/InsertPeg-v0",
    )
    output = tmp_path / "prejepa"
    plan = prepare_training(
        "prejepa", data=data, task="insert_peg", output_dir=output
    )

    assert plan.output_root == output / "checkpoints"
    assert plan.environment == (("STABLEWM_HOME", str(output)),)
    assert not any(override.startswith("output_dir=") for override in plan.overrides)
    assert plan.shell_command().startswith(f"STABLEWM_HOME={output} ")


@pytest.mark.parametrize("model", ["lewm", "sensorimotor", "prejepa"])
@pytest.mark.parametrize(
    ("task", "environment"),
    [
        ("pusht", "swm/PushT-v1"),
        ("cube", "ogbench/cube-single-play-v0"),
        ("tworoom", "swm/TwoRoom-v1"),
        ("insert_peg", "anygoal/InsertPeg-v0"),
    ],
)
def test_remaining_native_models_support_all_base_tasks(
    tmp_path, model, task, environment
):
    data = write_hdf5(tmp_path / f"{model}-{task}.h5", environment=environment)
    plan = prepare_training(model, data=data, task=task)

    assert any(str(data.resolve()) in override for override in plan.overrides)
    if model == "sensorimotor":
        assert plan.config.is_file()
        assert plan.environment == (("RUNS_ROOT", str(plan.output_root)),)
    else:
        assert plan.config is None
        assert plan.environment[0][0] == "STABLEWM_HOME"


@pytest.mark.parametrize(
    ("task", "seed", "inverse_weight"),
    [
        ("pusht", 0, 30.0),
        ("cube", 3072, 1.0),
        ("tworoom", 0, 0.1),
        ("insert_peg", 0, 1.0),
    ],
)
def test_sensorimotor_profiles_preserve_recorded_training_values(
    tmp_path, task, seed, inverse_weight
):
    environments = {
        "pusht": "swm/PushT-v1",
        "cube": "ogbench/cube-single-play-v0",
        "tworoom": "swm/TwoRoom-v1",
        "insert_peg": "anygoal/InsertPeg-v0",
    }
    data = write_hdf5(tmp_path / f"{task}.h5", environment=environments[task])
    plan = prepare_training("sensorimotor", data=data, task=task)
    config = OmegaConf.load(plan.config)

    assert config.seed == seed
    assert config.optimizer.lr == 1e-4
    assert config.loader.batch_size == 256
    assert config.trainer.max_epochs == 10
    assert config.loss.inverse.weight == inverse_weight
    assert config.data.dataset.frameskip == 5


@pytest.mark.parametrize(
    ("task", "data_group", "image_size"),
    [
        ("pusht", "pusht", 196),
        ("cube", "ogb", 196),
        ("tworoom", "tworoom", 196),
        ("insert_peg", "dmc", 224),
    ],
)
def test_lewm_and_prejepa_task_defaults_are_explicit(
    tmp_path, task, data_group, image_size
):
    environments = {
        "pusht": "swm/PushT-v1",
        "cube": "ogbench/cube-single-play-v0",
        "tworoom": "swm/TwoRoom-v1",
        "insert_peg": "anygoal/InsertPeg-v0",
    }
    data = write_hdf5(tmp_path / f"{task}.h5", environment=environments[task])

    lewm = prepare_training("lewm", data=data, task=task)
    prejepa = prepare_training("prejepa", data=data, task=task)

    assert f"data={data_group}" in lewm.overrides
    assert f"image_size={image_size}" in prejepa.overrides
    assert "~wm.encoding.proprio" in prejepa.overrides


@pytest.mark.parametrize("model", ["lewm", "prejepa"])
@pytest.mark.parametrize("task", ["pusht", "cube", "tworoom"])
def test_official_checkpoint_combinations_warn(model, task):
    with pytest.warns(UserWarning, match="official checkpoint released by LeWM"):
        _warn_for_official_checkpoint_training(model, task)


@pytest.mark.parametrize("model", ["lewm", "prejepa"])
def test_locally_trained_insertpeg_combinations_do_not_warn(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_for_official_checkpoint_training(model, "insertpeg")


@pytest.mark.parametrize(
    ("model", "task", "environment", "data_key"),
    [
        ("driftworld", "pusht", "swm/PushT-v1", "data.dataset_name"),
        ("driftworld", "cube", "ogbench/cube-single-play-v0", "data.dataset_name"),
        ("driftworld", "tworoom", "swm/TwoRoom-v1", "data.dataset_name"),
        ("driftworld", "insert_peg", "anygoal/InsertPeg-v0", "data.dataset_name"),
        ("irasim", "pusht", "swm/PushT-v1", "h5_dataset_name"),
        ("irasim", "cube", "ogbench/cube-single-play-v0", "h5_dataset_name"),
        ("irasim", "tworoom", "swm/TwoRoom-v1", "h5_dataset_name"),
        ("irasim", "insert_peg", "anygoal/InsertPeg-v0", "h5_dataset_name"),
    ],
)
def test_verified_native_profiles_only_override_the_data_path_by_default(
    tmp_path, model, task, environment, data_key
):
    data = write_hdf5(tmp_path / f"{model}-{task}.h5", environment=environment)
    plan = prepare_training(model, data=data, task=task)
    assert plan.config.is_file()
    assert len(plan.overrides) == 1
    assert plan.overrides[0] == f"{data_key}={data.resolve()}"


def test_post_training_loader_returns_an_actual_model(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """model:
  name: adaln
  _target_: anygoal.wm.toy_predictors.AdaLNActionPredictor
  state_dim: 2
  action_dim: 1
  action_frameskip: 1
  action_embed_dim: 2
  history_size: 2
  hidden_dim: 8
  depth: 1
  heads: 1
  dim_head: 8
  mlp_dim: 16
  dropout: 0.0
  emb_dropout: 0.0
"""
    )
    expected = AdaLNActionPredictor(
        state_dim=2,
        action_dim=1,
        history_size=2,
        action_embed_dim=2,
        hidden_dim=8,
        depth=1,
        heads=1,
        dim_head=8,
        mlp_dim=16,
    )
    checkpoint = tmp_path / "weights_epoch_001.pt"
    torch.save(expected.state_dict(), checkpoint)
    data = write_hdf5(tmp_path / "data.h5")

    loaded = load_trained_model(
        "toy-adaln", checkpoint, config, data, task_id="pusht", device="cpu"
    )

    assert isinstance(loaded, AdaLNActionPredictor)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


@pytest.mark.parametrize("model_id", ["prejepa", "lewm"])
def test_stable_worldmodel_post_training_loader_returns_an_actual_model(
    tmp_path, model_id
):
    config = tmp_path / "config.json"
    config.write_text(
        '{"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 2}'
    )
    expected = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "weights_epoch_1.pt"
    torch.save(expected.state_dict(), checkpoint)
    data = write_hdf5(tmp_path / "data.h5")

    loaded = load_trained_model(
        model_id, checkpoint, config, data, task_id="insertpeg", device="cpu"
    )

    assert isinstance(loaded, torch.nn.Linear)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
