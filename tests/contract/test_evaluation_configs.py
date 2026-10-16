"""Compose real runner configs across the public model/task matrix."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from anygoal import cli
from anygoal.artifacts import default_checkpoint_path
from anygoal.paths import resolve_path
from anygoal.registry import get_registry
from anygoal.wm.lewm_config import resolve_model_config
from tests.support.catalog import EVALUATION_CASES, case_id


@pytest.mark.parametrize(
    "model_id,dataset_id", EVALUATION_CASES, ids=map(case_id, EVALUATION_CASES)
)
def test_composed_planning_config_matches_loader_contract(
    model_id, dataset_id, monkeypatch
):
    commands = []
    monkeypatch.setattr(
        cli, "_execute", lambda command, **kwargs: commands.append(command) or 0
    )
    assert (
        cli.main(
            [
                "eval",
                "planning",
                "--model",
                model_id,
                "--dataset",
                dataset_id,
                "--seed",
                "0",
                "--dry-run",
            ]
        )
        == 0
    )
    command = commands[0]
    model = get_registry().model(model_id)
    task = get_registry().dataset(dataset_id)
    path = cli._planning_config(model, task)
    if model.planner_config_style == "hydra":
        start = (
            next(
                i
                for i, value in enumerate(command)
                if value.startswith("--config-name=")
            )
            + 1
        )
        with initialize_config_dir(config_dir=str(path.parent), version_base=None):
            cfg = compose(config_name=path.stem, overrides=command[start:])
    else:
        start = command.index("--config") + 2
        if command[start : start + 1] == ["--seed"]:
            start += 2
        cfg = OmegaConf.merge(
            OmegaConf.load(path), OmegaConf.from_dotlist(command[start:])
        )

    assert cfg.eval.dataset_name == Path(task.filename).stem
    assert cfg.world.env_name == task.environment or (
        task.id == "cube" and cfg.world.env_name == "swm/OGBCube-v0"
    )
    assert (
        Path(OmegaConf.select(cfg, model.planning_checkpoint_key))
        == default_checkpoint_path(model, task).resolve()
    )
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    )

    if model_id == "lewm":
        cfg.model = resolve_model_config(cfg.model)
        for key in (
            "encoder_scale",
            "patch_size",
            "embed_dim",
            "history_size",
            "action_dim",
            "predictor.depth",
            "predictor.heads",
            "predictor.mlp_dim",
            "predictor.dim_head",
            "predictor.dropout",
            "predictor.emb_dropout",
        ):
            assert (
                OmegaConf.select(cfg.model, key, throw_on_missing=True) is not None
            ), key
        assert cfg.model.action_dim == (
            5 if task.family in {"cube", "insert_peg"} else 2
        )
    elif model_id == "sensorimotor":
        training = OmegaConf.load(resolve_path(cfg.model.config))
        assert cfg.plan_config.action_block == training.data.dataset.frameskip
        assert "seeds" in cfg.eval or "seeds" in cfg.model
        assert "run_name" in cfg.model or "run_name_template" in cfg.model
        for key in (
            "encoder_scale",
            "patch_size",
            "img_size",
            "wm.action_dim",
            "predictor",
            "inverse",
        ):
            assert OmegaConf.select(training, key, throw_on_missing=True) is not None, (
                key
            )
    elif model_id == "prejepa":
        assert "preflight" in cfg
        assert cfg.eval.img_size == (224 if task.family == "insert_peg" else 196)
        if task.family == "insert_peg":
            assert resolve_path(cfg.model.config).is_file()
    elif model_id in {"driftworld", "irasim"}:
        assert cfg.plan_config.action_block == cfg.model.action_frameskip
        assert cfg.plan_config.history_len == cfg.model.num_history_frames
        keys = ["rollout_batch_size", "num_rollout_samples", "rollout_seed"]
        if model_id == "driftworld":
            keys += ["temp_list", "n_neg", "num_future_frames", "decay"]
            keys += (
                ["unet_type"]
                if cfg.model.get("source") == "robomimic"
                else ["unet_name", "action_scale", "absolute_action_max"]
            )
        else:
            keys += [
                "vae_model_path",
                "scheduler_path",
                "beta_start",
                "beta_end",
                "beta_schedule",
                "variance_type",
                "inference_steps",
                "guidance_scale",
                "cost_type",
            ]
        for key in keys:
            assert (
                OmegaConf.select(cfg.model, key, throw_on_missing=True) is not None
            ), key
    else:
        training = OmegaConf.load(cfg.model.config_path)
        assert cfg.plan_config.history_len == training.history_size
        assert cfg.plan_config.action_block == training.data.frameskip
        expected = {
            "toy-adaln": "adaln",
            "toy-action-duplicate": "action_duplicate",
            "toy-driftworld": "drift_state_predictor",
        }
        assert training.model.name == expected[model_id]
        assert training.model.action_dim == (
            5 if task.family in {"cube", "insert_peg"} else 2
        )
        assert resolve_path(
            "resource:train/" + training.data.normalization_path
        ).is_file()
