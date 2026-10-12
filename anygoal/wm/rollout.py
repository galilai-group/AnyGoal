"""Model-specific loaders for AnyGoal rollout evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from anygoal.evaluation._rollout_adapters import (
    DriftWorldRolloutAdapter,
    IRASimRolloutAdapter,
    LatentARRolloutAdapter,
    PreJEPARolloutAdapter,
    TrueStateRolloutAdapter,
    imagenet_preprocess,
)
from anygoal.paths import resolve_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_research_loaders_importable() -> None:
    plan_directory = REPO_ROOT / "scripts" / "plan"
    for source_directory in (REPO_ROOT, plan_directory):
        if str(source_directory) not in sys.path:
            sys.path.insert(0, str(source_directory))


def fit_processor(dataset, key: str) -> StandardScaler:
    values = np.asarray(dataset.get_col_data(key))
    values = values[~np.isnan(values).any(axis=1)]
    if not len(values):
        raise ValueError(f"No finite values available to fit processor for {key!r}.")
    return StandardScaler().fit(values)


def load_planning_config(path: str | Path) -> DictConfig:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Planning config not found: {path}")
    return OmegaConf.load(path)


def _object_checkpoint_location(checkpoint: Path) -> tuple[Path, str]:
    suffix = "_object.ckpt"
    if not checkpoint.name.endswith(suffix):
        raise ValueError(f"Expected an *_object.ckpt checkpoint, got {checkpoint}.")
    return checkpoint.parent, checkpoint.name[: -len(suffix)]


def build_adapter(
    model_name: str,
    model_cfg: DictConfig,
    dataset,
    *,
    device: torch.device,
    checkpoint_override: Path | None,
    training_config_override: Path | None,
    planning_config_override: Path | None,
    run_directory_override: Path | None,
    history_length: int,
):
    """Build the adapter used by ``scripts/eval/rollout.py``."""
    _make_research_loaders_importable()
    action_processor = fit_processor(dataset, "action")

    if model_name == "lewm":
        from planning import (
            patch_loaded_model_compat,
            patch_transformers_vit_pickle_compat,
        )
        from planning_lewm import load_lewm_cost_model

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        if checkpoint.name.endswith("_object.ckpt"):
            checkpoint_root, policy = _object_checkpoint_location(checkpoint)
            cfg.policy = policy
        else:
            checkpoint_root = checkpoint.parent
            cfg.model.checkpoint = str(checkpoint)
            if cfg.model.get("config"):
                cfg.model.config = str(resolve_path(cfg.model.config))
        patch_transformers_vit_pickle_compat()
        model = load_lewm_cost_model(cfg, checkpoint_root)
        patch_loaded_model_compat(model)
        model = model.to(device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        image_size = int(cfg.eval.img_size)
        return (
            LatentARRolloutAdapter(
                model=model,
                model_name="lewm",
                action_processor=action_processor,
                device=device,
                pixel_preprocessor=lambda pixels: imagenet_preprocess(
                    pixels, image_size
                ),
                history_length=history_length,
            ),
            checkpoint,
            {},
        )

    if model_name == "sensorimotor":
        from planning_sensorimotor import load_model

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        if run_directory_override is not None:
            run_directory = run_directory_override
        elif "run_directory" in model_cfg:
            run_directory = resolve_path(model_cfg.run_directory)
        else:
            run_directory = resolve_path(Path(model_cfg.checkpoint).parent.parent)
        source_root = resolve_path(
            model_cfg.get("source_root", cfg.model.source_root)
        )
        checkpoint = checkpoint_override or resolve_path(
            model_cfg.get("checkpoint", run_directory / "checkpoints" / "last.ckpt")
        )
        training_config = training_config_override or resolve_path(
            model_cfg.get("training_config", run_directory / "config.yaml")
        )
        model = load_model(
            run_directory,
            source_root,
            str(device),
            train_config_path=training_config,
            checkpoint_path=checkpoint,
        )
        return (
            LatentARRolloutAdapter(
                model=model,
                model_name="sensorimotor_wm",
                action_processor=action_processor,
                device=device,
                pixel_preprocessor=lambda pixels: pixels,
                history_length=history_length,
            ),
            checkpoint,
            {"run_directory": str(run_directory)},
        )

    if model_name == "prejepa":
        from planning import (
            patch_loaded_model_compat,
            patch_transformers_vit_pickle_compat,
        )
        from planning_prejepa import load_prejepa_cost_model

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        if checkpoint.name.endswith("_object.ckpt"):
            checkpoint_root, policy = _object_checkpoint_location(checkpoint)
            cfg.policy = policy
        else:
            checkpoint_root = checkpoint.parent
            cfg.model.checkpoint = str(checkpoint)
            if cfg.model.get("config"):
                cfg.model.config = str(resolve_path(cfg.model.config))
        patch_transformers_vit_pickle_compat()
        model = load_prejepa_cost_model(cfg, checkpoint_root)
        patch_loaded_model_compat(model)
        model = model.to(device).eval().requires_grad_(False)
        processors = {
            key: fit_processor(dataset, key)
            for key in model.extra_encoders
            if key != "action"
        }
        return (
            PreJEPARolloutAdapter(
                model=model,
                action_processor=action_processor,
                processors=processors,
                device=device,
                image_size=int(cfg.eval.img_size),
                history_length=history_length,
            ),
            checkpoint,
            {"variant": str(cfg.get("variant", cfg.policy))},
        )

    if model_name in {
        "adaln_predictor",
        "action_duplicated_predictor",
        "toy_driftworld",
    }:
        from planning_toy import load_predictor

        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        training_config = training_config_override or resolve_path(
            model_cfg.get("training_config", checkpoint.parent / "config.yaml")
        )
        cfg = OmegaConf.create(
            {
                "device": str(device),
                "model": {
                    "checkpoint_path": str(checkpoint),
                    "config_path": str(training_config),
                },
            }
        )
        model, state_adapter, state_processor, toy_action_processor, train_cfg = (
            load_predictor(cfg)
        )
        trained_model_name = str(train_cfg.model.name)
        expected = {
            "adaln_predictor": "adaln",
            "action_duplicated_predictor": "action_duplicate",
            "toy_driftworld": "drift_state_predictor",
        }[model_name]
        if trained_model_name != expected:
            raise ValueError(
                f"Checkpoint model is {trained_model_name!r}, expected {expected!r}."
            )
        return (
            TrueStateRolloutAdapter(
                model=model,
                model_name=model_name,
                state_adapter=state_adapter,
                state_processor=state_processor,
                action_processor=toy_action_processor,
                device=device,
                history_length=history_length,
            ),
            checkpoint,
            {"state_representation": str(train_cfg.data.name)},
        )

    if model_name == "driftworld":
        from planning_driftworld import load_model
        from anygoal.wm.driftworld import DriftWorldActionProcessor

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        cfg.model.checkpoint = str(checkpoint)
        driftworld_action_processor = DriftWorldActionProcessor.fit(
            dataset.get_col_data("action")
        )
        model, checkpoint_step = load_model(
            cfg, device, driftworld_action_processor
        )
        if int(model.action_frameskip) != 5:
            raise ValueError(
                "Prediction evaluation requires a DriftWorld frameskip-5 checkpoint."
            )
        return (
            DriftWorldRolloutAdapter(
                model=model,
                action_processor=driftworld_action_processor,
                device=device,
                history_length=history_length,
            ),
            checkpoint,
            {"checkpoint_step": checkpoint_step},
        )

    if model_name == "irasim":
        from planning_irasim import load_model
        from anygoal.wm.irasim import IRASimActionProcessor

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        scheduler_override = os.environ.get("IRASIM_SCHEDULER_PATH")
        if scheduler_override:
            cfg.model.scheduler_path = scheduler_override
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        cfg.model.checkpoint = str(checkpoint)
        if "vae_model_path" in model_cfg:
            cfg.model.vae_model_path = str(model_cfg.vae_model_path)
        if "rollout_batch_size" in model_cfg:
            cfg.model.rollout_batch_size = int(model_cfg.rollout_batch_size)
        stats_name = cfg.dataset.get("action_stats_name")
        stats_dataset = dataset
        if stats_name:
            stats_dataset = swm.data.HDF5Dataset(
                str(stats_name),
                cache_dir=Path(cfg.dataset.root).expanduser(),
                keys_to_cache=["action"],
            )
        irasim_action_processor = IRASimActionProcessor.fit(
            stats_dataset.get_col_data("action")
        )
        model, checkpoint = load_model(cfg, device, irasim_action_processor)
        if int(model.action_frameskip) != 5:
            raise ValueError(
                "Prediction evaluation requires an IRASim frameskip-5 checkpoint."
            )
        return (
            IRASimRolloutAdapter(
                model=model,
                action_processor=irasim_action_processor,
                device=device,
                history_length=history_length,
            ),
            checkpoint,
            {
                "inference_steps": int(model.inference_steps),
                "video_size": list(model.video_size),
            },
        )

    raise ValueError(f"Unsupported model: {model_name}")


__all__ = ["build_adapter", "resolve_path"]
