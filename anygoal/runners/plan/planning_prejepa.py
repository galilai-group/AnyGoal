"""Evaluate PushT-trained PreJEPA (DINO-WM) checkpoints with MPC."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from .planning import evaluate


def patch_prejepa_pickle_compat() -> None:
    """Expose legacy flattened PreJEPA symbols without editing site-packages."""
    package = importlib.import_module("stable_worldmodel.wm.prejepa")
    for module_name in ("module", "prejepa"):
        source = importlib.import_module(
            f"stable_worldmodel.wm.prejepa.{module_name}"
        )
        for name in dir(source):
            if not name.startswith("_"):
                setattr(package, name, getattr(source, name))


def load_prejepa_cost_model(cfg: DictConfig, checkpoint_root: str | Path):
    patch_prejepa_pickle_compat()
    explicit_checkpoint = cfg.get("model", {}).get("checkpoint")
    if explicit_checkpoint:
        checkpoint_path = Path(str(explicit_checkpoint)).expanduser()
        config_path = Path(str(cfg.model.config)).expanduser()
        if not checkpoint_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Missing PreJEPA checkpoint or config: {checkpoint_path}, {config_path}"
            )
        model = hydra.utils.instantiate(OmegaConf.load(config_path))
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        model_label = str(checkpoint_path)
    else:
        model = swm.policy.AutoCostModel(cfg.policy, cache_dir=checkpoint_root)
        model_label = str(cfg.policy)
    modality_names = list(model.extra_encoders)
    if "action" not in modality_names:
        raise RuntimeError(
            f"PreJEPA checkpoint {cfg.policy!r} has no action encoder: "
            f"{modality_names}"
        )
    patch_size = model.backbone.config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    patch_grid_size = math.isqrt(int(model.predictor.num_patches))
    if patch_grid_size**2 != int(model.predictor.num_patches):
        raise RuntimeError(
            f"Predictor has a non-square patch count: {model.predictor.num_patches}"
        )
    expected_image_size = patch_grid_size * int(patch_size)
    if int(cfg.eval.img_size) != expected_image_size:
        raise ValueError(
            f"Checkpoint {cfg.policy!r} expects {expected_image_size}x"
            f"{expected_image_size} inputs ({model.predictor.num_patches} patches, "
            f"patch size {patch_size}), but eval.img_size={cfg.eval.img_size}."
        )
    print(
        f"Loaded {model_label}: {type(model).__module__}.{type(model).__name__}; "
        f"extra encoders={modality_names}; image size={expected_image_size}"
    )
    return model


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def checkpoint_summary(cfg: DictConfig) -> dict:
    checkpoint_root = cfg.cache_dir or swm.data.utils.get_cache_dir()
    model = load_prejepa_cost_model(cfg, checkpoint_root)
    return {
        "policy": cfg.policy,
        "checkpoint_root": str(checkpoint_root),
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "extra_encoders": list(model.extra_encoders),
        "backbone_patch_size": model.backbone.config.patch_size,
        "predictor_num_patches": model.predictor.num_patches,
        "predictor_num_frames": model.predictor.num_frames,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


@hydra.main(
    version_base=None,
    config_path="../../resources/plan/prejepa",
    config_name="pusht",
)
def run(cfg: DictConfig):
    if cfg.preflight:
        summary = checkpoint_summary(cfg)
        print(json.dumps(summary, indent=2))
        return summary

    metrics, evaluation_time = evaluate(cfg, load_prejepa_cost_model)
    output_directory = Path(cfg.output.directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "prejepa",
        "alias": "dino-wm",
        "policy": cfg.policy,
        "seed": int(cfg.seed),
        "metrics": jsonable(metrics),
        "evaluation_time_seconds": evaluation_time,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved structured metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    run()
