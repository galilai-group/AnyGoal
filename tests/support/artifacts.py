"""Resolve artifact-backed matrix inputs from packaged configuration."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from anygoal.paths import resolve_path
from anygoal.registry import get_registry
from anygoal.resources import resource_path


def data_path(dataset_id: str, data_root: Path) -> Path:
    """Return the canonical artifact path for a registered dataset."""
    spec = get_registry().dataset(dataset_id)
    path = (data_root / spec.filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing matrix_e2e dataset {dataset_id!r}: {path}")
    return path


def evaluation_profile(model_id: str, dataset_id: str, config_path: Path | None = None):
    """Return the configured model profile for a dataset family."""
    registry = get_registry()
    model = registry.model(model_id)
    dataset = registry.dataset(dataset_id)
    config = OmegaConf.load(config_path or resource_path("eval", "rollout.yaml"))
    OmegaConf.resolve(config)
    return config.models[model.evaluation_id][dataset.family]


def configured_path(profile, key: str) -> Path | None:
    """Expand an optional filesystem path from an evaluation profile."""
    value = profile.get(key)
    if value is None:
        return None
    return resolve_path(str(value)).resolve()
