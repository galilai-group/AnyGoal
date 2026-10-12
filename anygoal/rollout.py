"""Public open-loop rollout evaluation API."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from anygoal.evaluation.rollout import (
    RolloutProtocol,
    evaluate_rollout as evaluate_rollout_protocol,
)
from anygoal.planning import (
    _dataset_path,
    _normalize_seeds,
    _resolve_task,
)
from anygoal.registry import get_registry


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "scripts/eval/config/rollout_eval.yaml"
)
BASE_TASKS = {"pusht", "cube", "tworoom", "insertpeg"}


@dataclass(frozen=True)
class HorizonMetrics:
    horizon_environment_steps: int
    rollout_steps: int
    available: bool
    cumulative_mse: float | None = None
    mean_step_mse: float | None = None
    terminal_mse: float | None = None
    start_weighted_cumulative_mse: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RolloutResult:
    model_id: str
    task_id: str
    seed: int
    error_space: str
    short_horizon_steps: int
    short_horizon_mse: float
    horizons: Mapping[int, HorizonMetrics]
    raw_metrics: Mapping[str, Any]
    output_dir: Path | None = None
    metrics_path: Path | None = None

    @property
    def five_step_mse(self) -> float:
        if self.short_horizon_steps != 5:
            raise ValueError(
                "five_step_mse is available only when frame_skip is 5; use "
                "short_horizon_mse for this result"
            )
        return self.short_horizon_mse


def _load_dataset(data: Any):
    if hasattr(data, "load_episode") and hasattr(data, "lengths"):
        return data
    path = _dataset_path(data)
    if path is None:
        raise TypeError("data must be an HDF5 path or StableWM HDF5Dataset")
    from stable_worldmodel.data import HDF5Dataset

    return HDF5Dataset(path=path, keys_to_cache=["action"])


def _load_config(config: str | Path | None):
    from anygoal.paths import resolve_path

    config_path = resolve_path(config or DEFAULT_CONFIG)
    if not config_path.is_file():
        raise FileNotFoundError(f"Rollout config not found: {config_path}")
    loaded = OmegaConf.load(config_path)
    OmegaConf.resolve(loaded)
    return loaded


def _registered_adapter(
    model_id: str,
    task_spec,
    dataset,
    *,
    config: str | Path | None,
    checkpoint: str | Path | None,
    training_config: str | Path | None,
    planning_config: str | Path | None,
    run_directory: str | Path | None,
    device: str,
    history_length: int,
    model_kwargs: dict[str, Any],
):
    from anygoal.wm.rollout import build_adapter

    model_spec = get_registry().model(model_id)
    model_name = model_spec.evaluation_id
    cfg = _load_config(config)
    base_environment = task_spec.family
    if (
        model_name not in cfg.models
        or base_environment not in cfg.models[model_name]
    ):
        raise ValueError(
            f"No rollout profile for {model_spec.id!r} on "
            f"{base_environment!r}."
        )
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.models[model_name][base_environment], resolve=True)
    )
    if "source_root" in cfg.models[model_name]:
        model_cfg.source_root = cfg.models[model_name].source_root
    unknown = set(model_kwargs) - {"rollout_batch_size"}
    if unknown:
        raise TypeError(
            "Unsupported model-specific rollout options: "
            + ", ".join(sorted(unknown))
        )
    if "rollout_batch_size" in model_kwargs:
        if model_name != "irasim":
            raise ValueError("rollout_batch_size is only supported for IRASim")
        model_cfg.rollout_batch_size = int(model_kwargs["rollout_batch_size"])

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    adapter, checkpoint_path, model_metadata = build_adapter(
        model_name,
        model_cfg,
        dataset,
        device=torch.device(device),
        checkpoint_override=(
            Path(checkpoint).expanduser().resolve() if checkpoint else None
        ),
        training_config_override=(
            Path(training_config).expanduser().resolve() if training_config else None
        ),
        planning_config_override=(
            Path(planning_config).expanduser().resolve() if planning_config else None
        ),
        run_directory_override=(
            Path(run_directory).expanduser().resolve() if run_directory else None
        ),
        history_length=history_length,
    )
    return adapter, model_spec.id, checkpoint_path, model_metadata


def _horizon_metrics(raw_result: Mapping[str, Any]) -> dict[int, HorizonMetrics]:
    converted = {}
    for raw_horizon, values in raw_result["metrics"]["long_horizon"].items():
        horizon = int(raw_horizon)
        available = values.get("available", True)
        converted[horizon] = HorizonMetrics(
            horizon_environment_steps=horizon,
            rollout_steps=int(values["rollout_steps"]),
            available=bool(available),
            cumulative_mse=(
                float(values["cumulative_error"]) if available else None
            ),
            mean_step_mse=(
                float(values["mean_step_error"]) if available else None
            ),
            terminal_mse=(float(values["terminal_error"]) if available else None),
            start_weighted_cumulative_mse=(
                float(values["start_weighted_cumulative_error"])
                if available
                else None
            ),
            reason=values.get("reason"),
        )
    return converted


def evaluate_rollout(
    model_or_adapter: str | Any,
    *,
    data: str | Path | Any,
    task: str | None = None,
    checkpoint: str | Path | None = None,
    seed: int | Sequence[int] = 0,
    history_length: int = 3,
    frame_skip: int = 5,
    horizons: Sequence[int] = (15, 25),
    num_episodes: int = 100,
    batch_size: int = 32,
    max_starts_per_episode: int | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    device: str | None = None,
    config: str | Path | None = None,
    training_config: str | Path | None = None,
    planning_config: str | Path | None = None,
    run_directory: str | Path | None = None,
    episode_start_position: int = 0,
    episode_end_position: int | None = None,
    **model_kwargs: Any,
) -> RolloutResult | dict[int, RolloutResult]:
    """Evaluate 5-step and autoregressive rollout MSE on HDF5 episodes."""
    task_spec, data_path = _resolve_task(task, data)
    seeds, scalar_seed = _normalize_seeds(seed)
    dataset = _load_dataset(data)
    protocol = RolloutProtocol(
        history_length=int(history_length),
        frame_skip=int(frame_skip),
        horizons=tuple(int(value) for value in horizons),
        num_episodes=int(num_episodes),
        batch_size=int(batch_size),
    )

    if isinstance(model_or_adapter, str):
        torch.manual_seed(seeds[0])
        np.random.seed(seeds[0])
        adapter, model_id, checkpoint_path, model_metadata = _registered_adapter(
            model_or_adapter,
            task_spec,
            dataset,
            config=config,
            checkpoint=checkpoint,
            training_config=training_config,
            planning_config=planning_config,
            run_directory=run_directory,
            device=device or "cuda",
            history_length=protocol.history_length,
            model_kwargs=model_kwargs,
        )
    else:
        if checkpoint is not None or any(
            value is not None
            for value in (config, training_config, planning_config, run_directory)
        ):
            raise ValueError(
                "checkpoint and loader configs cannot be used with a supplied adapter"
            )
        if model_kwargs:
            raise TypeError(
                "model-specific options cannot be used with a supplied adapter"
            )
        required = ("rollout_errors", "model_name", "error_space", "observation_keys")
        missing = [name for name in required if not hasattr(model_or_adapter, name)]
        if missing:
            raise TypeError(
                "model_or_adapter does not implement RolloutAdapter: "
                + ", ".join(missing)
            )
        adapter = model_or_adapter
        model_id = str(adapter.model_name)
        checkpoint_path = None
        model_metadata = {}

    output_root = Path(output_dir).expanduser().resolve() if output_dir else None
    run_directories = {
        run_seed: (
            output_root
            if scalar_seed or output_root is None
            else output_root / f"seed_{run_seed}"
        )
        for run_seed in seeds
    }
    for run_directory in run_directories.values():
        metrics_path = run_directory / "metrics.json" if run_directory else None
        if metrics_path is not None and metrics_path.exists() and not overwrite:
            raise FileExistsError(
                f"Rollout result already exists: {metrics_path}. "
                "Pass overwrite=True to replace it."
            )

    dataset_name = (
        Path(str(data_path)).stem
        if data_path is not None
        else Path(task_spec.filename).stem
    )
    metadata = {
        "environment": task_spec.evaluation_id,
        "dataset_name": dataset_name,
        "dataset_path": str(data_path) if data_path is not None else None,
        "base_environment": task_spec.family,
        "variant_identifier": "base" if task_spec.id in BASE_TASKS else task_spec.id,
        "is_base": task_spec.id in BASE_TASKS,
        "checkpoint": (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        ),
        **model_metadata,
    }

    results = {}
    for run_seed in seeds:
        run_directory = run_directories[run_seed]
        if run_directory is not None:
            run_directory.mkdir(parents=True, exist_ok=True)
        raw_result = evaluate_rollout_protocol(
            dataset,
            adapter,
            protocol=protocol,
            seed=run_seed,
            metadata=metadata,
            max_starts_per_episode=max_starts_per_episode,
            progress_path=(
                run_directory / "progress.json" if run_directory else None
            ),
            episode_start_position=episode_start_position,
            episode_end_position=episode_end_position,
        )
        metrics_path = run_directory / "metrics.json" if run_directory else None
        if metrics_path is not None:
            metrics_path.write_text(json.dumps(raw_result, indent=2) + "\n")
        results[run_seed] = RolloutResult(
            model_id=model_id,
            task_id=task_spec.id,
            seed=run_seed,
            error_space=str(raw_result["error_space"]),
            short_horizon_steps=protocol.frame_skip,
            short_horizon_mse=float(raw_result["short_horizon_error"]),
            horizons=_horizon_metrics(raw_result),
            raw_metrics=raw_result,
            output_dir=run_directory,
            metrics_path=metrics_path,
        )
    return results[seeds[0]] if scalar_seed else results


__all__ = ["HorizonMetrics", "RolloutResult", "evaluate_rollout"]
