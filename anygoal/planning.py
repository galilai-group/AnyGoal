"""Thin Stable Worldmodel planning evaluation API."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from anygoal.datasets.validation import validate_hdf5
from anygoal.registry import DatasetSpec, get_registry


DEFAULT_NUM_EVALS = 50
DEFAULT_GOAL_OFFSET_STEPS = 25
DEFAULT_EVAL_BUDGET = 50
DEFAULT_IMAGE_SHAPE = (224, 224)


@dataclass(frozen=True)
class PlanningResult:
    """Result and exact dataset trials used by :func:`plan`."""

    task_id: str
    data: Path | str | None
    metrics: Mapping[str, Any]
    episodes: tuple[int, ...]
    start_steps: tuple[int, ...]
    seed: int
    elapsed_seconds: float
    output_dir: Path | None = None
    metrics_path: Path | None = None


def _attr_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _dataset_path(data: Any) -> Path | str | None:
    raw = getattr(data, "h5_path", data)
    if isinstance(raw, str) and "://" in raw:
        return raw
    if isinstance(raw, (str, Path)):
        return Path(raw).expanduser().resolve()
    return None


def _infer_task(path: Path) -> DatasetSpec | None:
    registry = get_registry()
    filename_matches = [
        spec for spec in registry.datasets.values() if spec.filename == path.name
    ]
    inferred = filename_matches[0] if len(filename_matches) == 1 else None
    with h5py.File(path, "r", swmr=True) as handle:
        identifiers = {
            _attr_text(handle.attrs[name])
            for name in ("anygoal_dataset_id", "task", "environment")
            if name in handle.attrs
        }
    attribute_matches = []
    for spec in registry.datasets.values():
        known = {spec.id, spec.evaluation_id, spec.environment, *spec.aliases}
        if identifiers & known:
            attribute_matches.append(spec)
    if len(attribute_matches) == 1:
        attribute_task = attribute_matches[0]
        if inferred is not None and inferred.id != attribute_task.id:
            raise ValueError(
                f"Dataset filename implies {inferred.id!r}, but HDF5 attributes "
                f"imply {attribute_task.id!r}."
            )
        inferred = attribute_task
    return inferred


def _resolve_task(
    task: str | None, data: Any
) -> tuple[DatasetSpec, Path | str | None]:
    path = _dataset_path(data)
    inferred = None
    if isinstance(path, Path):
        if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
            raise FileNotFoundError(f"HDF5 planning data not found: {path}")
        inferred = _infer_task(path)
    requested = get_registry().dataset(task) if task else inferred
    if requested is None:
        raise ValueError(
            "task could not be inferred from the HDF5 filename or metadata; "
            "pass task explicitly"
        )
    if isinstance(path, Path):
        if inferred is not None and inferred.id != requested.id:
            raise ValueError(
                f"task={requested.id!r} does not match the dataset identity "
                f"{inferred.id!r} inferred from {path}."
            )
        report = validate_hdf5(path, dataset_id=requested.id)
        if not report.valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise ValueError(f"Planning dataset validation failed: {messages}")
    return requested, path


def _normalize_seeds(seed: int | Sequence[int]) -> tuple[tuple[int, ...], bool]:
    if isinstance(seed, Integral) and not isinstance(seed, bool):
        return (int(seed),), True
    if isinstance(seed, (str, bytes)):
        raise TypeError("seed must be an integer or a sequence of integers")
    try:
        raw_seeds = tuple(seed)
    except TypeError as error:
        raise TypeError("seed must be an integer or a sequence of integers") from error
    if not raw_seeds:
        raise ValueError("seed sequence must not be empty")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in raw_seeds
    ):
        raise TypeError("every seed must be an integer")
    seeds = tuple(int(value) for value in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed sequence must not contain duplicates")
    return seeds, False


def _load_dataset(data: Any):
    if hasattr(data, "load_chunk") and hasattr(data, "lengths"):
        return data
    path = _dataset_path(data)
    if path is None:
        raise TypeError("data must be an HDF5 path or StableWM HDF5Dataset")
    from stable_worldmodel.data import HDF5Dataset

    return HDF5Dataset(path=path)


def _select_starts(
    dataset,
    *,
    num_evals: int,
    goal_offset_steps: int,
    seed: int,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    if num_evals <= 0:
        raise ValueError("num_evals must be positive")
    if goal_offset_steps <= 0:
        raise ValueError("goal_offset_steps must be positive")
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    eligible = lengths - int(goal_offset_steps)

    if selection == "first_episodes":
        episodes = np.flatnonzero(eligible > 0)[:num_evals]
        if len(episodes) < num_evals:
            raise ValueError(
                f"Not enough eligible episodes: found {len(episodes)}, need {num_evals}."
            )
        return episodes.astype(np.int64), np.zeros(num_evals, dtype=np.int64)
    if selection != "random":
        raise ValueError("start_selection must be 'random' or 'first_episodes'")

    counts = np.maximum(eligible, 0)
    candidate_episodes = np.repeat(np.arange(len(lengths)), counts)
    candidate_starts = (
        np.concatenate([np.arange(count, dtype=np.int64) for count in counts])
        if len(counts)
        else np.empty(0, dtype=np.int64)
    )
    if len(candidate_episodes) < num_evals:
        raise ValueError(
            f"Not enough valid starting points: found {len(candidate_episodes)}, "
            f"need {num_evals}."
        )
    chosen = np.sort(
        np.random.default_rng(seed).choice(
            len(candidate_episodes), size=num_evals, replace=False
        )
    )
    return candidate_episodes[chosen], candidate_starts[chosen]


def _task_callables(task: DatasetSpec) -> list[dict[str, Any]]:
    if task.family == "pusht":
        return [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ]
    if task.family == "tworoom":
        return [
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_proprio"}},
            },
        ]
    if task.family == "insert_peg":
        return [
            {"method": "set_state", "args": {"state": {"value": "observation"}}},
            {
                "method": "set_evaluation_goal",
                "args": {"goal_state": {"value": "goal_observation"}},
            },
        ]
    if task.family == "cube":
        callables = [
            {
                "method": "set_state",
                "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
            }
        ]
        if task.id == "cube":
            callables.append(
                {
                    "method": "set_target_pos",
                    "args": {
                        "cube_id": {"value": 0, "in_dataset": False},
                        "target_pos": {"value": "goal_privileged_block_0_pos"},
                        "target_quat": {"value": "goal_privileged_block_0_quat"},
                    },
                }
            )
        else:
            callables.append(
                {
                    "method": "set_evaluation_goal",
                    "args": {
                        "effector_pos": {"value": "goal_proprio_effector_pos"},
                        "effector_yaw": {"value": "goal_proprio_effector_yaw"},
                        "cube_pos": {"value": "goal_privileged_block_0_pos"},
                    },
                }
            )
        return callables
    raise ValueError(f"No planning protocol is registered for task {task.id!r}.")


def _validate_planning_columns(dataset, task: DatasetSpec) -> None:
    required = {
        "pusht": {"pixels", "action", "state"},
        "tworoom": {"pixels", "action", "proprio"},
        "insert_peg": {"pixels", "action", "observation"},
        "cube": {"pixels", "action", "qpos", "qvel", "privileged_block_0_pos"},
    }[task.family]
    if task.id == "cube":
        required.add("privileged_block_0_quat")
    elif task.family == "cube":
        required.update({"proprio_effector_pos", "proprio_effector_yaw"})
    missing = sorted(required.difference(dataset.column_names))
    if missing:
        raise ValueError(
            f"Dataset is missing columns required to plan {task.id!r}: "
            + ", ".join(missing)
        )


def _world_kwargs(
    task: DatasetSpec, num_evals: int, eval_budget: int
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "env_name": "swm/OGBCube-v0" if task.id == "cube" else task.environment,
        "num_envs": num_evals,
        "max_episode_steps": 2 * eval_budget,
        "image_shape": DEFAULT_IMAGE_SHAPE,
    }
    if task.family == "cube":
        kwargs.update(
            env_type="single",
            ob_type="states",
            multiview=False,
            width=224,
            height=224,
            visualize_info=False,
            terminate_at_goal=True,
        )
        if task.id != "cube":
            kwargs.update(
                evaluation_ee_position_tolerance=0.04,
                evaluation_ee_yaw_tolerance=0.10,
                evaluation_cube_position_tolerance=0.04,
            )
    elif task.family == "insert_peg":
        kwargs.update(width=224, height=224, terminate_at_goal=True)
    return kwargs


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def plan(
    policy: Any,
    *,
    data: str | Path | Any,
    task: str | None = None,
    seed: int | Sequence[int] = 0,
    num_evals: int = DEFAULT_NUM_EVALS,
    goal_offset_steps: int = DEFAULT_GOAL_OFFSET_STEPS,
    eval_budget: int = DEFAULT_EVAL_BUDGET,
    start_selection: str = "random",
    output_dir: str | Path | None = None,
    save_video: bool = False,
    save_artifacts: bool = False,
    overwrite: bool = False,
    world: Any | None = None,
) -> PlanningResult | dict[int, PlanningResult]:
    """Evaluate a caller-built StableWM policy on an AnyGoal HDF5 task.

    Model, cost, solver, and ``PlanConfig`` construction deliberately remain
    Stable Worldmodel concerns. This function only selects deterministic
    dataset trials, configures the task environment, calls ``World.evaluate``,
    and attaches AnyGoal task metrics. ``seed`` controls dataset start-point
    sampling; solver randomness remains owned by the supplied policy.
    """
    if eval_budget <= 0:
        raise ValueError("eval_budget must be positive")
    task_spec, data_path = _resolve_task(task, data)
    seeds, scalar_seed = _normalize_seeds(seed)
    dataset = _load_dataset(data)
    _validate_planning_columns(dataset, task_spec)

    output_root = Path(output_dir).expanduser().resolve() if output_dir else None
    if (save_video or save_artifacts) and output_root is None:
        raise ValueError("output_dir is required when saving video or artifacts")
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
                f"Planning result already exists: {metrics_path}. "
                "Pass overwrite=True to replace it."
            )

    from stable_worldmodel import World
    from anygoal.envs.cube_hop.env import (
        add_cube_evaluation_metrics,
        prepare_cube_evaluation,
    )
    from anygoal.envs.insert_peg.env import add_insert_peg_evaluation_metrics
    from anygoal._planning_artifacts import PlanningArtifactRecorder

    owns_world = world is None
    if world is None:
        world = World(**_world_kwargs(task_spec, num_evals, eval_budget))
    elif world.num_envs != num_evals:
        raise ValueError(
            f"world.num_envs={world.num_envs} must equal num_evals={num_evals}."
        )

    policy_config = getattr(policy, "cfg", getattr(policy, "config", None))
    if policy_config is not None and all(
        hasattr(policy_config, name) for name in ("horizon", "action_block")
    ):
        planning_span = int(policy_config.horizon) * int(policy_config.action_block)
        if planning_span > eval_budget:
            raise ValueError(
                "policy config horizon * action_block must not exceed eval_budget"
            )

    results: dict[int, PlanningResult] = {}
    try:
        for run_seed in seeds:
            episodes, start_steps = _select_starts(
                dataset,
                num_evals=num_evals,
                goal_offset_steps=goal_offset_steps,
                seed=run_seed,
                selection=start_selection,
            )
            run_directory = run_directories[run_seed]
            metrics_path = run_directory / "metrics.json" if run_directory else None
            recorder = None
            try:
                world.set_policy(policy)
                callables = prepare_cube_evaluation(
                    world, _task_callables(task_spec)
                )
                if run_directory is not None:
                    run_directory.mkdir(parents=True, exist_ok=True)
                if save_artifacts:
                    recorder = PlanningArtifactRecorder(
                        world,
                        dataset,
                        episodes,
                        start_steps,
                        goal_offset_steps,
                        run_directory / "artifacts",
                        run_seed,
                    )
                    recorder.install()
                started = time.monotonic()
                metrics = world.evaluate(
                    dataset=dataset,
                    start_steps=start_steps.tolist(),
                    goal_offset=goal_offset_steps,
                    eval_budget=eval_budget,
                    episodes_idx=episodes.tolist(),
                    callables=callables,
                    video=run_directory / "videos" if save_video else None,
                )
                metrics = add_cube_evaluation_metrics(world, metrics)
                metrics = add_insert_peg_evaluation_metrics(world, metrics)
                elapsed = time.monotonic() - started
                if recorder is not None:
                    recorder.save(metrics)
                if metrics_path is not None:
                    payload = {
                        "task": task_spec.id,
                        "data": str(data_path) if data_path is not None else None,
                        "seed": run_seed,
                        "episodes": episodes.tolist(),
                        "start_steps": start_steps.tolist(),
                        "goal_offset_steps": goal_offset_steps,
                        "eval_budget": eval_budget,
                        "elapsed_seconds": elapsed,
                        "metrics": metrics,
                    }
                    metrics_path.write_text(
                        json.dumps(_json_value(payload), indent=2) + "\n"
                    )
                results[run_seed] = PlanningResult(
                    task_id=task_spec.id,
                    data=data_path,
                    metrics=metrics,
                    episodes=tuple(int(value) for value in episodes),
                    start_steps=tuple(int(value) for value in start_steps),
                    seed=run_seed,
                    elapsed_seconds=elapsed,
                    output_dir=run_directory,
                    metrics_path=metrics_path,
                )
            finally:
                if recorder is not None:
                    recorder.restore()
    finally:
        if owns_world:
            world.close()
    return results[seeds[0]] if scalar_seed else results


__all__ = ["PlanningResult", "plan"]
