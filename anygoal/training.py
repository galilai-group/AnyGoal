"""Unified, reproducible training entry point for verified AnyGoal profiles."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import hydra
import torch
from omegaconf import OmegaConf

from anygoal.datasets.validation import validate_hdf5
from anygoal.registry import DatasetSpec, ModelSpec, get_registry
from anygoal.resources import resource_path
from anygoal.sources import source_path


BASE_TASKS = {"pusht", "cube", "tworoom", "insertpeg"}
TOY_DATA_CONFIG = {
    "pusht": "pusht",
    "cube": "cube_everything",
    "tworoom": "tworoom_essential",
    "insertpeg": "insert_peg_essential",
}
TOY_DRIFT_DATA_CONFIG = {
    "pusht": "pusht",
    "cube": "cube",
    "tworoom": "tworoom",
    "insertpeg": "insert_peg_essential",
}

COMMON_OVERRIDES: dict[str, dict[str, str]] = {
    "lewm": {
        "learning_rate": "optimizer.lr",
        "batch_size": "loader.batch_size",
        "seed": "seed",
        "num_workers": "num_workers",
        "max_epochs": "trainer.max_epochs",
    },
    "sensorimotor": {
        "learning_rate": "optimizer.lr",
        "batch_size": "loader.batch_size",
        "seed": "seed",
        "num_workers": "num_workers",
        "max_epochs": "trainer.max_epochs",
    },
    "driftworld": {
        "learning_rate": "opt.lr",
        "batch_size": "data.batch_size",
        "seed": "train.seed",
        "num_workers": "dataloader.num_workers",
        "max_steps": "train.max_steps",
        "max_epochs": "train.num_epochs",
        "output_dir": "output.checkpoint_root",
    },
    "irasim": {
        "learning_rate": "learning_rate",
        "batch_size": "local_batch_size",
        "seed": "global_seed",
        "num_workers": "num_workers",
        "max_steps": "max_train_steps",
        "output_dir": "results_dir",
    },
    "prejepa": {
        "learning_rate": "optimizer.lr",
        "batch_size": "batch_size",
        "seed": "seed",
        "num_workers": "num_workers",
        "max_epochs": "trainer.max_epochs",
    },
    "toy-adaln": {
        "learning_rate": "optimizer.lr",
        "batch_size": "loader.batch_size",
        "seed": "seed",
        "num_workers": "loader.num_workers",
        "max_epochs": "trainer.max_epochs",
        "output_dir": "output.checkpoint_root",
    },
    "toy-action-duplicate": {
        "learning_rate": "optimizer.lr",
        "batch_size": "loader.batch_size",
        "seed": "seed",
        "num_workers": "loader.num_workers",
        "max_epochs": "trainer.max_epochs",
        "output_dir": "output.checkpoint_root",
    },
    "toy-driftworld": {
        "learning_rate": "optimizer.lr",
        "batch_size": "loader.batch_size",
        "seed": "seed",
        "num_workers": "loader.num_workers",
        "max_steps": "train.max_steps",
        "max_epochs": "train.num_epochs",
        "output_dir": "output.checkpoint_root",
    },
}

DATA_OVERRIDES = {
    "lewm": "data.dataset.name",
    "sensorimotor": "data.dataset.name",
    "driftworld": "data.dataset_name",
    "irasim": "h5_dataset_name",
    "prejepa": "dataset_name",
    "toy-adaln": "data.dataset_name",
    "toy-action-duplicate": "data.dataset_name",
    "toy-driftworld": "data.dataset_name",
}

LEWM_DATA_CONFIG = {
    "pusht": "pusht",
    "cube": "ogb",
    "tworoom": "tworoom",
    "insertpeg": "dmc",
}
OFFICIAL_CHECKPOINT_TASKS = {
    (model_id, task_id)
    for model_id in ("lewm", "prejepa")
    for task_id in ("pusht", "cube", "tworoom")
}


def _warn_for_official_checkpoint_training(model_id: str, task_id: str) -> None:
    if (model_id, task_id) not in OFFICIAL_CHECKPOINT_TASKS:
        return
    warnings.warn(
        "⚠️⚠️ AnyGoal used the official checkpoint released by LeWM for "
        f"{model_id} x {task_id}. A newly trained checkpoint may not reproduce "
        "the results reported in the AnyGoal paper. ⚠️⚠️",
        UserWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class TrainingPlan:
    model: ModelSpec
    task: DatasetSpec
    data: Path
    config: Path | None
    command: tuple[str, ...]
    overrides: tuple[str, ...]
    output_root: Path
    environment: tuple[tuple[str, str], ...] = ()

    def shell_command(self) -> str:
        command = shlex.join(self.command)
        if not self.environment:
            return command
        prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in self.environment
        )
        return f"{prefix} {command}"


@dataclass(frozen=True)
class TrainingResult:
    model_id: str
    task_id: str
    data: Path
    run_directory: Path
    checkpoint: Path
    config: Path
    command: tuple[str, ...]
    elapsed_seconds: float


def _data_path(data: str | Path | Any) -> Path:
    raw = getattr(data, "h5_path", data)
    if isinstance(raw, str) and "://" in raw:
        raise ValueError("AnyGoal training currently requires a local HDF5 file.")
    try:
        path = Path(raw).expanduser().resolve()
    except TypeError as error:
        raise TypeError(
            "data must be an HDF5 path or StableWM HDF5Dataset"
        ) from error
    if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
        raise FileNotFoundError(f"HDF5 training data not found: {path}")
    return path


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _infer_task(path: Path) -> DatasetSpec | None:
    registry = get_registry()
    filename_matches = [
        item for item in registry.datasets.values() if item.filename == path.name
    ]
    inferred = filename_matches[0] if len(filename_matches) == 1 else None
    with h5py.File(path, "r", swmr=True) as handle:
        values = {
            _attr_text(handle.attrs[name])
            for name in ("anygoal_dataset_id", "task", "environment")
            if name in handle.attrs
        }
    attribute_matches = []
    for item in registry.datasets.values():
        identifiers = {item.id, item.evaluation_id, item.environment, *item.aliases}
        if values & identifiers:
            attribute_matches.append(item)
    if len(attribute_matches) == 1:
        if inferred is not None and inferred.id != attribute_matches[0].id:
            raise ValueError(
                f"Dataset filename implies {inferred.id!r}, but HDF5 attributes imply "
                f"{attribute_matches[0].id!r}."
            )
        inferred = attribute_matches[0]
    return inferred


def _resolve_task(task: str, data_path: Path) -> DatasetSpec:
    if not task:
        raise ValueError("task is required for training")
    requested = get_registry().dataset(task)
    inferred = _infer_task(data_path)
    if inferred is not None and inferred.id != requested.id:
        raise ValueError(
            f"task={requested.id!r} does not match the dataset identity "
            f"{inferred.id!r} inferred from {data_path}."
        )
    report = validate_hdf5(data_path, dataset_id=requested.id)
    if not report.valid:
        messages = "; ".join(issue.message for issue in report.issues)
        raise ValueError(f"Training dataset validation failed: {messages}")
    return requested


def _format_override(key: str, value: Any) -> str:
    if isinstance(value, Path):
        value = str(value.expanduser().resolve())
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif value is None:
        encoded = "null"
    elif isinstance(value, (dict, list, tuple)):
        encoded = json.dumps(value, separators=(",", ":"))
    else:
        encoded = str(value)
    return f"{key}={encoded}"


def _default_output_root(model_id: str, config: Path | None) -> Path:
    if model_id in {"lewm", "prejepa"}:
        stablewm_home = Path(
            os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")
        ).expanduser()
        return (stablewm_home / "checkpoints").resolve()
    if config is not None:
        cfg = OmegaConf.load(config)
        candidates = {
            "driftworld": "output.checkpoint_root",
            "irasim": "results_dir",
            "toy-adaln": "output.checkpoint_root",
            "toy-action-duplicate": "output.checkpoint_root",
            "toy-driftworld": "output.checkpoint_root",
        }
        key = candidates.get(model_id)
        value = OmegaConf.select(cfg, key) if key else None
        if value is not None:
            try:
                return Path(str(value)).expanduser().resolve()
            except TypeError:
                pass
    return Path("~/.cache/anygoal/checkpoints").expanduser().resolve()


def prepare_training(
    model: str,
    *,
    data: str | Path | Any,
    task: str,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    num_workers: int | None = None,
    output_dir: str | Path | None = None,
    max_steps: int | None = None,
    max_epochs: int | None = None,
    **kwargs: Any,
) -> TrainingPlan:
    registry = get_registry()
    model_spec = registry.model(model)
    data_path = _data_path(data)
    task_spec = _resolve_task(task, data_path)
    if task_spec.family not in model_spec.families:
        raise ValueError(
            f"{model_spec.id} does not support the {task_spec.family} task family."
        )
    if task_spec.id not in BASE_TASKS:
        raise ValueError(
            f"{task_spec.id} has no verified training profile; "
            "use its base task explicitly."
        )
    if task_spec.id not in model_spec.training_profiles:
        raise ValueError(
            f"No verified training profile for {model_spec.id} x {task_spec.id}."
        )
    profile = model_spec.training_profiles[task_spec.id]
    config = None if profile is None else resource_path(*profile.split("/"))
    if config is not None and not config.is_file():
        raise FileNotFoundError(config)

    common_values = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "num_workers": num_workers,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "max_epochs": max_epochs,
    }
    mapping = COMMON_OVERRIDES[model_spec.id]
    supported_common = set(mapping)
    if model_spec.id in {"lewm", "prejepa", "sensorimotor"}:
        supported_common.add("output_dir")
    unsupported = [
        name
        for name, value in common_values.items()
        if value is not None and name not in supported_common
    ]
    if unsupported:
        raise ValueError(
            f"{model_spec.id} does not support common option(s): "
            f"{', '.join(unsupported)}"
        )
    overrides = []
    if model_spec.id == "lewm":
        overrides.extend(
            [
                f"data={LEWM_DATA_CONFIG[task_spec.id]}",
                f"output_model_name=anygoal_lewm_{task_spec.id}",
                f"subdir=anygoal_lewm_{task_spec.id}",
            ]
        )
        if task_spec.id == "insertpeg" and max_epochs is None:
            overrides.append("trainer.max_epochs=10")
    elif model_spec.id == "toy-adaln":
        overrides.extend(
            [f"data={TOY_DATA_CONFIG[task_spec.id]}", "model=adaln"]
        )
    elif model_spec.id == "toy-action-duplicate":
        overrides.extend(
            [f"data={TOY_DATA_CONFIG[task_spec.id]}", "model=action_duplicate"]
        )
    elif model_spec.id == "toy-driftworld":
        overrides.append(f"data={TOY_DRIFT_DATA_CONFIG[task_spec.id]}")
    elif model_spec.id == "prejepa":
        overrides.extend(
            [
                "~wm.encoding.proprio",
                "wm.history_size=3",
                "wm.num_preds=1",
                "frameskip=5",
                f"image_size={196 if task_spec.id != 'insertpeg' else 224}",
                f"output_model_name=anygoal_prejepa_{task_spec.id}",
            ]
        )
    overrides.append(_format_override(DATA_OVERRIDES[model_spec.id], data_path))
    for name, value in common_values.items():
        if value is not None and name in mapping:
            overrides.append(_format_override(mapping[name], value))
    for name, value in kwargs.items():
        key = name.replace("__", ".")
        protected = {DATA_OVERRIDES[model_spec.id], *mapping.values()}
        if key in protected:
            common_name = next(
                (
                    option
                    for option, target in mapping.items()
                    if target == key
                ),
                "data",
            )
            raise ValueError(
                f"Use the common {common_name} "
                f"argument instead of model-specific override {key!r}."
            )
        overrides.append(_format_override(key, value))

    command = [sys.executable, "-m", model_spec.trainer]
    if config is not None:
        if model_spec.trainer_config_style == "argparse":
            command.extend(["--config", str(config)])
        else:
            command.extend(
                ["--config-path", str(config.parent), "--config-name", config.stem]
            )
    command.extend(overrides)
    environment: tuple[tuple[str, str], ...] = ()
    if model_spec.id in {"lewm", "prejepa"}:
        stablewm_home = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel"))
            .expanduser()
            .resolve()
        )
        environment = (("STABLEWM_HOME", str(stablewm_home)),)
        output_root = stablewm_home / "checkpoints"
    elif model_spec.id == "sensorimotor":
        output_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else _default_output_root(model_spec.id, config)
        )
        environment = (("RUNS_ROOT", str(output_root)),)
    else:
        output_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else _default_output_root(model_spec.id, config)
        )
    return TrainingPlan(
        model=model_spec,
        task=task_spec,
        data=data_path,
        config=config,
        command=tuple(command),
        overrides=tuple(overrides),
        output_root=output_root,
        environment=environment,
    )


def _snapshot_checkpoints(root: Path) -> dict[Path, tuple[int, int]]:
    if not root.exists():
        return {}
    suffixes = {".pt", ".pth", ".ckpt"}
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }


def _select_checkpoint(model_id: str, candidates: set[Path]) -> Path:
    if not candidates:
        raise RuntimeError(
            "Training completed without producing a discoverable checkpoint."
        )
    preferences = {
        "driftworld": ("ckpt-latest.pth",),
        "lewm": ("weights_epoch_",),
        "sensorimotor": ("last.ckpt",),
        "toy-adaln": ("weights_epoch_",),
        "toy-action-duplicate": ("weights_epoch_",),
        "toy-driftworld": ("weights_step_",),
        "prejepa": ("weights_epoch_",),
    }.get(model_id, ())
    preferred = [
        path
        for path in candidates
        if any(
            path.name == item or path.name.startswith(item)
            for item in preferences
        )
    ]
    pool = preferred or list(candidates)
    return max(pool, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _find_saved_config(
    checkpoint: Path,
    fallback: Path | None,
    *,
    prefer_model_config: bool = False,
) -> Path:
    names = (
        ("config.json", "config.yaml", "config.yml")
        if prefer_model_config
        else ("config.yaml", "config.yml", "config.json")
    )
    for directory in (checkpoint.parent, *list(checkpoint.parents)[1:5]):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"No saved training config found for {checkpoint}")


def _load_toy_model(checkpoint: Path, config_path: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("name", None)
    model = hydra.utils.instantiate(model_config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _load_driftworld_model(
    checkpoint: Path, config_path: Path, data: Path, device: torch.device
):
    import stable_worldmodel as swm
    from anygoal.wm.driftworld import (
        DriftWorldActionProcessor,
        DriftWorldCubeWorldModel,
        DriftWorldWorldModel,
    )

    cfg = OmegaConf.load(config_path)
    dataset = swm.data.HDF5Dataset(path=data, keys_to_cache=["action"])
    processor = DriftWorldActionProcessor.fit(dataset.get_col_data("action"))
    common = dict(
        temp_list=tuple(cfg.model.temp_list),
        n_neg=int(cfg.model.n_neg),
        num_future_frames=int(cfg.model.num_future_frames),
        num_history_frames=int(cfg.model.num_history_frames),
        decay=float(cfg.train.decay),
        action_frameskip=int(cfg.data.frameskip),
        action_mode=str(cfg.data.action_mode),
        action_processor=processor,
    )
    if str(cfg.model.get("source", "pusht")) == "robomimic":
        model = DriftWorldCubeWorldModel(unet_type=str(cfg.model.unet_type), **common)
    else:
        model = DriftWorldWorldModel(unet_name=str(cfg.model.unet_name), **common)
    model.load_native_checkpoint(checkpoint)
    return model.to(device).eval().requires_grad_(False)


def _load_irasim_model(
    checkpoint: Path,
    config_path: Path,
    data: Path,
    task_id: str,
    device: torch.device,
):
    import stable_worldmodel as swm
    from anygoal.runners.plan import planning_irasim

    stem = "insert_peg" if task_id == "insertpeg" else task_id
    planning_config = resource_path("plan", "irasim", f"{stem}.yaml")
    training_config = OmegaConf.load(config_path)
    training_to_planning = {
        "vae_model_path": "model.vae_model_path",
        "scheduler_path": "model.scheduler_path",
        "frameskip": "model.action_frameskip",
        "mask_frame_num": "model.num_history_frames",
    }
    resource_overrides = []
    for training_key, planning_key in training_to_planning.items():
        value = OmegaConf.select(training_config, training_key)
        if value is not None:
            resource_overrides.append(_format_override(planning_key, value))
    cfg = planning_irasim.load_config(
        planning_config,
        [
            _format_override("model.checkpoint", checkpoint),
            _format_override("eval.dataset_name", data),
            *resource_overrides,
        ],
    )
    dataset = swm.data.HDF5Dataset(path=data, keys_to_cache=["action"])
    action_processor = planning_irasim.get_action_processor(cfg, dataset)
    loaded, _ = planning_irasim.load_model(cfg, device, action_processor)
    return loaded


def _load_prejepa_model(
    checkpoint: Path, config_path: Path, device: torch.device
):
    package = importlib.import_module("stable_worldmodel.wm.prejepa")
    for module_name in ("module", "prejepa"):
        source = importlib.import_module(
            f"stable_worldmodel.wm.prejepa.{module_name}"
        )
        for name in dir(source):
            if not name.startswith("_"):
                setattr(package, name, getattr(source, name))
    model = hydra.utils.instantiate(OmegaConf.load(config_path))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _load_lewm_model(checkpoint: Path, config_path: Path, device: torch.device):
    source_root = source_path("lewm")
    previous_modules = {
        name: sys.modules.pop(name, None) for name in ("jepa", "module")
    }
    sys.path.insert(0, str(source_root))
    try:
        importlib.import_module("jepa")
        importlib.import_module("module")
        model = hydra.utils.instantiate(OmegaConf.load(config_path))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    finally:
        sys.path.pop(0)
        for name in ("jepa", "module"):
            sys.modules.pop(name, None)
            if previous_modules[name] is not None:
                sys.modules[name] = previous_modules[name]
    return model.to(device).eval().requires_grad_(False)


def _load_sensorimotor_model(
    checkpoint: Path, config_path: Path, device: torch.device
):
    from anygoal.runners.plan import planning_sensorimotor

    source_root = source_path("sensorimotor") / "planning"
    return planning_sensorimotor.load_model(
        checkpoint.parent.parent,
        source_root,
        str(device),
        train_config_path=config_path,
        checkpoint_path=checkpoint,
    )


def load_trained_model(
    model_id: str,
    checkpoint: Path,
    config_path: Path,
    data: Path,
    *,
    task_id: str | None = None,
    device: str | torch.device | None = None,
):
    target = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if model_id == "driftworld":
        return _load_driftworld_model(checkpoint, config_path, data, target)
    if model_id == "irasim":
        if task_id is None:
            raise ValueError("task_id is required to load an IRASim model")
        return _load_irasim_model(
            checkpoint, config_path, data, task_id, target
        )
    if model_id == "prejepa":
        return _load_prejepa_model(checkpoint, config_path, target)
    if model_id == "lewm":
        return _load_lewm_model(checkpoint, config_path, target)
    if model_id == "sensorimotor":
        return _load_sensorimotor_model(checkpoint, config_path, target)
    if model_id in {"toy-adaln", "toy-action-duplicate", "toy-driftworld"}:
        return _load_toy_model(checkpoint, config_path, target)
    raise NotImplementedError(
        f"Post-training model loading is not yet defined for {model_id}."
    )


def train(
    model: str,
    *,
    data: str | Path | Any,
    task: str,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    num_workers: int | None = None,
    output_dir: str | Path | None = None,
    max_steps: int | None = None,
    max_epochs: int | None = None,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> tuple[Any, TrainingResult]:
    """Train a verified model/task profile and return the loaded model and run."""
    plan = prepare_training(
        model,
        data=data,
        task=task,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        output_dir=output_dir,
        max_steps=max_steps,
        max_epochs=max_epochs,
        **kwargs,
    )
    if plan.model.id not in {
        "driftworld",
        "irasim",
        "lewm",
        "sensorimotor",
        "prejepa",
        "toy-adaln",
        "toy-action-duplicate",
        "toy-driftworld",
    }:
        raise NotImplementedError(
            f"{plan.model.id} training is not exposed until its native checkpoint "
            "loader can return a verified in-memory model."
        )
    _warn_for_official_checkpoint_training(plan.model.id, plan.task.id)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    before = _snapshot_checkpoints(plan.output_root)
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(dict(plan.environment))
    subprocess.run(
        plan.command,
        env=environment,
        check=True,
    )
    after = _snapshot_checkpoints(plan.output_root)
    candidates = {
        path for path, fingerprint in after.items() if before.get(path) != fingerprint
    }
    checkpoint = _select_checkpoint(plan.model.id, candidates)
    saved_config = _find_saved_config(
        checkpoint,
        plan.config,
        prefer_model_config=plan.model.id in {"lewm", "prejepa"},
    )
    loaded_model = load_trained_model(
        plan.model.id,
        checkpoint,
        saved_config,
        plan.data,
        task_id=plan.task.id,
        device=device,
    )
    result = TrainingResult(
        model_id=plan.model.id,
        task_id=plan.task.id,
        data=plan.data,
        run_directory=checkpoint.parent,
        checkpoint=checkpoint,
        config=saved_config,
        command=plan.command,
        elapsed_seconds=time.monotonic() - started,
    )
    return loaded_model, result


__all__ = [
    "TrainingPlan",
    "TrainingResult",
    "load_trained_model",
    "prepare_training",
    "train",
]
