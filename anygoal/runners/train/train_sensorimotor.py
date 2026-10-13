"""Run the vendored sensorimotor trainer with local-HDF5 path support."""

from __future__ import annotations

import argparse
import math
import re
import runpy
import sys
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
from stable_pretraining.optim import lr_scheduler as spt_lr_scheduler
import stable_worldmodel as swm

from anygoal.sources import source_path


_DATASET_ALIASES: dict[str, Path] = {}


def parse_wrapper_arguments(arguments: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--sensorimotor-source",
        type=Path,
        default=source_path("sensorimotor"),
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return parsed.sensorimotor_source.expanduser().resolve(), remaining


def _local_path_hdf5_dataset(*args, **kwargs):
    """Route an absolute ``name`` to HDF5Dataset's explicit ``path`` API."""
    name = kwargs.get("name")
    alias_path = _DATASET_ALIASES.get(str(name))
    if not args and alias_path is not None:
        kwargs.pop("name")
        kwargs["path"] = str(alias_path)
        kwargs.pop("cache_dir", None)
    elif not args and name is not None and Path(str(name)).expanduser().is_absolute():
        kwargs["path"] = str(Path(str(kwargs.pop("name"))).expanduser().resolve())
        kwargs.pop("cache_dir", None)
    return _local_path_hdf5_dataset.original(*args, **kwargs)


_local_path_hdf5_dataset.original = swm.data.HDF5Dataset


class _SaveFinalCheckpoint(pl.Callback):
    def __init__(self, path: Path) -> None:
        self.path = path

    def on_train_end(self, trainer, pl_module) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.path))


def _compatible_manager(*args, **kwargs):
    """Adapt the legacy save-path convention to current stable-pretraining."""
    checkpoint = kwargs.get("ckpt_path")
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            trainer = kwargs.get("trainer")
            if trainer is None and args:
                trainer = args[0]
            trainer.callbacks.append(_SaveFinalCheckpoint(checkpoint))
            kwargs["ckpt_path"] = None
        else:
            kwargs["ckpt_path"] = checkpoint
    return _compatible_manager.original(*args, **kwargs)


_compatible_manager.original = spt.Manager


def _sensorimotor_scheduler_defaults(module, optimizer) -> dict[str, float | int]:
    """Reproduce the step-based scheduler defaults used by upstream training."""
    del optimizer
    trainer = module.trainer
    batches_per_epoch = trainer.num_training_batches
    if not math.isfinite(batches_per_epoch):
        train_loader = trainer.train_dataloader
        if train_loader is None and trainer.datamodule is not None:
            train_loader = trainer.datamodule.train_dataloader()
        if train_loader is None:
            raise ValueError("Sensorimotor training dataloader is not available.")
        batches_per_epoch = len(train_loader)
        batch_limit = trainer.limit_train_batches
        if isinstance(batch_limit, int):
            batches_per_epoch = min(batches_per_epoch, batch_limit)
        else:
            batches_per_epoch = int(batches_per_epoch * float(batch_limit))
    batches_per_epoch = int(batches_per_epoch)
    accumulate_grad_batches = max(int(trainer.accumulate_grad_batches), 1)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulate_grad_batches)
    max_steps = steps_per_epoch * int(trainer.max_epochs)
    trainer_max_steps = int(getattr(trainer, "max_steps", -1))
    if trainer_max_steps >= 0:
        max_steps = min(max_steps, trainer_max_steps)
    if max_steps < 1:
        raise ValueError("Sensorimotor training requires at least one optimizer step.")
    return {
        "warmup_steps": max(1, int(0.01 * max_steps)),
        "max_steps": max_steps,
        "warmup_start_lr": 0.0,
        "eta_min": 0.0,
    }


def _alias_local_training_dataset(
    hydra_arguments: list[str],
) -> tuple[list[str], dict[str, Path]]:
    """Map upstream's train/eval names to one explicitly supplied local HDF5."""
    rewritten = list(hydra_arguments)
    aliases: dict[str, Path] = {}
    prefix = "data.dataset.name="
    for index, argument in enumerate(rewritten):
        if not argument.startswith(prefix):
            continue
        dataset_path = Path(argument[len(prefix) :]).expanduser()
        if not dataset_path.is_absolute():
            continue
        dataset_path = dataset_path.resolve()
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_path.stem)
        train_name = f"__anygoal_{safe_stem}"
        if not train_name.endswith("_train"):
            train_name += "_train"
        eval_name = train_name[: -len("_train")] + "_eval"
        aliases = {train_name: dataset_path, eval_name: dataset_path}
        rewritten[index] = f"{prefix}{train_name}"
        break
    return rewritten, aliases


def main(arguments: list[str] | None = None) -> int:
    source_root, hydra_arguments = parse_wrapper_arguments(
        list(sys.argv[1:] if arguments is None else arguments)
    )
    hydra_arguments, dataset_aliases = _alias_local_training_dataset(hydra_arguments)
    planning_source = (
        source_root
        if (source_root / "train.py").is_file()
        else source_root / "planning"
    )
    trainer = planning_source / "train.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Sensorimotor training source not found: {trainer}")

    original_dataset = swm.data.HDF5Dataset
    original_manager = spt.Manager
    original_scheduler_factory = spt_lr_scheduler.DEFAULT_SCHEDULER_FACTORIES[
        "LinearWarmupCosineAnnealingLR"
    ]
    original_argv = sys.argv
    original_aliases = dict(_DATASET_ALIASES)
    sys.path.insert(0, str(planning_source))
    swm.data.HDF5Dataset = _local_path_hdf5_dataset
    spt.Manager = _compatible_manager
    spt_lr_scheduler.DEFAULT_SCHEDULER_FACTORIES["LinearWarmupCosineAnnealingLR"] = (
        _sensorimotor_scheduler_defaults
    )
    _DATASET_ALIASES.clear()
    _DATASET_ALIASES.update(dataset_aliases)
    sys.argv = [str(trainer), *hydra_arguments]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    finally:
        sys.argv = original_argv
        swm.data.HDF5Dataset = original_dataset
        spt.Manager = original_manager
        spt_lr_scheduler.DEFAULT_SCHEDULER_FACTORIES[
            "LinearWarmupCosineAnnealingLR"
        ] = original_scheduler_factory
        _DATASET_ALIASES.clear()
        _DATASET_ALIASES.update(original_aliases)
        sys.path.pop(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
