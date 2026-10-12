"""Run the vendored sensorimotor trainer with local-HDF5 path support."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm

from anygoal.sources import source_path


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
    if not args and name is not None and Path(str(name)).expanduser().is_absolute():
        kwargs["path"] = kwargs.pop("name")
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


def main(arguments: list[str] | None = None) -> int:
    source_root, hydra_arguments = parse_wrapper_arguments(
        list(sys.argv[1:] if arguments is None else arguments)
    )
    planning_source = (
        source_root if (source_root / "train.py").is_file() else source_root / "planning"
    )
    trainer = planning_source / "train.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Sensorimotor training source not found: {trainer}")

    original_dataset = swm.data.HDF5Dataset
    original_manager = spt.Manager
    original_argv = sys.argv
    sys.path.insert(0, str(planning_source))
    swm.data.HDF5Dataset = _local_path_hdf5_dataset
    spt.Manager = _compatible_manager
    sys.argv = [str(trainer), *hydra_arguments]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    finally:
        sys.argv = original_argv
        swm.data.HDF5Dataset = original_dataset
        spt.Manager = original_manager
        sys.path.pop(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
