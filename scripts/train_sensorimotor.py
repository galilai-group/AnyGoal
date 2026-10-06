"""Run sensorimotor-world-model training with the AnyGoal compatibility patch."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "worldmodels" / "sensorimotor-world-model"
PATCH_PATH = REPOSITORY_ROOT / "patches" / "sensorimotor" / "train-anygoal.patch"


def parse_wrapper_arguments(arguments: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--sensorimotor-source",
        type=Path,
        default=Path(os.environ.get("SENSORIMOTOR_SOURCE_ROOT", DEFAULT_SOURCE)),
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return parsed.sensorimotor_source.expanduser().resolve(), remaining


def main(arguments: list[str] | None = None) -> int:
    source_root, hydra_arguments = parse_wrapper_arguments(
        list(sys.argv[1:] if arguments is None else arguments)
    )
    planning_source = (
        source_root if (source_root / "train.py").is_file() else source_root / "planning"
    )
    if not (planning_source / "train.py").is_file():
        raise FileNotFoundError(f"Sensorimotor training source not found: {planning_source}")
    if not PATCH_PATH.is_file():
        raise FileNotFoundError(f"AnyGoal compatibility patch not found: {PATCH_PATH}")

    with tempfile.TemporaryDirectory(prefix="anygoal-sensorimotor-") as temporary:
        temporary_root = Path(temporary)
        planning_copy = temporary_root / "planning"
        shutil.copytree(planning_source, planning_copy)
        subprocess.run(
            ["git", "apply", "--check", str(PATCH_PATH)],
            cwd=temporary_root,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(PATCH_PATH)],
            cwd=temporary_root,
            check=True,
        )
        return subprocess.run(
            [sys.executable, str(planning_copy / "train.py"), *hydra_arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
