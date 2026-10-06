"""Run LeWM training with the AnyGoal HDF5 compatibility patch."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "worldmodels" / "lewm"
PATCH_PATH = REPOSITORY_ROOT / "patches" / "lewm" / "train-anygoal.patch"


def parse_wrapper_arguments(arguments: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--lewm-source",
        type=Path,
        default=Path(os.environ.get("LEWM_SOURCE_ROOT", DEFAULT_SOURCE)),
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return parsed.lewm_source.expanduser().resolve(), remaining


def main(arguments: list[str] | None = None) -> int:
    source_root, hydra_arguments = parse_wrapper_arguments(
        list(sys.argv[1:] if arguments is None else arguments)
    )
    if not (source_root / "train.py").is_file():
        raise FileNotFoundError(f"LeWM training source not found: {source_root}")
    if not PATCH_PATH.is_file():
        raise FileNotFoundError(f"AnyGoal compatibility patch not found: {PATCH_PATH}")

    with tempfile.TemporaryDirectory(prefix="anygoal-lewm-") as temporary:
        temporary_root = Path(temporary)
        lewm_copy = temporary_root / "lewm"
        shutil.copytree(source_root, lewm_copy, ignore=shutil.ignore_patterns(".git"))
        subprocess.run(
            ["git", "apply", "--check", str(PATCH_PATH)],
            cwd=lewm_copy,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(PATCH_PATH)],
            cwd=lewm_copy,
            check=True,
        )
        return subprocess.run(
            [sys.executable, str(lewm_copy / "train.py"), *hydra_arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
