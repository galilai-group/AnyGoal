"""Run the Stable Worldmodel LeWM trainer on AnyGoal HDF5 data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from anygoal.sources import source_path


def _default_source() -> Path:
    return source_path("lewm")


def parse_wrapper_arguments(arguments: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--lewm-source",
        type=Path,
        default=_default_source(),
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return parsed.lewm_source.expanduser().resolve(), remaining


def main(arguments: list[str] | None = None) -> int:
    source_root, hydra_arguments = parse_wrapper_arguments(
        list(sys.argv[1:] if arguments is None else arguments)
    )
    trainer = (
        source_root / "train.py"
        if (source_root / "train.py").is_file()
        else source_root / "scripts/train/lewm.py"
    )
    if not trainer.is_file():
        raise FileNotFoundError(f"Stable Worldmodel LeWM trainer not found: {trainer}")
    return subprocess.run(
        [sys.executable, str(trainer), *hydra_arguments],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
