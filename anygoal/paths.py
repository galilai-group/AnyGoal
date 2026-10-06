"""User-overridable filesystem roots shared by package tools."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(
        os.environ.get("ANYGOAL_DATA_ROOT", "~/.stable_worldmodel/datasets")
    ).expanduser()


def checkpoint_root() -> Path:
    return Path(
        os.environ.get("ANYGOAL_CHECKPOINT_ROOT", "~/.cache/anygoal/checkpoints")
    ).expanduser()


def cache_root() -> Path:
    return Path(os.environ.get("ANYGOAL_CACHE_ROOT", "~/.cache/anygoal")).expanduser()


def source_root() -> Path:
    return Path(
        os.environ.get("ANYGOAL_SOURCE_ROOT", "~/.cache/anygoal/sources")
    ).expanduser()
