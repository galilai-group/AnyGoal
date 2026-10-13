"""User-overridable filesystem roots shared by package tools."""

from __future__ import annotations

import os
from pathlib import Path

from stable_worldmodel.data.utils import get_cache_dir


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    """Expand a path, resolving relative values against a caller-selected base."""
    text = str(value)
    resource_prefix = "resource:"
    if text.startswith(resource_prefix):
        from anygoal.resources import resource_path

        relative = text.removeprefix(resource_prefix).lstrip("/")
        if not relative:
            raise ValueError("resource path must name a packaged file")
        return resource_path(*Path(relative).parts)
    path = Path(text).expanduser()
    root = Path.cwd() if base is None else Path(base).expanduser()
    return path if path.is_absolute() else root / path


def data_root() -> Path:
    override = os.environ.get("ANYGOAL_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return get_cache_dir(sub_folder="datasets")


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
