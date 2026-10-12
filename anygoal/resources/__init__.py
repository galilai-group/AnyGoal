"""Packaged registry, runner profiles, and CLI defaults."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return a filesystem path to an installed AnyGoal resource."""
    return Path(str(files(__package__).joinpath(*parts)))


__all__ = ["resource_path"]
