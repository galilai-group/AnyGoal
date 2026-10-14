"""Central filesystem policy for AnyGoal and Stable Worldmodel artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from stable_worldmodel.data.utils import get_cache_dir


def anygoal_home() -> Path:
    """Return the root for files owned by AnyGoal."""
    override = os.environ.get("ANYGOAL_HOME")
    return Path(override).expanduser() if override else Path.home() / ".anygoal"


def stablewm_root() -> Path:
    """Delegate the shared Stable Worldmodel root and its default to upstream."""
    return get_cache_dir()


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


def dataset_file(name: str | Path, *, root: str | Path) -> Path:
    """Resolve a dataset name under its data directory, with legacy-root support."""
    name_path = Path(name).expanduser()
    if name_path.is_absolute() or name_path.suffix in {".h5", ".hdf5"}:
        return name_path

    data_directory = Path(root).expanduser()
    direct = data_directory / f"{name_path}.h5"
    legacy = data_directory / "datasets" / f"{name_path}.h5"
    return legacy if legacy.is_file() and not direct.is_file() else direct


def checkpoint_root() -> Path:
    override = os.environ.get("ANYGOAL_CHECKPOINT_ROOT")
    if override:
        return Path(override).expanduser()
    return anygoal_home() / "checkpoints"


def output_root() -> Path:
    override = os.environ.get("ANYGOAL_OUTPUT_ROOT")
    return Path(override).expanduser() if override else Path("outputs")


def cache_root() -> Path:
    override = os.environ.get("ANYGOAL_CACHE_ROOT")
    return Path(override).expanduser() if override else anygoal_home() / "cache"


def source_root() -> Path:
    override = os.environ.get("ANYGOAL_SOURCE_ROOT")
    return Path(override).expanduser() if override else anygoal_home() / "sources"


def path_value(name: str) -> str:
    """Resolve a named root for OmegaConf package configurations."""
    roots = {
        "home": anygoal_home,
        "stablewm": stablewm_root,
        "data": data_root,
        "checkpoints": checkpoint_root,
        "outputs": output_root,
        "sources": source_root,
        "cache": cache_root,
    }
    try:
        return str(roots[name]())
    except KeyError as error:
        raise ValueError(f"Unknown AnyGoal path name {name!r}.") from error


def register_omegaconf_resolvers() -> None:
    """Register the lazy path resolver when evaluation dependencies exist."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return
    if not OmegaConf.has_resolver("anygoal.path"):
        OmegaConf.register_new_resolver("anygoal.path", path_value)


register_omegaconf_resolvers()


__all__ = [
    "anygoal_home",
    "cache_root",
    "checkpoint_root",
    "data_root",
    "dataset_file",
    "output_root",
    "path_value",
    "register_omegaconf_resolvers",
    "resolve_path",
    "source_root",
    "stablewm_root",
]
