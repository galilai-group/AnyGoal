"""Configuration loading with package, site, and experiment precedence."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

from anygoal.paths import (
    anygoal_home,
    cache_root,
    checkpoint_root,
    data_root,
    output_root,
    source_root,
)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return content


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(2))

    expanded = _ENV_PATTERN.sub(replace, value)
    return str(Path(expanded).expanduser()) if expanded.startswith("~") else expanded


def load_config(
    *,
    site: str | Path | None = None,
    experiment: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load defaults < site < experiment < explicit overrides."""

    default_resource = resources.files("anygoal.resources").joinpath("defaults.yaml")
    with default_resource.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    default_site = Path(
        os.environ.get("ANYGOAL_SITE_CONFIG", "~/.config/anygoal/site.yaml")
    ).expanduser()
    site_path = Path(site).expanduser() if site is not None else default_site
    if site is not None or site_path.is_file():
        config = _merge(config, _read_yaml(site_path))
    if experiment is not None:
        config = _merge(config, _read_yaml(Path(experiment).expanduser()))
    if overrides:
        config = _merge(config, overrides)
    config["paths"] = {
        "home": str(anygoal_home()),
        "data": str(data_root()),
        "checkpoints": str(checkpoint_root()),
        "outputs": str(output_root()),
        "sources": str(source_root()),
        "cache": str(cache_root()),
    }
    return _expand(config)
