"""Pinned, explicit installation of third-party model source checkouts."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml

from anygoal.paths import source_root
from anygoal.resources import resource_path


@dataclass(frozen=True)
class SourceSpec:
    id: str
    repository: str
    revision: str
    directory: str
    environment: str
    marker: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceStatus:
    source: SourceSpec
    path: Path
    state: str
    revision: str | None


@lru_cache(maxsize=1)
def get_sources() -> Mapping[str, SourceSpec]:
    payload = yaml.safe_load(resource_path("sources.yaml").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema version.")
    specs = {}
    aliases = {}
    for source_id, raw in payload["sources"].items():
        spec = SourceSpec(
            id=source_id,
            repository=str(raw["repository"]),
            revision=str(raw["revision"]),
            directory=str(raw["directory"]),
            environment=str(raw["environment"]),
            marker=str(raw["marker"]),
            aliases=tuple(raw.get("aliases", ())),
        )
        specs[source_id] = spec
        for alias in spec.aliases:
            aliases[alias] = source_id
    return {
        **specs,
        **{alias: specs[source_id] for alias, source_id in aliases.items()},
    }


def get_source(name: str) -> SourceSpec:
    try:
        return get_sources()[name]
    except KeyError as error:
        canonical = sorted({spec.id for spec in get_sources().values()})
        raise KeyError(
            f"Unknown model source {name!r}; choose from {canonical}."
        ) from error


def _development_source(spec: SourceSpec) -> Path:
    return Path(__file__).resolve().parents[1] / "worldmodels" / spec.directory


def source_path(name: str, *, require: bool = True) -> Path:
    """Resolve an explicit override, managed checkout, or development submodule."""
    spec = get_source(name)
    configured = os.environ.get(spec.environment)
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        candidates = [source_root() / spec.directory, _development_source(spec)]
    for candidate in candidates:
        if (candidate / spec.marker).exists():
            return candidate.resolve()
    target = candidates[0].resolve()
    if require:
        raise FileNotFoundError(
            f"{spec.id} source was not found at {target}. Install the pinned original "
            f"source with `anygoal models install {spec.id}` or set {spec.environment}."
        )
    return target


def source_status(name: str) -> SourceStatus:
    spec = get_source(name)
    path = source_path(spec.id, require=False)
    if not (path / spec.marker).exists():
        return SourceStatus(spec, path, "missing", None)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip() if result.returncode == 0 else None
    state = "ready" if revision == spec.revision else "unverified"
    return SourceStatus(spec, path, state, revision)


def install_source(name: str) -> SourceStatus:
    """Clone one pinned original repository without installing its dependencies."""
    spec = get_source(name)
    if os.environ.get(spec.environment):
        current = source_status(spec.id)
        if current.state == "ready":
            return current
        raise RuntimeError(
            f"{spec.environment} is set to {current.path}, but that checkout is "
            "not at the pinned revision. Unset it before using the managed installer."
        )
    destination = (source_root() / spec.directory).resolve()
    current = source_status(spec.id)
    if current.path == destination and current.state == "ready":
        return current
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing source directory {destination}. "
            "Move it aside or set the corresponding source-root environment variable."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{spec.directory}-", dir=destination.parent)
    )
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                spec.repository,
                str(temporary),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary), "checkout", "--detach", spec.revision],
            check=True,
        )
        temporary.rename(destination)
    except subprocess.CalledProcessError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(
            f"Could not install {spec.id} from {spec.repository} at {spec.revision}."
        ) from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return source_status(spec.id)


__all__ = [
    "SourceSpec",
    "SourceStatus",
    "get_source",
    "get_sources",
    "install_source",
    "source_path",
    "source_status",
]
