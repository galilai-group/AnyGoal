"""Public dataset collection API."""

from __future__ import annotations

import argparse
import inspect
import types
import typing
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import h5py

from anygoal.registry import get_registry


def _stamp_dataset_id(path: Path, dataset_id: str) -> Path:
    """Record identity without changing trajectory arrays or collection results."""
    with h5py.File(path, "a") as handle:
        existing = handle.attrs.get("anygoal_dataset_id")
        if isinstance(existing, bytes):
            existing = existing.decode()
        if existing is not None and str(existing) != dataset_id:
            raise ValueError(
                f"Dataset identity mismatch: {existing!r} != {dataset_id!r}"
            )
        handle.attrs["anygoal_dataset_id"] = dataset_id
    return path


def _native_episode_option(dataset_id: str, count: int) -> tuple[str, int]:
    if count <= 0:
        raise ValueError("num_episodes must be positive")
    if dataset_id in {"avoidt", "pusht-antigoal"}:
        if count % 4:
            raise ValueError(f"{dataset_id} num_episodes must be divisible by 4")
        return "episodes_per_goal", count // 4
    if dataset_id == "tworoom_route":
        raise ValueError("tworoom_route uses fixed balanced condition counts")
    return "episodes", count


def collect(dataset: str, num_episodes: int | None = None, **options: Any) -> Path:
    """Collect a generated dataset and return its HDF5 path."""
    spec = get_registry().dataset(dataset)
    if spec.source != "generated":
        raise ValueError(
            f"{spec.id} is an externally provided base dataset, not a collection target"
        )
    module = import_module(spec.collector)
    options.setdefault("seed", spec.default_seed)
    if num_episodes is not None:
        name, value = _native_episode_option(spec.id, int(num_episodes))
        if name in options:
            raise ValueError(f"Specify only num_episodes or {name}, not both")
        options[name] = value
    path = Path(module.collect_dataset(**options)).expanduser().resolve()
    return _stamp_dataset_id(path, spec.id)


def _argument_type(annotation: Any) -> tuple[Any, tuple[Any, ...] | None]:
    origin = get_origin(annotation)
    if origin is Literal:
        choices = get_args(annotation)
        return type(choices[0]), choices
    if origin in (types.UnionType, typing.Union):
        members = tuple(
            member for member in get_args(annotation) if member is not type(None)
        )
        if len(members) == 1:
            return _argument_type(members[0])
    return annotation, None


def _parse_cli_options(
    dataset_id: str, module: Any, arguments: Sequence[str]
) -> dict[str, Any]:
    collector = module.collect_dataset
    signature = inspect.signature(collector)
    annotations = get_type_hints(collector)
    parser = argparse.ArgumentParser(prog=f"anygoal collect {dataset_id}")
    parser.add_argument("--num-episodes", type=int, default=argparse.SUPPRESS)
    for name, parameter in signature.parameters.items():
        if name == "num_episodes" or parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        annotation, choices = _argument_type(annotations.get(name, str))
        kwargs: dict[str, Any] = {"default": argparse.SUPPRESS}
        if choices is not None:
            kwargs["choices"] = choices
        if annotation is bool:
            kwargs["action"] = (
                argparse.BooleanOptionalAction
                if parameter.default is True
                else "store_true"
            )
        else:
            kwargs["type"] = annotation
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    return vars(parser.parse_args(list(arguments)))


def collect_from_cli(dataset: str, arguments: Sequence[str]) -> Path:
    """Run collection with the collector's native command-line options."""
    spec = get_registry().dataset(dataset)
    if spec.source != "generated":
        raise ValueError(
            f"{spec.id} is an externally provided base dataset, not a collection target"
        )
    module = import_module(spec.collector)
    options = _parse_cli_options(spec.id, module, arguments)
    return collect(spec.id, **options)
