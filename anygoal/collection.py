"""Public dataset collection API."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

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


def _arguments_from_options(dataset_id: str, options: dict[str, Any]) -> list[str]:
    options = dict(options)
    if "num_episodes" in options:
        name, value = _native_episode_option(dataset_id, int(options.pop("num_episodes")))
        if name in options:
            raise ValueError(f"Specify only num_episodes or {name}, not both")
        options[name] = value
    arguments: list[str] = []
    for name, value in options.items():
        if value is None or value is False:
            continue
        flag = f"--{name.replace('_', '-')}"
        arguments.append(flag)
        if value is True:
            continue
        if isinstance(value, (list, tuple)):
            arguments.extend(str(item) for item in value)
        else:
            arguments.append(str(value))
    return arguments


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
        options["num_episodes"] = num_episodes
    arguments = _arguments_from_options(spec.id, options)
    path = Path(module.run(module.parse_args(arguments))).expanduser().resolve()
    return _stamp_dataset_id(path, spec.id)


def collect_from_cli(dataset: str, arguments: Sequence[str]) -> Path:
    """Run collection with the collector's native command-line options."""
    spec = get_registry().dataset(dataset)
    arguments = list(arguments)
    if spec.source != "generated":
        raise ValueError(
            f"{spec.id} is an externally provided base dataset, not a collection target"
        )
    if "--num-episodes" in arguments:
        index = arguments.index("--num-episodes")
        count = int(arguments[index + 1])
        name, value = _native_episode_option(spec.id, count)
        arguments[index : index + 2] = [f"--{name.replace('_', '-')}", str(value)]
    if not any(
        argument == "--seed" or argument.startswith("--seed=")
        for argument in arguments
    ):
        arguments.extend(["--seed", str(spec.default_seed)])
    module = import_module(spec.collector)
    path = Path(module.run(module.parse_args(arguments))).expanduser().resolve()
    return _stamp_dataset_id(path, spec.id)
