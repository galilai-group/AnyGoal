"""Download published AnyGoal datasets and checkpoints from Hugging Face."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

from anygoal.paths import checkpoint_root, data_root
from anygoal.registry import DatasetSpec, ModelSpec, get_registry
from anygoal.resources import resource_path


@lru_cache(maxsize=1)
def artifact_manifest() -> Mapping[str, Any]:
    manifest = yaml.safe_load(
        resource_path("artifacts.yaml").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest schema version.")
    return manifest


def _download(filename: str, *, kind: str, destination: Path) -> Path:
    target = destination / filename
    if target.is_file():
        return target.resolve()
    repository = artifact_manifest()["repositories"][kind]
    destination.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repository["repo_id"],
            repo_type=repository["repo_type"],
            filename=filename,
            local_dir=destination,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        raise RuntimeError(
            f"Could not download {filename} from {repository['repo_id']}. "
            "Authenticate with `hf auth login` if the repository is private."
        ) from error
    return Path(downloaded).resolve()


def download_dataset(dataset: str | DatasetSpec) -> Path:
    spec = get_registry().dataset(dataset) if isinstance(dataset, str) else dataset
    filename = artifact_manifest()["datasets"].get(spec.id)
    if filename is None:
        raise ValueError(
            f"Dataset {spec.id!r} is not published in the AnyGoal Hugging Face "
            "repository; download it from its upstream publisher."
        )
    if filename != spec.filename:
        raise ValueError(f"Artifact filename mismatch for dataset {spec.id!r}.")
    return _download(filename, kind="datasets", destination=data_root())


def download_checkpoint(model: str | ModelSpec, dataset: str | DatasetSpec) -> Path:
    registry = get_registry()
    model_spec = registry.model(model) if isinstance(model, str) else model
    dataset_spec = registry.dataset(dataset) if isinstance(dataset, str) else dataset
    if dataset_spec.family not in model_spec.families:
        raise ValueError(
            f"{model_spec.id} does not declare support for {dataset_spec.family} datasets."
        )
    filename = (
        artifact_manifest()["checkpoints"]
        .get(model_spec.id, {})
        .get(dataset_spec.family)
    )
    if filename is None:
        raise ValueError(
            f"The {model_spec.id}/{dataset_spec.family} checkpoint is not published "
            "in the AnyGoal Hugging Face repository."
        )
    return _download(filename, kind="checkpoints", destination=checkpoint_root())


def download_pair(model: str, dataset: str) -> tuple[Path, Path]:
    """Download the dataset and its model checkpoint after validating the pair."""
    registry = get_registry()
    model_spec = registry.model(model)
    dataset_spec = registry.dataset(dataset)
    manifest = artifact_manifest()
    dataset_filename = manifest["datasets"].get(dataset_spec.id)
    checkpoint_filename = (
        manifest["checkpoints"].get(model_spec.id, {}).get(dataset_spec.family)
    )
    if dataset_filename is None:
        raise ValueError(
            f"Dataset {dataset_spec.id!r} is not published in the AnyGoal Hugging "
            "Face repository; download it from its upstream publisher."
        )
    if checkpoint_filename is None:
        raise ValueError(
            f"The {model_spec.id}/{dataset_spec.family} checkpoint is not published "
            "in the AnyGoal Hugging Face repository."
        )
    return (
        _download(dataset_filename, kind="datasets", destination=data_root()),
        _download(
            checkpoint_filename,
            kind="checkpoints",
            destination=checkpoint_root(),
        ),
    )


def download_all(*, datasets: bool = False, checkpoints: bool = False) -> list[Path]:
    manifest = artifact_manifest()
    downloaded = []
    if datasets:
        for filename in manifest["datasets"].values():
            downloaded.append(
                _download(filename, kind="datasets", destination=data_root())
            )
    if checkpoints:
        filenames = {
            filename
            for families in manifest["checkpoints"].values()
            for filename in families.values()
        }
        for filename in sorted(filenames):
            downloaded.append(
                _download(filename, kind="checkpoints", destination=checkpoint_root())
            )
    return downloaded


__all__ = [
    "artifact_manifest",
    "download_all",
    "download_checkpoint",
    "download_dataset",
    "download_pair",
]
