"""Download published AnyGoal datasets and checkpoints from Hugging Face."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping

import yaml
import zstandard
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

from anygoal.paths import checkpoint_root, data_root
from anygoal.registry import DatasetSpec, ModelSpec, get_registry
from anygoal.resources import resource_path


_ISSUES_URL = "https://github.com/galilai-group/AnyGoal/issues/new"
_logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def artifact_manifest() -> Mapping[str, Any]:
    manifest = yaml.safe_load(
        resource_path("artifacts.yaml").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest schema version.")
    return manifest


def _entry(value: str | Mapping[str, Any]) -> dict[str, Any]:
    return {"filename": value} if isinstance(value, str) else dict(value)


def _extract_archive(archive: Path, output, *, compression: str, filename: str) -> None:
    with archive.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            if compression == "zstd":
                shutil.copyfileobj(reader, output)
            elif compression == "tar.zstd":
                with tarfile.open(fileobj=reader, mode="r|") as contents:
                    for member in contents:
                        if member.isfile() and Path(member.name).name == filename:
                            with contents.extractfile(member) as extracted:
                                shutil.copyfileobj(extracted, output)
                            return
                raise RuntimeError(
                    f"Archive {archive.name} does not contain {filename}."
                )
            else:
                raise ValueError(f"Unsupported artifact compression: {compression!r}.")


def _download(
    artifact: str | Mapping[str, Any], *, kind: str, destination: Path
) -> Path:
    entry = _entry(artifact)
    filename = entry["filename"]
    local_filename = entry.get("local_filename", filename)
    target = destination / local_filename
    if target.is_file():
        return target.resolve()
    external = "repo_id" in entry
    repository = entry if external else artifact_manifest()["repositories"][kind]
    destination.mkdir(parents=True, exist_ok=True)
    if external:
        _logger.warning(
            "Warning: downloading external artifact %s from https://huggingface.co/%s%s "
            "to %s. This source is maintained by its original publisher, not AnyGoal.",
            filename,
            "datasets/" if repository["repo_type"] == "dataset" else "",
            repository["repo_id"],
            target,
        )
    # Isolate upstream filenames (e.g. weights.pt) to prevent model collisions.
    download_directory = (
        destination / ".cache" / "anygoal" / repository["repo_id"]
        if external
        else destination
    )
    try:
        downloaded = hf_hub_download(
            repo_id=repository["repo_id"],
            repo_type=repository["repo_type"],
            filename=filename,
            local_dir=download_directory,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if external and status in {404, 410}:
            raise RuntimeError(
                f"External artifact {repository['repo_id']}/{filename} is unavailable "
                f"(HTTP {status}). Its publisher may have moved or removed it. "
                f"Please report the source and this error to AnyGoal: {_ISSUES_URL}"
            ) from error
        raise RuntimeError(
            f"Could not download {filename} from {repository['repo_id']}. "
            "Check your network connection and authenticate with `hf auth login` "
            f"if the repository is private or gated. Original error: {error}"
        ) from error
    downloaded = Path(downloaded)
    if not external:
        return downloaded.resolve()
    # Publish only completed files; interrupted extraction must not look cached.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination, prefix=".anygoal-", delete=False
        ) as output:
            temporary_path = Path(output.name)
            if entry.get("compression"):
                print(f"Extracting {filename} to {target} ...", file=sys.stderr)
                try:
                    _extract_archive(
                        downloaded,
                        output,
                        compression=entry["compression"],
                        filename=local_filename,
                    )
                except (RuntimeError, tarfile.TarError, zstandard.ZstdError) as error:
                    raise RuntimeError(
                        f"Could not extract {filename}: {error}. "
                        f"The upstream archive layout may have changed; report it at {_ISSUES_URL}"
                    ) from error
            else:
                with downloaded.open("rb") as source:
                    shutil.copyfileobj(source, output)
        try:
            target.hardlink_to(temporary_path)
        except FileExistsError:
            if not target.is_file():
                raise
        return target.resolve()
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _dataset_entry(spec: DatasetSpec) -> dict[str, Any]:
    artifact = artifact_manifest()["datasets"].get(spec.id)
    if artifact is None:
        raise ValueError(
            f"Dataset {spec.id!r} is not published in the AnyGoal Hugging Face "
            "repository; download it from its upstream publisher."
        )
    entry = _entry(artifact)
    if entry.get("local_filename", entry["filename"]) != spec.filename:
        raise ValueError(f"Artifact filename mismatch for dataset {spec.id!r}.")
    return entry


def download_dataset(dataset: str | DatasetSpec) -> Path:
    spec = get_registry().dataset(dataset) if isinstance(dataset, str) else dataset
    return _download(_dataset_entry(spec), kind="datasets", destination=data_root())


def _checkpoint_entry(
    model_spec: ModelSpec, dataset_spec: DatasetSpec
) -> dict[str, Any]:
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
    return _entry(filename)


def download_checkpoint(model: str | ModelSpec, dataset: str | DatasetSpec) -> Path:
    registry = get_registry()
    model_spec = registry.model(model) if isinstance(model, str) else model
    dataset_spec = registry.dataset(dataset) if isinstance(dataset, str) else dataset
    return _download(
        _checkpoint_entry(model_spec, dataset_spec),
        kind="checkpoints",
        destination=checkpoint_root(),
    )


def default_checkpoint_path(model: str | ModelSpec, dataset: str | DatasetSpec) -> Path:
    """Resolve the shared local checkpoint convention without downloading files."""
    registry = get_registry()
    model_spec = registry.model(model) if isinstance(model, str) else model
    dataset_spec = registry.dataset(dataset) if isinstance(dataset, str) else dataset
    if dataset_spec.family not in model_spec.families:
        raise ValueError(
            f"{model_spec.id} does not declare support for {dataset_spec.family} datasets."
        )
    manifest = artifact_manifest()
    artifact = manifest["checkpoints"].get(model_spec.id, {}).get(dataset_spec.family)
    if artifact is None:
        artifact = (
            manifest.get("local_checkpoints", {})
            .get(model_spec.id, {})
            .get(dataset_spec.family)
        )
    if artifact is None:
        raise ValueError(
            f"No default checkpoint is registered for {model_spec.id}/{dataset_spec.family}. "
            "Specify --checkpoint or --model-config explicitly."
        )
    entry = _entry(artifact)
    return checkpoint_root() / entry.get("local_filename", entry["filename"])


def download_model(model: str) -> list[Path]:
    """Download every published checkpoint family for one model."""
    spec = get_registry().model(model)
    families = artifact_manifest()["checkpoints"].get(spec.id, {})
    if not families:
        raise ValueError(f"No downloadable checkpoints are registered for {spec.id!r}.")
    return [
        _download(artifact, kind="checkpoints", destination=checkpoint_root())
        for artifact in families.values()
    ]


def download_pair(model: str, dataset: str) -> tuple[Path, Path]:
    """Download the dataset and its model checkpoint after validating the pair."""
    registry = get_registry()
    model_spec = registry.model(model)
    dataset_spec = registry.dataset(dataset)
    dataset_entry = _dataset_entry(dataset_spec)
    checkpoint_entry = _checkpoint_entry(model_spec, dataset_spec)
    return (
        _download(dataset_entry, kind="datasets", destination=data_root()),
        _download(
            checkpoint_entry,
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
        entries = {
            _entry(artifact).get(
                "local_filename", _entry(artifact)["filename"]
            ): artifact
            for families in manifest["checkpoints"].values()
            for artifact in families.values()
        }
        for filename in sorted(entries):
            downloaded.append(
                _download(
                    entries[filename], kind="checkpoints", destination=checkpoint_root()
                )
            )
    return downloaded


__all__ = [
    "artifact_manifest",
    "download_all",
    "download_checkpoint",
    "default_checkpoint_path",
    "download_dataset",
    "download_model",
    "download_pair",
]
