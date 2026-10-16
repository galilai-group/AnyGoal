from __future__ import annotations

from pathlib import Path
import io
import tarfile

import httpx
import pytest
import zstandard

from anygoal import artifacts
from anygoal.cli import main
from anygoal.registry import get_registry
from anygoal.resources import resource_path


def test_download_pair_uses_dataset_family_for_checkpoint(tmp_path, monkeypatch):
    data_directory = tmp_path / "datasets"
    checkpoint_directory = tmp_path / "checkpoints"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
        return str(path)

    monkeypatch.setattr(artifacts, "data_root", lambda: data_directory)
    monkeypatch.setattr(artifacts, "checkpoint_root", lambda: checkpoint_directory)
    monkeypatch.setattr(artifacts, "hf_hub_download", fake_download)

    dataset = artifacts.download_dataset("cubehop")
    checkpoint = artifacts.download_checkpoint("driftworld", "cubehop")

    assert dataset.name == "cube_hop_test.h5"
    assert checkpoint.name == "driftworld_cube.pt"
    assert [call["repo_type"] for call in calls] == ["dataset", "model"]


def test_existing_artifact_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "data_root", lambda: tmp_path)
    existing = tmp_path / "avoidt_test.h5"
    existing.write_bytes(b"existing")

    def unexpected_download(**kwargs):
        raise AssertionError(f"unexpected download: {kwargs}")

    monkeypatch.setattr(artifacts, "hf_hub_download", unexpected_download)
    assert artifacts.download_dataset("avoidt") == existing.resolve()


def test_artifact_cli_downloads_one_pair(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "cube_hop_test.h5"
    checkpoint = tmp_path / "driftworld_cube.pt"
    monkeypatch.setattr(artifacts, "download_pair", lambda *_: (dataset, checkpoint))

    assert (
        main(
            [
                "artifacts",
                "download",
                "--model",
                "driftworld",
                "--dataset",
                "cubehop",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [str(dataset), str(checkpoint)]


def test_eval_download_passes_resolved_checkpoint(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "driftworld_cube.pt"
    monkeypatch.setattr(artifacts, "download_dataset", lambda _: tmp_path / "data.h5")
    monkeypatch.setattr(artifacts, "download_checkpoint", lambda *_: checkpoint)

    assert (
        main(
            [
                "eval",
                "rollout",
                "--model",
                "driftworld",
                "--dataset",
                "cubehop",
                "--download",
                "--dry-run",
            ]
        )
        == 0
    )
    assert f"--checkpoint {checkpoint.resolve()}" in capsys.readouterr().out


@pytest.mark.parametrize("dataset", ["pusht", "cube", "tworoom"])
def test_external_dataset_download_and_extraction(
    dataset, tmp_path, monkeypatch, caplog
):
    entry = artifacts.artifact_manifest()["datasets"][dataset]
    payload = b"test hdf5 payload"
    if entry["compression"] == "tar.zstd":
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as contents:
            # Extract only the registered dataset, never paths from the archive.
            unwanted = tarfile.TarInfo("../unwanted.txt")
            unwanted.size = 3
            contents.addfile(unwanted, io.BytesIO(b"bad"))
            member = tarfile.TarInfo("nested/" + entry["local_filename"])
            member.size = len(payload)
            contents.addfile(member, io.BytesIO(payload))
        compressed = zstandard.ZstdCompressor().compress(archive.getvalue())
    else:
        compressed = zstandard.ZstdCompressor().compress(payload)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
        return path

    monkeypatch.setattr(artifacts, "data_root", lambda: tmp_path)
    monkeypatch.setattr(artifacts, "hf_hub_download", fake_download)
    target = artifacts.download_dataset(dataset)
    assert target == tmp_path / entry["local_filename"]
    assert target.read_bytes() == payload
    assert calls[0]["repo_id"] == entry["repo_id"]
    assert calls[0]["repo_type"] == "dataset"
    assert "external artifact" in caplog.text
    assert not (tmp_path.parent / "unwanted.txt").exists()
    assert artifacts.download_dataset(dataset) == target
    assert len(calls) == 1


def test_external_checkpoints_are_saved_at_configured_paths(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(artifacts, "checkpoint_root", lambda: tmp_path)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(kwargs["repo_id"].encode())
        return path

    monkeypatch.setattr(artifacts, "hf_hub_download", fake_download)
    paths = artifacts.download_model("lewm")
    assert {path.name for path in paths} == {
        "lewm_pusht.ckpt",
        "lewm_cube.ckpt",
        "lewm_tworoom.ckpt",
        "lewm_insertpeg.pt",
    }
    assert (tmp_path / "lewm_cube.ckpt").read_bytes() == b"quentinll/lewm-cube"
    assert (tmp_path / "lewm_pusht.ckpt").read_bytes() == b"quentinll/lewm-pusht"
    assert (
        artifacts.download_checkpoint("lewm", "cubehop") == tmp_path / "lewm_cube.ckpt"
    )
    assert len(calls) == 4
    assert "external artifact" in caplog.text


@pytest.mark.parametrize("status", [401, 403, 404, 410, 500])
def test_external_http_error_guidance(status, tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "checkpoint_root", lambda: tmp_path)

    def fail_download(**kwargs):
        response = httpx.Response(
            status, request=httpx.Request("GET", "https://huggingface.co/test")
        )
        raise artifacts.HfHubHTTPError("upstream error", response=response)

    monkeypatch.setattr(artifacts, "hf_hub_download", fail_download)
    with pytest.raises(RuntimeError) as caught:
        artifacts.download_checkpoint("lewm", "cube")
    message = str(caught.value)
    assert "quentinll/lewm-cube" in message
    if status in {404, 410}:
        assert "moved or removed" in message
        assert "https://github.com/galilai-group/AnyGoal/issues/new" in message
    else:
        assert "network" in message
        assert "hf auth login" in message
        assert "moved or removed" not in message


def test_corrupt_archive_leaves_no_partial_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "data_root", lambda: tmp_path)
    archive = tmp_path / "bad.zst"
    archive.write_bytes(b"corrupt zstandard content")
    monkeypatch.setattr(artifacts, "hf_hub_download", lambda **_: archive)
    with pytest.raises(RuntimeError, match="Could not extract"):
        artifacts.download_dataset("pusht")
    assert not (tmp_path / "pusht_expert_train.h5").exists()
    assert not list(tmp_path.glob(".anygoal-*"))


def test_invalid_pair_does_not_start_download(monkeypatch):
    manifest = dict(artifacts.artifact_manifest())
    manifest["checkpoints"] = {}
    monkeypatch.setattr(artifacts, "artifact_manifest", lambda: manifest)
    def unexpected_download(*args, **kwargs):
        raise AssertionError("invalid pair must not download anything")

    monkeypatch.setattr(artifacts, "_download", unexpected_download)
    with pytest.raises(ValueError, match="not published"):
        artifacts.download_pair("prejepa", "pusht")


@pytest.mark.parametrize("selection", ["model", "dataset"])
def test_artifact_cli_single_selector(selection, monkeypatch, tmp_path, capsys):
    path = tmp_path / "artifact"
    if selection == "model":
        monkeypatch.setattr(artifacts, "download_model", lambda _: [path])
        identifier = "lewm"
    else:
        monkeypatch.setattr(artifacts, "download_dataset", lambda _: path)
        identifier = "pusht"
    assert main(["artifacts", "download", f"--{selection}", identifier]) == 0
    assert capsys.readouterr().out.strip() == str(path)


def test_download_all_includes_external_artifacts(monkeypatch, tmp_path):
    calls = []

    def fake_download(entry, *, kind, destination):
        entry = artifacts._entry(entry)
        calls.append((kind, entry))
        return destination / entry.get("local_filename", entry["filename"])

    monkeypatch.setattr(artifacts, "_download", fake_download)
    monkeypatch.setattr(artifacts, "checkpoint_root", lambda: tmp_path / "checkpoints")
    monkeypatch.setattr(artifacts, "data_root", lambda: tmp_path / "datasets")
    paths = artifacts.download_all(datasets=True, checkpoints=True)
    assert tmp_path / "datasets/pusht_expert_train.h5" in paths
    assert tmp_path / "checkpoints/lewm_cube.ckpt" in paths
    assert sum("repo_id" in entry for _, entry in calls) == 6


def test_shared_checkpoint_paths_match_all_evaluation_profiles(monkeypatch, tmp_path):
    profiles = artifacts.yaml.safe_load(
        resource_path("eval", "rollout.yaml").read_text()
    )["models"]
    monkeypatch.setattr(artifacts, "checkpoint_root", lambda: tmp_path)
    registry = get_registry()
    for model in registry.models.values():
        for family, profile in profiles[model.evaluation_id].items():
            task = next(
                spec for spec in registry.datasets.values() if spec.family == family
            )
            expected = profile["checkpoint"].replace(
                "${anygoal.path:checkpoints}", str(tmp_path)
            )
            assert artifacts.default_checkpoint_path(model, task) == Path(expected), (
                model.id,
                family,
            )
