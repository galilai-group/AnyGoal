from __future__ import annotations

from pathlib import Path

from anygoal import artifacts
from anygoal.cli import main


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
    monkeypatch.setattr(
        artifacts, "download_pair", lambda *_: (dataset, checkpoint)
    )

    assert main(
        [
            "artifacts",
            "download",
            "--model",
            "driftworld",
            "--dataset",
            "cubehop",
        ]
    ) == 0
    assert capsys.readouterr().out.splitlines() == [str(dataset), str(checkpoint)]


def test_eval_download_passes_resolved_checkpoint(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "driftworld_cube.pt"
    monkeypatch.setattr(artifacts, "download_dataset", lambda _: tmp_path / "data.h5")
    monkeypatch.setattr(artifacts, "download_checkpoint", lambda *_: checkpoint)

    assert main(
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
    ) == 0
    assert f"--checkpoint {checkpoint.resolve()}" in capsys.readouterr().out
