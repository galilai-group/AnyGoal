from __future__ import annotations

import subprocess

import pytest

import anygoal.sources as sources
from anygoal.sources import SourceSpec


def test_source_manifest_pins_the_four_external_models():
    specs = {spec.id: spec for spec in sources.get_sources().values()}
    assert set(specs) == {"driftworld", "irasim", "lewm", "sensorimotor"}
    assert all(len(spec.revision) == 40 for spec in specs.values())


def test_explicit_missing_source_reports_the_install_command(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_SOURCE_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="anygoal models install lewm"):
        sources.source_path("lewm")


def test_installer_clones_and_checks_out_the_pinned_revision(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    marker = upstream / "model.py"
    marker.write_text("MODEL = 'original'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(upstream), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=AnyGoal Test",
            "-c",
            "user.email=anygoal@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = SourceSpec(
        id="test-model",
        repository=str(upstream),
        revision=revision,
        directory="test-model",
        environment="ANYGOAL_TEST_MODEL_SOURCE",
        marker="model.py",
    )
    monkeypatch.setenv("ANYGOAL_SOURCE_ROOT", str(tmp_path / "managed"))
    monkeypatch.setattr(sources, "get_source", lambda name: spec)

    status = sources.install_source("test-model")

    assert status.state == "ready"
    assert status.revision == revision
    assert status.path == (tmp_path / "managed" / "test-model").resolve()
    assert (status.path / "model.py").read_text(
        encoding="utf-8"
    ) == "MODEL = 'original'\n"
