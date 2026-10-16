from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from anygoal.cli import main
from anygoal.config import load_config
from anygoal.paths import (
    anygoal_home,
    cache_root,
    checkpoint_root,
    data_root,
    dataset_file,
    output_root,
    source_root,
)
from anygoal.resources import resource_path


def test_anygoal_home_derives_owned_roots(monkeypatch, tmp_path):
    home = tmp_path / "anygoal"
    monkeypatch.setenv("ANYGOAL_HOME", str(home))
    for name in (
        "ANYGOAL_CHECKPOINT_ROOT",
        "ANYGOAL_OUTPUT_ROOT",
        "ANYGOAL_SOURCE_ROOT",
        "ANYGOAL_CACHE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert anygoal_home() == home
    assert checkpoint_root() == home / "checkpoints"
    assert source_root() == home / "sources"
    assert cache_root() == home / "cache"
    assert output_root() == Path("outputs")


def test_specific_roots_override_anygoal_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYGOAL_HOME", str(tmp_path / "home"))
    roots = {
        "ANYGOAL_DATA_ROOT": tmp_path / "data",
        "ANYGOAL_CHECKPOINT_ROOT": tmp_path / "checkpoints",
        "ANYGOAL_OUTPUT_ROOT": tmp_path / "outputs",
        "ANYGOAL_SOURCE_ROOT": tmp_path / "sources",
        "ANYGOAL_CACHE_ROOT": tmp_path / "cache",
    }
    for name, path in roots.items():
        monkeypatch.setenv(name, str(path))

    assert data_root() == roots["ANYGOAL_DATA_ROOT"]
    assert checkpoint_root() == roots["ANYGOAL_CHECKPOINT_ROOT"]
    assert output_root() == roots["ANYGOAL_OUTPUT_ROOT"]
    assert source_root() == roots["ANYGOAL_SOURCE_ROOT"]
    assert cache_root() == roots["ANYGOAL_CACHE_ROOT"]


def test_dataset_file_accepts_data_directory_and_legacy_stablewm_root(tmp_path):
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    assert dataset_file("cube", root=data_directory) == data_directory / "cube.h5"

    legacy_directory = tmp_path / "stablewm" / "datasets"
    legacy_directory.mkdir(parents=True)
    legacy_file = legacy_directory / "cube.h5"
    legacy_file.touch()
    assert dataset_file("cube", root=tmp_path / "stablewm") == legacy_file


def test_packaged_configs_resolve_canonical_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYGOAL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ANYGOAL_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("ANYGOAL_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("ANYGOAL_SOURCE_ROOT", str(tmp_path / "sources"))
    for name in (
        "DRIFTWORLD_CUBE_CHECKPOINT",
        "DRIFTWORLD_OUTPUT_ROOT",
        "IRASIM_SCHEDULER_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    config = OmegaConf.load(resource_path("plan", "driftworld", "cube_lewm.yaml"))
    assert Path(config.dataset.root) == tmp_path / "data"
    assert Path(config.model.checkpoint) == tmp_path / "checkpoints/driftworld_cube.pt"
    assert Path(config.output.directory) == tmp_path / "outputs/driftworld/cube"

    irasim = OmegaConf.load(resource_path("plan", "irasim", "cube.yaml"))
    assert Path(irasim.model.scheduler_path) == (
        tmp_path / "sources/IRASim/pretrained_models/scheduler"
    )


def test_config_path_prints_effective_value(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANYGOAL_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    assert main(["config", "path", "checkpoints"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "checkpoints")


def test_default_config_reports_all_effective_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYGOAL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANYGOAL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("ANYGOAL_SITE_CONFIG", raising=False)
    config = load_config()

    assert config["paths"] == {
        "home": str(tmp_path / "home"),
        "data": str(tmp_path / "data"),
        "checkpoints": str(tmp_path / "home/checkpoints"),
        "outputs": "outputs",
        "sources": str(tmp_path / "home/sources"),
        "cache": str(tmp_path / "home/cache"),
    }


def test_repository_does_not_repeat_legacy_anygoal_default():
    repository = Path(__file__).resolve().parents[2]
    # Relative download staging directories are distinct from the legacy home root.
    forbidden = ("~/.cache/" + "anygoal",)
    roots = [repository / "anygoal", repository / "scripts", repository / "README.md"]
    offenders = []
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if not path.is_file() or "worldmodels" in path.parts:
                continue
            if path.suffix not in {".py", ".yaml", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(value in text for value in forbidden):
                offenders.append(str(path.relative_to(repository)))
    assert offenders == []
