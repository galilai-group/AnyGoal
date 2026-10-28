from pathlib import Path

from omegaconf import OmegaConf

from anygoal.wm import rollout


def test_load_stats_dataset_uses_configured_training_dataset(monkeypatch, tmp_path):
    captured = {}

    class Dataset:
        def __init__(self, path, keys_to_cache):
            captured["path"] = path
            captured["keys_to_cache"] = keys_to_cache

    monkeypatch.setattr(rollout.swm.data, "HDF5Dataset", Dataset)
    cfg = OmegaConf.create(
        {"dataset": {"root": str(tmp_path), "stats": "cube_single_expert"}}
    )

    result = rollout.load_stats_dataset(cfg, object(), ["action", "state"])

    assert isinstance(result, Dataset)
    assert captured == {
        "path": Path(tmp_path) / "cube_single_expert.h5",
        "keys_to_cache": ["action", "state"],
    }


def test_load_stats_dataset_falls_back_to_evaluation_dataset():
    dataset = object()
    cfg = OmegaConf.create({"dataset": {}})

    assert rollout.load_stats_dataset(cfg, dataset, ["action"]) is dataset
