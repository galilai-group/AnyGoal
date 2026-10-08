from argparse import Namespace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

import anygoal.collection as collection
from anygoal.collections.cube_diagonal import parse_args as parse_cube_diagonal_args


def test_python_collection_api_maps_common_episode_count(monkeypatch, tmp_path):
    captured = {}

    def parse_args(arguments):
        captured["arguments"] = arguments
        return Namespace(output=tmp_path / "avoidt.h5")

    module = SimpleNamespace(
        parse_args=parse_args,
        run=lambda args: args.output,
    )
    monkeypatch.setattr(collection, "import_module", lambda _: module)
    monkeypatch.setattr(collection, "_stamp_dataset_id", lambda path, _: path)
    output = collection.collect(
        "avoidt",
        num_episodes=400,
        num_parallel=8,
        output=tmp_path / "avoidt.h5",
    )
    assert output == (tmp_path / "avoidt.h5").resolve()
    assert captured["arguments"] == [
        "--num-parallel",
        "8",
        "--output",
        str(tmp_path / "avoidt.h5"),
        "--seed",
        "0",
        "--episodes-per-goal",
        "100",
    ]


def test_python_collection_api_rejects_unbalanced_episode_count():
    try:
        collection.collect("avoidt", num_episodes=399)
    except ValueError as error:
        assert "divisible by 4" in str(error)
    else:
        raise AssertionError("Expected an unbalanced episode count to fail")


@pytest.mark.parametrize("dataset_id", ["pusht", "cube", "insertpeg", "tworoom"])
def test_base_datasets_are_not_collection_targets(dataset_id):
    with pytest.raises(ValueError, match="externally provided base dataset"):
        collection.collect(dataset_id)


def test_generated_collectors_live_in_collection_package():
    registry = collection.get_registry()
    for spec in registry.datasets.values():
        if spec.source == "generated":
            assert spec.collector.startswith("anygoal.collections.")
            module = import_module(spec.collector)
            assert callable(module.parse_args)
            assert callable(module.run)


def test_cube_diagonal_defaults_collect_one_canonical_batch():
    args = parse_cube_diagonal_args([])
    assert args.episodes == 400
    assert args.seed == 10065
    assert args.num_parallel == 5
    assert args.gif_episodes == 0
    assert args.require_diagonal_lift is True
    assert args.min_lifted_targets == 2
    assert args.max_abs_block_quat_xy == 0.2


def test_python_collection_api_uses_catalog_seed_unless_overridden(
    monkeypatch, tmp_path
):
    captured = []

    def parse_args(arguments):
        captured.append(arguments)
        return Namespace(output=tmp_path / "cube-diagonal.h5")

    module = SimpleNamespace(parse_args=parse_args, run=lambda args: args.output)
    monkeypatch.setattr(collection, "import_module", lambda _: module)
    monkeypatch.setattr(collection, "_stamp_dataset_id", lambda path, _: path)

    collection.collect("cubediagonal")
    collection.collect("cubediagonal", seed=7)

    assert captured == [["--seed", "10065"], ["--seed", "7"]]


def test_cli_collection_uses_catalog_seed(monkeypatch, tmp_path):
    captured = {}

    def parse_args(arguments):
        captured["arguments"] = arguments
        return Namespace(output=tmp_path / "cube-diagonal.h5")

    module = SimpleNamespace(parse_args=parse_args, run=lambda args: args.output)
    monkeypatch.setattr(collection, "import_module", lambda _: module)
    monkeypatch.setattr(collection, "_stamp_dataset_id", lambda path, _: path)

    collection.collect_from_cli("cubediagonal", [])

    assert captured["arguments"] == ["--seed", "10065"]


def test_collection_result_records_dataset_identity(tmp_path):
    path = tmp_path / "dataset.h5"
    with h5py.File(path, "w"):
        pass
    assert collection._stamp_dataset_id(path, "avoidt") == path
    with h5py.File(path, "r") as handle:
        assert handle.attrs["anygoal_dataset_id"] == "avoidt"
