from importlib import import_module
from inspect import signature
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from anygoal import collection
from anygoal.collections.cube_diagonal import collect_dataset as collect_cube_diagonal
from anygoal.collections.extract_peg import _trim_episode_prefix
from anygoal.datasets.insert_peg.support import write_variant_dataset
from anygoal.envs.insert_peg import env as insert_peg_env
from anygoal.paths import data_root


def test_python_collection_api_maps_common_episode_count(monkeypatch, tmp_path):
    captured = {}

    def collect_dataset(**options):
        captured["options"] = options
        return options["output"]

    module = SimpleNamespace(collect_dataset=collect_dataset)
    monkeypatch.setattr(collection, "import_module", lambda _: module)
    monkeypatch.setattr(collection, "_stamp_dataset_id", lambda path, _: path)
    output = collection.collect(
        "avoidt",
        num_episodes=400,
        num_parallel=8,
        output=tmp_path / "avoidt.h5",
    )
    assert output == (tmp_path / "avoidt.h5").resolve()
    assert captured["options"] == {
        "num_parallel": 8,
        "output": tmp_path / "avoidt.h5",
        "seed": 0,
        "episodes_per_goal": 100,
    }


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
            assert callable(module.collect_dataset)


def test_cube_diagonal_defaults_collect_one_canonical_batch():
    parameters = signature(collect_cube_diagonal).parameters
    assert parameters["episodes"].default == 400
    assert parameters["seed"].default == 10065
    assert parameters["num_parallel"].default == 5
    assert parameters["require_diagonal_lift"].default is True
    assert parameters["min_lifted_targets"].default == 2
    assert parameters["max_abs_block_quat_xy"].default == 0.2


def test_python_collection_api_uses_catalog_seed_unless_overridden(
    monkeypatch, tmp_path
):
    captured = []

    def collect_dataset(**options):
        captured.append(options)
        return tmp_path / "cube-diagonal.h5"

    module = SimpleNamespace(collect_dataset=collect_dataset)
    monkeypatch.setattr(collection, "import_module", lambda _: module)
    monkeypatch.setattr(collection, "_stamp_dataset_id", lambda path, _: path)

    collection.collect("cubediagonal")
    collection.collect("cubediagonal", seed=7)

    assert captured == [{"seed": 10065}, {"seed": 7}]


def test_cli_collection_uses_catalog_seed(monkeypatch, tmp_path):
    captured = {}

    def collect_dataset(*, seed=10065):
        captured["seed"] = seed
        return tmp_path / "cube-diagonal.h5"

    module = SimpleNamespace(collect_dataset=collect_dataset)
    monkeypatch.setattr(collection, "import_module", lambda _: module)
    monkeypatch.setattr(collection, "_stamp_dataset_id", lambda path, _: path)

    collection.collect_from_cli("cubediagonal", [])

    assert captured["seed"] == 10065


def test_collection_cli_parses_typed_collector_options(tmp_path):
    module = import_module("anygoal.collections.cube_diagonal")
    options = collection._parse_cli_options(
        "cubediagonal",
        module,
        [
            "--episodes",
            "10",
            "--output",
            str(tmp_path / "cube.h5"),
            "--no-require-diagonal-lift",
        ],
    )
    assert options == {
        "episodes": 10,
        "output": tmp_path / "cube.h5",
        "require_diagonal_lift": False,
    }


def test_collection_cli_rejects_unknown_collector_option():
    module = import_module("anygoal.collections.avoidt")
    with pytest.raises(SystemExit):
        collection._parse_cli_options("avoidt", module, ["--not-an-option", "1"])


def test_collection_result_records_dataset_identity(tmp_path):
    path = tmp_path / "dataset.h5"
    with h5py.File(path, "w"):
        pass
    assert collection._stamp_dataset_id(path, "avoidt") == path
    with h5py.File(path, "r") as handle:
        assert handle.attrs["anygoal_dataset_id"] == "avoidt"


def test_data_root_uses_stablewm_home(monkeypatch, tmp_path):
    monkeypatch.delenv("ANYGOAL_DATA_ROOT", raising=False)
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    assert data_root() == tmp_path / "datasets"


def test_insert_peg_variant_writer_preserves_canonical_schema(tmp_path):
    output = tmp_path / "variant.h5"
    first_states = np.arange(3 * 44, dtype=np.float64).reshape(3, 44)
    second_states = np.arange(2 * 44, dtype=np.float64).reshape(2, 44) + 1000
    episodes = [
        {"states": first_states, "actions": np.ones((2, 5))},
        {"states": second_states, "actions": np.full((1, 5), 2.0)},
    ]
    write_variant_dataset(
        output,
        episodes,
        {"task": "test", "seed": 7},
        include_pixels=False,
        env=None,
    )

    expected_observation = np.concatenate(
        (first_states[:-1], second_states[:-1])
    ).astype(np.float32)
    with h5py.File(output, "r") as dataset:
        np.testing.assert_array_equal(dataset["ep_len"][:], [2, 1])
        np.testing.assert_array_equal(dataset["ep_offset"][:], [0, 2])
        np.testing.assert_array_equal(dataset["ep_idx"][:], [0, 0, 1])
        np.testing.assert_array_equal(dataset["fragment_start"][:], [True, False, True])
        np.testing.assert_array_equal(dataset["fragment_end"][:], [False, True, True])
        np.testing.assert_array_equal(dataset["observation"][:], expected_observation)
        assert dataset["privilege"].id == dataset["observation"].id
        for name, columns in insert_peg_env.STATE_FIELDS.items():
            np.testing.assert_array_equal(
                dataset[name][:], expected_observation[:, columns]
            )


def test_extract_peg_trimming_removes_each_episode_prefix():
    rollout = {
        "states": np.arange(11 * 2).reshape(11, 2),
        "actions": np.arange(10 * 5).reshape(10, 5),
        "state_support": np.arange(10),
        "action_support": np.arange(10) + 10,
        "floor_clearance": np.arange(10) + 20,
        "target_contacts": np.arange(10) + 30,
    }

    trimmed, removed = _trim_episode_prefix(rollout, 0.4)

    assert removed == 4
    np.testing.assert_array_equal(trimmed["states"], rollout["states"][4:])
    for name in rollout.keys() - {"states"}:
        np.testing.assert_array_equal(trimmed[name], rollout[name][4:])
    assert len(trimmed["states"]) == len(trimmed["actions"]) + 1


def test_extract_peg_trimming_uses_floor_rounding():
    rollout = {
        "states": np.zeros((9, 44)),
        "actions": np.zeros((8, 5)),
    }

    trimmed, removed = _trim_episode_prefix(rollout, 0.4)

    assert removed == 3
    assert len(trimmed["actions"]) == 5
