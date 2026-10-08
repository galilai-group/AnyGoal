"""Check shuffled linkage, ambiguity, terminal edges, and data round trips."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "convert_rlu",
    Path(__file__).resolve().parents[1] / "scripts/data/convert_rlu_insert_peg.py",
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def columns(starts, ends, discounts=None):
    n = len(starts)
    return {
        "observation": np.repeat(np.array(starts, dtype=np.float32)[:, None], 44, axis=1),
        "next_observation": np.repeat(np.array(ends, dtype=np.float32)[:, None], 44, axis=1),
        "action": np.zeros((n, 5), dtype=np.float32),
        "next_action": np.zeros((n, 5), dtype=np.float32),
        "discount": np.ones(n, dtype=np.float32) if discounts is None else np.array(discounts, dtype=np.float32),
        "reward": np.arange(n, dtype=np.float32),
    }


def test_shuffled_chains_and_round_trip(tmp_path):
    source = columns([2, 10, 0, 1], [3, 11, 1, 2])
    order, lengths, stats = converter.recover_fragments(source)
    np.testing.assert_array_equal(order, [1, 2, 3, 0])
    np.testing.assert_array_equal(lengths, [1, 3])
    assert stats["original_episode_count"] is None
    path = tmp_path / "test.h5"
    converter.write_hdf5(path, source, order, lengths, stats)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["ep_offset"][:], [0, 1])
        np.testing.assert_array_equal(handle["step_idx"][:], [0, 0, 1, 2])
        np.testing.assert_array_equal(handle["reward"][:], source["reward"][order])
        np.testing.assert_array_equal(handle["next_observation"][:], source["next_observation"][order])
        assert handle["privilege"].id == handle["observation"].id
    with pytest.raises(FileExistsError):
        converter.write_hdf5(path, source, order, lengths, stats)


def test_duplicate_starts_and_ambiguous_predecessors_are_cut():
    for source in (columns([0, 1, 1], [1, 2, 2]), columns([0, 5, 1], [1, 1, 2])):
        order, lengths, _ = converter.recover_fragments(source)
        np.testing.assert_array_equal(np.sort(order), [0, 1, 2])
        np.testing.assert_array_equal(lengths, [1, 1, 1])


def test_terminal_and_action_mismatch_do_not_link():
    source = columns([0, 1], [1, 2], [0, 1])
    assert converter.recover_fragments(source)[1].tolist() == [1, 1]
    source["discount"][0] = 1
    source["next_action"][0, 0] = 0.5
    assert converter.recover_fragments(source)[1].tolist() == [1, 1]


def test_cycles_fail_without_inventing_boundaries():
    with pytest.raises(ValueError, match="cycles"):
        converter.recover_fragments(columns([0, 1], [1, 0]))


def test_pack_rejects_missing_state_and_preserves_order():
    state = {field: np.full(dim, i, dtype=np.float32)
             for i, (field, dim) in enumerate(converter.FIELDS.items())}
    row = dict(s=state, s_=state.copy(), a=np.arange(5, dtype=np.float32),
               a_=np.ones(5, dtype=np.float32), r=np.float32(0), discount=np.float32(1))
    packed = converter.pack_records(np.array([row], dtype=object))
    np.testing.assert_array_equal(packed["observation"][0], np.concatenate(list(state.values())))
    row["s_"] = {"arm_pos": state["arm_pos"]}
    with pytest.raises(ValueError, match="observation fields"):
        converter.pack_records(np.array([row], dtype=object))
