from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import anygoal
from anygoal.planning import _select_starts, _task_callables
from anygoal.registry import get_registry


class FakeDataset:
    column_names = ["pixels", "action", "state"]

    def __init__(self, lengths=(30, 31, 40)):
        self.lengths = np.asarray(lengths, dtype=np.int64)

    def load_chunk(self, episodes, start, end):
        raise AssertionError("The fake World should own dataset extraction in this test.")


class FakeWorld:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = SimpleNamespace(envs=[])
        self.policy = None
        self.evaluate_kwargs = None
        self.evaluations = []
        self.closed = False

    def set_policy(self, policy):
        self.policy = policy

    def evaluate(self, **kwargs):
        self.evaluate_kwargs = kwargs
        self.evaluations.append(kwargs)
        return {
            "success_rate": 50.0,
            "episode_successes": np.array([True, False]),
            "seeds": None,
        }

    def close(self):
        self.closed = True


def test_select_starts_is_seeded_and_uses_all_valid_offsets():
    dataset = FakeDataset((27, 28))
    first = _select_starts(
        dataset, num_evals=4, goal_offset_steps=25, seed=7, selection="random"
    )
    second = _select_starts(
        dataset, num_evals=4, goal_offset_steps=25, seed=7, selection="random"
    )
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert set(zip(first[0], first[1])).issubset(
        {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)}
    )


def test_first_episode_selection_skips_short_episodes():
    episodes, starts = _select_starts(
        FakeDataset((20, 26, 25, 40)),
        num_evals=2,
        goal_offset_steps=25,
        seed=0,
        selection="first_episodes",
    )
    assert episodes.tolist() == [1, 3]
    assert starts.tolist() == [0, 0]


def test_plan_passes_policy_and_protocol_directly_to_stablewm_world(tmp_path):
    dataset = FakeDataset()
    policy = object()
    world = FakeWorld(num_envs=2)

    result = anygoal.plan(
        policy,
        data=dataset,
        task="avoid_t",
        seed=3,
        num_evals=2,
        output_dir=tmp_path,
        world=world,
    )

    assert world.policy is policy
    assert world.closed is False
    assert result.task_id == "avoidt"
    assert result.metrics["success_rate"] == 50.0
    assert world.evaluate_kwargs["dataset"] is dataset
    assert world.evaluate_kwargs["goal_offset"] == 25
    assert world.evaluate_kwargs["eval_budget"] == 50
    assert world.evaluate_kwargs["video"] is None
    assert world.evaluate_kwargs["callables"] == _task_callables(
        get_registry().dataset("avoidt")
    )
    assert world.evaluate_kwargs["episodes_idx"] == list(result.episodes)
    assert world.evaluate_kwargs["start_steps"] == list(result.start_steps)

    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["episodes"] == list(result.episodes)
    assert payload["metrics"]["episode_successes"] == [True, False]


def test_plan_refuses_to_replace_existing_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        anygoal.plan(
            object(),
            data=FakeDataset(),
            task="avoidt",
            num_evals=2,
            output_dir=tmp_path,
            world=FakeWorld(num_envs=2),
        )


def test_plan_accepts_a_local_stablewm_hdf5_path(tmp_path):
    path = tmp_path / "avoidt_test.h5"
    lengths = np.array([26, 27], dtype=np.int32)
    rows = int(lengths.sum())
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=lengths)
        handle.create_dataset("ep_offset", data=np.array([0, 26], dtype=np.int64))
        handle.create_dataset("action", data=np.zeros((rows, 2), dtype=np.float32))
        handle.create_dataset("pixels", data=np.zeros((rows, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("state", data=np.zeros((rows, 5), dtype=np.float32))

    world = FakeWorld(num_envs=2)
    result = anygoal.plan(
        object(),
        data=path,
        num_evals=2,
        world=world,
    )

    assert result.data == path.resolve()
    assert world.evaluate_kwargs["dataset"].h5_path == path.resolve()


def test_explicit_task_must_match_inferred_planning_data(tmp_path):
    path = tmp_path / "avoidt_test.h5"
    lengths = np.array([26, 27], dtype=np.int32)
    rows = int(lengths.sum())
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=lengths)
        handle.create_dataset("ep_offset", data=np.array([0, 26], dtype=np.int64))
        handle.create_dataset("action", data=np.zeros((rows, 2), dtype=np.float32))
        handle.create_dataset("pixels", data=np.zeros((rows, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("state", data=np.zeros((rows, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match"):
        anygoal.plan(
            object(),
            data=path,
            task="pusht",
            num_evals=2,
            world=FakeWorld(num_envs=2),
        )


def test_plan_requires_task_when_dataset_identity_is_unavailable():
    with pytest.raises(ValueError, match="could not be inferred"):
        anygoal.plan(
            object(),
            data=FakeDataset(),
            num_evals=2,
            world=FakeWorld(num_envs=2),
        )


def test_seed_sequence_returns_seed_keyed_results_and_separate_outputs(tmp_path):
    world = FakeWorld(num_envs=2)
    results = anygoal.plan(
        object(),
        data=FakeDataset(),
        task="avoidt",
        seed=[7, 3],
        num_evals=2,
        output_dir=tmp_path,
        world=world,
    )

    assert list(results) == [7, 3]
    assert [result.seed for result in results.values()] == [7, 3]
    assert len(world.evaluations) == 2
    assert results[7].metrics_path == tmp_path / "seed_7" / "metrics.json"
    assert results[3].metrics_path == tmp_path / "seed_3" / "metrics.json"
    assert results[7].metrics_path.is_file()
    assert results[3].metrics_path.is_file()


@pytest.mark.parametrize("seed", [[], [1, 1], [0, "1"], "0", True])
def test_invalid_seed_sequences_fail_early(seed):
    with pytest.raises((TypeError, ValueError)):
        anygoal.plan(
            object(),
            data=FakeDataset(),
            task="avoidt",
            seed=seed,
            num_evals=2,
            world=FakeWorld(num_envs=2),
        )


def test_plan_requires_task_specific_dataset_columns():
    with pytest.raises(ValueError, match="observation"):
        anygoal.plan(
            object(),
            data=FakeDataset(),
            task="insert_peg",
            num_evals=2,
            world=FakeWorld(num_envs=2),
        )


def test_plan_checks_policy_span_before_evaluation():
    policy = SimpleNamespace(cfg=SimpleNamespace(horizon=11, action_block=5))
    world = FakeWorld(num_envs=2)
    with pytest.raises(ValueError, match="must not exceed"):
        anygoal.plan(
            policy,
            data=FakeDataset(),
            task="avoidt",
            num_evals=2,
            eval_budget=50,
            world=world,
        )
    assert world.policy is None
