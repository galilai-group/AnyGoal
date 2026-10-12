from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

import anygoal
from anygoal.evaluation.rollout import (
    RolloutProtocol,
    evaluate_rollout,
    sample_episode_indices,
)


class FakeDataset:
    def __init__(self, lengths):
        self.lengths = np.asarray(lengths)

    def load_episode(self, episode_index):
        length = int(self.lengths[episode_index])
        return {
            "value": torch.arange(length, dtype=torch.float32)[:, None],
            "action": torch.zeros(length, 1),
            "episode_idx": torch.full((length,), episode_index),
        }


class ZeroRolloutAdapter:
    model_name = "zero"
    error_space = "state_mse"
    observation_keys = ("value",)

    def rollout_errors(self, observations, action_blocks, *, rollout_seed):
        del action_blocks, rollout_seed
        targets = observations["value"][:, 3:, 0]
        return targets.square()


def test_episode_sampling_is_reproducible_and_filters_short_episodes():
    lengths = np.asarray([8, 9, 10, 11, 12])
    selected_a, eligible_a = sample_episode_indices(
        lengths, required_length=9, num_episodes=3, seed=7
    )
    selected_b, eligible_b = sample_episode_indices(
        lengths, required_length=9, num_episodes=3, seed=7
    )
    np.testing.assert_array_equal(eligible_a, np.asarray([1, 2, 3, 4]))
    np.testing.assert_array_equal(eligible_a, eligible_b)
    np.testing.assert_array_equal(selected_a, selected_b)


def test_shared_evaluator_uses_all_starts_and_episode_balanced_averages():
    protocol = RolloutProtocol(
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
    )
    result = evaluate_rollout(
        FakeDataset([10, 12]),
        ZeroRolloutAdapter(),
        protocol=protocol,
        seed=0,
        metadata={"environment": "fake"},
    )

    assert result["sampling"]["num_starting_points"] == 6
    assert [episode["num_starting_points"] for episode in result["episodes"]] == [
        2,
        4,
    ]
    assert result["short_horizon_error"] == pytest.approx(50.0)
    assert result["long_horizon_error"]["2"] == pytest.approx(50.0)
    assert result["long_horizon_error"]["4"] == pytest.approx(132.0)


def test_shared_evaluator_falls_back_to_short_horizon_for_short_episodes():
    protocol = RolloutProtocol(
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
    )
    result = evaluate_rollout(
        FakeDataset([7, 7]),
        ZeroRolloutAdapter(),
        protocol=protocol,
        seed=0,
        metadata={"environment": "fake"},
    )

    assert result["protocol"]["evaluation_mode"] == "short_only"
    assert result["sampling"]["full_rollout_eligible_episode_count"] == 0
    assert result["sampling"]["num_starting_points"] == 2
    assert result["short_horizon_error"] == pytest.approx(36.0)
    assert result["long_horizon_error"] == {"2": None, "4": None}
    assert result["metrics"]["long_horizon"]["4"]["available"] is False


def test_public_rollout_api_preserves_shared_protocol_metrics():
    result = anygoal.evaluate_rollout(
        ZeroRolloutAdapter(),
        data=FakeDataset([10, 12]),
        task="avoidt",
        seed=0,
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
    )

    assert result.task_id == "avoidt"
    assert result.error_space == "state_mse"
    assert result.short_horizon_steps == 2
    assert result.short_horizon_mse == pytest.approx(50.0)
    assert result.horizons[4].cumulative_mse == pytest.approx(132.0)
    with pytest.raises(ValueError, match="frame_skip is 5"):
        _ = result.five_step_mse


def test_public_rollout_api_returns_seed_results_and_separate_outputs(tmp_path):
    results = anygoal.evaluate_rollout(
        ZeroRolloutAdapter(),
        data=FakeDataset([10, 12]),
        task="avoidt",
        seed=[7, 3],
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
        output_dir=tmp_path,
    )

    assert list(results) == [7, 3]
    for seed, result in results.items():
        assert result.seed == seed
        assert result.metrics_path == tmp_path / f"seed_{seed}" / "metrics.json"
        payload = json.loads(result.metrics_path.read_text())
        assert payload["seed"] == seed


def test_public_rollout_api_accepts_registered_model_ids(monkeypatch):
    captured = {}

    def fake_registered_adapter(model_id, task_spec, dataset, **kwargs):
        captured.update(
            model_id=model_id,
            task_id=task_spec.id,
            dataset=dataset,
            kwargs=kwargs,
        )
        return ZeroRolloutAdapter(), "prejepa", None, {"loaded": True}

    monkeypatch.setattr(
        "anygoal.rollout._registered_adapter", fake_registered_adapter
    )
    dataset = FakeDataset([10, 12])
    result = anygoal.evaluate_rollout(
        "prejepa",
        data=dataset,
        task="avoidt",
        seed=0,
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
        device="cpu",
    )

    assert result.model_id == "prejepa"
    assert result.raw_metrics["loaded"] is True
    assert captured["dataset"] is dataset


def test_public_rollout_api_infers_task_from_hdf5_filename(tmp_path):
    path = tmp_path / "avoidt_test.h5"
    lengths = np.array([10, 12], dtype=np.int32)
    rows = int(lengths.sum())
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=lengths)
        handle.create_dataset("ep_offset", data=np.array([0, 10], dtype=np.int64))
        handle.create_dataset("action", data=np.zeros((rows, 1), dtype=np.float32))
        handle.create_dataset("pixels", data=np.zeros((rows, 2, 2, 3), dtype=np.uint8))
        handle.create_dataset("value", data=np.arange(rows, dtype=np.float32)[:, None])

    result = anygoal.evaluate_rollout(
        ZeroRolloutAdapter(),
        data=path,
        seed=0,
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
    )

    assert result.task_id == "avoidt"


def test_five_step_convenience_matches_default_short_horizon():
    result = anygoal.evaluate_rollout(
        ZeroRolloutAdapter(),
        data=FakeDataset([40]),
        task="avoidt",
        seed=0,
        num_episodes=1,
        max_starts_per_episode=1,
    )

    assert result.short_horizon_steps == 5
    assert result.five_step_mse == result.short_horizon_mse


def test_research_script_uses_the_shared_rollout_loader():
    from anygoal.paths import resolve_path
    from anygoal.wm.rollout import build_adapter
    from anygoal.runners import rollout as evaluation_script

    assert evaluation_script.build_adapter is build_adapter
    assert evaluation_script.resolve_path is resolve_path


def test_registered_models_have_all_base_rollout_profiles():
    from anygoal.rollout import _load_config
    from anygoal.registry import get_registry

    config = _load_config(None)
    for model in get_registry().models.values():
        assert model.evaluation_id in config.models
        assert {"pusht", "cube", "tworoom", "insert_peg"}.issubset(
            config.models[model.evaluation_id]
        )
