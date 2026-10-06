from __future__ import annotations

import numpy as np
import pytest
import torch

from anygoal.evaluation.prediction import (
    PredictionProtocol,
    evaluate_prediction,
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


class ZeroPredictionAdapter:
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
    protocol = PredictionProtocol(
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
    )
    result = evaluate_prediction(
        FakeDataset([10, 12]),
        ZeroPredictionAdapter(),
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
    protocol = PredictionProtocol(
        history_length=3,
        frame_skip=2,
        horizons=(2, 4),
        num_episodes=2,
        batch_size=2,
    )
    result = evaluate_prediction(
        FakeDataset([7, 7]),
        ZeroPredictionAdapter(),
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
