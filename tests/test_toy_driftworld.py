from __future__ import annotations

import torch

from anygoal.wm.toy_predictors import (
    DriftWorldStatePredictor,
    count_trainable_parameters,
)


SMALL_MODEL = {
    "history_size": 3,
    "hidden_dim": 32,
    "num_res_blocks": 2,
    "action_embed_dim": 16,
    "dropout": 0.0,
    "n_neg": 4,
}


def make_batch(batch_size=3, state_dim=5, action_dim=6):
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(batch_size, 4, state_dim, generator=generator)
    actions = torch.randn(batch_size, 4, action_dim, generator=generator)
    return states, actions


def test_state_driftworld_shapes_and_official_objective_gradients():
    model = DriftWorldStatePredictor(
        state_dim=5,
        action_dim=2,
        action_frameskip=3,
        **SMALL_MODEL,
    )
    states, actions = make_batch()
    output = model.loss(
        states,
        actions,
        generator=torch.Generator().manual_seed(9),
    )

    assert output["prediction"].shape == (3, 5)
    assert output["generated"].shape == (3, 4, 5)
    assert torch.isfinite(output["loss"])
    assert set(output["drift_metrics"]) >= {
        "loss_0.02",
        "loss_0.05",
        "loss_0.2",
    }

    output["loss"].backward()
    for parameter in model.inner_model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert all(parameter.grad is None for parameter in model.ema_model.parameters())


def test_lower_level_and_public_inference_preserve_noise_semantics():
    model = DriftWorldStatePredictor(
        state_dim=5,
        action_dim=2,
        action_frameskip=3,
        **SMALL_MODEL,
    ).eval()
    states, actions = make_batch()
    history = states[:, :3]
    action = actions[:, 2]
    noise = torch.randn(3, 5, generator=torch.Generator().manual_seed(10))

    generated = model(noise, history, action)
    predicted = model.predict_next_state(
        history,
        action,
        noise=noise,
        use_ema=False,
    )
    torch.testing.assert_close(generated, predicted)

    first = model.predict_next_state(
        history,
        action,
        generator=torch.Generator().manual_seed(11),
    )
    second = model.predict_next_state(
        history,
        action,
        generator=torch.Generator().manual_seed(11),
    )
    torch.testing.assert_close(first, second)
    assert not torch.equal(
        model.predict_next_state(history, action),
        model.predict_next_state(history, action),
    )


def test_action_film_changes_prediction_for_fixed_history_and_noise():
    model = DriftWorldStatePredictor(
        state_dim=5,
        action_dim=2,
        action_frameskip=3,
        **SMALL_MODEL,
    ).eval()
    states, _ = make_batch()
    noise = torch.randn(3, 5, generator=torch.Generator().manual_seed(12))
    low = torch.full((3, 6), -2.0)
    high = torch.full((3, 6), 2.0)

    low_prediction = model.predict_next_state(states[:, :3], low, noise=noise)
    high_prediction = model.predict_next_state(states[:, :3], high, noise=noise)

    assert not torch.equal(low_prediction, high_prediction)


def test_ema_update_matches_official_parameter_lerp():
    model = DriftWorldStatePredictor(
        state_dim=5,
        action_dim=2,
        action_frameskip=3,
        decay=0.75,
        **SMALL_MODEL,
    )
    inner_parameter = next(model.inner_model.parameters())
    ema_parameter = next(model.ema_model.parameters())
    original = ema_parameter.detach().clone()
    with torch.no_grad():
        inner_parameter.add_(2.0)
    model.update_ema()
    torch.testing.assert_close(ema_parameter, original + 0.5)


def test_default_and_large_parameter_counts_are_reportable_and_smaller_than_pixels():
    default = DriftWorldStatePredictor(
        state_dim=10,
        action_dim=2,
        action_frameskip=5,
    )
    large = DriftWorldStatePredictor(
        state_dim=10,
        action_dim=2,
        action_frameskip=5,
        hidden_dim=512,
        num_res_blocks=6,
    )

    assert count_trainable_parameters(default) == 1_426_058
    assert count_trainable_parameters(large) == 7_706_506
    assert count_trainable_parameters(default) < 8_700_000


def test_planning_returns_one_reproducible_cost_per_candidate():
    model = DriftWorldStatePredictor(
        state_dim=5,
        action_dim=2,
        action_frameskip=3,
        rollout_seed=29,
        **SMALL_MODEL,
    ).eval()
    generator = torch.Generator().manual_seed(13)
    info = {
        "state": torch.randn(2, 4, 3, 5, generator=generator),
        "action": torch.randn(2, 4, 2, 6, generator=generator),
        "goal_state": torch.randn(2, 4, 1, 5, generator=generator),
    }
    candidates = torch.randn(2, 4, 2, 6, generator=generator)

    first = model.get_cost(info, candidates)
    second = model.get_cost(info, candidates)

    assert first.shape == (2, 4)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_time_embedding_is_rejected_because_pusht_has_no_time_variable():
    try:
        DriftWorldStatePredictor(
            state_dim=5,
            action_dim=2,
            time_embed_dim=128,
            **SMALL_MODEL,
        )
    except ValueError as error:
        assert "no drifting-time input" in str(error)
    else:
        raise AssertionError("A spurious time-conditioning path was accepted.")
