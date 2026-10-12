from __future__ import annotations

import gc
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from anygoal.datasets import (
    CubeEverythingStateAdapter,
    CubeTrueStateAdapter,
    MinMaxTransform,
    PushTTrueStateAdapter,
    TwoRoomTrueStateAdapter,
    ZScoreTransform,
    adapted_zscore_statistics,
)
from anygoal.evaluation._rollout_adapters import standardize_tensor
from anygoal.wm.toy_predictors import (
    ActionDuplicateCausalPredictor,
    AdaLNActionPredictor,
    count_trainable_parameters,
    FixedMinMaxScaler,
    FixedStandardScaler,
    ToyWorldModelPolicy,
)


SMALL_TRANSFORMER = {
    "hidden_dim": 32,
    "depth": 2,
    "heads": 4,
    "dim_head": 8,
    "mlp_dim": 64,
    "dropout": 0.0,
    "emb_dropout": 0.0,
}


def sequence_batch(batch_size=2):
    generator = torch.Generator().manual_seed(5)
    return (
        torch.randn(batch_size, 4, 10, generator=generator),
        torch.randn(batch_size, 4, 10, generator=generator),
    )


def test_pusht_adapter_uses_canonical_state_order_and_sine_cosine_angles():
    adapter = PushTTrueStateAdapter()
    raw = torch.tensor([[10.0, 20.0, 30.0, 40.0, np.pi / 2, 5.0, 6.0]])

    encoded = adapter.encode_state(raw)

    expected = torch.tensor(
        [
            [
                256.0,
                256.0,
                np.sqrt(0.5),
                np.sqrt(0.5),
                30.0,
                40.0,
                1.0,
                0.0,
                10.0,
                20.0,
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(encoded, expected, atol=1e-6, rtol=1e-6)
    decoded = adapter.decode_state(encoded)
    torch.testing.assert_close(decoded[:, :5], raw[:, :5], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded[:, 5:], torch.zeros(1, 2))


def test_cube_adapters_use_requested_essential_and_extra_state_order():
    state = {
        "proprio_effector_pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "proprio_effector_yaw": torch.tensor([[np.pi / 2]]),
        "proprio_gripper_opening": torch.tensor([[0.25]]),
        "privileged_block_0_pos": torch.tensor([[4.0, 5.0, 6.0]]),
        "privileged_block_0_yaw": torch.tensor([[np.pi]]),
        "proprio_joint_pos": torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]),
    }
    expected_essential = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 1.0, 0.25, 4.0, 5.0, 6.0, -1.0, 0.0]]
    )

    essential = CubeTrueStateAdapter(include_joint_positions=False)
    extra = CubeTrueStateAdapter(include_joint_positions=True)

    torch.testing.assert_close(
        essential.encode_state(state), expected_essential, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        extra.encode_state(state),
        torch.cat((expected_essential, state["proprio_joint_pos"]), dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert essential.state_dim == 11
    assert extra.state_dim == 17


def test_cube_contact_adapter_uses_raw_yaw_and_nine_state_dimensions():
    state = {
        "proprio_effector_pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "proprio_effector_yaw": torch.tensor([[np.pi / 2]]),
        "proprio_gripper_opening": torch.tensor([[0.25]]),
        "proprio_gripper_contact": torch.tensor([[1.0]]),
        "privileged_block_0_pos": torch.tensor([[4.0, 5.0, 6.0]]),
    }
    adapter = CubeTrueStateAdapter(include_gripper_contact=True)

    encoded = adapter.encode_state(state)

    expected = torch.tensor(
        [[1.0, 2.0, 3.0, np.pi / 2, 0.25, 1.0, 4.0, 5.0, 6.0]]
    )
    torch.testing.assert_close(encoded, expected)
    assert adapter.state_dim == 9
    assert "privileged_block_0_yaw" not in adapter.state_keys


def test_cube_everything_adapters_use_requested_state_order():
    state = {
        "proprio_effector_pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "proprio_effector_yaw": torch.tensor([[np.pi / 2]]),
        "proprio_gripper_opening": torch.tensor([[0.25]]),
        "proprio_gripper_contact": torch.tensor([[1.0]]),
        "proprio_gripper_vel": torch.tensor([[-0.5]]),
        "privileged_block_0_pos": torch.tensor([[4.0, 5.0, 6.0]]),
        "privileged_block_0_quat": torch.tensor([[0.7, 0.1, 0.2, 0.3]]),
        "privileged_block_0_yaw": torch.tensor([[np.pi]]),
        "privileged_target_block_pos": torch.tensor([[7.0, 8.0, 9.0]]),
        "privileged_target_block_yaw": torch.tensor([[-np.pi / 2]]),
    }
    base = CubeEverythingStateAdapter()
    extreme = CubeEverythingStateAdapter(include_target=True)

    expected_base = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                0.0,
                1.0,
                0.25,
                1.0,
                -0.5,
                4.0,
                5.0,
                6.0,
                0.7,
                0.1,
                0.2,
                0.3,
            ]
        ]
    )
    expected_extreme = torch.cat(
        (expected_base, torch.tensor([[-1.0, 0.0, 7.0, 8.0, 9.0, 0.0, -1.0]])),
        dim=-1,
    )
    torch.testing.assert_close(
        base.encode_state(state), expected_base, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        extreme.encode_state(state), expected_extreme, atol=1e-6, rtol=1e-6
    )
    assert base.state_dim == 15
    assert extreme.state_dim == 22


def test_tworoom_adapters_select_agent_and_door_coordinates_in_order():
    observation = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0, 112.0, 49.0, 0.0, 0.0, 0.0, 0.0]]
    )
    essential = TwoRoomTrueStateAdapter(include_doors=False)
    extra = TwoRoomTrueStateAdapter(include_doors=True)

    torch.testing.assert_close(
        essential.encode_state(observation), torch.tensor([[10.0, 20.0]])
    )
    torch.testing.assert_close(
        extra.encode_state(observation),
        torch.tensor([[10.0, 20.0, 112.0, 49.0]]),
    )
    assert essential.state_dim == 2
    assert extra.state_dim == 4
    assert essential.action_dim == extra.action_dim == 2


def test_tworoom_adapter_accepts_planning_state_dict():
    observation = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    adapter = TwoRoomTrueStateAdapter(include_doors=True)

    encoded = adapter.encode_state({"observation": observation})

    torch.testing.assert_close(encoded, observation[:, [0, 1, 4, 5]])


def test_tworoom_fixed_minmax_scales_every_state_dimension_by_224():
    values = torch.tensor([[0.0, 56.0, 112.0, 224.0]])
    transform = MinMaxTransform(
        torch.zeros(4), torch.full((4,), 224.0), source="state"
    )

    normalized = transform({"state": values})["state"]

    torch.testing.assert_close(
        normalized, torch.tensor([[0.0, 0.25, 0.5, 1.0]])
    )


def test_zscore_transform_keeps_constant_dimensions_finite():
    transform = ZScoreTransform(
        mean=torch.tensor([256.0, 0.0]),
        std=torch.tensor([0.0, 2.0]),
        source="state",
    )
    sample = {"state": torch.tensor([[256.0, 4.0]])}

    transformed = transform(sample)["state"]

    assert torch.isfinite(transformed).all()
    torch.testing.assert_close(transformed, torch.tensor([[0.0, 2.0]]))


def test_minmax_training_and_planning_transforms_match():
    minimum = torch.tensor([-2.0, 10.0])
    maximum = torch.tensor([2.0, 30.0])
    values = torch.tensor([[0.0, 15.0]])

    transformed = MinMaxTransform(minimum, maximum, "state")(
        {"state": values.clone()}
    )["state"]
    processor = FixedMinMaxScaler(minimum.numpy(), maximum.numpy())

    torch.testing.assert_close(transformed, torch.tensor([[0.5, 0.25]]))
    np.testing.assert_allclose(processor.transform(values.numpy()), transformed)
    np.testing.assert_allclose(
        processor.inverse_transform(transformed.numpy()), values.numpy()
    )
    torch.testing.assert_close(
        standardize_tensor(values, processor), transformed
    )


def test_adapted_statistics_map_hardcoded_goal_exactly_to_zero():
    class Dataset:
        def get_col_data(self, source):
            assert source == "state"
            row = np.array([10.0, 20.0, 30.0, 40.0, 0.5, 0.0, 0.0])
            return np.repeat(row[None], 10_000, axis=0)

    adapter = PushTTrueStateAdapter()
    mean, std = adapted_zscore_statistics(Dataset(), adapter, "state")
    encoded = adapter.encode_state(torch.tensor(Dataset().get_col_data("state")[:1]))
    transformed = ZScoreTransform(mean, std, "state")({"state": encoded})["state"]

    torch.testing.assert_close(transformed[:, :4], torch.zeros(1, 4))
    torch.testing.assert_close(std[:4], torch.ones(4))


def test_adaln_predictor_returns_one_future_state_and_backpropagates():
    states, actions = sequence_batch()
    model = AdaLNActionPredictor(
        state_dim=10,
        action_dim=2,
        action_frameskip=5,
        history_size=3,
        **SMALL_TRANSFORMER,
    )

    output = model.loss(states, actions)

    assert output["prediction"].shape == (2, 10)
    assert output["shifted_predictions"].shape == (2, 3, 10)
    expected = F.mse_loss(output["shifted_predictions"], states[:, 1:4])
    torch.testing.assert_close(output["loss"], expected)
    torch.testing.assert_close(
        output["future_loss"],
        F.mse_loss(output["prediction"], states[:, 3]),
    )
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cube_adaln_action_embedder_matches_lewm_conditioning_dimensions():
    model = AdaLNActionPredictor(
        state_dim=17,
        action_dim=5,
        action_frameskip=5,
        history_size=3,
        **SMALL_TRANSFORMER,
    )

    assert model.action_dim == 25
    assert model.action_embedding.projection.in_channels == 25
    assert model.action_embedding.projection.out_channels == 10
    assert model.action_embedding.embedding[0].in_features == 10


def test_action_duplicate_layout_and_state_only_loss():
    states, actions = sequence_batch()
    model = ActionDuplicateCausalPredictor(
        state_dim=10,
        action_dim=2,
        action_frameskip=5,
        history_size=3,
        **SMALL_TRANSFORMER,
    )
    tokens = model.build_tokens(states[:, :3], actions[:, :3])

    assert tokens.shape == (2, 3, 10, 11)
    torch.testing.assert_close(tokens[..., 0], states[:, :3])
    torch.testing.assert_close(
        tokens[..., 1:], actions[:, :3, None].expand(-1, -1, 10, -1)
    )

    output = model.loss(states, actions)
    assert output["shifted_predictions"].shape == (2, 3, 10, 11)
    expected = F.mse_loss(
        output["shifted_predictions"][..., 0], states[:, 1:4]
    )
    torch.testing.assert_close(output["loss"], expected)
    torch.testing.assert_close(
        output["future_loss"],
        F.mse_loss(output["prediction"][..., 0], states[:, 3]),
    )
    assert output["prediction"].shape == (2, 10, 11)


def test_default_architectures_match_lewm_size_and_frameskip_one():
    configurations = (
        (
            AdaLNActionPredictor,
            {"state_dim": 10, "action_dim": 2},
        ),
        (
            ActionDuplicateCausalPredictor,
            {"state_dim": 10, "action_dim": 2},
        ),
    )
    for model_type, kwargs in configurations:
        model = model_type(**kwargs)
        assert model.action_dim == 2
        assert len(model.blocks) == 6
        assert count_trainable_parameters(model) > 0
        del model
        gc.collect()


def test_all_predictors_return_one_cost_per_cem_candidate():
    models = (
        AdaLNActionPredictor(
            state_dim=10,
            action_dim=2,
            action_frameskip=5,
            history_size=3,
            **SMALL_TRANSFORMER,
        ),
        ActionDuplicateCausalPredictor(
            state_dim=10,
            action_dim=2,
            action_frameskip=5,
            history_size=3,
            **SMALL_TRANSFORMER,
        ),
    )
    generator = torch.Generator().manual_seed(19)
    info = {
        "state": torch.randn(2, 3, 3, 10, generator=generator),
        "action": torch.randn(2, 3, 2, 10, generator=generator),
        "goal_state": torch.randn(2, 3, 1, 10, generator=generator),
    }
    candidates = torch.randn(2, 3, 2, 10, generator=generator)

    for model in models:
        model.eval()
        costs = model.get_cost(info, candidates)
        assert costs.shape == (2, 3)
        assert torch.isfinite(costs).all()


def test_toy_policy_builds_training_aligned_state_action_history():
    adapter = PushTTrueStateAdapter()
    policy = ToyWorldModelPolicy(
        solver=object(),
        config=type("Config", (), {"history_len": 3, "action_block": 5})(),
        adapter=adapter,
        state_processor=FixedStandardScaler(np.zeros(10), np.ones(10)),
        action_processor=FixedStandardScaler(np.zeros(2), np.ones(2)),
    )
    policy._state_history = [deque(maxlen=3)]
    policy._transition_history = [deque(maxlen=2)]
    policy._pending_actions = [deque(maxlen=5)]
    initial = np.array([10, 20, 30, 40, 0.0, 0, 0], dtype=np.float32)
    next_state = initial.copy()
    next_state[0] = 11
    goal = np.array([50, 60, 256, 256, np.pi / 4, 0, 0], dtype=np.float32)

    first = policy._prepare_info(
        {
            "state": initial[None, None],
            "goal_state": goal[None, None],
            "action": np.array([[[0.9, 0.9]]], dtype=np.float32),
        }
    )
    prepared = first
    for step in range(5):
        next_state[0] = 11 + step
        prepared = policy._prepare_info(
            {
                "state": next_state[None, None],
                "goal_state": goal[None, None],
                "action": np.array(
                    [[[0.25 + step, -0.5 - step]]], dtype=np.float32
                ),
            }
        )
        if step < 4:
            torch.testing.assert_close(prepared["state"], first["state"])

    assert first["state"].shape == (1, 3, 10)
    torch.testing.assert_close(first["state"][:, 0], first["state"][:, 2])
    torch.testing.assert_close(first["action"], torch.zeros(1, 2, 10))
    torch.testing.assert_close(prepared["state"][:, 0], prepared["state"][:, 1])
    torch.testing.assert_close(
        prepared["action"],
        torch.tensor(
            [
                [
                    [0.0] * 10,
                    [
                        0.25,
                        -0.5,
                        1.25,
                        -1.5,
                        2.25,
                        -2.5,
                        3.25,
                        -3.5,
                        4.25,
                        -4.5,
                    ],
                ]
            ]
        ),
    )


def test_toy_policy_builds_cube_state_and_goal_from_named_columns():
    adapter = CubeTrueStateAdapter(include_joint_positions=True)
    policy = ToyWorldModelPolicy(
        solver=object(),
        config=type("Config", (), {"history_len": 3, "action_block": 5})(),
        adapter=adapter,
        state_processor=FixedStandardScaler(np.zeros(17), np.ones(17)),
        action_processor=FixedStandardScaler(np.zeros(5), np.ones(5)),
    )
    policy._state_history = [deque(maxlen=3)]
    policy._transition_history = [deque(maxlen=2)]
    policy._pending_actions = [deque(maxlen=5)]
    canonical_current = {
        "proprio_effector_pos": np.array([[[1.0, 2.0, 3.0]]]),
        "proprio_effector_yaw": np.array([[[np.pi / 2]]]),
        "proprio_gripper_opening": np.array([[[0.25]]]),
        "privileged_block_0_pos": np.array([[[4.0, 5.0, 6.0]]]),
        "privileged_block_0_yaw": np.array([[[np.pi]]]),
        "proprio_joint_pos": np.array([[[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]),
    }
    current = {
        adapter.environment_state_keys[key]: value
        for key, value in canonical_current.items()
    }
    goal = {
        f"goal_{key}": value + 0.1 for key, value in canonical_current.items()
    }

    prepared = policy._prepare_info(
        {
            **current,
            **goal,
            "action": np.zeros((1, 1, 5), dtype=np.float32),
        }
    )

    expected_current = adapter.encode_state(
        {
            key: torch.as_tensor(value[:, -1])
            for key, value in canonical_current.items()
        }
    )
    expected_goal = adapter.encode_state(
        {
            key: torch.as_tensor(goal[f"goal_{key}"][:, -1])
            for key in canonical_current
        }
    )
    assert prepared["state"].shape == (1, 3, 17)
    assert prepared["goal_state"].shape == (1, 1, 17)
    torch.testing.assert_close(prepared["state"][:, 0], expected_current)
    torch.testing.assert_close(prepared["goal_state"][:, 0], expected_goal)


def test_toy_policy_builds_nine_dimensional_cube_contact_state_and_goal():
    adapter = CubeTrueStateAdapter(include_gripper_contact=True)
    minimum = np.array([0.0, 0.0, 0.0, -np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    maximum = np.array([2.0, 4.0, 6.0, np.pi, 1.0, 1.0, 8.0, 10.0, 12.0])
    policy = ToyWorldModelPolicy(
        solver=object(),
        config=type("Config", (), {"history_len": 3, "action_block": 5})(),
        adapter=adapter,
        state_processor=FixedMinMaxScaler(minimum, maximum),
        action_processor=FixedStandardScaler(np.zeros(5), np.ones(5)),
    )
    policy._state_history = [deque(maxlen=3)]
    policy._transition_history = [deque(maxlen=2)]
    policy._pending_actions = [deque(maxlen=5)]
    canonical_current = {
        "proprio_effector_pos": np.array([[[1.0, 2.0, 3.0]]]),
        "proprio_effector_yaw": np.array([[[0.0]]]),
        "proprio_gripper_opening": np.array([[[0.25]]]),
        "proprio_gripper_contact": np.array([[[1.0]]]),
        "privileged_block_0_pos": np.array([[[4.0, 5.0, 6.0]]]),
    }
    current = {
        adapter.environment_state_keys[key]: value
        for key, value in canonical_current.items()
    }
    goal = {
        f"goal_{key}": value.copy() for key, value in canonical_current.items()
    }

    prepared = policy._prepare_info(
        {
            **current,
            **goal,
            "action": np.zeros((1, 1, 5), dtype=np.float32),
        }
    )

    assert prepared["state"].shape == (1, 3, 9)
    assert prepared["goal_state"].shape == (1, 1, 9)
    expected = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5, 0.25, 1.0, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(prepared["state"][:, 0], expected)
    torch.testing.assert_close(prepared["goal_state"][:, 0], expected)
