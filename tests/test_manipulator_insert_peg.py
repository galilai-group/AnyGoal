from __future__ import annotations

import numpy as np

from anygoal.envs.manipulator_insert_peg import (
    ManipulatorInsertPegEnv,
    manipulator_goal_errors,
    manipulator_goal_reached,
    planar_angle,
    wrapped_angle_error,
)


def test_planar_angle_and_wrapping_are_quaternion_sign_invariant():
    angle = 2.7
    quaternion = np.array([np.cos(angle / 2), np.sin(angle / 2)])
    assert np.isclose(wrapped_angle_error(planar_angle(quaternion), angle), 0.0)
    assert np.isclose(wrapped_angle_error(planar_angle(-quaternion), angle), 0.0)


def test_goal_criterion_checks_hand_and_object_pose():
    goal = np.zeros(44, dtype=np.float32)
    goal[[31, 35, 42]] = 1.0
    current = goal.copy()
    current[29] = 0.03
    current[34] = 0.03
    tolerances = {
        "hand_position": 0.04,
        "hand_angle": 0.10,
        "object_position": 0.04,
        "object_angle": 0.10,
    }
    assert manipulator_goal_reached(
        manipulator_goal_errors(goal, current), tolerances
    )
    current[33] = 0.05
    assert not manipulator_goal_reached(
        manipulator_goal_errors(goal, current), tolerances
    )


def test_environment_reset_restore_step_and_render():
    env = ManipulatorInsertPegEnv(render_mode="rgb_array")
    try:
        state, info = env.reset(seed=7)
        assert state.shape == (44,)
        assert info["proprio"].shape == (33,)
        assert info["privilege"].shape == (44,)
        assert env.action_space.shape == (5,)
        assert env.render().shape == (224, 224, 3)

        restored_state, restored_info = env.reset(
            seed=11, options={"state": state, "goal_state": state}
        )
        render_relevant = np.r_[0:24, 29:44]
        restoration_error = np.max(
            np.abs(restored_state[render_relevant] - state[render_relevant])
        )
        assert restoration_error < 2e-5
        assert restored_info["evaluation_success"]

        action = np.zeros(5, dtype=np.float32)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        assert next_state.shape == (44,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["qpos"].shape == (11,)
        assert step_info["qvel"].shape == (11,)

        env.reset(seed=19, options={"state": state})
        replayed_state, *_ = env.step(action)
        np.testing.assert_array_equal(replayed_state, next_state)
    finally:
        env.close()


def test_dataset_rows_map_to_reset_options():
    env = ManipulatorInsertPegEnv()
    try:
        state, _ = env.reset(seed=3)
        options = env.reset_options_from_dataset(
            {"observation": state}, {"goal_observation": state}
        )
        assert np.array_equal(options["state"], state)
        assert np.array_equal(options["goal_state"], state)
    finally:
        env.close()
