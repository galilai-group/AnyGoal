import gymnasium as gym
import numpy as np

from anygoal.envs.pusht_antigoal.env import (
    AGENT_POSITION_BOUNDS,
    ANTIGOAL_POSITION_ARRAY,
    ANTIGOAL_V3_POSITION_ARRAY,
    INVISIBLE_GOAL_ANGLE,
    PHYSICAL_T_ANGLES,
    PHYSICAL_T_LOCATION_ANGLES,
    PHYSICAL_T_RADII,
    VISIBLE_GOAL_T_POSE,
    PushTAntigoal,
    PushTAntigoalEpisodeRandomizationWrapper,
)
from anygoal.envs.pusht_antigoal.policy import (
    PushTAntigoalPolicy,
    PushTAntigoalV3Policy,
)
from anygoal.envs.stackt.env import (
    T_CENTER_OF_GRAVITY_LOCAL,
    V2_ANGLE_TOLERANCE,
)


class ResetRecorder(gym.Env):
    def reset(self, *, seed=None, options=None):
        self.options = options
        return options["state"], {}


def test_antigoal_reset_uses_requested_discrete_distributions():
    base_env = ResetRecorder()
    wrapper = PushTAntigoalEpisodeRandomizationWrapper(base_env)

    _, info = wrapper.reset(seed=9)

    initial_pose = info["antigoal_initial_block_pose"]
    displacement = initial_pose[:2] - VISIBLE_GOAL_T_POSE[:2]
    sampled_location_angle = np.arctan2(displacement[1], displacement[0])
    sampled_location_angle %= 2.0 * np.pi
    assert np.any(np.isclose(np.linalg.norm(displacement), PHYSICAL_T_RADII))
    assert np.any(
        np.isclose(sampled_location_angle, PHYSICAL_T_LOCATION_ANGLES)
    )
    assert np.any(np.isclose(initial_pose[2], PHYSICAL_T_ANGLES))
    assert len(PHYSICAL_T_RADII) == 2
    assert len(PHYSICAL_T_LOCATION_ANGLES) == 36
    assert len(PHYSICAL_T_ANGLES) == 8


def test_antigoal_reset_selects_nearest_corner_and_keeps_goal_t_visible():
    base_env = ResetRecorder()
    wrapper = PushTAntigoalEpisodeRandomizationWrapper(base_env)

    _, info = wrapper.reset(seed=4)

    initial_position = info["antigoal_initial_block_pose"][:2]
    target_pose = info["invisible_goal_pose"]
    distances = np.linalg.norm(
        ANTIGOAL_POSITION_ARRAY - initial_position[None], axis=1
    )
    assert info["antigoal_corner_index"] in np.flatnonzero(
        np.isclose(distances, distances.min())
    )
    np.testing.assert_allclose(
        target_pose[:2],
        ANTIGOAL_POSITION_ARRAY[info["antigoal_corner_index"]],
    )
    assert np.isclose(target_pose[2], INVISIBLE_GOAL_ANGLE)
    np.testing.assert_allclose(
        base_env.options["variation_values"]["goal.position"],
        VISIBLE_GOAL_T_POSE[:2],
    )
    assert np.isclose(
        base_env.options["variation_values"]["goal.angle"],
        VISIBLE_GOAL_T_POSE[2],
    )


def test_antigoal_agent_matches_original_pusht_uniform_bounds():
    base_env = ResetRecorder()
    wrapper = PushTAntigoalEpisodeRandomizationWrapper(base_env)

    samples = np.asarray(
        [wrapper.reset(seed=seed)[0][:2] for seed in range(100)]
    )

    assert np.all(samples >= AGENT_POSITION_BOUNDS[0])
    assert np.all(samples <= AGENT_POSITION_BOUNDS[1])
    assert np.ptp(samples[:, 0]) > 300.0
    assert np.ptp(samples[:, 1]) > 300.0


def test_antigoal_can_target_one_corner_for_balanced_collection():
    wrapper = PushTAntigoalEpisodeRandomizationWrapper(
        ResetRecorder(), target_corner="top-right"
    )

    for seed in range(20):
        _, info = wrapper.reset(seed=seed)
        assert info["antigoal_corner"] == "top-right"


def test_antigoal_success_checks_agent_final_position():
    env = PushTAntigoal()
    goal_state = np.array(
        [0.0, 0.0, 106.0, 106.0, INVISIBLE_GOAL_ANGLE, 0.0, 0.0]
    )
    current_state = goal_state.copy()
    current_state[:2] = (500.0, 500.0)

    success, distance = env.eval_state(goal_state, current_state)

    assert not success
    assert distance > 0.0

    success, distance = env.eval_state(goal_state, goal_state)
    assert success
    assert distance == 0.0


def test_registered_antigoal_v3_uses_supported_corners_and_v1_actions():
    env = gym.make("anygoal/PushTAntigoal-v3", target_corner="top-right")
    try:
        _, info = env.reset(seed=3)
        assert np.isclose(env.unwrapped.angle_tolerance, V2_ANGLE_TOLERANCE)
        np.testing.assert_allclose(
            info["invisible_goal_pose"][:2],
            ANTIGOAL_V3_POSITION_ARRAY[2],
        )
    finally:
        env.close()

    policy = PushTAntigoalV3Policy()
    assert policy.speed == 0.65
    assert policy.theta == 0.7
    assert policy.rotation_speed == 0.45
    assert policy.sigma == 0.01


def test_antigoal_rotation_contacts_apply_requested_torque_sign():
    policy = PushTAntigoalPolicy()

    for turn_error in (1.0, -1.0, 0.3, -0.3):
        contact, push_direction = policy._rotation_contact_local(turn_error)
        lever_arm = contact - T_CENTER_OF_GRAVITY_LOCAL
        torque = (
            lever_arm[0] * push_direction[1]
            - lever_arm[1] * push_direction[0]
        )

        assert np.sign(torque) == np.sign(turn_error)
