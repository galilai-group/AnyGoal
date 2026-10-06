import gymnasium as gym
import numpy as np

from anygoal.envs.stack_t import (
    GOAL_T_POSE,
    T_HEAD_CENTER_LOCAL,
    StackT,
    StackTEpisodeRandomizationWrapper,
    StackTPolicy,
    StackTV2EpisodeRandomizationWrapper,
    StackTV2Policy,
    V2_ANGLE_TOLERANCE,
    V2_INITIAL_BLOCK_ANGLE_BOUNDS,
    rotate,
    stacked_block_pose,
)


class ResetRecorder(gym.Env):
    def reset(self, *, seed=None, options=None):
        self.options = options
        return options["state"], {}


def test_stacked_pose_aligns_heads_and_opposes_stems():
    block_pose = stacked_block_pose(GOAL_T_POSE)
    goal_head = GOAL_T_POSE[:2] + rotate(
        T_HEAD_CENTER_LOCAL, GOAL_T_POSE[2]
    )
    block_head = block_pose[:2] + rotate(
        T_HEAD_CENTER_LOCAL, block_pose[2]
    )

    np.testing.assert_allclose(block_head, goal_head)
    assert np.isclose(
        (block_pose[2] - GOAL_T_POSE[2]) % (2.0 * np.pi), np.pi
    )


def test_stackt_success_checks_agent_final_position():
    env = StackT()
    block_pose = stacked_block_pose()
    goal_state = np.array([0.0, 0.0, *block_pose, 0.0, 0.0])
    current_state = goal_state.copy()
    current_state[:2] = (500.0, 500.0)

    success, distance = env.eval_state(goal_state, current_state)

    assert not success
    assert distance > 0.0

    success, distance = env.eval_state(goal_state, goal_state)
    assert success
    assert distance == 0.0


def test_stackt_collection_completion_uses_terminal_block_pose():
    env = StackT()
    block_pose = stacked_block_pose()
    goal_state = np.array([0.0, 0.0, *block_pose, 0.0, 0.0])
    current_state = goal_state.copy()
    current_state[:2] = (500.0, 500.0)

    assert env.block_goal_reached(goal_state, current_state)
    assert not env.eval_state(goal_state, current_state)[0]


def test_stackt_reset_samples_full_block_pose_and_avoidt_ball_annulus():
    base_env = ResetRecorder()
    wrapper = StackTEpisodeRandomizationWrapper(base_env, clearance=20.0)

    _, info = wrapper.reset(seed=7)

    block_pose = info["stack_initial_block_pose"]
    ball_position = base_env.options["state"][:2]
    ball_displacement = ball_position - block_pose[:2]
    assert np.all((100.0 <= block_pose[:2]) & (block_pose[:2] <= 400.0))
    assert 0.0 <= block_pose[2] < 2.0 * np.pi
    assert np.all((50.0 <= ball_position) & (ball_position <= 450.0))
    assert np.isclose(np.linalg.norm(ball_displacement), info["ball_spawn_radius"])
    assert info["ball_spawn_inner_radius"] <= info["ball_spawn_radius"]
    assert info["ball_spawn_radius"] <= info["ball_spawn_outer_radius"]


def test_stackt_v2_uses_precise_tolerance_and_supported_initial_angles():
    env = gym.make("anygoal/StackT-v2")
    try:
        assert np.isclose(env.unwrapped.angle_tolerance, V2_ANGLE_TOLERANCE)
    finally:
        env.close()

    wrapper = StackTV2EpisodeRandomizationWrapper(ResetRecorder())
    angles = np.asarray(
        [wrapper.reset(seed=seed)[1]["stack_initial_block_pose"][2] for seed in range(100)]
    )
    wrapped = (angles + np.pi) % (2.0 * np.pi) - np.pi
    assert np.all(wrapped >= V2_INITIAL_BLOCK_ANGLE_BOUNDS[0])
    assert np.all(wrapped <= V2_INITIAL_BLOCK_ANGLE_BOUNDS[1])
    assert StackTV2Policy().sigma == 0.01


def test_stackt_approach_enters_aligned_rear_waypoint():
    block_cog = np.array((256.0, 256.0))
    rear_waypoint = np.array((326.0, 256.0))
    agent_position = np.array((356.0, 256.0))

    waypoint = StackTPolicy._approach_waypoint(
        agent_position, block_cog, rear_waypoint, 512.0
    )

    np.testing.assert_allclose(waypoint, rear_waypoint)


def test_stackt_approach_moves_tangentially_when_wall_clips_orbit():
    block_cog = np.array((390.0, 256.0))
    agent_position = np.array((492.0, 256.0))
    rear_waypoint = np.array((390.0, 366.0))

    waypoint = StackTPolicy._approach_waypoint(
        agent_position, block_cog, rear_waypoint, 512.0
    )

    assert np.linalg.norm(waypoint - agent_position) > 20.0
    assert np.all((20.0 <= waypoint) & (waypoint <= 492.0))
