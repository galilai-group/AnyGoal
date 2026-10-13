import gymnasium as gym
import numpy as np

import anygoal  # noqa: F401


def test_avoidt_success_checks_agent_and_block_state():
    env = gym.make("anygoal/AvoidT-v1")
    try:
        goal_state = np.array([75.0, 75.0, 256.0, 256.0, 0.5, 0.0, 0.0])
        current_state = goal_state.copy()
        current_state[:2] = (400.0, 400.0)

        success, distance = env.unwrapped.eval_state(goal_state, current_state)

        assert not success
        assert distance > 0.0

        success, distance = env.unwrapped.eval_state(goal_state, goal_state)
        assert success
        assert distance == 0.0
    finally:
        env.close()
