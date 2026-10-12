"""OU-process policy that moves the PushT agent around the T block."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

# PushT keeps pymunk.pygame_util's default downward-positive rendered y axis.
WINDOW_SIZE = 512.0
T_PIVOT_RADIUS = float(np.hypot(15.0, 120.0))
# Exact polygon clipping gives 90.0% worst-case containment at 99px.
GOAL_EDGE_MARGIN = 100.0
GOAL_POSITIONS = {
    'top-left': (GOAL_EDGE_MARGIN, GOAL_EDGE_MARGIN),
    'bottom-left': (GOAL_EDGE_MARGIN, WINDOW_SIZE - GOAL_EDGE_MARGIN),
    'top-right': (WINDOW_SIZE - GOAL_EDGE_MARGIN, GOAL_EDGE_MARGIN),
    'bottom-right': (
        WINDOW_SIZE - GOAL_EDGE_MARGIN,
        WINDOW_SIZE - GOAL_EDGE_MARGIN,
    ),
}
BLOCK_POSITION = (256.0, 206.0)
BALL_RADIUS = 15.0
SPAWN_ANNULUS_WIDTH = 100.0
CORNER_GOAL_BOUNDS = {
    'top-left': ((50.0, 100.0), (50.0, 100.0)),
    'top-right': ((400.0, 450.0), (50.0, 100.0)),
    'bottom-left': ((50.0, 100.0), (400.0, 450.0)),
    'bottom-right': ((400.0, 450.0), (400.0, 450.0)),
}
FIXED_GOAL_T_POSE = np.array((256.0, 256.0, np.pi / 4))
PHYSICAL_T_POSITION_BOUNDS = np.array(((100.0, 400.0), (100.0, 400.0)))


def resolve_goal(goal: str | Sequence[float]) -> np.ndarray:
    """Return a validated ``(x, y)`` PushT goal position."""
    if isinstance(goal, str):
        if goal not in GOAL_POSITIONS:
            choices = ', '.join(GOAL_POSITIONS)
            raise ValueError(f'Unknown goal {goal!r}. Choose one of: {choices}')
        goal = GOAL_POSITIONS[goal]

    goal_array = np.asarray(goal, dtype=np.float64)
    if goal_array.shape != (2,):
        raise ValueError(f'goal must have shape (2,), got {goal_array.shape}')
    return goal_array


class EpisodeRandomizationWrapper(gym.Wrapper):
    """Sample independent T rotations and a ball start in an annulus."""

    def __init__(self, env, goal_position, block_position, clearance):
        super().__init__(env)
        self.goal_position = np.asarray(goal_position, dtype=np.float64)
        self.block_position = np.asarray(block_position, dtype=np.float64)
        self.inner_spawn_radius = (
            T_PIVOT_RADIUS + BALL_RADIUS + float(clearance)
        )
        self.outer_spawn_radius = self.inner_spawn_radius + SPAWN_ANNULUS_WIDTH
        self.episode_randomization = {}
        self._randomization_rng = np.random.default_rng()

        if np.any(self.block_position < 50.0) or np.any(
            self.block_position > 450.0
        ):
            raise ValueError(
                'block-position must stay inside the agent range [50, 450].'
            )

    def _sample_agent_position(self, rng):
        for rejected_samples in range(10_000):
            spawn_angle = float(rng.uniform(0.0, 2.0 * np.pi))
            spawn_radius = float(
                np.sqrt(
                    rng.uniform(
                        self.inner_spawn_radius**2,
                        self.outer_spawn_radius**2,
                    )
                )
            )
            agent_position = self.block_position + spawn_radius * np.array(
                [np.cos(spawn_angle), np.sin(spawn_angle)]
            )
            if np.all(agent_position >= 50.0) and np.all(
                agent_position <= 450.0
            ):
                return (
                    agent_position,
                    spawn_angle,
                    spawn_radius,
                    rejected_samples,
                )
        raise RuntimeError(
            'Could not sample a ball position from the valid annulus region.'
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._randomization_rng = np.random.default_rng(seed)
        rng = self._randomization_rng
        physical_t_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        goal_t_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        (
            agent_position,
            spawn_angle,
            spawn_radius,
            rejected_samples,
        ) = self._sample_agent_position(rng)

        randomized_options = dict(options or {})
        randomized_options.update(
            {
                'variation': [],
                'variation_values': {
                    'agent.start_position': agent_position,
                    'block.start_position': self.block_position,
                    'block.angle': physical_t_angle,
                    'goal.position': self.goal_position,
                    'goal.angle': goal_t_angle,
                },
                'state': np.concatenate(
                    (
                        agent_position,
                        self.block_position,
                        [physical_t_angle, 0.0, 0.0],
                    )
                ),
                'goal_state': np.concatenate(
                    (
                        self.goal_position,
                        self.block_position,
                        [physical_t_angle, 0.0, 0.0],
                    )
                ),
            }
        )
        observation, info = self.env.reset(
            seed=seed, options=randomized_options
        )
        self.episode_randomization = {
            'physical_t_angle': np.float64(physical_t_angle),
            'goal_t_angle': np.float64(goal_t_angle),
            'ball_spawn_angle': np.float64(spawn_angle),
            'ball_spawn_radius': np.float64(spawn_radius),
            'ball_spawn_inner_radius': np.float64(self.inner_spawn_radius),
            'ball_spawn_outer_radius': np.float64(self.outer_spawn_radius),
            'ball_spawn_rejections': np.int64(rejected_samples),
        }
        info.update(self.episode_randomization)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.episode_randomization)
        return observation, reward, terminated, truncated, info


class AvoidTEpisodeRandomizationWrapper(EpisodeRandomizationWrapper):
    """Sample a ball target in one corner while keeping the goal T fixed."""

    def __init__(self, env, corner, clearance):
        if corner not in CORNER_GOAL_BOUNDS:
            raise ValueError(f'Unknown corner: {corner}')
        self.corner = corner
        self.goal_bounds = np.asarray(CORNER_GOAL_BOUNDS[corner])
        super().__init__(
            env,
            goal_position=self.goal_bounds.mean(axis=1),
            block_position=PHYSICAL_T_POSITION_BOUNDS.mean(axis=1),
            clearance=clearance,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._randomization_rng = np.random.default_rng(seed)
        rng = self._randomization_rng
        physical_t_position = rng.uniform(
            PHYSICAL_T_POSITION_BOUNDS[:, 0],
            PHYSICAL_T_POSITION_BOUNDS[:, 1],
        )
        physical_t_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        ball_goal_position = rng.uniform(
            self.goal_bounds[:, 0], self.goal_bounds[:, 1]
        )
        self.block_position = physical_t_position
        (
            agent_position,
            spawn_angle,
            spawn_radius,
            rejected_samples,
        ) = self._sample_agent_position(rng)

        randomized_options = dict(options or {})
        randomized_options.update(
            {
                'variation': [],
                'variation_values': {
                    'agent.start_position': agent_position,
                    'block.start_position': self.block_position,
                    'block.angle': physical_t_angle,
                    'goal.position': FIXED_GOAL_T_POSE[:2],
                    'goal.angle': FIXED_GOAL_T_POSE[2],
                },
                'state': np.concatenate(
                    (
                        agent_position,
                        physical_t_position,
                        [physical_t_angle, 0.0, 0.0],
                    )
                ),
                'goal_state': np.concatenate(
                    (
                        ball_goal_position,
                        physical_t_position,
                        [physical_t_angle, 0.0, 0.0],
                    )
                ),
            }
        )
        observation, info = self.env.reset(
            seed=seed, options=randomized_options
        )
        self.episode_randomization = {
            'physical_t_position': physical_t_position.copy(),
            'physical_t_angle': np.float64(physical_t_angle),
            'goal_t_pose': FIXED_GOAL_T_POSE.copy(),
            'ball_goal_position': ball_goal_position.copy(),
            'ball_spawn_angle': np.float64(spawn_angle),
            'ball_spawn_radius': np.float64(spawn_radius),
            'ball_spawn_inner_radius': np.float64(self.inner_spawn_radius),
            'ball_spawn_outer_radius': np.float64(self.outer_spawn_radius),
            'ball_spawn_rejections': np.int64(rejected_samples),
        }
        info.update(self.episode_randomization)
        return observation, info


def make_avoidt_env(
    corner: str = 'top-left', clearance: float = 20.0, **kwargs
):
    """Create a PushT environment with the sampled-goal AvoidT reset."""
    from stable_worldmodel.envs.pusht.env import PushT

    return AvoidTEpisodeRandomizationWrapper(
        PushT(**kwargs), corner=corner, clearance=clearance
    )


__all__ = [
    'AvoidTEpisodeRandomizationWrapper',
    'BLOCK_POSITION',
    'CORNER_GOAL_BOUNDS',
    'EpisodeRandomizationWrapper',
    'FIXED_GOAL_T_POSE',
    'GOAL_EDGE_MARGIN',
    'GOAL_POSITIONS',
    'PHYSICAL_T_POSITION_BOUNDS',
    'T_PIVOT_RADIUS',
    'make_avoidt_env',
    'resolve_goal',
]
