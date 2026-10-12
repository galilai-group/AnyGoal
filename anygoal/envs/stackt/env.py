"""StackT environment and physics-based OU expert policy."""

from __future__ import annotations

import numpy as np

from stable_worldmodel.envs.pusht.env import PushT

from ..avoidt.env import (
    PHYSICAL_T_POSITION_BOUNDS,
    EpisodeRandomizationWrapper,
)


GOAL_T_POSE = np.array((256.0, 256.0, np.pi / 4), dtype=np.float64)
T_HEAD_CENTER_LOCAL = np.array((0.0, 15.0), dtype=np.float64)
T_CENTER_OF_GRAVITY_LOCAL = np.array((0.0, 45.0), dtype=np.float64)
DEFAULT_POSITION_TOLERANCE = 20.0
DEFAULT_ANGLE_TOLERANCE = np.pi / 9
V2_ANGLE_TOLERANCE = np.deg2rad(1.0)
V2_INITIAL_BLOCK_ANGLE_BOUNDS = np.deg2rad(
    np.array((-125.0, -115.0), dtype=np.float64)
)


def rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D vector counter-clockwise in the PushT coordinate frame."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=np.float64,
    )


def wrapped_angle_error(target: float, current: float) -> float:
    """Return the signed shortest rotation from ``current`` to ``target``."""
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


def stacked_block_pose(goal_t_pose: np.ndarray = GOAL_T_POSE) -> np.ndarray:
    """Return the physical-T pose whose head exactly overlays the goal-T head.

    The two T stems point in opposite directions. Their shared head bar and two
    opposing stems therefore render as a plus sign.
    """
    goal_t_pose = np.asarray(goal_t_pose, dtype=np.float64)
    if goal_t_pose.shape != (3,):
        raise ValueError(f"goal_t_pose must have shape (3,), got {goal_t_pose.shape}")
    goal_angle = float(goal_t_pose[2])
    block_angle = (goal_angle + np.pi) % (2.0 * np.pi)
    block_position = goal_t_pose[:2] + rotate(
        2.0 * T_HEAD_CENTER_LOCAL, goal_angle
    )
    return np.array((*block_position, block_angle), dtype=np.float64)


class StackT(PushT):
    """PushT physics with the StackT reset distribution and goal rendering."""

    def __init__(
        self,
        position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
        angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
        angle_distance_scale: float = 40.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if position_tolerance <= 0.0 or angle_tolerance <= 0.0:
            raise ValueError("StackT tolerances must be positive.")
        if angle_distance_scale <= 0.0:
            raise ValueError("angle_distance_scale must be positive.")
        self.position_tolerance = float(position_tolerance)
        self.angle_tolerance = float(angle_tolerance)
        self.angle_distance_scale = float(angle_distance_scale)
        self.env_name = "StackT"

    def eval_state(self, goal_state, cur_state):
        """Use the same pusher-and-block success criterion as PushT."""
        return super().eval_state(goal_state, cur_state)

    def block_goal_reached(self, goal_state, cur_state) -> bool:
        """Return whether the physical T reached its target pose."""
        goal_state = np.asarray(goal_state, dtype=np.float64)
        cur_state = np.asarray(cur_state, dtype=np.float64)
        position_error = float(np.linalg.norm(goal_state[2:4] - cur_state[2:4]))
        angle_error = abs(wrapped_angle_error(goal_state[4], cur_state[4]))
        return (
            position_error < self.position_tolerance
            and angle_error < self.angle_tolerance
        )

    def _get_info(self):
        info = super()._get_info()
        state = self._get_obs()
        position_error = float(
            np.linalg.norm(self.goal_state[2:4] - state[2:4])
        )
        angle_error = abs(
            wrapped_angle_error(self.goal_state[4], state[4])
        )
        success, _ = self.eval_state(self.goal_state, state)
        info.update(
            {
                "stack_position_error": np.float64(position_error),
                "stack_angle_error": np.float64(angle_error),
                "stack_success": np.bool_(success),
            }
        )
        return info


class StackTEpisodeRandomizationWrapper(EpisodeRandomizationWrapper):
    """Sample the physical T and ball independently for each episode."""

    def __init__(
        self,
        env,
        goal_t_pose=GOAL_T_POSE,
        clearance: float = 20.0,
    ) -> None:
        self.goal_t_pose = np.asarray(goal_t_pose, dtype=np.float64)
        self.block_target_pose = stacked_block_pose(self.goal_t_pose)
        super().__init__(
            env,
            goal_position=self.goal_t_pose[:2],
            block_position=PHYSICAL_T_POSITION_BOUNDS.mean(axis=1),
            clearance=clearance,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._randomization_rng = np.random.default_rng(seed)
        rng = self._randomization_rng
        block_position = rng.uniform(
            PHYSICAL_T_POSITION_BOUNDS[:, 0],
            PHYSICAL_T_POSITION_BOUNDS[:, 1],
        )
        block_angle = self._sample_block_angle(rng)
        self.block_position = block_position
        (
            agent_position,
            spawn_angle,
            spawn_radius,
            rejected_samples,
        ) = self._sample_agent_position(rng)
        goal_state = np.concatenate(
            (
                agent_position,
                self.block_target_pose,
                [0.0, 0.0],
            )
        )
        state = np.concatenate(
            (agent_position, block_position, [block_angle, 0.0, 0.0])
        )
        randomized_options = dict(options or {})
        randomized_options.update(
            {
                "variation": [],
                "variation_values": {
                    "agent.start_position": agent_position,
                    "block.start_position": block_position,
                    "block.angle": block_angle,
                    "goal.position": self.goal_t_pose[:2],
                    "goal.angle": self.goal_t_pose[2],
                },
                "state": state,
                "goal_state": goal_state,
            }
        )
        observation, info = self.env.reset(seed=seed, options=randomized_options)
        self.episode_randomization = {
            "goal_t_pose": self.goal_t_pose.copy(),
            "stack_block_target_pose": self.block_target_pose.copy(),
            "stack_initial_block_pose": np.array(
                (*block_position, block_angle), dtype=np.float64
            ),
            "ball_spawn_angle": np.float64(spawn_angle),
            "ball_spawn_radius": np.float64(spawn_radius),
            "ball_spawn_inner_radius": np.float64(self.inner_spawn_radius),
            "ball_spawn_outer_radius": np.float64(self.outer_spawn_radius),
            "ball_spawn_rejections": np.int64(rejected_samples),
        }
        info.update(self.episode_randomization)
        return observation, info

    @staticmethod
    def _sample_block_angle(rng: np.random.Generator) -> float:
        return float(rng.uniform(0.0, 2.0 * np.pi))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.episode_randomization)
        return observation, reward, terminated, truncated, info


class StackTV2EpisodeRandomizationWrapper(StackTEpisodeRandomizationWrapper):
    """StackT reset distribution with a supported 10--20 degree rotation."""

    @staticmethod
    def _sample_block_angle(rng: np.random.Generator) -> float:
        return float(rng.uniform(*V2_INITIAL_BLOCK_ANGLE_BOUNDS))


def make_stackt_env(**kwargs):
    """Create the registered StackT benchmark environment."""
    wrapper_keys = {
        "goal_t_pose",
        "clearance",
    }
    wrapper_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in wrapper_keys
    }
    return StackTEpisodeRandomizationWrapper(
        StackT(**kwargs), **wrapper_kwargs
    )


def make_stackt_v2_env(**kwargs):
    """Create StackT with precise rotation and supported initial angles."""
    wrapper_keys = {"goal_t_pose", "clearance"}
    wrapper_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in wrapper_keys
    }
    kwargs.setdefault("angle_tolerance", V2_ANGLE_TOLERANCE)
    return StackTV2EpisodeRandomizationWrapper(
        StackT(**kwargs), **wrapper_kwargs
    )


__all__ = [
    "GOAL_T_POSE",
    "StackT",
    "StackTV2EpisodeRandomizationWrapper",
    "V2_ANGLE_TOLERANCE",
    "V2_INITIAL_BLOCK_ANGLE_BOUNDS",
    "make_stackt_v2_env",
    "stacked_block_pose",
    "wrapped_angle_error",
]
