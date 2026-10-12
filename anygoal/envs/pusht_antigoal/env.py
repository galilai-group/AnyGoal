"""PushT Antigoal environment and physics-based expert policy."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from stable_worldmodel.envs.pusht.env import PushT

from ..stackt.env import (
    DEFAULT_ANGLE_TOLERANCE,
    DEFAULT_POSITION_TOLERANCE,
    StackT,
    V2_ANGLE_TOLERANCE,
    V2_INITIAL_BLOCK_ANGLE_BOUNDS,
    wrapped_angle_error,
)


VISIBLE_GOAL_T_POSE = np.array(
    (256.0, 256.0, np.pi / 4), dtype=np.float64
)
INVISIBLE_GOAL_ANGLE = 5.0 * np.pi / 4.0
ANTIGOAL_POSITIONS = {
    "top-left": (106.0, 106.0),
    "bottom-left": (106.0, 406.0),
    "top-right": (406.0, 106.0),
    "bottom-right": (406.0, 406.0),
}
ANTIGOAL_NAMES = tuple(ANTIGOAL_POSITIONS)
ANTIGOAL_POSITION_ARRAY = np.asarray(
    tuple(ANTIGOAL_POSITIONS.values()), dtype=np.float64
)
ANTIGOAL_V3_POSITIONS = {
    "top-left": (126.0, 126.0),
    "bottom-left": (126.0, 386.0),
    "top-right": (386.0, 126.0),
    "bottom-right": (386.0, 386.0),
}
ANTIGOAL_V3_POSITION_ARRAY = np.asarray(
    tuple(ANTIGOAL_V3_POSITIONS.values()), dtype=np.float64
)
PHYSICAL_T_RADII = np.array((40.0, 70.0), dtype=np.float64)
PHYSICAL_T_LOCATION_ANGLES = np.deg2rad(
    np.arange(0.0, 360.0, 10.0, dtype=np.float64)
)
# Eight values: 0, pi/16, ..., 7*pi/16. The pi/2 endpoint is excluded.
PHYSICAL_T_ANGLES = np.arange(8, dtype=np.float64) * np.pi / 16.0
AGENT_POSITION_BOUNDS = np.array((50.0, 450.0), dtype=np.float64)


def nearest_antigoal(
    block_position: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[int, str, np.ndarray]:
    """Return the corner closest to an initial physical-T position.

    Location angles on a cardinal axis are exactly tied between two corners.
    Reset randomization breaks only those ties uniformly.
    """
    block_position = np.asarray(block_position, dtype=np.float64)
    if block_position.shape != (2,):
        raise ValueError(
            f"block_position must have shape (2,), got {block_position.shape}"
        )
    distances = np.linalg.norm(
        ANTIGOAL_POSITION_ARRAY - block_position[None], axis=1
    )
    tied_indices = np.flatnonzero(
        np.isclose(distances, distances.min(), rtol=0.0, atol=1e-10)
    )
    if rng is None or len(tied_indices) == 1:
        corner_index = int(tied_indices[0])
    else:
        corner_index = int(rng.choice(tied_indices))
    return (
        corner_index,
        ANTIGOAL_NAMES[corner_index],
        ANTIGOAL_POSITION_ARRAY[corner_index].copy(),
    )


class PushTAntigoal(StackT):
    """PushT physics with an invisible corner target for the physical T."""

    def __init__(
        self,
        position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
        angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
        angle_distance_scale: float = 40.0,
        **kwargs,
    ) -> None:
        super().__init__(
            position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance,
            angle_distance_scale=angle_distance_scale,
            **kwargs,
        )
        self.env_name = "PushTAntigoal"

    def _get_info(self):
        info = PushT._get_info(self)
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
                "antigoal_position_error": np.float64(position_error),
                "antigoal_angle_error": np.float64(angle_error),
                "antigoal_success": np.bool_(success),
            }
        )
        return info


class PushTAntigoalEpisodeRandomizationWrapper(gym.Wrapper):
    """Apply the discrete Antigoal reset distribution to every episode."""

    def __init__(
        self,
        env,
        target_corner: str | None = None,
        antigoal_position_array: np.ndarray = ANTIGOAL_POSITION_ARRAY,
    ) -> None:
        super().__init__(env)
        if target_corner is not None and target_corner not in ANTIGOAL_NAMES:
            raise ValueError(f"Unknown antigoal corner: {target_corner}")
        self.target_corner = target_corner
        self.antigoal_position_array = np.asarray(
            antigoal_position_array, dtype=np.float64
        )
        if self.antigoal_position_array.shape != (len(ANTIGOAL_NAMES), 2):
            raise ValueError("antigoal_position_array must have shape (4, 2)")
        self._randomization_rng = np.random.default_rng()
        self.episode_randomization = {}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._randomization_rng = np.random.default_rng(seed)
        rng = self._randomization_rng

        while True:
            block_radius = float(rng.choice(PHYSICAL_T_RADII))
            block_location_angle = float(
                rng.choice(PHYSICAL_T_LOCATION_ANGLES)
            )
            block_position = VISIBLE_GOAL_T_POSE[:2] + block_radius * np.array(
                (
                    np.cos(block_location_angle),
                    np.sin(block_location_angle),
                ),
                dtype=np.float64,
            )
            distances = np.linalg.norm(
                self.antigoal_position_array - block_position[None], axis=1
            )
            tied_indices = np.flatnonzero(
                np.isclose(distances, distances.min(), rtol=0.0, atol=1e-10)
            )
            corner_index = int(rng.choice(tied_indices))
            corner_name = ANTIGOAL_NAMES[corner_index]
            invisible_goal_position = self.antigoal_position_array[
                corner_index
            ].copy()
            if self.target_corner is None or corner_name == self.target_corner:
                break
        block_angle = self._sample_block_angle(rng)
        invisible_goal_pose = np.array(
            (*invisible_goal_position, INVISIBLE_GOAL_ANGLE),
            dtype=np.float64,
        )
        agent_position = rng.uniform(
            AGENT_POSITION_BOUNDS[0], AGENT_POSITION_BOUNDS[1], size=2
        )

        state = np.concatenate(
            (agent_position, block_position, [block_angle, 0.0, 0.0])
        )
        goal_state = np.concatenate(
            (agent_position, invisible_goal_pose, [0.0, 0.0])
        )
        randomized_options = dict(options or {})
        randomized_options.update(
            {
                "variation": [],
                "variation_values": {
                    "agent.start_position": agent_position,
                    "block.start_position": block_position,
                    "block.angle": block_angle,
                    "goal.position": VISIBLE_GOAL_T_POSE[:2],
                    "goal.angle": VISIBLE_GOAL_T_POSE[2],
                },
                "state": state,
                "goal_state": goal_state,
            }
        )
        observation, info = self.env.reset(
            seed=seed, options=randomized_options
        )
        self.episode_randomization = {
            "visible_goal_t_pose": VISIBLE_GOAL_T_POSE.copy(),
            "antigoal_corner_index": np.int64(corner_index),
            "antigoal_corner": corner_name,
            "invisible_goal_pose": invisible_goal_pose.copy(),
            "antigoal_initial_block_pose": np.array(
                (*block_position, block_angle), dtype=np.float64
            ),
            "antigoal_block_radius": np.float64(block_radius),
            "antigoal_block_location_angle": np.float64(
                block_location_angle
            ),
        }
        info.update(self.episode_randomization)
        return observation, info

    @staticmethod
    def _sample_block_angle(rng: np.random.Generator) -> float:
        return float(rng.choice(PHYSICAL_T_ANGLES))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(self.episode_randomization)
        return observation, reward, terminated, truncated, info


class PushTAntigoalV2EpisodeRandomizationWrapper(
    PushTAntigoalEpisodeRandomizationWrapper
):
    """Sample a supported initial angle requiring 10--20 degrees of rotation."""

    @staticmethod
    def _sample_block_angle(rng: np.random.Generator) -> float:
        return float(rng.uniform(*V2_INITIAL_BLOCK_ANGLE_BOUNDS))


class PushTAntigoalV3EpisodeRandomizationWrapper(
    PushTAntigoalV2EpisodeRandomizationWrapper
):
    """Use supported initial angles and inward, PushT-supported corners."""


def make_push_t_antigoal_env(**kwargs):
    """Create the registered PushT Antigoal benchmark environment."""
    target_corner = kwargs.pop("target_corner", None)
    return PushTAntigoalEpisodeRandomizationWrapper(
        PushTAntigoal(**kwargs), target_corner=target_corner
    )


def make_push_t_antigoal_v2_env(**kwargs):
    """Create PushT Antigoal with precise opposite-angle completion."""
    target_corner = kwargs.pop("target_corner", None)
    kwargs.setdefault("angle_tolerance", V2_ANGLE_TOLERANCE)
    return PushTAntigoalV2EpisodeRandomizationWrapper(
        PushTAntigoal(**kwargs), target_corner=target_corner
    )


def make_push_t_antigoal_v3_env(**kwargs):
    """Create the all-factor-support PushT Antigoal environment."""
    target_corner = kwargs.pop("target_corner", None)
    kwargs.setdefault("angle_tolerance", V2_ANGLE_TOLERANCE)
    return PushTAntigoalV3EpisodeRandomizationWrapper(
        PushTAntigoal(**kwargs),
        target_corner=target_corner,
        antigoal_position_array=ANTIGOAL_V3_POSITION_ARRAY,
    )


__all__ = [
    "AGENT_POSITION_BOUNDS",
    "ANTIGOAL_NAMES",
    "ANTIGOAL_POSITION_ARRAY",
    "ANTIGOAL_POSITIONS",
    "ANTIGOAL_V3_POSITIONS",
    "ANTIGOAL_V3_POSITION_ARRAY",
    "INVISIBLE_GOAL_ANGLE",
    "PHYSICAL_T_ANGLES",
    "PHYSICAL_T_LOCATION_ANGLES",
    "PHYSICAL_T_RADII",
    "PushTAntigoal",
    "PushTAntigoalV3EpisodeRandomizationWrapper",
    "VISIBLE_GOAL_T_POSE",
    "make_push_t_antigoal_v3_env",
    "nearest_antigoal",
]
