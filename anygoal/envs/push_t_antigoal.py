"""PushT Antigoal environment and physics-based expert policy."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from stable_worldmodel.envs.pusht.env import PushT

from .stack_t import (
    DEFAULT_ANGLE_TOLERANCE,
    DEFAULT_POSITION_TOLERANCE,
    StackT,
    StackTPolicy,
    V2_ANGLE_TOLERANCE,
    V2_INITIAL_BLOCK_ANGLE_BOUNDS,
    rotate,
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


class PushTAntigoalPolicy(StackTPolicy):
    """Physics-based OU expert that rotates and pushes T to its antigoal.

    Coarse rotation pushes the lower side of the stem, where the long lever
    arm produces much more torque than translation. Fine rotation pushes up
    on the underside of the head before the inherited center-of-gravity
    controller translates the rotated T to its goal.
    """

    def __init__(
        self,
        speed: float = 0.65,
        theta: float = 0.7,
        sigma: float = 0.015,
        push_through_distance: float = 25.0,
        rotation_speed: float = 0.6,
        fine_rotation_threshold: float = 0.35,
        rotation_completion_fraction: float = 0.65,
        top_right_staging_margin: float = 50.0,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            speed=speed,
            theta=theta,
            sigma=sigma,
            push_through_distance=push_through_distance,
            seed=seed,
            **kwargs,
        )
        if not 0.0 < rotation_speed <= 1.0:
            raise ValueError("rotation_speed must be in (0, 1].")
        if fine_rotation_threshold <= 0.0:
            raise ValueError("fine_rotation_threshold must be positive.")
        if not 0.0 < rotation_completion_fraction < 1.0:
            raise ValueError(
                "rotation_completion_fraction must be in (0, 1)."
            )
        if top_right_staging_margin < 0.0:
            raise ValueError("top_right_staging_margin must be non-negative.")
        self.rotation_speed = float(rotation_speed)
        self.fine_rotation_threshold = float(fine_rotation_threshold)
        self.rotation_completion_fraction = float(
            rotation_completion_fraction
        )
        self.top_right_staging_margin = float(top_right_staging_margin)
        self.top_right_final_stage = np.empty(0, dtype=bool)

    def set_env(self, env) -> None:
        super().set_env(env)
        for base_env in self._unwrapped_envs():
            if not isinstance(base_env, PushTAntigoal):
                raise ValueError(
                    "PushTAntigoalPolicy requires a PushTAntigoal environment."
                )
        self.top_right_final_stage = np.zeros(
            len(self._unwrapped_envs()), dtype=bool
        )
    def _rotation_action(
        self, base_env, turn_error: float
    ) -> np.ndarray:
        current_angle = float(base_env.block.angle)
        block_position = np.asarray(base_env.block.position, dtype=np.float64)
        contact_local, push_direction_local = self._rotation_contact_local(
            turn_error
        )
        push_direction = rotate(push_direction_local, current_angle)
        contact_point = block_position + rotate(contact_local, current_angle)
        rear_waypoint = contact_point - push_direction * 38.0
        agent_position = np.asarray(base_env.agent.position, dtype=np.float64)
        relative_agent = agent_position - contact_point
        longitudinal = float(np.dot(relative_agent, push_direction))
        lateral_direction = np.array(
            (-push_direction[1], push_direction[0]), dtype=np.float64
        )
        lateral_error = float(np.dot(relative_agent, lateral_direction))
        aligned = -55.0 < longitudinal < -8.0 and abs(lateral_error) < 18.0

        if aligned:
            waypoint = contact_point + push_direction * 42.0
        else:
            block_cog = np.asarray(
                base_env.block.local_to_world(base_env.block.center_of_gravity),
                dtype=np.float64,
            )
            waypoint = self._approach_waypoint(
                agent_position,
                block_cog,
                rear_waypoint,
                base_env.window_size,
            )

        displacement = waypoint - agent_position
        distance = float(np.linalg.norm(displacement))
        if distance < 1e-8:
            return np.zeros(2, dtype=np.float32)
        magnitude = min(
            self.rotation_speed, distance / float(base_env.action_scale)
        )
        action = displacement / distance * magnitude
        action += self.sigma * self.rng.normal(size=2)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _rotation_contact_local(
        self, turn_error: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return local contact and force directions with the desired torque."""
        turn_sign = float(np.sign(turn_error))
        if abs(turn_error) < self.fine_rotation_threshold:
            contact_local = np.array(
                (-turn_sign * 45.0, 30.0), dtype=np.float64
            )
            push_direction_local = np.array(
                (0.0, -1.0), dtype=np.float64
            )
        else:
            contact_local = np.array(
                (turn_sign * 15.0, 108.0), dtype=np.float64
            )
            push_direction_local = np.array(
                (-turn_sign, 0.0), dtype=np.float64
            )
        return contact_local, push_direction_local

    def get_action(self, info_dict, **kwargs) -> np.ndarray:
        envs = self._unwrapped_envs()
        if self.top_right_final_stage.shape != (len(envs),):
            self.top_right_final_stage = np.zeros(len(envs), dtype=bool)
        step_indices = self._step_indices(info_dict, len(envs))
        saved_goal_states = [base_env.goal_state for base_env in envs]
        controller_goal_states = list(saved_goal_states)
        rotating = np.zeros(len(envs), dtype=bool)
        turn_errors = np.zeros(len(envs), dtype=np.float64)

        for env_index, (base_env, final_goal_state) in enumerate(
            zip(envs, saved_goal_states)
        ):
            if step_indices is not None and step_indices[env_index] == 0:
                self.top_right_final_stage[env_index] = False
            current_angle = float(base_env.block.angle)
            turn_error = wrapped_angle_error(final_goal_state[4], current_angle)
            turn_errors[env_index] = turn_error
            rotation_stage_threshold = (
                base_env.angle_tolerance
                * self.rotation_completion_fraction
                if self.rotation_stage_threshold is None
                else self.rotation_stage_threshold
            )
            rotating[env_index] = (
                abs(turn_error) >= rotation_stage_threshold
            )
            if rotating[env_index]:
                staged_goal_state = np.asarray(
                    final_goal_state, dtype=np.float64
                ).copy()
                staged_goal_state[2:4] = np.asarray(
                    base_env.block.position, dtype=np.float64
                )
                staged_goal_state[4] = current_angle
                base_env.goal_state = staged_goal_state
                controller_goal_states[env_index] = staged_goal_state
            elif (
                self.top_right_staging_margin
                and np.allclose(
                    final_goal_state[2:4], ANTIGOAL_POSITIONS["top-right"]
                )
                and not self.top_right_final_stage[env_index]
            ):
                staged_goal_state = np.asarray(
                    final_goal_state, dtype=np.float64
                ).copy()
                staged_goal_state[2:4] += np.array(
                    (-self.top_right_staging_margin, self.top_right_staging_margin),
                    dtype=np.float64,
                )
                distance_to_stage = float(
                    np.linalg.norm(
                        staged_goal_state[2:4]
                        - np.asarray(base_env.block.position, dtype=np.float64)
                    )
                )
                if distance_to_stage < base_env.position_tolerance * 0.7:
                    self.top_right_final_stage[env_index] = True
                else:
                    base_env.goal_state = staged_goal_state
                    controller_goal_states[env_index] = staged_goal_state

        try:
            actions = super().get_action(info_dict, **kwargs)
        finally:
            for base_env, final_goal_state in zip(envs, saved_goal_states):
                base_env.goal_state = final_goal_state

        for env_index, base_env in enumerate(envs):
            if rotating[env_index]:
                actions[env_index] = self._rotation_action(
                    base_env,
                    turn_errors[env_index],
                )
                continue

            target_pose = np.asarray(
                controller_goal_states[env_index][2:5], dtype=np.float64
            )
            block_position = np.asarray(
                base_env.block.position, dtype=np.float64
            )
            target_displacement = target_pose[:2] - block_position
            position_error = float(np.linalg.norm(target_displacement))
            if not 1e-8 < position_error < 60.0:
                continue
            push_direction = target_displacement / position_error
            lateral_direction = np.array(
                (-push_direction[1], push_direction[0]), dtype=np.float64
            )
            angle_error = wrapped_angle_error(
                target_pose[2], float(base_env.block.angle)
            )
            lateral_offset = float(
                np.clip(
                    -self.torque_gain * angle_error,
                    -self.max_lateral_offset,
                    self.max_lateral_offset,
                )
            )
            block_cog = np.asarray(
                base_env.block.local_to_world(base_env.block.center_of_gravity),
                dtype=np.float64,
            )
            relative_agent = (
                np.asarray(base_env.agent.position, dtype=np.float64)
                - block_cog
            )
            longitudinal = float(np.dot(relative_agent, push_direction))
            lateral_error = float(
                np.dot(relative_agent, lateral_direction) - lateral_offset
            )
            pushing = (
                -70.0 < longitudinal < self.push_through_distance
                and abs(lateral_error) < self.alignment_tolerance
            )
            if pushing:
                push_through = np.clip(
                    position_error - base_env.position_tolerance * 0.8,
                    5.0,
                    self.push_through_distance,
                )
                waypoint = (
                    block_cog
                    + push_direction * push_through
                    + lateral_direction * lateral_offset
                )
                actions[env_index] = self._mean_action(
                    np.asarray(base_env.agent.position, dtype=np.float64),
                    waypoint,
                    base_env.action_scale,
                )
        return actions


class PushTAntigoalV2Policy(PushTAntigoalPolicy):
    """Faster, lower-noise rotation with precise orientation locking."""

    def __init__(self, speed: float = 0.75, **kwargs) -> None:
        kwargs.setdefault("theta", 0.85)
        kwargs.setdefault("sigma", 0.008)
        kwargs.setdefault("rotation_speed", 0.9)
        kwargs.setdefault("fine_rotation_threshold", 0.2)
        kwargs.setdefault("rotation_completion_fraction", 0.9)
        kwargs.setdefault("rotation_stage_threshold", np.deg2rad(13.0))
        super().__init__(speed=speed, **kwargs)


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


class PushTAntigoalV3Policy(PushTAntigoalPolicy):
    """Retain V1 action statistics with precise V2 angle completion."""

    def __init__(
        self,
        speed: float = 0.65,
        angle_control_offset: float = np.deg2rad(0.75),
        **kwargs,
    ) -> None:
        if not 0.0 <= angle_control_offset < V2_ANGLE_TOLERANCE:
            raise ValueError(
                "angle_control_offset must be within the V3 angle tolerance"
            )
        kwargs.setdefault("rotation_stage_threshold", np.deg2rad(3.0))
        kwargs.setdefault("rotation_speed", 0.45)
        kwargs.setdefault("sigma", 0.01)
        super().__init__(speed=speed, **kwargs)
        self.angle_control_offset = float(angle_control_offset)

    def get_action(self, info_dict, **kwargs) -> np.ndarray:
        envs = self._unwrapped_envs()
        saved_goal_states = [base_env.goal_state for base_env in envs]
        try:
            for base_env, goal_state in zip(envs, saved_goal_states):
                controller_goal = np.asarray(goal_state).copy()
                controller_goal[4] += self.angle_control_offset
                base_env.goal_state = controller_goal
            return super().get_action(info_dict, **kwargs)
        finally:
            for base_env, goal_state in zip(envs, saved_goal_states):
                base_env.goal_state = goal_state


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
    "PushTAntigoalV3Policy",
    "PushTAntigoalV3EpisodeRandomizationWrapper",
    "VISIBLE_GOAL_T_POSE",
    "make_push_t_antigoal_v3_env",
    "nearest_antigoal",
]
