"""Policy for the Cube-Hop environment."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from stable_worldmodel.policy import BasePolicy

from .env import (
    MAX_TRANSLATION_ACTION_NORM,
    MAX_YAW_ACTION,
    POSITION_TOLERANCE,
    RIGHT_YAW_DELTA,
    SIDE_SAMPLING_BOUNDS,
    YAW_TOLERANCE,
    CubeHopEnv,
    wrapped_angle_error,
)


class ActionSupport(Protocol):
    """Project proposed actions into calibrated Cube training support."""

    def constrain(self, action: np.ndarray) -> np.ndarray: ...


class CubeHopPolicy(BasePolicy):
    """Move between fresh floor targets and elevated waypoints forever."""

    def __init__(
        self,
        action_support: ActionSupport,
        position_tolerance: float = POSITION_TOLERANCE,
        yaw_tolerance: float = YAW_TOLERANCE,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if position_tolerance <= 0.0 or yaw_tolerance <= 0.0:
            raise ValueError("CubeHop tolerances must be positive.")
        self.action_support = action_support
        self.position_tolerance = float(position_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.seed = seed
        self.discrete = False
        self.destinations = np.empty((0, 3), dtype=np.float64)
        self.destination_is_waypoint = np.empty(0, dtype=bool)
        self.rotation_yaws = np.empty(0, dtype=np.float64)
        self.phases: list[str] = []
        self.sampled_gripper_targets: list[list[np.ndarray]] = []
        self.sampled_waypoints: list[list[np.ndarray]] = []
        self.completed_gripper_targets = np.empty(0, dtype=np.int64)
        self.completed_waypoints = np.empty(0, dtype=np.int64)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed

    def set_env(self, env) -> None:
        self.env = env
        base_envs = self._unwrapped_envs()
        if not all(isinstance(base_env, CubeHopEnv) for base_env in base_envs):
            raise ValueError("CubeHopPolicy requires CubeHopEnv environments.")
        count = len(base_envs)
        self.destinations = np.zeros((count, 3), dtype=np.float64)
        self.destination_is_waypoint = np.zeros(count, dtype=bool)
        self.rotation_yaws = np.zeros(count, dtype=np.float64)
        self.phases = ["reset"] * count
        self.sampled_gripper_targets = [[] for _ in range(count)]
        self.sampled_waypoints = [[] for _ in range(count)]
        self.completed_gripper_targets = np.zeros(count, dtype=np.int64)
        self.completed_waypoints = np.zeros(count, dtype=np.int64)

    def get_action(self, info_dict, **kwargs) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Environment not set for CubeHopPolicy.")
        base_envs = self._unwrapped_envs()
        step_indices = np.asarray(info_dict["step_idx"])[:, 0]
        actions = np.zeros((len(base_envs), 5), dtype=np.float32)

        for env_idx, base_env in enumerate(base_envs):
            if step_indices[env_idx] == 0:
                self._begin_episode(env_idx)
                self._sample_destination(env_idx, base_env, waypoint=False)

            effector_position = np.asarray(
                info_dict["proprio/effector_pos"][env_idx, 0],
                dtype=np.float64,
            )
            effector_yaw = float(info_dict["proprio/effector_yaw"][env_idx, 0, 0])
            gripper_opening = float(info_dict["proprio/gripper_opening"][env_idx, 0, 0])

            if self.phases[env_idx] == "move":
                displacement = self.destinations[env_idx] - effector_position
                if np.linalg.norm(displacement) <= self.position_tolerance:
                    if self.destination_is_waypoint[env_idx]:
                        self.completed_waypoints[env_idx] += 1
                    else:
                        self.completed_gripper_targets[env_idx] += 1
                    self.rotation_yaws[env_idx] = effector_yaw + RIGHT_YAW_DELTA
                    self.phases[env_idx] = "rotate"
                else:
                    actions[env_idx] = self._translation_action(
                        displacement, gripper_opening
                    )
                    continue

            yaw_error = wrapped_angle_error(self.rotation_yaws[env_idx], effector_yaw)
            if abs(yaw_error) <= self.yaw_tolerance:
                self._sample_destination(
                    env_idx,
                    base_env,
                    waypoint=not self.destination_is_waypoint[env_idx],
                )
                displacement = self.destinations[env_idx] - effector_position
                actions[env_idx] = self._translation_action(
                    displacement, gripper_opening
                )
            else:
                action = np.zeros(5, dtype=np.float64)
                action[3] = np.clip(yaw_error / 0.3, -MAX_YAW_ACTION, MAX_YAW_ACTION)
                action[4] = -gripper_opening
                actions[env_idx] = self.action_support.constrain(action)

        return actions

    def episode_summary(self, env_idx: int) -> dict:
        """Return the current episode's sampled points and completed arrivals."""
        return {
            "side": self._unwrapped_envs()[env_idx].cube_hop_side,
            "sampled_gripper_targets": np.asarray(
                self.sampled_gripper_targets[env_idx], dtype=np.float64
            ).reshape(-1, 3),
            "sampled_waypoints": np.asarray(
                self.sampled_waypoints[env_idx], dtype=np.float64
            ).reshape(-1, 3),
            "completed_gripper_targets": int(self.completed_gripper_targets[env_idx]),
            "completed_waypoints": int(self.completed_waypoints[env_idx]),
        }

    def _begin_episode(self, env_idx: int) -> None:
        self.sampled_gripper_targets[env_idx] = []
        self.sampled_waypoints[env_idx] = []
        self.completed_gripper_targets[env_idx] = 0
        self.completed_waypoints[env_idx] = 0
        self.phases[env_idx] = "reset"

    def _sample_destination(
        self, env_idx: int, base_env: CubeHopEnv, waypoint: bool
    ) -> None:
        side_bounds = SIDE_SAMPLING_BOUNDS[base_env.cube_hop_side]
        if waypoint:
            destination = base_env.np_random.uniform(*side_bounds["waypoint"])
            self.sampled_waypoints[env_idx].append(destination.copy())
        else:
            destination = base_env.np_random.uniform(*side_bounds["gripper_target"])
            self.sampled_gripper_targets[env_idx].append(destination.copy())
        self.destinations[env_idx] = destination
        self.destination_is_waypoint[env_idx] = waypoint
        self.phases[env_idx] = "move"

    def _translation_action(
        self, displacement: np.ndarray, gripper_opening: float = 0.0
    ) -> np.ndarray:
        normalized = np.asarray(displacement, dtype=np.float64) / 0.05
        norm = float(np.linalg.norm(normalized))
        if norm > MAX_TRANSLATION_ACTION_NORM:
            normalized *= MAX_TRANSLATION_ACTION_NORM / norm
        action = np.zeros(5, dtype=np.float64)
        action[:3] = normalized
        # ManipSpace actions are relative, so this tracks absolute fully-open=0.
        action[4] = -float(np.clip(gripper_opening, 0.0, 1.0))
        return self.action_support.constrain(action)

    def _unwrapped_envs(self) -> list[CubeHopEnv]:
        if hasattr(self.env, "envs"):
            return [env.unwrapped for env in self.env.envs]
        return [self.env.unwrapped]


__all__ = ["ActionSupport", "CubeHopPolicy"]
