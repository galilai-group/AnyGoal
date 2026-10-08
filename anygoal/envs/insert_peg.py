"""Gymnasium adapter for DeepMind Control Insert Peg."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from dm_control import suite


ARM_JOINTS = (
    "arm_root",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
    "finger",
    "fingertip",
    "thumb",
    "thumbtip",
)
OBJECT_JOINTS = ("peg_x", "peg_z", "peg_y")

STATE_DIM = 44
PROPRIO_DIM = 33
HAND_POSITION_SLICE = slice(29, 31)
HAND_QUATERNION_SLICE = slice(31, 33)
OBJECT_POSITION_SLICE = slice(33, 35)
OBJECT_QUATERNION_SLICE = slice(35, 37)

EVALUATION_HAND_POSITION_TOLERANCE = 0.04
EVALUATION_HAND_ANGLE_TOLERANCE = 0.10
EVALUATION_OBJECT_POSITION_TOLERANCE = 0.04
EVALUATION_OBJECT_ANGLE_TOLERANCE = 0.10


def planar_angle(quaternion_wy: np.ndarray) -> float:
    """Convert a planar ``(qw, qy)`` quaternion to its joint angle."""
    quaternion_wy = np.asarray(quaternion_wy, dtype=np.float64).reshape(2)
    return float(2.0 * np.arctan2(quaternion_wy[1], quaternion_wy[0]))


def wrapped_angle_error(target: float, current: float) -> float:
    """Return the signed shortest angular change from ``current`` to ``target``."""
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


def insert_peg_goal_errors(
    goal_state: np.ndarray, state: np.ndarray
) -> dict[str, float]:
    """Measure the render-relevant hand and peg errors between two states."""
    goal = np.asarray(goal_state, dtype=np.float64).reshape(STATE_DIM)
    current = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
    return {
        "hand_position": float(
            np.linalg.norm(
                goal[HAND_POSITION_SLICE] - current[HAND_POSITION_SLICE]
            )
        ),
        "hand_angle": abs(
            wrapped_angle_error(
                planar_angle(goal[HAND_QUATERNION_SLICE]),
                planar_angle(current[HAND_QUATERNION_SLICE]),
            )
        ),
        "object_position": float(
            np.linalg.norm(
                goal[OBJECT_POSITION_SLICE] - current[OBJECT_POSITION_SLICE]
            )
        ),
        "object_angle": abs(
            wrapped_angle_error(
                planar_angle(goal[OBJECT_QUATERNION_SLICE]),
                planar_angle(current[OBJECT_QUATERNION_SLICE]),
            )
        ),
    }


def insert_peg_goal_reached(
    errors: Mapping[str, float], tolerances: Mapping[str, float]
) -> bool:
    """Return whether the hand and peg match a dataset future-state goal."""
    return bool(all(errors[name] <= tolerances[name] for name in tolerances))


class InsertPegEnv(gym.Env):
    """Planning environment matching the RL Unplugged Insert Peg transitions.

    One Gymnasium step advances one dm_control control step. Observations use the
    44-value order from the dm_control Insert-Peg observation and
    from the converted RL Unplugged dataset. A reset can restore such a state and
    independently install a future dataset state as the planning goal.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 224,
        height: int = 224,
        camera_id: int = 0,
        terminate_at_goal: bool = True,
        evaluation_hand_position_tolerance: float = (
            EVALUATION_HAND_POSITION_TOLERANCE
        ),
        evaluation_hand_angle_tolerance: float = EVALUATION_HAND_ANGLE_TOLERANCE,
        evaluation_object_position_tolerance: float = (
            EVALUATION_OBJECT_POSITION_TOLERANCE
        ),
        evaluation_object_angle_tolerance: float = (
            EVALUATION_OBJECT_ANGLE_TOLERANCE
        ),
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        if width <= 0 or height <= 0:
            raise ValueError("Render width and height must be positive.")

        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.camera_id = int(camera_id)
        self.terminate_at_goal = bool(terminate_at_goal)
        self.env_name = "InsertPeg"
        self.dmc_env = suite.load(
            domain_name="manipulator",
            task_name="insert_peg",
        )

        action_spec = self.dmc_env.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=action_spec.dtype),
            high=np.asarray(action_spec.maximum, dtype=action_spec.dtype),
            dtype=action_spec.dtype,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        self._tolerances = {
            "hand_position": float(evaluation_hand_position_tolerance),
            "hand_angle": float(evaluation_hand_angle_tolerance),
            "object_position": float(evaluation_object_position_tolerance),
            "object_angle": float(evaluation_object_angle_tolerance),
        }
        if any(value <= 0.0 for value in self._tolerances.values()):
            raise ValueError("Evaluation tolerances must be positive.")

        self._evaluation_goal: np.ndarray | None = None
        self._evaluation_success = False
        self._evaluation_errors = {name: np.inf for name in self._tolerances}
        self._task_success = False
        self._cumulative_reward = 0.0

    @property
    def physics(self):
        """Expose the dm_control physics object for inspection and planning."""
        return self.dmc_env.physics

    @staticmethod
    def _observation_to_array(observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(value).reshape(-1) for value in observation.values()]
        ).astype(np.float32, copy=False)

    def get_state(self) -> np.ndarray:
        """Return the current state in the converted dataset's 44-D order."""
        return self._observation_to_array(
            self.dmc_env.task.get_observation(self.physics)
        )

    def set_state(self, state: np.ndarray) -> None:
        """Restore a converted RL Unplugged observation into MuJoCo."""
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"State must have shape ({STATE_DIM},), got {state.shape}."
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("State must contain only finite values.")

        arm_pose = state[:16].reshape(8, 2)
        self.physics.named.data.qpos[list(ARM_JOINTS)] = np.arctan2(
            arm_pose[:, 0], arm_pose[:, 1]
        )
        self.physics.named.data.qvel[list(ARM_JOINTS)] = state[16:24]
        self.physics.named.data.qpos[list(OBJECT_JOINTS)] = (
            state[33],
            state[34],
            planar_angle(state[35:37]),
        )
        self.physics.named.data.qvel[list(OBJECT_JOINTS)] = state[37:40]

        for body in ("target_peg", "slot"):
            self.physics.named.model.body_pos[body, ["x", "z"]] = state[40:42]
            self.physics.named.model.body_quat[body, ["qw", "qy"]] = state[42:44]

        # The 44-D observation does not contain MuJoCo's solver warm start.
        # Clear it so identical restored states have identical rollouts.
        self.physics.data.qacc_warmstart[:] = 0.0
        self.physics.data.ctrl[:] = 0.0
        if self.physics.model.na:
            self.physics.data.act[:] = 0.0
        self.physics.forward()

    @staticmethod
    def _row_state(row: Mapping[str, Any], goal: bool) -> np.ndarray:
        candidates = (
            ("goal_observation", "goal_privilege", "goal_proprio")
            if goal
            else ("observation", "privilege", "proprio")
        )
        for key in candidates:
            if key in row:
                state = np.asarray(row[key])
                if state.shape == (STATE_DIM,):
                    return state.astype(np.float32, copy=False)
        kind = "goal" if goal else "initial"
        raise KeyError(
            f"Could not find a ({STATE_DIM},) {kind} state in keys {candidates}."
        )

    def reset_options_from_dataset(
        self,
        init_row: Mapping[str, Any],
        goal_row: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Convert stable-worldmodel HDF5 rows into reset options."""
        return {
            "state": self._row_state(init_row, goal=False).copy(),
            "goal_state": self._row_state(goal_row, goal=True).copy(),
        }

    def set_evaluation_goal(self, goal_state: np.ndarray) -> None:
        """Set a future-state planning goal without changing the physical slot."""
        goal = np.asarray(goal_state, dtype=np.float32)
        if goal.shape != (STATE_DIM,):
            raise ValueError(
                f"Evaluation goal must have shape ({STATE_DIM},), got {goal.shape}."
            )
        self._evaluation_goal = goal.copy()
        self._evaluation_success = False
        self._evaluation_errors = {name: np.inf for name in self._tolerances}

    def _update_evaluation(self, state: np.ndarray) -> bool:
        if self._evaluation_goal is None:
            return False
        self._evaluation_errors = insert_peg_goal_errors(self._evaluation_goal, state)
        current_success = insert_peg_goal_reached(
            self._evaluation_errors, self._tolerances
        )
        self._evaluation_success |= current_success
        return current_success

    def get_evaluation_metrics(self) -> dict[str, Any]:
        """Return ever-achieved dataset-goal and native-task success metrics."""
        if self._evaluation_goal is None:
            raise RuntimeError("An evaluation goal has not been configured.")
        return {
            "success": self._evaluation_success,
            "task_success": self._task_success,
            "final_errors": self._evaluation_errors.copy(),
            "tolerances": self._tolerances.copy(),
        }

    def _info(self, state: np.ndarray, reward: float) -> dict[str, Any]:
        current_evaluation_success = self._update_evaluation(state)
        success = (
            current_evaluation_success
            if self._evaluation_goal is not None
            else self._task_success
        )
        return {
            "env_name": self.env_name,
            "success": bool(success),
            "task_success": bool(self._task_success),
            "evaluation_success": bool(current_evaluation_success),
            "score": float(self._cumulative_reward / 1000.0),
            "discount": np.float32(1.0),
            "qpos": self.physics.data.qpos.copy(),
            "qvel": self.physics.data.qvel.copy(),
            "proprio": state[:PROPRIO_DIM].copy(),
            "privilege": state.copy(),
            "proprio_arm_pos": state[:16].copy(),
            "proprio_arm_vel": state[16:24].copy(),
            "proprio_touch": state[24:29].copy(),
            "proprio_hand_pos": state[29:33].copy(),
            "privileged_object_pos": state[33:37].copy(),
            "privileged_object_vel": state[37:40].copy(),
            "privileged_target_pos": state[40:44].copy(),
            "native_reward": np.float32(reward),
            **{
                f"evaluation_{name}_error": np.float32(value)
                for name, value in self._evaluation_errors.items()
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        if seed is not None:
            self.dmc_env.task._random = np.random.RandomState(seed)

        timestep = self.dmc_env.reset()
        self._cumulative_reward = 0.0
        self._task_success = False
        self._evaluation_goal = None
        self._evaluation_success = False
        self._evaluation_errors = {name: np.inf for name in self._tolerances}

        if options.get("state") is not None:
            self.set_state(options["state"])
            state = self.get_state()
        else:
            state = self._observation_to_array(timestep.observation)
        native_reward = float(self.dmc_env.task.get_reward(self.physics))
        self._task_success = native_reward >= 1.0 - 1e-7
        if options.get("goal_state") is not None:
            self.set_evaluation_goal(options["goal_state"])
        return state, self._info(state, reward=native_reward)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action must have shape {self.action_space.shape}, got {action.shape}."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        timestep = self.dmc_env.step(action)
        reward = float(timestep.reward or 0.0)
        self._cumulative_reward += reward
        self._task_success |= reward >= 1.0 - 1e-7
        state = self._observation_to_array(timestep.observation)
        info = self._info(state, reward)
        terminated = bool(self.terminate_at_goal and info["success"])
        truncated = bool(timestep.last() and not terminated)
        return state, reward, terminated, truncated, info

    def render(
        self,
        width: int | None = None,
        height: int | None = None,
        camera_id: int | None = None,
    ) -> np.ndarray:
        """Render the same camera used for the converted dataset RGB frames."""
        return self.physics.render(
            height=self.height if height is None else int(height),
            width=self.width if width is None else int(width),
            camera_id=self.camera_id if camera_id is None else int(camera_id),
        )

    def close(self) -> None:
        self.dmc_env.close()


def make_insert_peg_env(**kwargs: Any) -> InsertPegEnv:
    """Construct the registered Insert Peg environment."""
    return InsertPegEnv(**kwargs)


def add_insert_peg_evaluation_metrics(world, metrics: dict) -> dict:
    """Attach dataset-goal errors and native Insert-Peg task success."""
    base_envs = [env.unwrapped for env in world.envs.envs]
    if not base_envs or not all(
        isinstance(env, InsertPegEnv) for env in base_envs
    ):
        return metrics

    episode_metrics = [env.get_evaluation_metrics() for env in base_envs]
    task_successes = [bool(episode["task_success"]) for episode in episode_metrics]
    metrics["dataset_goal_criterion"] = {
        "definition": "hand and peg pose match the sampled 25-step future state",
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": metrics["episode_successes"],
    }
    metrics["native_insert_peg_criterion"] = {
        "definition": "dm_control insert_peg reward reaches 1.0",
        "success_rate": float(np.mean(task_successes) * 100.0),
        "episode_successes": task_successes,
    }
    metrics["insert_peg_evaluation_tolerances"] = episode_metrics[0]["tolerances"]
    metrics["insert_peg_final_errors"] = {
        error: [float(episode["final_errors"][error]) for episode in episode_metrics]
        for error in episode_metrics[0]["final_errors"]
    }
    return metrics


__all__ = [
    "EVALUATION_HAND_ANGLE_TOLERANCE",
    "EVALUATION_HAND_POSITION_TOLERANCE",
    "EVALUATION_OBJECT_ANGLE_TOLERANCE",
    "EVALUATION_OBJECT_POSITION_TOLERANCE",
    "InsertPegEnv",
    "add_insert_peg_evaluation_metrics",
    "insert_peg_goal_errors",
    "insert_peg_goal_reached",
    "make_insert_peg_env",
    "planar_angle",
    "wrapped_angle_error",
]
