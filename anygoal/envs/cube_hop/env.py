"""OGBench CubeHop environment and conservative expert policy."""

from __future__ import annotations

from types import MethodType
import gymnasium as gym
import mujoco
import numpy as np

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


EE_START_LOW = np.array((0.25, -0.35, 0.20), dtype=np.float64)
EE_START_HIGH = np.array((0.60, 0.35, 0.35), dtype=np.float64)
RIGHT_CUBE_POSITION_LOW = np.array((0.30, 0.10, 0.02), dtype=np.float64)
RIGHT_CUBE_POSITION_HIGH = np.array((0.55, 0.30, 0.02), dtype=np.float64)
RIGHT_GRIPPER_TARGET_LOW = np.array((0.30, -0.15, 0.02), dtype=np.float64)
RIGHT_GRIPPER_TARGET_HIGH = np.array((0.55, 0.05, 0.02), dtype=np.float64)
RIGHT_WAYPOINT_LOW = np.array((0.30, -0.30, 0.20), dtype=np.float64)
RIGHT_WAYPOINT_HIGH = np.array((0.55, -0.15, 0.35), dtype=np.float64)

LEFT_CUBE_POSITION_LOW = np.array((0.30, -0.30, 0.02), dtype=np.float64)
LEFT_CUBE_POSITION_HIGH = np.array((0.55, -0.10, 0.02), dtype=np.float64)
LEFT_GRIPPER_TARGET_LOW = np.array((0.30, -0.05, 0.02), dtype=np.float64)
LEFT_GRIPPER_TARGET_HIGH = np.array((0.55, 0.15, 0.02), dtype=np.float64)
LEFT_WAYPOINT_LOW = np.array((0.30, 0.15, 0.20), dtype=np.float64)
LEFT_WAYPOINT_HIGH = np.array((0.55, 0.30, 0.35), dtype=np.float64)

CUBE_HOP_SIDES = ("right", "left")
SIDE_SAMPLING_BOUNDS = {
    "right": {
        "cube": (RIGHT_CUBE_POSITION_LOW, RIGHT_CUBE_POSITION_HIGH),
        "gripper_target": (RIGHT_GRIPPER_TARGET_LOW, RIGHT_GRIPPER_TARGET_HIGH),
        "waypoint": (RIGHT_WAYPOINT_LOW, RIGHT_WAYPOINT_HIGH),
    },
    "left": {
        "cube": (LEFT_CUBE_POSITION_LOW, LEFT_CUBE_POSITION_HIGH),
        "gripper_target": (LEFT_GRIPPER_TARGET_LOW, LEFT_GRIPPER_TARGET_HIGH),
        "waypoint": (LEFT_WAYPOINT_LOW, LEFT_WAYPOINT_HIGH),
    },
}

# Backward-compatible aliases for the original right-side case.
CUBE_POSITION_LOW = RIGHT_CUBE_POSITION_LOW
CUBE_POSITION_HIGH = RIGHT_CUBE_POSITION_HIGH
GRIPPER_TARGET_LOW = RIGHT_GRIPPER_TARGET_LOW
GRIPPER_TARGET_HIGH = RIGHT_GRIPPER_TARGET_HIGH
WAYPOINT_LOW = RIGHT_WAYPOINT_LOW
WAYPOINT_HIGH = RIGHT_WAYPOINT_HIGH

# Match the original Cube Markov oracle's 3-D position-alignment criterion.
POSITION_TOLERANCE = 0.02
YAW_TOLERANCE = 0.01
RIGHT_YAW_DELTA = -0.2
MAX_TRANSLATION_ACTION_NORM = 0.80
MAX_YAW_ACTION = 2.0 / 3.0

EVALUATION_EE_POSITION_TOLERANCE = 0.04
EVALUATION_EE_YAW_TOLERANCE = 0.10
EVALUATION_CUBE_POSITION_TOLERANCE = 0.04

def wrapped_angle_error(target: float, current: float) -> float:
    """Return the signed shortest yaw change from ``current`` to ``target``."""
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


def cube_hop_success_criterion(
    ee_position_error: float,
    ee_yaw_error: float,
    cube_position_error: float,
    ee_position_tolerance: float,
    ee_yaw_tolerance: float,
    cube_position_tolerance: float,
) -> bool:
    """Return whether the CubeHop end-effector, yaw, and cube goal is met."""
    return bool(
        ee_position_error <= ee_position_tolerance
        and ee_yaw_error <= ee_yaw_tolerance
        and cube_position_error <= cube_position_tolerance
    )


class CubeHopEnv(CubeEnv):
    """Single-cube scene for the CubeHop end-effector task."""

    def __init__(
        self,
        side: str = "right",
        evaluation_ee_position_tolerance: float = (
            EVALUATION_EE_POSITION_TOLERANCE
        ),
        evaluation_ee_yaw_tolerance: float = EVALUATION_EE_YAW_TOLERANCE,
        evaluation_cube_position_tolerance: float = (
            EVALUATION_CUBE_POSITION_TOLERANCE
        ),
        **kwargs,
    ) -> None:
        self._cube_hop_touched_cube = False
        self._cube_hop_cube_geom_ids: set[int] = set()
        self._cube_hop_robot_geom_ids: set[int] = set()
        tolerances = {
            "ee_position": float(evaluation_ee_position_tolerance),
            "ee_yaw": float(evaluation_ee_yaw_tolerance),
            "cube_position": float(evaluation_cube_position_tolerance),
        }
        if any(value <= 0.0 for value in tolerances.values()):
            raise ValueError("CubeHop evaluation tolerances must be positive.")
        self._evaluation_tolerances = tolerances
        self._evaluation_goal: dict[str, np.ndarray | float] | None = None
        self._criterion_3_success = False
        self._evaluation_errors = {
            "ee_position": np.inf,
            "ee_yaw": np.inf,
            "cube_position": np.inf,
        }
        self._original_cube_success = False
        self.set_cube_hop_side(side)
        kwargs.setdefault("env_type", "single")
        kwargs.setdefault("ob_type", "states")
        kwargs.setdefault("mode", "data_collection")
        kwargs.setdefault("terminate_at_goal", False)
        kwargs.setdefault("visualize_info", False)
        super().__init__(**kwargs)
        if self._env_type != "single":
            raise ValueError("CubeHop requires env_type='single'.")

        self._target_task = "cube_hop"
        self.env_name = "CubeHop"

    def set_cube_hop_side(self, side: str) -> None:
        if side not in CUBE_HOP_SIDES:
            raise ValueError(
                f"CubeHop side must be one of {CUBE_HOP_SIDES}, got {side!r}."
            )
        self.cube_hop_side = side

    def reset(self, seed=None, options=None, *args, **kwargs):
        """Sample all requested episode variables from seeded uniform laws."""
        self._cube_hop_touched_cube = False
        self._evaluation_goal = None
        self._criterion_3_success = False
        self._evaluation_errors = {
            "ee_position": np.inf,
            "ee_yaw": np.inf,
            "cube_position": np.inf,
        }
        self._original_cube_success = False
        gym.Env.reset(self, seed=seed)
        cube_bounds = SIDE_SAMPLING_BOUNDS[self.cube_hop_side]["cube"]
        cube_position = self.np_random.uniform(*cube_bounds)
        effector_position = self.np_random.uniform(EE_START_LOW, EE_START_HIGH)
        cube_yaw = float(self.np_random.uniform(0.0, 2.0 * np.pi))

        reset_options = dict(options or {})
        reset_options.update(
            {
                "variation": [],
                "variation_values": {
                    "agent.ee_start_position": effector_position,
                    "cube.start_position": cube_position[None, :2],
                    "cube.start_yaw": np.array([cube_yaw]),
                    # Compatibility-only target fields are anchored to the cube.
                    "cube.goal_position": cube_position[None, :2],
                    "cube.goal_yaw": np.array([cube_yaw]),
                },
            }
        )
        # CubeEnv currently consumes but does not forward its seed.  The call to
        # gym.Env.reset above has already initialized the environment RNG.
        return super().reset(seed=None, options=reset_options, *args, **kwargs)

    def post_compilation_objects(self) -> None:
        """Cache geometry IDs for contact auditing without changing physics."""
        super().post_compilation_objects()
        self._cube_hop_cube_geom_ids = set(self._cube_geom_ids_list[0])
        self._cube_hop_robot_geom_ids = {
            geom_id
            for geom_id in range(self._model.ngeom)
            if (
                (
                    name := mujoco.mj_id2name(
                        self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                )
                and name.startswith("ur5e/")
            )
        }

    def set_new_target(self, return_info=True, **kwargs):
        """Keep compatibility target fields at the cube without sampling a goal."""
        cube_joint = self._data.joint("object_joint_0")
        self._target_block = 0
        self._data.mocap_pos[self._cube_target_mocap_ids[0]] = cube_joint.qpos[:3]
        self._data.mocap_quat[self._cube_target_mocap_ids[0]] = cube_joint.qpos[3:]
        for geom_id in self._cube_target_geom_ids_list[0]:
            self._model.geom(geom_id).rgba[3] = 0.0
        if return_info:
            return self.compute_observation(), self.get_reset_info()
        return None

    def set_evaluation_goal(
        self,
        effector_pos,
        effector_yaw,
        cube_pos,
    ) -> None:
        """Set a future-state goal used only for evaluation bookkeeping."""
        self._evaluation_goal = {
            "ee_position": np.asarray(effector_pos, dtype=np.float64).reshape(3).copy(),
            "ee_yaw": float(np.asarray(effector_yaw).reshape(-1)[0]),
            "cube_position": np.asarray(cube_pos, dtype=np.float64).reshape(3).copy(),
        }
        self._criterion_3_success = False
        self._evaluation_errors = {
            "ee_position": np.inf,
            "ee_yaw": np.inf,
            "cube_position": np.inf,
        }
        self._original_cube_success = False

    def get_evaluation_metrics(self) -> dict:
        """Return per-episode, ever-achieved planning metrics."""
        if self._evaluation_goal is None:
            raise RuntimeError("CubeHop evaluation goal was not configured.")
        return {
            "criterion_3_success": self._criterion_3_success,
            "original_cube_success": self._original_cube_success,
            "final_errors": self._evaluation_errors.copy(),
            "tolerances": self._evaluation_tolerances.copy(),
        }

    def step(self, action):
        """Run the original control step while auditing every physics substep."""
        if self._reset_next_step:
            return self.reset()

        if self._success_timing == "pre":
            success = self._success
            terminated = self.terminate_episode()
            reward = self.compute_reward()

        action = np.array(action)
        self.set_control(action)
        self.pre_step()
        for _ in range(self._n_steps):
            mujoco.mj_step(self._model, self._data)
            self._audit_cube_contact()
        mujoco.mj_rnePostConstraint(self._model, self._data)
        self.post_step()

        if self._success_timing == "post":
            success = self._success
            terminated = self.terminate_episode()
            reward = self.compute_reward()

        truncated = self.truncate_episode()
        observation = self.compute_observation()
        info = self.get_step_info()
        info["success"] = success
        return observation, reward, terminated, truncated, info

    def post_step(self) -> None:
        super().post_step()
        self._audit_cube_contact()
        self._update_evaluation_metrics()
        # Planning metrics are reporting-only and must not terminate the rollout.
        self._success = False

    def _update_evaluation_metrics(self) -> None:
        if self._evaluation_goal is None:
            return

        ob_info = self.compute_ob_info()
        ee_position = np.asarray(ob_info["proprio/effector_pos"])
        ee_yaw = float(np.asarray(ob_info["proprio/effector_yaw"]).reshape(-1)[0])
        cube_position = np.asarray(ob_info["privileged/block_0_pos"])
        self._evaluation_errors = {
            "ee_position": float(
                np.linalg.norm(ee_position - self._evaluation_goal["ee_position"])
            ),
            "ee_yaw": abs(
                wrapped_angle_error(float(self._evaluation_goal["ee_yaw"]), ee_yaw)
            ),
            "cube_position": float(
                np.linalg.norm(
                    cube_position - self._evaluation_goal["cube_position"]
                )
            ),
        }
        current_success = cube_hop_success_criterion(
            ee_position_error=self._evaluation_errors["ee_position"],
            ee_yaw_error=self._evaluation_errors["ee_yaw"],
            cube_position_error=self._evaluation_errors["cube_position"],
            ee_position_tolerance=self._evaluation_tolerances["ee_position"],
            ee_yaw_tolerance=self._evaluation_tolerances["ee_yaw"],
            cube_position_tolerance=self._evaluation_tolerances["cube_position"],
        )
        self._criterion_3_success |= current_success
        self._original_cube_success |= (
            self._evaluation_errors["cube_position"]
            <= self._evaluation_tolerances["cube_position"]
        )

    def _audit_cube_contact(self) -> None:
        for contact in self._data.contact[: self._data.ncon]:
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            if (
                geom_pair & self._cube_hop_cube_geom_ids
                and geom_pair & self._cube_hop_robot_geom_ids
            ):
                self._cube_hop_touched_cube = True
                break


def make_cube_hop_env(**kwargs) -> CubeHopEnv:
    return CubeHopEnv(**kwargs)


def _install_cube_evaluation(env: CubeEnv) -> None:
    """Install full-budget Cube metrics on a StableWM Cube environment."""
    if isinstance(env, CubeHopEnv) or getattr(
        env, "_cube_evaluation_installed", False
    ):
        return
    env._evaluation_tolerances = {
        "ee_position": EVALUATION_EE_POSITION_TOLERANCE,
        "ee_yaw": EVALUATION_EE_YAW_TOLERANCE,
        "cube_position": EVALUATION_CUBE_POSITION_TOLERANCE,
    }
    env._evaluation_goal = None
    env._criterion_3_success = False
    env._original_cube_success = False
    env._evaluation_errors = {
        "ee_position": np.inf,
        "ee_yaw": np.inf,
        "cube_position": np.inf,
    }
    env.set_evaluation_goal = MethodType(CubeHopEnv.set_evaluation_goal, env)
    env.get_evaluation_metrics = MethodType(CubeHopEnv.get_evaluation_metrics, env)
    env._update_evaluation_metrics = MethodType(
        CubeHopEnv._update_evaluation_metrics, env
    )
    original_post_step = env.post_step

    def post_step_with_evaluation(instance) -> None:
        original_post_step()
        instance._update_evaluation_metrics()
        instance._success = False

    env.post_step = MethodType(post_step_with_evaluation, env)
    env._cube_evaluation_installed = True


def prepare_cube_evaluation(world, callables: list[dict] | None):
    """Enable the two Cube planning metrics and add their dataset goal."""
    prepared = list(callables or [])
    base_envs = [env.unwrapped for env in world.envs.envs]
    if not base_envs or not all(isinstance(env, CubeEnv) for env in base_envs):
        return prepared
    for env in base_envs:
        _install_cube_evaluation(env)
    if any(
        spec.get("method") == "set_evaluation_goal" for spec in prepared
    ):
        return prepared
    prepared.append(
        {
            "method": "set_evaluation_goal",
            "args": {
                "effector_pos": {"value": "goal_proprio_effector_pos"},
                "effector_yaw": {"value": "goal_proprio_effector_yaw"},
                "cube_pos": {"value": "goal_privileged_block_0_pos"},
            },
        }
    )
    return prepared


def add_cube_evaluation_metrics(world, metrics: dict) -> dict:
    """Attach original Cube and CubeHop criterion 3 planning metrics."""
    base_envs = [env.unwrapped for env in world.envs.envs]
    if not base_envs or not all(isinstance(env, CubeEnv) for env in base_envs):
        return metrics

    episode_metrics = [env.get_evaluation_metrics() for env in base_envs]
    metrics.pop("success_rate")
    metrics.pop("episode_successes")
    original_cube_successes = [
        bool(episode["original_cube_success"]) for episode in episode_metrics
    ]
    metrics["original_cube_criterion"] = {
        "definition": "cube position within the original Cube 0.04 m tolerance",
        "success_rate": float(np.mean(original_cube_successes) * 100.0),
        "episode_successes": original_cube_successes,
    }
    criterion_3_successes = [
        bool(episode["criterion_3_success"]) for episode in episode_metrics
    ]
    metrics["cube_hop_criterion_3"] = {
        "definition": "end-effector position and yaw, and cube position",
        "success_rate": float(np.mean(criterion_3_successes) * 100.0),
        "episode_successes": criterion_3_successes,
    }
    metrics["cube_evaluation_tolerances"] = episode_metrics[0]["tolerances"]
    metrics["cube_final_errors"] = {
        error: [float(episode["final_errors"][error]) for episode in episode_metrics]
        for error in episode_metrics[0]["final_errors"]
    }
    return metrics


__all__ = [
    "CUBE_HOP_SIDES",
    "CUBE_POSITION_HIGH",
    "CUBE_POSITION_LOW",
    "CubeHopEnv",
    "EE_START_HIGH",
    "EE_START_LOW",
    "EVALUATION_CUBE_POSITION_TOLERANCE",
    "EVALUATION_EE_POSITION_TOLERANCE",
    "EVALUATION_EE_YAW_TOLERANCE",
    "GRIPPER_TARGET_HIGH",
    "GRIPPER_TARGET_LOW",
    "LEFT_CUBE_POSITION_HIGH",
    "LEFT_CUBE_POSITION_LOW",
    "LEFT_GRIPPER_TARGET_HIGH",
    "LEFT_GRIPPER_TARGET_LOW",
    "LEFT_WAYPOINT_HIGH",
    "LEFT_WAYPOINT_LOW",
    "MAX_TRANSLATION_ACTION_NORM",
    "MAX_YAW_ACTION",
    "RIGHT_CUBE_POSITION_HIGH",
    "RIGHT_CUBE_POSITION_LOW",
    "RIGHT_GRIPPER_TARGET_HIGH",
    "RIGHT_GRIPPER_TARGET_LOW",
    "RIGHT_WAYPOINT_HIGH",
    "RIGHT_WAYPOINT_LOW",
    "RIGHT_YAW_DELTA",
    "SIDE_SAMPLING_BOUNDS",
    "WAYPOINT_HIGH",
    "WAYPOINT_LOW",
    "add_cube_evaluation_metrics",
    "cube_hop_success_criterion",
    "make_cube_hop_env",
    "prepare_cube_evaluation",
    "wrapped_angle_error",
]
