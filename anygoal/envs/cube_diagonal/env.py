"""OGBench Cube with an expert that grasps opposite vertical cube edges."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from stable_worldmodel.envs.ogbench.cube_env import DEFAULT_VARIATIONS, CubeEnv

from anygoal.envs.cube_hop.env import (
    EVALUATION_CUBE_POSITION_TOLERANCE,
    EVALUATION_EE_POSITION_TOLERANCE,
    EVALUATION_EE_YAW_TOLERANCE,
)


DIAGONAL_GRASP_OFFSET = np.pi / 4.0
DEFAULT_CUBE_YAW_CENTER = np.deg2rad(85.0)
DEFAULT_CUBE_YAW_HALF_WIDTH = np.deg2rad(5.0)
CUBE_HALF_EXTENT = 0.02
CUBE_SIDE_LENGTH = 2.0 * CUBE_HALF_EXTENT
EXPECTED_PAD_GAP = np.sqrt(2.0) * CUBE_SIDE_LENGTH
DIAGONAL_CLOSED_GRASP = 0.83
EDGE_POSITION_TOLERANCE = 0.006
PAD_GAP_TOLERANCE = 0.008
DIAGONAL_YAW_TOLERANCE = np.deg2rad(6.0)
LIFT_HEIGHT_THRESHOLD = 0.06
STABLE_EDGE_CONTACT_STEPS = 3


def wrapped_angle_error(target: float, current: float) -> float:
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


class CubeDiagonalEnv(CubeEnv):
    """Original single-cube task with diagonal-grasp contact auditing."""

    def __init__(self, **kwargs) -> None:
        self._evaluation_tolerances = {
            "ee_position": float(
                kwargs.pop(
                    "evaluation_ee_position_tolerance",
                    EVALUATION_EE_POSITION_TOLERANCE,
                )
            ),
            "ee_yaw": float(
                kwargs.pop(
                    "evaluation_ee_yaw_tolerance",
                    EVALUATION_EE_YAW_TOLERANCE,
                )
            ),
            "cube_position": float(
                kwargs.pop(
                    "evaluation_cube_position_tolerance",
                    EVALUATION_CUBE_POSITION_TOLERANCE,
                )
            ),
        }
        if any(value <= 0.0 for value in self._evaluation_tolerances.values()):
            raise ValueError("CubeDiagonal evaluation tolerances must be positive.")
        self._cube_yaw_center = kwargs.pop("cube_yaw_center", DEFAULT_CUBE_YAW_CENTER)
        self._cube_yaw_half_width = kwargs.pop(
            "cube_yaw_half_width", DEFAULT_CUBE_YAW_HALF_WIDTH
        )
        if self._cube_yaw_half_width < 0.0:
            raise ValueError("cube_yaw_half_width must be nonnegative.")
        kwargs.setdefault("env_type", "single")
        kwargs.setdefault("ob_type", "states")
        kwargs.setdefault("mode", "data_collection")
        kwargs.setdefault("terminate_at_goal", False)
        kwargs.setdefault("visualize_info", False)
        super().__init__(**kwargs)
        if self._env_type != "single":
            raise ValueError("CubeDiagonal requires env_type='single'.")
        if self._mode != "data_collection":
            raise ValueError("CubeDiagonal requires mode='data_collection'.")
        self._target_task = "cube_diagonal"
        self.env_name = "CubeDiagonal"

    def reset(self, seed=None, options=None, *args, **kwargs):
        """Use the configured cube-yaw band while honoring the supplied seed."""
        gym.Env.reset(self, seed=seed)
        self.variation_space.seed(seed)
        self.variation_space.reset()
        self.variation_space.update(DEFAULT_VARIATIONS)
        variation_values = {}
        for key in DEFAULT_VARIATIONS:
            space = self.variation_space
            for part in key.split("."):
                space = space[part]
            variation_values[key] = np.asarray(space.value).copy()
        if self._cube_yaw_center is not None:
            for key in ("cube.start_yaw", "cube.goal_yaw"):
                variation_values[key] = np.mod(
                    self._cube_yaw_center
                    + self.np_random.uniform(
                        -self._cube_yaw_half_width,
                        self._cube_yaw_half_width,
                        size=self._num_cubes,
                    ),
                    2.0 * np.pi,
                )
        reset_options = dict(options or {})
        variation_values.update(reset_options.get("variation_values", {}))
        reset_options["variation"] = []
        reset_options["variation_values"] = variation_values
        return super().reset(seed=None, options=reset_options, *args, **kwargs)

    def post_compilation_objects(self) -> None:
        super().post_compilation_objects()
        self._diagonal_cube_geom_id = self._cube_geom_ids_list[0][0]
        self._diagonal_pad_geom_sides: dict[int, str] = {}
        for geom_id in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and "robotiq/right_pad" in name:
                self._diagonal_pad_geom_sides[geom_id] = "right"
            elif name and "robotiq/left_pad" in name:
                self._diagonal_pad_geom_sides[geom_id] = "left"

    def initialize_episode(self) -> None:
        self._reset_diagonal_audit()
        super().initialize_episode()

    def post_step(self) -> None:
        super().post_step()
        self._audit_diagonal_contacts()
        cube_z = float(self._data.joint("object_joint_0").qpos[2])
        self._diagonal_max_cube_z = max(self._diagonal_max_cube_z, cube_z)
        if (
            cube_z >= LIFT_HEIGHT_THRESHOLD
            and self._diagonal_consecutive_edge_steps >= STABLE_EDGE_CONTACT_STEPS
        ):
            self._diagonal_lifted_with_edge_grasp = True
            self._diagonal_lifted_target_indices.add(
                len(self._diagonal_target_positions) - 1
            )
        if self._success:
            self._diagonal_ever_goal_success = True
            current_target = len(self._diagonal_target_positions) - 1
            gripper_contact = float(
                self.compute_ob_info()["proprio/gripper_contact"][0]
            )
            if (
                current_target in self._diagonal_lifted_target_indices
                and gripper_contact < 0.1
            ):
                self._diagonal_completed_target_indices.add(current_target)

    def diagonal_grasp_summary(self) -> dict:
        separations = np.asarray(self._diagonal_edge_separations, dtype=np.float64)
        return {
            "expected_pad_gap_m": EXPECTED_PAD_GAP,
            "ever_bilateral_pad_contact": self._diagonal_ever_bilateral_contact,
            "ever_opposite_edge_contact": self._diagonal_ever_edge_contact,
            "max_consecutive_opposite_edge_steps": (
                self._diagonal_max_consecutive_edge_steps
            ),
            "lifted_with_opposite_edge_grasp": (self._diagonal_lifted_with_edge_grasp),
            "ever_goal_success": self._diagonal_ever_goal_success,
            "max_cube_z_m": self._diagonal_max_cube_z,
            "opposite_edge_contact_steps": len(separations),
            "observed_pad_gap_median": (
                float(np.median(separations)) if len(separations) else None
            ),
            "observed_pad_gap_min": (
                float(separations.min()) if len(separations) else None
            ),
            "observed_pad_gap_max": (
                float(separations.max()) if len(separations) else None
            ),
            "targets_started": len(self._diagonal_target_positions),
            "retarget_count": max(len(self._diagonal_target_positions) - 1, 0),
            "target_positions": [
                position.tolist() for position in self._diagonal_target_positions
            ],
            "targets_with_diagonal_lift": sorted(self._diagonal_lifted_target_indices),
            "targets_completed": sorted(self._diagonal_completed_target_indices),
            "completed_diagonal_tasks": len(
                self._diagonal_lifted_target_indices
                & self._diagonal_completed_target_indices
            ),
        }

    def _reset_diagonal_audit(self) -> None:
        self._diagonal_ever_bilateral_contact = False
        self._diagonal_ever_edge_contact = False
        self._diagonal_consecutive_edge_steps = 0
        self._diagonal_max_consecutive_edge_steps = 0
        self._diagonal_lifted_with_edge_grasp = False
        self._diagonal_ever_goal_success = False
        self._diagonal_max_cube_z = CUBE_HALF_EXTENT
        self._diagonal_edge_separations: list[float] = []
        self._diagonal_target_positions: list[np.ndarray] = []
        self._diagonal_lifted_target_indices: set[int] = set()
        self._diagonal_completed_target_indices: set[int] = set()

    def set_new_target(self, return_info=True, **kwargs):
        """Record each original fixed-target PlanOracle cycle."""
        super().set_new_target(return_info=False, **kwargs)
        target_position = self._data.mocap_pos[self._cube_target_mocap_ids[0]].copy()
        self._diagonal_target_positions.append(target_position)
        if return_info:
            return self.compute_observation(), self.get_reset_info()
        return None

    def _audit_diagonal_contacts(self) -> None:
        contacts: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        cube_id = self._diagonal_cube_geom_id
        cube_center = self._data.geom_xpos[cube_id]
        cube_rotation = self._data.geom_xmat[cube_id].reshape(3, 3)
        for contact_index in range(self._data.ncon):
            contact = self._data.contact[contact_index]
            if contact.geom1 == cube_id:
                pad_geom_id = int(contact.geom2)
            elif contact.geom2 == cube_id:
                pad_geom_id = int(contact.geom1)
            else:
                continue
            side = self._diagonal_pad_geom_sides.get(pad_geom_id)
            if side is None:
                continue
            local_position = cube_rotation.T @ (contact.pos - cube_center)
            contacts[side].append(local_position)

        if not contacts["left"] or not contacts["right"]:
            self._diagonal_consecutive_edge_steps = 0
            return
        self._diagonal_ever_bilateral_contact = True

        best_separation = None
        for left_position in contacts["left"]:
            for right_position in contacts["right"]:
                left_xy = left_position[:2]
                right_xy = right_position[:2]
                left_at_edge = np.all(
                    np.abs(np.abs(left_xy) - CUBE_HALF_EXTENT)
                    <= EDGE_POSITION_TOLERANCE
                )
                right_at_edge = np.all(
                    np.abs(np.abs(right_xy) - CUBE_HALF_EXTENT)
                    <= EDGE_POSITION_TOLERANCE
                )
                opposite_edges = np.all(np.sign(left_xy) == -np.sign(right_xy))
                separation = float(np.linalg.norm(left_xy - right_xy))
                gap_matches = abs(separation - EXPECTED_PAD_GAP) <= PAD_GAP_TOLERANCE
                if left_at_edge and right_at_edge and opposite_edges and gap_matches:
                    if best_separation is None or abs(
                        separation - EXPECTED_PAD_GAP
                    ) < abs(best_separation - EXPECTED_PAD_GAP):
                        best_separation = separation

        cube_yaw = float(self.compute_ob_info()["privileged/block_0_yaw"][0])
        effector_yaw = float(self.compute_ob_info()["proprio/effector_yaw"][0])
        yaw_residual = abs(
            wrapped_angle_error(cube_yaw + DIAGONAL_GRASP_OFFSET, effector_yaw)
        )
        yaw_residual = min(
            yaw_residual,
            abs(yaw_residual - np.pi / 2.0),
            abs(yaw_residual - np.pi),
        )
        edge_contact = (
            best_separation is not None and yaw_residual <= DIAGONAL_YAW_TOLERANCE
        )
        if edge_contact:
            self._diagonal_ever_edge_contact = True
            self._diagonal_edge_separations.append(best_separation)
            self._diagonal_consecutive_edge_steps += 1
            self._diagonal_max_consecutive_edge_steps = max(
                self._diagonal_max_consecutive_edge_steps,
                self._diagonal_consecutive_edge_steps,
            )
        else:
            self._diagonal_consecutive_edge_steps = 0


def make_cube_diagonal_env(**kwargs) -> CubeDiagonalEnv:
    return CubeDiagonalEnv(**kwargs)


__all__ = [
    "CUBE_SIDE_LENGTH",
    "DEFAULT_CUBE_YAW_CENTER",
    "DEFAULT_CUBE_YAW_HALF_WIDTH",
    "DIAGONAL_CLOSED_GRASP",
    "DIAGONAL_GRASP_OFFSET",
    "EXPECTED_PAD_GAP",
    "CubeDiagonalEnv",
    "make_cube_diagonal_env",
]
