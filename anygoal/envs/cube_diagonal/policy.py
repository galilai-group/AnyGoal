"""Policies and oracles for the Cube-Diagonal environment."""

from __future__ import annotations

import numpy as np
from ogbench.manipspace import lie
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy

from .env import (
    DIAGONAL_CLOSED_GRASP,
    DIAGONAL_GRASP_OFFSET,
    CubeDiagonalEnv,
)


class DiagonalCubeMarkovOracle(CubeMarkovOracle):
    """Original Cube Markov oracle with only the grasp yaw shifted by 45 degrees."""

    def select_action(self, ob, info):
        effector_pos = info["proprio/effector_pos"]
        effector_yaw = info["proprio/effector_yaw"][0]

        target_block = info["privileged/target_block"]
        block_pos = info[f"privileged/block_{target_block}_pos"]
        block_yaw = self.shortest_yaw(
            effector_yaw,
            info[f"privileged/block_{target_block}_yaw"][0]
            + DIAGONAL_GRASP_OFFSET,
        )
        target_pos = info["privileged/target_block_pos"]
        target_yaw = self.shortest_yaw(
            effector_yaw,
            info["privileged/target_block_yaw"][0]
            + DIAGONAL_GRASP_OFFSET,
        )

        block_above_offset = np.array([0, 0, 0.18])
        above_threshold = 0.16
        gripper_closed = info["proprio/gripper_contact"] > 0.5
        gripper_open = info["proprio/gripper_contact"] < 0.1
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.02
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.04
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not target_pos_aligned:
            if not xy_aligned:
                action = np.zeros(5)
                diff = self.shape_diff(block_pos + block_above_offset - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                diff = self.shape_diff(block_pos - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif pos_aligned and not gripper_closed:
                diff = self.shape_diff(block_pos - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and not above and not target_xy_aligned:
                diff = (
                    np.array(
                        [block_pos[0], block_pos[1], block_above_offset[2] * 2]
                    )
                    - effector_pos
                )
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            elif pos_aligned and gripper_closed and above and not target_xy_aligned:
                diff = self.shape_diff(target_pos + block_above_offset - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
            else:
                diff = self.shape_diff(target_pos - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = 1
        else:
            if not gripper_open:
                diff = self.shape_diff(target_pos - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (target_yaw - block_yaw) * gain_yaw
                action[4] = -1
            elif gripper_open and not above:
                diff = (
                    np.array([block_pos[0], block_pos[1], above_threshold * 2])
                    - effector_pos
                )
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            else:
                diff = self.shape_diff(self._final_pos - effector_pos)
                action[:3] = diff[:3] * gain_pos
                action[3] = (self._final_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            if final_pos_aligned:
                self._done = True

        action = np.clip(action, -1, 1)
        self._step += 1
        if self._step == self._max_step:
            self._done = True
        return action


class DiagonalCubePlanOracle(CubePlanOracle):
    """Original Cube plan with pick and place gripper yaw shifted by 45 degrees."""

    def compute_keyframes(self, plan_input):
        diagonal_input = dict(plan_input)
        for key in ("block_initial", "block_goal"):
            pose = plan_input[key]
            diagonal_input[key] = lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3.from_z_radians(
                    self.get_yaw(pose) + DIAGONAL_GRASP_OFFSET
                ),
                translation=pose.translation(),
            )
        times, poses, grasps = super().compute_keyframes(diagonal_input)
        grasps = {
            name: value * DIAGONAL_CLOSED_GRASP
            for name, value in grasps.items()
        }
        return times, poses, grasps


class CubeDiagonalPolicy(ExpertPolicy):
    """Stable-world-model Cube expert using the diagonal-grasp oracle."""

    def __init__(self, policy_type="plan_oracle", **kwargs) -> None:
        super().__init__(policy_type=policy_type, **kwargs)

    def set_env(self, env) -> None:
        self.env = env
        base_envs = [item.unwrapped for item in env.envs]
        if not all(isinstance(item, CubeDiagonalEnv) for item in base_envs):
            raise ValueError("CubeDiagonalPolicy requires CubeDiagonalEnv.")
        if self.type == "markov_oracle":
            agents = [
                DiagonalCubeMarkovOracle(
                    env=base_envs[index], min_norm=self.min_norm
                )
                for index in range(env.num_envs)
            ]
        else:
            agents = [
                DiagonalCubePlanOracle(
                    env=base_envs[index],
                    noise=self.action_noise,
                    noise_smoothing=self.noise_smoothing,
                )
                for index in range(env.num_envs)
            ]
        self._oracle_agents = {"cube_diagonal": agents}
        self._p_stack = np.zeros(env.num_envs)
        self._xi = np.zeros(env.num_envs)
        self._agents = [None] * env.num_envs

    def episode_summary(self, env_idx: int) -> dict:
        base_env = self.env.envs[env_idx].unwrapped
        return {
            **base_env.diagonal_grasp_summary(),
            "policy_type": self.type,
            "action_noise": self.action_noise,
            "grasp_yaw_offset_rad": DIAGONAL_GRASP_OFFSET,
            "closed_grasp_target": DIAGONAL_CLOSED_GRASP,
        }


__all__ = [
    "CubeDiagonalPolicy",
    "DiagonalCubeMarkovOracle",
    "DiagonalCubePlanOracle",
]

