"""Cube adapters for canonical true-state predictor inputs."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class CubeTrueStateAdapter:
    """Build a configured Cube state representation from named HDF5 columns."""

    action_dim = 5
    _essential_state_keys = (
        "proprio_effector_pos",
        "proprio_effector_yaw",
        "proprio_gripper_opening",
        "privileged_block_0_pos",
        "privileged_block_0_yaw",
    )
    _environment_state_keys = {
        "proprio_effector_pos": "proprio/effector_pos",
        "proprio_effector_yaw": "proprio/effector_yaw",
        "proprio_gripper_opening": "proprio/gripper_opening",
        "proprio_gripper_contact": "proprio/gripper_contact",
        "privileged_block_0_pos": "privileged/block_0_pos",
        "privileged_block_0_yaw": "privileged/block_0_yaw",
        "proprio_joint_pos": "proprio/joint_pos",
    }

    def __init__(
        self,
        include_joint_positions: bool = False,
        include_gripper_contact: bool = False,
    ) -> None:
        self.include_joint_positions = bool(include_joint_positions)
        self.include_gripper_contact = bool(include_gripper_contact)
        if self.include_joint_positions and self.include_gripper_contact:
            raise ValueError(
                "The 9D contact state and legacy 17D joint state are distinct "
                "Cube representations."
            )
        if self.include_gripper_contact:
            self.state_dim = 9
            self.state_keys = (
                "proprio_effector_pos",
                "proprio_effector_yaw",
                "proprio_gripper_opening",
                "proprio_gripper_contact",
                "privileged_block_0_pos",
            )
            self.state_names = (
                "proprio_effector_pos_x",
                "proprio_effector_pos_y",
                "proprio_effector_pos_z",
                "proprio_effector_yaw",
                "proprio_gripper_opening",
                "proprio_gripper_contact",
                "privileged_block_0_pos_x",
                "privileged_block_0_pos_y",
                "privileged_block_0_pos_z",
            )
        else:
            self.state_dim = 17 if self.include_joint_positions else 11
            self.state_keys = self._essential_state_keys + (
                ("proprio_joint_pos",) if self.include_joint_positions else ()
            )
            self.state_names = (
                "proprio_effector_pos_x",
                "proprio_effector_pos_y",
                "proprio_effector_pos_z",
                "proprio_effector_yaw_cos",
                "proprio_effector_yaw_sin",
                "proprio_gripper_opening",
                "privileged_block_0_pos_x",
                "privileged_block_0_pos_y",
                "privileged_block_0_pos_z",
                "privileged_block_0_yaw_cos",
                "privileged_block_0_yaw_sin",
            ) + tuple(
                f"proprio_joint_pos_{index}"
                for index in range(6 if self.include_joint_positions else 0)
            )
        self.environment_state_keys = {
            key: self._environment_state_keys[key] for key in self.state_keys
        }

    @staticmethod
    def _require_column(
        state: Mapping[str, torch.Tensor],
        key: str,
        dimension: int,
    ) -> torch.Tensor:
        if key not in state:
            raise KeyError(f"Cube state input is missing {key!r}.")
        values = torch.as_tensor(state[key]).float()
        if values.shape[-1] != dimension:
            raise ValueError(
                f"Cube column {key!r} must have final dimension {dimension}, "
                f"got {values.shape}."
            )
        return values

    def encode_state(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(state, Mapping):
            raise TypeError("Cube state encoding requires a mapping of named columns.")
        effector_position = self._require_column(state, "proprio_effector_pos", 3)
        effector_yaw = self._require_column(state, "proprio_effector_yaw", 1)
        gripper_opening = self._require_column(state, "proprio_gripper_opening", 1)
        block_position = self._require_column(state, "privileged_block_0_pos", 3)
        if self.include_gripper_contact:
            gripper_contact = self._require_column(
                state, "proprio_gripper_contact", 1
            )
            return torch.cat(
                [
                    effector_position,
                    effector_yaw,
                    gripper_opening,
                    gripper_contact,
                    block_position,
                ],
                dim=-1,
            )
        block_yaw = self._require_column(state, "privileged_block_0_yaw", 1)
        components = [
            effector_position,
            torch.cos(effector_yaw),
            torch.sin(effector_yaw),
            gripper_opening,
            block_position,
            torch.cos(block_yaw),
            torch.sin(block_yaw),
        ]
        if self.include_joint_positions:
            components.append(self._require_column(state, "proprio_joint_pos", 6))
        encoded = torch.cat(components, dim=-1)
        if encoded.shape[-1] != self.state_dim:
            raise RuntimeError(
                f"Cube adapter produced {encoded.shape[-1]} values, expected "
                f"{self.state_dim}."
            )
        return encoded

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action).float()
        if action.shape[-1] < self.action_dim or action.shape[-1] % self.action_dim:
            raise ValueError(
                "Cube action blocks must contain a whole number of raw 5D "
                f"actions, got final dimension {action.shape[-1]}."
            )
        return action

    def __call__(self, sample: dict) -> dict:
        sample["state"] = self.encode_state(sample)
        sample["action"] = self.encode_action(sample["action"])
        return sample


class CubeEverythingStateAdapter:
    """Build the broad Cube state from named proprioceptive/privileged columns."""

    action_dim = 5
    _base_state_keys = (
        "proprio_effector_pos",
        "proprio_effector_yaw",
        "proprio_gripper_opening",
        "proprio_gripper_contact",
        "proprio_gripper_vel",
        "privileged_block_0_pos",
        "privileged_block_0_quat",
    )
    _extreme_state_keys = (
        "privileged_block_0_yaw",
        "privileged_target_block_pos",
        "privileged_target_block_yaw",
    )
    _environment_state_keys = {
        "proprio_effector_pos": "proprio/effector_pos",
        "proprio_effector_yaw": "proprio/effector_yaw",
        "proprio_gripper_opening": "proprio/gripper_opening",
        "proprio_gripper_contact": "proprio/gripper_contact",
        "proprio_gripper_vel": "proprio/gripper_vel",
        "privileged_block_0_pos": "privileged/block_0_pos",
        "privileged_block_0_quat": "privileged/block_0_quat",
        "privileged_block_0_yaw": "privileged/block_0_yaw",
        "privileged_target_block_pos": "privileged/target_block_pos",
        "privileged_target_block_yaw": "privileged/target_block_yaw",
    }

    def __init__(self, include_target: bool = False) -> None:
        self.include_target = bool(include_target)
        self.state_dim = 22 if self.include_target else 15
        self.static_goal_state_keys = (
            (
                "privileged_target_block_pos",
                "privileged_target_block_yaw",
            )
            if self.include_target
            else ()
        )
        self.state_keys = self._base_state_keys + (
            self._extreme_state_keys if self.include_target else ()
        )
        self.state_names = (
            "proprio_effector_pos_x",
            "proprio_effector_pos_y",
            "proprio_effector_pos_z",
            "proprio_effector_yaw_cos",
            "proprio_effector_yaw_sin",
            "proprio_gripper_opening",
            "proprio_gripper_contact",
            "proprio_gripper_vel",
            "privileged_block_0_pos_x",
            "privileged_block_0_pos_y",
            "privileged_block_0_pos_z",
            "privileged_block_0_quat_w",
            "privileged_block_0_quat_x",
            "privileged_block_0_quat_y",
            "privileged_block_0_quat_z",
        ) + (
            (
                "privileged_block_0_yaw_cos",
                "privileged_block_0_yaw_sin",
                "privileged_target_block_pos_x",
                "privileged_target_block_pos_y",
                "privileged_target_block_pos_z",
                "privileged_target_block_yaw_cos",
                "privileged_target_block_yaw_sin",
            )
            if self.include_target
            else ()
        )
        self.environment_state_keys = {
            key: self._environment_state_keys[key] for key in self.state_keys
        }

    def encode_state(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(state, Mapping):
            raise TypeError("Cube state encoding requires a mapping of named columns.")
        effector_yaw = CubeTrueStateAdapter._require_column(
            state, "proprio_effector_yaw", 1
        )
        components = [
            CubeTrueStateAdapter._require_column(state, "proprio_effector_pos", 3),
            torch.cos(effector_yaw),
            torch.sin(effector_yaw),
            CubeTrueStateAdapter._require_column(
                state, "proprio_gripper_opening", 1
            ),
            CubeTrueStateAdapter._require_column(
                state, "proprio_gripper_contact", 1
            ),
            CubeTrueStateAdapter._require_column(state, "proprio_gripper_vel", 1),
            CubeTrueStateAdapter._require_column(
                state, "privileged_block_0_pos", 3
            ),
            CubeTrueStateAdapter._require_column(
                state, "privileged_block_0_quat", 4
            ),
        ]
        if self.include_target:
            block_yaw = CubeTrueStateAdapter._require_column(
                state, "privileged_block_0_yaw", 1
            )
            target_yaw = CubeTrueStateAdapter._require_column(
                state, "privileged_target_block_yaw", 1
            )
            components.extend(
                (
                    torch.cos(block_yaw),
                    torch.sin(block_yaw),
                    CubeTrueStateAdapter._require_column(
                        state, "privileged_target_block_pos", 3
                    ),
                    torch.cos(target_yaw),
                    torch.sin(target_yaw),
                )
            )
        encoded = torch.cat(components, dim=-1)
        if encoded.shape[-1] != self.state_dim:
            raise RuntimeError(
                f"Cube adapter produced {encoded.shape[-1]} values, expected "
                f"{self.state_dim}."
            )
        return encoded

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action).float()
        if action.shape[-1] < self.action_dim or action.shape[-1] % self.action_dim:
            raise ValueError(
                "Cube action blocks must contain a whole number of raw 5D "
                f"actions, got final dimension {action.shape[-1]}."
            )
        return action

    def __call__(self, sample: dict) -> dict:
        sample["state"] = self.encode_state(sample)
        sample["action"] = self.encode_action(sample["action"])
        return sample


__all__ = ["CubeEverythingStateAdapter", "CubeTrueStateAdapter"]
