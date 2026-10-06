"""Manipulator Insert Peg adapters for canonical true-state predictors."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class ManipulatorInsertPegStateAdapter:
    """Select the requested 33D essential or full 44D observation state."""

    action_dim = 5
    state_keys = ("observation",)
    environment_state_keys = {"observation": "observation"}
    _essential_indices = tuple(range(16)) + tuple(range(24, 37)) + tuple(
        range(40, 44)
    )
    _all_state_names = (
        *(f"proprio_arm_pos_{index}" for index in range(16)),
        *(f"proprio_arm_vel_{index}" for index in range(8)),
        *(f"proprio_touch_{index}" for index in range(5)),
        "proprio_hand_pos_x",
        "proprio_hand_pos_z",
        "proprio_hand_pos_qw",
        "proprio_hand_pos_qy",
        "privileged_object_pos_x",
        "privileged_object_pos_z",
        "privileged_object_pos_qw",
        "privileged_object_pos_qy",
        "privileged_object_vel_x",
        "privileged_object_vel_z",
        "privileged_object_vel_angular_y",
        "privileged_target_pos_x",
        "privileged_target_pos_z",
        "privileged_target_pos_qw",
        "privileged_target_pos_qy",
    )

    def __init__(self, include_velocities: bool = False) -> None:
        self.include_velocities = bool(include_velocities)
        self.state_dim = 44 if self.include_velocities else 33
        indices = range(44) if self.include_velocities else self._essential_indices
        self.state_names = tuple(self._all_state_names[index] for index in indices)

    @staticmethod
    def _observation(
        state: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(state, Mapping):
            if "observation" not in state:
                raise KeyError("Manipulator state input is missing 'observation'.")
            state = state["observation"]
        observation = torch.as_tensor(state).float()
        if observation.shape[-1] != 44:
            raise ValueError(
                "Manipulator observation must have final dimension 44, got "
                f"{tuple(observation.shape)}."
            )
        return observation

    def encode_state(
        self,
        state: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        observation = self._observation(state)
        if self.include_velocities:
            return observation
        return observation[..., self._essential_indices]

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action).float()
        if action.shape[-1] < self.action_dim or action.shape[-1] % self.action_dim:
            raise ValueError(
                "Manipulator action blocks must contain a whole number of raw "
                f"5D actions, got final dimension {action.shape[-1]}."
            )
        return action

    def __call__(self, sample: dict) -> dict:
        sample["state"] = self.encode_state(sample["observation"])
        sample["action"] = self.encode_action(sample["action"])
        return sample


__all__ = ["ManipulatorInsertPegStateAdapter"]
