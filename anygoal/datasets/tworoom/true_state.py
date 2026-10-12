"""TwoRoom adapters for canonical true-state predictor inputs."""

from __future__ import annotations

import torch


class TwoRoomTrueStateAdapter:
    """Select agent position and optional door coordinates from the 10D state."""

    action_dim = 2
    state_keys = ("observation",)

    def __init__(self, include_doors: bool = False) -> None:
        self.include_doors = bool(include_doors)
        self.state_dim = 4 if self.include_doors else 2
        self.state_names = (
            "agent_x",
            "agent_y",
        ) + (
            (
                "door_1_x",
                "door_1_y",
            )
            if self.include_doors
            else ()
        )

    def encode_state(self, observation: torch.Tensor | dict) -> torch.Tensor:
        if isinstance(observation, dict):
            observation = observation["observation"]
        observation = torch.as_tensor(observation).float()
        if observation.shape[-1] != 10:
            raise ValueError(
                "TwoRoom observation must have final dimension 10, got "
                f"{observation.shape}."
            )
        if self.include_doors:
            return torch.cat((observation[..., :2], observation[..., 4:6]), dim=-1)
        return observation[..., :2]

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action).float()
        if action.shape[-1] < self.action_dim or action.shape[-1] % self.action_dim:
            raise ValueError(
                "TwoRoom action blocks must contain a whole number of raw 2D "
                f"actions, got final dimension {action.shape[-1]}."
            )
        return action

    def __call__(self, sample: dict) -> dict:
        sample["state"] = self.encode_state(sample["observation"])
        sample["action"] = self.encode_action(sample["action"])
        return sample


__all__ = ["TwoRoomTrueStateAdapter"]
