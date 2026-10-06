"""PushT adapter for canonical true-state predictor inputs."""

from __future__ import annotations

import numpy as np
import torch


class PushTTrueStateAdapter:
    """Map PushT's stored state to the canonical 10-variable representation.

    The output order is goal position, goal angle (sin, cos), physical-block
    position, physical-block angle (sin, cos), and agent position. Agent
    velocity is intentionally excluded because the three-frame history
    already contains motion information.
    """

    state_dim = 10
    action_dim = 2
    state_keys = ("state",)
    state_names = (
        "goal_x",
        "goal_y",
        "goal_sin",
        "goal_cos",
        "block_x",
        "block_y",
        "block_sin",
        "block_cos",
        "agent_x",
        "agent_y",
    )

    def __init__(
        self,
        goal_position: tuple[float, float] = (256.0, 256.0),
        goal_angle: float = np.pi / 4,
    ) -> None:
        self.goal_position = tuple(float(value) for value in goal_position)
        self.goal_angle = float(goal_angle)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(state).float()
        if state.shape[-1] != 7:
            raise ValueError(
                f"PushT state must have final dimension 7, got {state.shape}."
            )

        goal_x = torch.full_like(state[..., 0], self.goal_position[0])
        goal_y = torch.full_like(state[..., 0], self.goal_position[1])
        goal_sin = torch.full_like(state[..., 0], np.sin(self.goal_angle))
        goal_cos = torch.full_like(state[..., 0], np.cos(self.goal_angle))
        block_angle = state[..., 4]

        return torch.stack(
            (
                goal_x,
                goal_y,
                goal_sin,
                goal_cos,
                state[..., 2],
                state[..., 3],
                torch.sin(block_angle),
                torch.cos(block_angle),
                state[..., 0],
                state[..., 1],
            ),
            dim=-1,
        )

    def decode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Map a canonical prediction back to the renderer's raw 7D state."""
        state = torch.as_tensor(state).float()
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Canonical PushT state must have final dimension "
                f"{self.state_dim}, got {state.shape}."
            )
        block_angle = torch.atan2(state[..., 6], state[..., 7])
        zero_velocity = torch.zeros_like(state[..., 0])
        return torch.stack(
            (
                state[..., 8],
                state[..., 9],
                state[..., 4],
                state[..., 5],
                block_angle,
                zero_velocity,
                zero_velocity,
            ),
            dim=-1,
        )

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action).float()
        if action.shape[-1] < self.action_dim or action.shape[-1] % self.action_dim:
            raise ValueError(
                "PushT action blocks must contain a whole number of raw 2D "
                f"actions, got final dimension {action.shape[-1]}."
            )
        return action

    def __call__(self, sample: dict) -> dict:
        sample["state"] = self.encode_state(sample["state"])
        sample["action"] = self.encode_action(sample["action"])
        return sample


__all__ = [
    "PushTTrueStateAdapter",
]
