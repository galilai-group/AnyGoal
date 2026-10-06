"""Action-duplicating causal true-state predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .common import (
    TransformerBlock,
    append_rollout_step,
    prepare_planning_inputs,
    terminal_state_cost,
    validate_sequences,
)


class ActionDuplicateCausalPredictor(nn.Module):
    """DINO-WM-style scalar state tokens with duplicated action features."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_frameskip: int = 1,
        history_size: int = 3,
        hidden_dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.raw_action_dim = int(action_dim)
        self.action_frameskip = int(action_frameskip)
        if self.raw_action_dim <= 0 or self.action_frameskip <= 0:
            raise ValueError("action_dim and action_frameskip must be positive.")
        self.action_dim = self.raw_action_dim * self.action_frameskip
        self.history_size = int(history_size)
        self.token_dim = 1 + self.action_dim
        self.input_embedding = nn.Linear(self.token_dim, hidden_dim)
        self.time_embedding = nn.Parameter(torch.randn(1, history_size, 1, hidden_dim))
        self.variable_embedding = nn.Parameter(torch.randn(1, 1, state_dim, hidden_dim))
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.token_dim)
        token_times = torch.arange(history_size).repeat_interleave(state_dim)
        self.register_buffer(
            "allowed_attention",
            token_times[:, None] >= token_times[None, :],
            persistent=False,
        )

    def build_tokens(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        validate_sequences(
            states,
            actions,
            self.state_dim,
            self.action_dim,
            self.history_size,
        )
        duplicated_actions = actions.unsqueeze(2).expand(-1, -1, self.state_dim, -1)
        return torch.cat((states.unsqueeze(-1), duplicated_actions), dim=-1)

    def predict_shifted(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.build_tokens(states.float(), actions.float())
        values = self.input_embedding(tokens)
        values = self.embedding_dropout(
            values + self.time_embedding + self.variable_embedding
        )
        values = rearrange(values, "b t s d -> b (t s) d")
        for block in self.blocks:
            values = block(values, allowed_attention=self.allowed_attention)
        values = rearrange(
            self.norm(values),
            "b (t s) d -> b t s d",
            t=self.history_size,
            s=self.state_dim,
        )
        return self.output(values)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_shifted(states, actions)[:, -1]

    def loss(self, states: torch.Tensor, actions: torch.Tensor) -> dict:
        shifted_predictions = self.predict_shifted(
            states[:, : self.history_size], actions[:, : self.history_size]
        )
        shifted_targets = states[:, 1 : self.history_size + 1].float()
        loss = F.mse_loss(shifted_predictions[..., 0], shifted_targets)
        prediction = shifted_predictions[:, -1]
        future_loss = F.mse_loss(prediction[..., 0], shifted_targets[:, -1])
        return {
            "loss": loss,
            "future_loss": future_loss.detach(),
            "prediction": prediction,
            "shifted_predictions": shifted_predictions,
        }

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out state channel zero and discard predicted action channels."""
        (
            states,
            history_actions,
            candidates,
            goal,
            batch_size,
            num_samples,
        ) = prepare_planning_inputs(self, info_dict, action_candidates)
        prediction = states[:, -1]
        for action in candidates.unbind(dim=1):
            action_window = torch.cat(
                (history_actions[:, -(self.history_size - 1) :], action[:, None]),
                dim=1,
            )
            prediction = self(states[:, -self.history_size :], action_window)[..., 0]
            states, history_actions = append_rollout_step(
                states, history_actions, action, prediction
            )
        return terminal_state_cost(prediction, goal, batch_size, num_samples)


__all__ = ["ActionDuplicateCausalPredictor"]
