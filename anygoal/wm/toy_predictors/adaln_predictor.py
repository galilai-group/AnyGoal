"""AdaLN-conditioned true-state predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    ActionEmbedder,
    ConditionalBlock,
    append_rollout_step,
    prepare_planning_inputs,
    terminal_state_cost,
    validate_sequences,
)


class AdaLNActionPredictor(nn.Module):
    """Causal temporal predictor conditioned on actions through AdaLN-zero."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_frameskip: int = 1,
        action_embed_dim: int = 10,
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
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = ActionEmbedder(
            self.action_dim,
            hidden_dim,
            smoothed_dim=action_embed_dim,
        )
        self.position_embedding = nn.Parameter(torch.randn(1, history_size, hidden_dim))
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            ConditionalBlock(
                hidden_dim,
                heads,
                dim_head,
                mlp_dim,
                dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, state_dim)

    def predict_shifted(
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
        values = self.state_embedding(states.float())
        values = self.embedding_dropout(values + self.position_embedding)
        condition = self.action_embedding(actions.float())
        for block in self.blocks:
            values = block(values, condition)
        return self.output(self.norm(values))

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
        loss = F.mse_loss(shifted_predictions, shifted_targets)
        prediction = shifted_predictions[:, -1]
        future_loss = F.mse_loss(prediction, shifted_targets[:, -1])
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
        """Roll out candidates and return terminal true-state goal MSE."""
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
            prediction = self(states[:, -self.history_size :], action_window)
            states, history_actions = append_rollout_step(
                states, history_actions, action, prediction
            )
        return terminal_state_cost(prediction, goal, batch_size, num_samples)


__all__ = ["AdaLNActionPredictor"]
