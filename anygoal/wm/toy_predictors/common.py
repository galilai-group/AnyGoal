"""Shared transformer blocks and planning utilities for toy predictors."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _modulate(
    values: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return values * (1 + scale) + shift


def validate_sequences(
    states: torch.Tensor,
    actions: torch.Tensor,
    state_dim: int,
    action_dim: int,
    expected_frames: int,
) -> None:
    if states.ndim != 3 or states.shape[-1] != state_dim:
        raise ValueError(
            f"Expected states shaped (B,T,{state_dim}), got {tuple(states.shape)}."
        )
    if actions.ndim != 3 or actions.shape[-1] != action_dim:
        raise ValueError(
            f"Expected actions shaped (B,T,{action_dim}), got {tuple(actions.shape)}."
        )
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError("State and action batch/time dimensions must match.")
    if states.shape[1] != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {states.shape[1]}.")


class MultiHeadSelfAttention(nn.Module):
    """LeWM-style attention with an explicit per-head dimension."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = float(dropout)

    def forward(
        self,
        values: torch.Tensor,
        *,
        causal: bool = False,
        allowed_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.norm(values)
        query, key, value = self.to_qkv(values).chunk(3, dim=-1)
        query, key, value = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads)
            for tensor in (query, key, value)
        )
        attention_mask = None
        if allowed_attention is not None:
            attention_mask = allowed_attention.to(values.device)[None, None]
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        return self.to_out(rearrange(output, "b h n d -> b n (h d)"))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)

    def forward(
        self,
        values: torch.Tensor,
        *,
        causal: bool = False,
        allowed_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = values + self.attention(
            values,
            causal=causal,
            allowed_attention=allowed_attention,
        )
        return values + self.feed_forward(values)


class ConditionalBlock(nn.Module):
    """AdaLN-zero block matching the conditioning path used by LeWM."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(condition).chunk(6, dim=-1)
        attention_input = _modulate(
            self.norm_attention(values), shift_attention, scale_attention
        )
        values = values + gate_attention * self.attention(attention_input, causal=True)
        mlp_input = _modulate(self.norm_mlp(values), shift_mlp, scale_mlp)
        return values + gate_mlp * self.feed_forward(mlp_input)


class ActionEmbedder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        smoothed_dim: int = 10,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv1d(action_dim, smoothed_dim, kernel_size=1)
        self.embedding = nn.Sequential(
            nn.Linear(smoothed_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        actions = self.projection(actions.transpose(1, 2)).transpose(1, 2)
        return self.embedding(actions)


def prepare_planning_inputs(
    model: nn.Module,
    info_dict: dict,
    action_candidates: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    """Flatten CEM's batch/sample axes and validate canonical planning inputs."""
    if "state" not in info_dict or "goal_state" not in info_dict:
        raise KeyError("Planning requires state and goal_state in info_dict.")
    if action_candidates.ndim != 4:
        raise ValueError("action_candidates must have shape (B,N,horizon,action_dim).")

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    candidates = action_candidates.to(device=device, dtype=dtype)
    batch_size, num_samples, _, candidate_dim = candidates.shape
    if candidate_dim != model.action_dim:
        raise ValueError(
            f"Expected candidate action-block dimension {model.action_dim}, "
            f"got {candidate_dim}. plan_config.action_block must match the "
            "predictor's action_frameskip."
        )

    states = info_dict["state"].to(device=device, dtype=dtype)
    goal = info_dict["goal_state"].to(device=device, dtype=dtype)
    if states.ndim == 3:
        states = states[:, None].expand(-1, num_samples, -1, -1)
    if goal.ndim == 3:
        goal = goal[:, None].expand(-1, num_samples, -1, -1)
    if states.ndim != 4 or states.shape[:2] != (batch_size, num_samples):
        raise ValueError("state must have CEM-expanded shape (B,N,T,state_dim).")
    if goal.ndim != 4 or goal.shape[:2] != (batch_size, num_samples):
        raise ValueError("goal_state must have CEM-expanded shape (B,N,T,state_dim).")
    if states.shape[-1] != model.state_dim or goal.shape[-1] != model.state_dim:
        raise ValueError(
            f"Planning states must use the model's {model.state_dim}D representation."
        )

    if states.shape[2] < model.history_size:
        padding = states[:, :, :1].expand(
            -1, -1, model.history_size - states.shape[2], -1
        )
        states = torch.cat((padding, states), dim=2)
    states = states[:, :, -model.history_size :]

    required_actions = max(model.history_size - 1, 0)
    history_actions = info_dict.get("action")
    if history_actions is None:
        history_actions = states.new_zeros(
            batch_size, num_samples, required_actions, model.action_dim
        )
    else:
        history_actions = history_actions.to(device=device, dtype=dtype)
        if history_actions.ndim == 3:
            history_actions = history_actions[:, None].expand(-1, num_samples, -1, -1)
        if history_actions.ndim != 4 or history_actions.shape[:2] != (
            batch_size,
            num_samples,
        ):
            raise ValueError(
                "action history must have CEM-expanded shape (B,N,T,action_dim)."
            )
        if history_actions.shape[-1] != model.action_dim:
            raise ValueError(
                f"Expected action history dim {model.action_dim}, got "
                f"{history_actions.shape[-1]}."
            )
        history_actions = history_actions[:, :, -required_actions:]
        if history_actions.shape[2] < required_actions:
            padding = history_actions.new_zeros(
                batch_size,
                num_samples,
                required_actions - history_actions.shape[2],
                model.action_dim,
            )
            history_actions = torch.cat((padding, history_actions), dim=2)

    states = rearrange(states, "b n t d -> (b n) t d").contiguous()
    history_actions = rearrange(history_actions, "b n t d -> (b n) t d").contiguous()
    candidates = rearrange(candidates, "b n t d -> (b n) t d").contiguous()
    goal = rearrange(goal[:, :, -1], "b n d -> (b n) d").contiguous()
    return (
        states,
        history_actions,
        candidates,
        goal,
        batch_size,
        num_samples,
    )


def append_rollout_step(
    states: torch.Tensor,
    history_actions: torch.Tensor,
    action: torch.Tensor,
    predicted_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat((states, predicted_state[:, None]), dim=1),
        torch.cat((history_actions, action[:, None]), dim=1),
    )


def terminal_state_cost(
    predicted_state: torch.Tensor,
    goal_state: torch.Tensor,
    batch_size: int,
    num_samples: int,
) -> torch.Tensor:
    cost = (predicted_state - goal_state).square().sum(dim=-1)
    return rearrange(cost, "(b n) -> b n", b=batch_size, n=num_samples)


__all__ = [
    "ActionEmbedder",
    "ConditionalBlock",
    "TransformerBlock",
    "append_rollout_step",
    "count_trainable_parameters",
    "prepare_planning_inputs",
    "terminal_state_cost",
    "validate_sequences",
]
