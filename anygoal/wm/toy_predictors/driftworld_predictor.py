"""Predictor-only DriftWorld operating on normalized true-state vectors."""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .common import prepare_planning_inputs, terminal_state_cost


@lru_cache(maxsize=1)
def _official_drift_loss_function():
    source = (
        Path(__file__).resolve().parents[3]
        / "worldmodels"
        / "driftworld"
        / "pusht_driftworld"
        / "drifting"
        / "drift_loss_indep.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Official PushT drifting loss not found: {source}")
    spec = importlib.util.spec_from_file_location(
        "anygoal.wm._official_drift_loss_indep",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official PushT drifting loss: {source}")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module.drift_loss


def _official_drift_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    temperatures: tuple[float, ...],
) -> tuple[torch.Tensor, dict]:
    """Call the vendored PushT loss without copying or changing its math."""
    return _official_drift_loss_function()(
        gen=generated,
        fixed_pos=target,
        R_list=temperatures,
    )


class ResidualFiLMBlock(nn.Module):
    """Layer-normalized residual MLP with per-block action FiLM."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        residual = self.norm(values)
        residual = (1.0 + gamma) * residual + beta
        residual = self.linear_in(residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.linear_out(residual)
        return values + residual


class DriftWorldStateNetwork(nn.Module):
    """Small vector backbone replacing DriftWorld's image-space 2D U-Net."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_size: int = 3,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        action_embed_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, history_size, hidden_dim) <= 0:
            raise ValueError(
                "State, action, history, and hidden dimensions must be positive."
            )
        if num_res_blocks <= 0 or action_embed_dim <= 0:
            raise ValueError(
                "Residual-block and action-embedding sizes must be positive."
            )
        if activation.lower() != "silu" or norm.lower() != "layernorm":
            raise ValueError("The DriftWorld state backbone supports SiLU and LayerNorm.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(self.history_size * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.noise_encoder = nn.Linear(self.state_dim, self.hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_dim, action_embed_dim),
            nn.SiLU(),
            nn.Linear(action_embed_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            ResidualFiLMBlock(self.hidden_dim, dropout=dropout)
            for _ in range(num_res_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def forward(
        self,
        noise: torch.Tensor,
        state_history: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noise.shape[0]
        if noise.shape != (batch_size, self.state_dim):
            raise ValueError(
                f"Expected noise shaped (B,{self.state_dim}), got {tuple(noise.shape)}."
            )
        if state_history.shape != (
            batch_size,
            self.history_size,
            self.state_dim,
        ):
            raise ValueError(
                "Expected state history shaped "
                f"(B,{self.history_size},{self.state_dim}), got "
                f"{tuple(state_history.shape)}."
            )
        if action.shape != (batch_size, self.action_dim):
            raise ValueError(
                f"Expected action shaped (B,{self.action_dim}), got "
                f"{tuple(action.shape)}."
            )

        history = self.state_encoder(state_history).flatten(start_dim=1)
        history = self.history_projection(history)
        noisy_state = self.noise_encoder(noise)
        values = self.input_projection(torch.cat((noisy_state, history), dim=-1))
        condition = self.action_embedding(action)
        for block in self.blocks:
            values = block(values, condition)
        return self.output(values)


class DriftWorldStatePredictor(nn.Module):
    """Stochastic true-state DriftWorld generator and drifting objective.

    The official PushT model is a one-pass generator: it maps independent
    Gaussian noise, history, and actions to generated samples. Its drifting
    loss then moves the generated particle set toward the observed target.
    There is no diffusion timestep or target-corruption schedule in that
    implementation, so this predictor deliberately does not invent one.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_frameskip: int = 1,
        history_size: int = 3,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        time_embed_dim: int | None = None,
        action_embed_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm: str = "layernorm",
        temperatures: Sequence[float] = (0.02, 0.05, 0.2),
        n_neg: int = 16,
        decay: float = 0.999,
        rollout_seed: int = 0,
        num_rollout_samples: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.raw_action_dim = int(action_dim)
        self.action_frameskip = int(action_frameskip)
        if self.raw_action_dim <= 0 or self.action_frameskip <= 0:
            raise ValueError("action_dim and action_frameskip must be positive.")
        self.action_dim = self.raw_action_dim * self.action_frameskip
        self.history_size = int(history_size)
        self.n_neg = int(n_neg)
        self.decay = float(decay)
        self.rollout_seed = int(rollout_seed)
        self.num_rollout_samples = int(num_rollout_samples)
        self.temperatures = tuple(float(value) for value in temperatures)
        if self.n_neg <= 1:
            raise ValueError("n_neg must exceed one for the drifting particle set.")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        if self.num_rollout_samples <= 0:
            raise ValueError("num_rollout_samples must be positive.")
        if not self.temperatures or any(value <= 0 for value in self.temperatures):
            raise ValueError("Drifting temperatures must be positive.")
        if time_embed_dim is not None:
            raise ValueError(
                "Official PushT DriftWorld has no drifting-time input. Set "
                "time_embed_dim=null rather than adding a new conditioning variable."
            )

        self.inner_model = DriftWorldStateNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            history_size=self.history_size,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            action_embed_dim=action_embed_dim,
            dropout=dropout,
            activation=activation,
            norm=norm,
        )
        self.ema_model = copy.deepcopy(self.inner_model)
        self.ema_model.requires_grad_(False)

    def forward(
        self,
        noise: torch.Tensor,
        state_history: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the lower-level drifting generator used by the objective."""
        return self.inner_model(noise, state_history, action)

    def _validate_training_sequences(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        expected_steps = self.history_size + 1
        if states.ndim != 3 or states.shape[1:] != (
            expected_steps,
            self.state_dim,
        ):
            raise ValueError(
                f"Expected states shaped (B,{expected_steps},{self.state_dim}), "
                f"got {tuple(states.shape)}."
            )
        if actions.ndim != 3 or actions.shape[0] != states.shape[0]:
            raise ValueError("Actions must have shape (B,T,action_dim).")
        if actions.shape[1] < self.history_size or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected at least {self.history_size} actions with final dimension "
                f"{self.action_dim}, got {tuple(actions.shape)}."
            )

    def drifting_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if generated.ndim != 3 or generated.shape[-1] != self.state_dim:
            raise ValueError("Generated particles must have shape (B,N,state_dim).")
        if target.shape != (generated.shape[0], 1, self.state_dim):
            raise ValueError("Drifting targets must have shape (B,1,state_dim).")
        losses, metrics = _official_drift_loss(
            generated,
            target,
            self.temperatures,
        )
        return losses.mean(), metrics

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Apply the official particle drifting objective to one future state."""
        self._validate_training_sequences(states, actions)
        states = states.float()
        actions = actions.float()
        batch_size = states.shape[0]
        history = states[:, : self.history_size]
        action = actions[:, self.history_size - 1]
        target = states[:, self.history_size : self.history_size + 1]

        repeated_history = history.repeat_interleave(self.n_neg, dim=0)
        repeated_action = action.repeat_interleave(self.n_neg, dim=0)
        noise = torch.randn(
            batch_size * self.n_neg,
            self.state_dim,
            device=states.device,
            dtype=states.dtype,
            generator=generator,
        )
        generated = self.inner_model(noise, repeated_history, repeated_action)
        generated = generated.reshape(batch_size, self.n_neg, self.state_dim)
        loss, drift_metrics = self.drifting_loss(generated, target)
        prediction = generated[:, 0]
        future_loss = F.mse_loss(prediction, target[:, 0])
        return {
            "loss": loss,
            "future_loss": future_loss.detach(),
            "prediction": prediction,
            "generated": generated,
            "drift_metrics": drift_metrics,
        }

    @torch.no_grad()
    def update_ema(self) -> None:
        """Match the official parameter-only EMA update."""
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.inner_model.parameters()
        ):
            ema_parameter.lerp_(parameter, 1.0 - self.decay)

    @torch.inference_mode()
    def predict_next_state(
        self,
        state_history: torch.Tensor,
        action: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Generate one normalized next state from a fresh Gaussian sample."""
        if action.ndim == 3:
            action = action[:, -1]
        parameter = next(self.inner_model.parameters())
        history = state_history.to(device=parameter.device, dtype=parameter.dtype)
        action = action.to(device=parameter.device, dtype=parameter.dtype)
        if history.ndim != 3:
            raise ValueError("state_history must have shape (B,history_size,state_dim).")
        if noise is None:
            noise = torch.randn(
                history.shape[0],
                self.state_dim,
                device=history.device,
                dtype=history.dtype,
                generator=generator,
            )
        else:
            noise = noise.to(device=history.device, dtype=history.dtype)
        network = self.ema_model if use_ema else self.inner_model
        return network(noise, history, action)

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Stochastically roll out candidates with reproducible common noise."""
        (
            initial_states,
            _,
            candidates,
            goal,
            batch_size,
            num_samples,
        ) = prepare_planning_inputs(self, info_dict, action_candidates)
        costs = []
        for rollout_index in range(self.num_rollout_samples):
            states = initial_states
            prediction = states[:, -1]
            generator = torch.Generator(device=states.device).manual_seed(
                self.rollout_seed + rollout_index
            )
            for action in candidates.unbind(dim=1):
                base_noise = torch.randn(
                    batch_size,
                    1,
                    self.state_dim,
                    generator=generator,
                    device=states.device,
                    dtype=states.dtype,
                )
                noise = base_noise.expand(-1, num_samples, -1).reshape(
                    batch_size * num_samples, self.state_dim
                )
                prediction = self.predict_next_state(
                    states[:, -self.history_size :],
                    action,
                    noise=noise,
                )
                states = torch.cat((states, prediction[:, None]), dim=1)
            costs.append(
                terminal_state_cost(prediction, goal, batch_size, num_samples)
            )
        return torch.stack(costs).mean(dim=0)


__all__ = [
    "DriftWorldStateNetwork",
    "DriftWorldStatePredictor",
    "ResidualFiLMBlock",
]
