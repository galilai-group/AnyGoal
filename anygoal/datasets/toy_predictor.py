"""Shared dataset utilities for true-state toy predictors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class TrueStateAdapter(Protocol):
    """Convert an environment dataset schema to state/action tensors."""

    state_dim: int
    action_dim: int
    state_keys: tuple[str, ...]
    state_names: tuple[str, ...]

    def encode_state(
        self,
        state: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...

    def encode_action(self, action: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, sample: dict) -> dict: ...


class ZScoreTransform:
    """Picklable, finite z-score transform with safe constant dimensions."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        source: str,
        epsilon: float = 1e-8,
    ) -> None:
        self.mean = torch.as_tensor(mean).float()
        self.std = torch.as_tensor(std).float().clamp_min(epsilon)
        self.source = source

    def __call__(self, sample: dict) -> dict:
        values = sample[self.source].float()
        mean, std = self.mean, self.std
        if values.shape[-1] != mean.numel():
            if values.shape[-1] % mean.numel():
                raise ValueError(
                    f"{self.source} has dimension {values.shape[-1]}, which "
                    f"is not divisible by statistic dimension {mean.numel()}."
                )
            repeats = values.shape[-1] // mean.numel()
            mean = mean.repeat(repeats)
            std = std.repeat(repeats)
        sample[self.source] = (values - mean) / std
        return sample


class MinMaxTransform:
    """Scale each variable to [0, 1] using fixed dataset extrema."""

    def __init__(
        self,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        source: str,
        epsilon: float = 1e-8,
    ) -> None:
        self.minimum = torch.as_tensor(minimum).float()
        self.scale = (
            torch.as_tensor(maximum).float() - self.minimum
        ).clamp_min(epsilon)
        self.source = source

    def __call__(self, sample: dict) -> dict:
        values = sample[self.source].float()
        minimum, scale = self.minimum, self.scale
        if values.shape[-1] != minimum.numel():
            if values.shape[-1] % minimum.numel():
                raise ValueError(
                    f"{self.source} has dimension {values.shape[-1]}, which "
                    f"is not divisible by statistic dimension {minimum.numel()}."
                )
            repeats = values.shape[-1] // minimum.numel()
            minimum = minimum.repeat(repeats)
            scale = scale.repeat(repeats)
        sample[self.source] = (values - minimum) / scale
        return sample


def _adapted_values(dataset, adapter: TrueStateAdapter, source: str) -> torch.Tensor:
    """Read and adapt one complete dataset column group."""
    if source == "state":
        state_columns = {
            key: torch.from_numpy(np.asarray(dataset.get_col_data(key))).float()
            for key in adapter.state_keys
        }
        state_input = (
            next(iter(state_columns.values()))
            if len(state_columns) == 1
            else state_columns
        )
        values = adapter.encode_state(state_input)
    elif source == "action":
        values = torch.from_numpy(np.asarray(dataset.get_col_data(source))).float()
        values = adapter.encode_action(values)
    else:
        raise ValueError(f"Unsupported canonical input column: {source!r}")

    finite = torch.isfinite(values).all(dim=-1)
    values = values[finite]
    if not len(values):
        raise ValueError(f"No finite values found in dataset column {source!r}.")
    return values


def adapted_zscore_statistics(
    dataset,
    adapter: TrueStateAdapter,
    source: str,
    constant_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-variable statistics after applying an environment adapter."""
    values = _adapted_values(dataset, adapter, source)
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    constant = (values.amax(dim=0) - values.amin(dim=0)) <= constant_epsilon
    mean[constant] = values[0, constant]
    std[constant] = 1.0
    return mean, std


def adapted_minmax_statistics(
    dataset,
    adapter: TrueStateAdapter,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-variable minima and maxima after applying an adapter."""
    values = _adapted_values(dataset, adapter, source)
    return values.amin(dim=0), values.amax(dim=0)


__all__ = [
    "MinMaxTransform",
    "TrueStateAdapter",
    "ZScoreTransform",
    "adapted_minmax_statistics",
    "adapted_zscore_statistics",
]
