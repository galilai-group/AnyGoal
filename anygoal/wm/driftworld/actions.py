"""Action normalization for the DriftWorld integration."""

from __future__ import annotations

import numpy as np
import torch
from sklearn import preprocessing


class DriftWorldActionProcessor:
    """Standardize CEM actions while keeping DriftWorld and the env in raw units."""

    def __init__(self, mean=None, scale=None) -> None:
        mean = np.zeros(2, dtype=np.float32) if mean is None else mean
        scale = np.ones(2, dtype=np.float32) if scale is None else scale
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != self.scale.shape or self.mean.ndim != 1:
            raise ValueError("Action mean and scale must be same-sized 1-D arrays.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Action mean and scale must be finite.")
        if np.any(self.scale <= 0):
            raise ValueError("Action scale must be positive.")

    @classmethod
    def fit(cls, values) -> "DriftWorldActionProcessor":
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("Action statistics require a 2-D array.")
        valid = values[~np.isnan(values).any(axis=1)]
        if not len(valid):
            raise ValueError("Cannot fit action statistics without non-NaN rows.")
        scaler = preprocessing.StandardScaler().fit(valid)
        return cls(mean=scaler.mean_, scale=scaler.scale_)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / self.scale

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return values * self.scale + self.mean

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
        return values * scale + mean
