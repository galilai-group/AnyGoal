"""Reusable directional-support metrics used by dataset collectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


BINS = 256
MAHALANOBIS_REFERENCE_QUANTILE = 0.99
MAHALANOBIS_TARGET_FRACTION = 0.95
KDE_REFERENCE_QUANTILE = 0.01
KDE_TARGET_FRACTION = 0.95
RANGE_TARGET_FRACTION = 0.99


@dataclass
class Feature:
    label: str
    values: np.ndarray
    periodic: bool = False


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def pooled_limits(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    lower = np.minimum(reference.min(axis=0), target.min(axis=0))
    upper = np.maximum(reference.max(axis=0), target.max(axis=0))
    span = upper - lower
    fallback = np.maximum(np.maximum(np.abs(lower), np.abs(upper)) * 0.03, 1e-6)
    padding = np.where(span > 0.0, 0.03 * span, fallback)
    return np.column_stack((lower - padding, upper + padding))


def scott_bandwidth(points: np.ndarray) -> np.ndarray:
    dimensions = points.shape[1]
    return points.std(axis=0, ddof=0) * len(points) ** (-1.0 / (dimensions + 4.0))


def density_at_points(
    points: np.ndarray, density: np.ndarray, edges: list[np.ndarray]
) -> np.ndarray:
    indices = [
        np.searchsorted(axis_edges, points[:, axis], side="right") - 1
        for axis, axis_edges in enumerate(edges)
    ]
    inside = np.ones(len(points), dtype=bool)
    for axis, axis_indices in enumerate(indices):
        inside &= (axis_indices >= 0) & (axis_indices < density.shape[axis])
    values = np.zeros(len(points), dtype=np.float64)
    values[inside] = density[tuple(axis_indices[inside] for axis_indices in indices)]
    return values


def direct_kde_support(
    reference: np.ndarray, target: np.ndarray, *, periodic: bool
) -> dict:
    dimensions = reference.shape[1]
    if periodic:
        if dimensions != 1:
            raise ValueError("Periodic KDE is only implemented for scalar angles")
        reference = wrap_radians(reference)
        target = wrap_radians(target)
        limits = np.array([[-np.pi, np.pi]])
        mean_angle = np.arctan2(
            np.sin(reference[:, 0]).mean(), np.cos(reference[:, 0]).mean()
        )
        bandwidth_points = wrap_radians(reference - mean_angle)
        mode = "wrap"
    else:
        limits = pooled_limits(reference, target)
        bandwidth_points = reference
        mode = "constant"

    histogram, edges = np.histogramdd(reference, bins=BINS, range=limits.tolist())
    widths = np.array([axis_edges[1] - axis_edges[0] for axis_edges in edges])
    bandwidth = scott_bandwidth(bandwidth_points)
    sigma_bins = np.maximum(bandwidth / widths, 0.5)
    density = gaussian_filter(
        histogram.astype(np.float32),
        sigma=sigma_bins,
        mode=mode,
        truncate=4.0,
    )
    normalization = float(density.sum(dtype=np.float64) * np.prod(widths))
    if normalization <= 0.0:
        raise ValueError("KDE has zero mass")
    density /= normalization
    reference_density = density_at_points(reference, density, edges)
    target_density = density_at_points(target, density, edges)
    threshold = float(np.quantile(reference_density, KDE_REFERENCE_QUANTILE))
    return {
        "mode": f"{dimensions}D histogram KDE",
        "bandwidth": bandwidth.tolist(),
        "sigma_bins": sigma_bins.tolist(),
        "density_threshold": threshold,
        "target_fraction": float(np.mean(target_density >= threshold)),
    }


def quantize_to_reference_bins(
    values: np.ndarray, limits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    widths = (limits[:, 1] - limits[:, 0]) / BINS
    inside = ((values >= limits[:, 0]) & (values <= limits[:, 1])).all(axis=1)
    indices = np.floor((values - limits[:, 0]) / widths).astype(np.int64)
    indices = np.clip(indices, 0, BINS - 1)
    centers = limits[:, 0] + (indices + 0.5) * widths
    return centers, inside


def weighted_log_gaussian_kde(
    reference: np.ndarray,
    log_weights: np.ndarray,
    query: np.ndarray,
    *,
    total_weight: int,
    bandwidth: float,
    log_determinant: float,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    reference_tensor = torch.from_numpy(reference.astype(np.float32, copy=False)).to(
        device
    )
    reference_norm = (reference_tensor * reference_tensor).sum(dim=1)
    log_weight_tensor = torch.from_numpy(log_weights.astype(np.float32, copy=False)).to(
        device
    )
    dimensions = reference.shape[1]
    log_normalizer = (
        np.log(total_weight)
        + 0.5 * dimensions * np.log(2.0 * np.pi)
        + dimensions * np.log(bandwidth)
        + log_determinant
    )
    inverse_two_bandwidth_squared = 0.5 / bandwidth**2
    output = np.empty(len(query), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(query), batch_size):
            stop = min(start + batch_size, len(query))
            query_tensor = torch.from_numpy(
                query[start:stop].astype(np.float32, copy=False)
            ).to(device)
            squared_distance = (
                (query_tensor * query_tensor).sum(dim=1)[:, None]
                + reference_norm[None, :]
                - 2.0 * query_tensor @ reference_tensor.T
            ).clamp_min_(0.0)
            output[start:stop] = (
                torch.logsumexp(
                    log_weight_tensor[None, :]
                    - squared_distance * inverse_two_bandwidth_squared,
                    dim=1,
                )
                .cpu()
                .numpy()
                .astype(np.float64)
                - log_normalizer
            )
    return output


def full_gaussian_kde_support(reference: np.ndarray, target: np.ndarray) -> dict:
    limits = pooled_limits(reference, reference)
    reference_quantized, reference_inside = quantize_to_reference_bins(
        reference, limits
    )
    if not reference_inside.all():
        raise RuntimeError("Reference bin limits do not contain every reference row")
    unique_reference, inverse, counts = np.unique(
        reference_quantized, axis=0, return_inverse=True, return_counts=True
    )

    mean = reference_quantized.mean(axis=0)
    covariance = np.atleast_2d(np.cov(reference_quantized, rowvar=False, bias=False))
    covariance_scale = float(np.trace(covariance) / covariance.shape[0])
    regularizer = max(covariance_scale * 1e-10, 1e-12)
    covariance = covariance + np.eye(covariance.shape[0]) * regularizer
    cholesky = np.linalg.cholesky(covariance)
    log_determinant = float(np.log(np.diag(cholesky)).sum())

    def whiten(values: np.ndarray) -> np.ndarray:
        return np.linalg.solve(cholesky, (values - mean).T).T.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    bandwidth = len(reference_quantized) ** (-1.0 / (reference.shape[1] + 4.0))
    unique_reference_white = whiten(unique_reference)
    unique_reference_density = weighted_log_gaussian_kde(
        unique_reference_white,
        np.log(counts),
        unique_reference_white,
        total_weight=len(reference_quantized),
        bandwidth=bandwidth,
        log_determinant=log_determinant,
        device=device,
    )
    threshold = float(
        np.quantile(unique_reference_density[inverse], KDE_REFERENCE_QUANTILE)
    )

    target_quantized, target_inside = quantize_to_reference_bins(target, limits)
    target_density = np.full(len(target), -np.inf, dtype=np.float64)
    inside_indices = np.flatnonzero(target_inside)
    unique_target, target_inverse = np.unique(
        target_quantized[target_inside], axis=0, return_inverse=True
    )
    if len(unique_target):
        unique_target_density = weighted_log_gaussian_kde(
            unique_reference_white,
            np.log(counts),
            whiten(unique_target),
            total_weight=len(reference_quantized),
            bandwidth=bandwidth,
            log_determinant=log_determinant,
            device=device,
        )
        target_density[inside_indices] = unique_target_density[target_inverse]
    return {
        "mode": (
            f"full-covariance {reference.shape[1]}D Gaussian KDE after "
            "256-bin quantization"
        ),
        "target_fraction": float(np.mean(target_density >= threshold)),
        "log_density_threshold": threshold,
        "bandwidth": bandwidth,
        "covariance_regularizer": regularizer,
        "reference_unique_centers": int(len(unique_reference)),
        "target_unique_centers": int(len(unique_target)),
        "device": str(device),
    }


def kde_support(reference: np.ndarray, target: np.ndarray, periodic: bool) -> dict:
    dimensions = reference.shape[1]
    if dimensions <= 3:
        return direct_kde_support(reference, target, periodic=periodic)
    if periodic:
        raise ValueError(
            "Periodic features with more than one dimension are unsupported"
        )
    return full_gaussian_kde_support(reference, target)


def regularized_covariance(points: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = np.atleast_2d(np.cov(points, rowvar=False, bias=True))
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularizer = max(scale * 1e-8, 1e-12)
    return covariance + np.eye(covariance.shape[0]) * regularizer, regularizer


def mahalanobis_support(
    reference: np.ndarray, target: np.ndarray, periodic: bool
) -> dict:
    if periodic:
        reference_angles = wrap_radians(reference)[:, 0]
        target_angles = wrap_radians(target)[:, 0]
        reference = np.column_stack(
            (np.cos(reference_angles), np.sin(reference_angles))
        )
        target = np.column_stack((np.cos(target_angles), np.sin(target_angles)))
        representation = "cosine/sine circular embedding"
    else:
        representation = "stored feature coordinates"
    mean = reference.mean(axis=0)
    covariance, regularizer = regularized_covariance(reference)
    inverse = np.linalg.inv(covariance)

    def squared_distance(points: np.ndarray) -> np.ndarray:
        centered = points - mean
        return np.einsum("ni,ij,nj->n", centered, inverse, centered, optimize=True)

    threshold = float(
        np.quantile(squared_distance(reference), MAHALANOBIS_REFERENCE_QUANTILE)
    )
    return {
        "representation": representation,
        "covariance_regularizer": regularizer,
        "squared_distance_threshold": threshold,
        "target_fraction": float(np.mean(squared_distance(target) <= threshold)),
    }


def evaluate_feature(reference: Feature, target: Feature) -> dict:
    if reference.label != target.label:
        raise ValueError(f"Feature mismatch: {reference.label} != {target.label}")
    if reference.values.shape[1] != target.values.shape[1]:
        raise ValueError(f"Dimension mismatch for {reference.label}")
    if reference.periodic != target.periodic:
        raise ValueError(f"Periodicity mismatch for {reference.label}")

    mahalanobis = mahalanobis_support(
        reference.values, target.values, reference.periodic
    )
    kde = kde_support(reference.values, target.values, reference.periodic)
    if reference.periodic:
        range_result = {"ignored_for_periodic_variable": True, "target_fraction": None}
        range_pass = None
    else:
        lower = reference.values.min(axis=0)
        upper = reference.values.max(axis=0)
        inside = np.all((target.values >= lower) & (target.values <= upper), axis=1)
        range_fraction = float(np.mean(inside))
        range_result = {
            "ignored_for_periodic_variable": False,
            "reference_min": lower.tolist(),
            "reference_max": upper.tolist(),
            "target_fraction": range_fraction,
        }
        range_pass = range_fraction >= RANGE_TARGET_FRACTION

    criteria = {
        "mahalanobis": mahalanobis["target_fraction"] >= MAHALANOBIS_TARGET_FRACTION,
        "kde": kde["target_fraction"] >= KDE_TARGET_FRACTION,
        "range": range_pass,
    }
    verdict = all(value for value in criteria.values() if value is not None)
    return {
        "dimensions": int(reference.values.shape[1]),
        "periodic": reference.periodic,
        "reference_rows": int(len(reference.values)),
        "target_rows": int(len(target.values)),
        "mahalanobis": mahalanobis,
        "kde": kde,
        "range": range_result,
        "criteria": criteria,
        "final_verdict": bool(verdict),
    }
