"""Diagnose TwoRoom triangle support failures by route and condition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import distribution_support as support  # noqa: E402


def support_masks(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0)
    covariance, _ = support.regularized_covariance(reference)
    inverse = np.linalg.inv(covariance)

    def squared_distance(points: np.ndarray) -> np.ndarray:
        centered = points - mean
        return np.einsum("ni,ij,nj->n", centered, inverse, centered, optimize=True)

    mahalanobis_threshold = np.quantile(
        squared_distance(reference), support.MAHALANOBIS_REFERENCE_QUANTILE
    )
    mahalanobis_mask = squared_distance(target) <= mahalanobis_threshold

    limits = support.pooled_limits(reference, target)
    histogram, edges = np.histogramdd(
        reference, bins=support.BINS, range=limits.tolist()
    )
    widths = np.asarray([edge[1] - edge[0] for edge in edges])
    bandwidth = support.scott_bandwidth(reference)
    density = gaussian_filter(
        histogram.astype(np.float32),
        sigma=np.maximum(bandwidth / widths, 0.5),
        mode="constant",
        truncate=4.0,
    )
    density /= float(density.sum(dtype=np.float64) * np.prod(widths))
    reference_density = support.density_at_points(reference, density, edges)
    target_density = support.density_at_points(target, density, edges)
    density_threshold = np.quantile(reference_density, support.KDE_REFERENCE_QUANTILE)
    kde_mask = target_density >= density_threshold
    return mahalanobis_mask, kde_mask


def describe(name: str, values: np.ndarray) -> None:
    quantiles = np.quantile(values, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], axis=0)
    print(f"{name} rows={len(values)}")
    print("  quantiles [min,p01,p05,p50,p95,p99,max]=")
    print(np.array2string(quantiles, precision=3, suppress_small=True))


def group_report(
    label: str,
    mahalanobis_mask: np.ndarray,
    kde_mask: np.ndarray,
    route_ids: np.ndarray,
    condition_ids: np.ndarray,
    waypoint_indices: np.ndarray,
) -> None:
    print(f"\n{label} overall mahalanobis={mahalanobis_mask.mean():.6f} kde={kde_mask.mean():.6f}")
    for route_id in np.unique(route_ids):
        mask = route_ids == route_id
        print(
            f"  route={int(route_id)} rows={int(mask.sum())} "
            f"mahalanobis={mahalanobis_mask[mask].mean():.6f} "
            f"kde={kde_mask[mask].mean():.6f}"
        )
    for condition_id in np.unique(condition_ids):
        mask = condition_ids == condition_id
        print(
            f"  condition={int(condition_id)} rows={int(mask.sum())} "
            f"mahalanobis={mahalanobis_mask[mask].mean():.6f} "
            f"kde={kde_mask[mask].mean():.6f}"
        )
    for waypoint_index in np.unique(waypoint_indices):
        mask = waypoint_indices == waypoint_index
        print(
            f"  segment_target={int(waypoint_index)} rows={int(mask.sum())} "
            f"mahalanobis={mahalanobis_mask[mask].mean():.6f} "
            f"kde={kde_mask[mask].mean():.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.reference) as reference_file:
        reference_position = np.asarray(reference_file["proprio"][:], dtype=np.float64)
        reference_action = np.asarray(reference_file["action"][:], dtype=np.float64)
    with h5py.File(args.target) as target_file:
        target_position = np.asarray(target_file["proprio"][:], dtype=np.float64)
        target_action = np.asarray(target_file["action"][:], dtype=np.float64)
        route_ids = target_file["route_id"][:]
        condition_ids = target_file["condition_id"][:]
        waypoint_indices = target_file["waypoint_index"][:]

    reference_position = reference_position[np.isfinite(reference_position).all(axis=1)]
    reference_action = reference_action[np.isfinite(reference_action).all(axis=1)]

    describe("reference position", reference_position)
    describe("target position", target_position)
    describe("reference action", reference_action)
    describe("target action", target_action)

    position_mahalanobis, position_kde = support_masks(
        reference_position, target_position
    )
    action_mahalanobis, action_kde = support_masks(reference_action, target_action)
    group_report(
        "agent pos (2D)",
        position_mahalanobis,
        position_kde,
        route_ids,
        condition_ids,
        waypoint_indices,
    )

    print("\nisotropic inward coordinate candidates")
    for scale in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        candidate = 112.0 + scale * (target_position - 112.0)
        candidate_mahalanobis, candidate_kde = support_masks(
            reference_position, candidate
        )
        print(
            f"  scale={scale:.2f} mahalanobis={candidate_mahalanobis.mean():.6f} "
            f"kde={candidate_kde.mean():.6f}"
        )

    clean_direction = np.asarray(target_action, dtype=np.float64).copy()
    waypoint_delta = np.zeros_like(target_position)
    with h5py.File(args.target) as target_file:
        waypoint_delta[:] = target_file["waypoint"][:] - target_position
    waypoint_norm = np.linalg.norm(waypoint_delta, axis=1, keepdims=True)
    valid_direction = waypoint_norm[:, 0] > 1e-8
    clean_direction[valid_direction] = (
        waypoint_delta[valid_direction] / waypoint_norm[valid_direction]
    )
    magnitudes = np.linalg.norm(target_action, axis=1, keepdims=True)
    noisy_rows = condition_ids == 1
    print("\nnoise-direction shrinkage candidates (alpha=old noisy direction weight)")
    for alpha in (0.75, 0.50, 0.25, 0.0):
        candidate = target_action.copy()
        blended = (
            (1.0 - alpha) * clean_direction[noisy_rows]
            + alpha * target_action[noisy_rows]
        )
        blended_norm = np.linalg.norm(blended, axis=1, keepdims=True)
        candidate[noisy_rows] = (
            blended / np.maximum(blended_norm, 1e-8) * magnitudes[noisy_rows]
        )
        candidate_mahalanobis, candidate_kde = support_masks(
            reference_action, candidate
        )
        print(
            f"  alpha={alpha:.2f} mahalanobis={candidate_mahalanobis.mean():.6f} "
            f"kde={candidate_kde.mean():.6f}"
        )
    group_report(
        "action (2D)",
        action_mahalanobis,
        action_kde,
        route_ids,
        condition_ids,
        waypoint_indices,
    )

    position_failed = ~(position_mahalanobis & position_kde)
    if position_failed.any():
        failed = target_position[position_failed]
        print("\nposition rows failing either criterion by coordinate region")
        for axis, axis_name in enumerate("xy"):
            low = failed[:, axis] <= np.quantile(target_position[:, axis], 0.1)
            high = failed[:, axis] >= np.quantile(target_position[:, axis], 0.9)
            print(
                f"  {axis_name}: failed_low_decile={int(low.sum())} "
                f"failed_high_decile={int(high.sum())}"
            )

    for name, actions in (("reference", reference_action), ("target", target_action)):
        magnitudes = np.linalg.norm(actions, axis=1)
        angles = np.arctan2(actions[:, 1], actions[:, 0])
        bins = np.linspace(-np.pi, np.pi, 9)
        counts, _ = np.histogram(angles, bins=bins)
        print(
            f"\n{name} action magnitude q01/q50/q99="
            f"{np.quantile(magnitudes, [0.01, 0.5, 0.99])}"
        )
        print(f"{name} action octant fractions={counts / counts.sum()}")


if __name__ == "__main__":
    main()
