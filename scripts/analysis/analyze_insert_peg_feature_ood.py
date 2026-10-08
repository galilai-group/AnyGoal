"""Report marginal OOD values and phases for selected insert-peg concepts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis import distribution_support as support


def support_masks(reference: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    mean = reference.mean(axis=0)
    covariance, _ = support.regularized_covariance(reference)
    inverse = np.linalg.inv(covariance)

    def squared_distance(points: np.ndarray) -> np.ndarray:
        centered = points - mean
        return np.einsum("ni,ij,nj->n", centered, inverse, centered, optimize=True)

    mahalanobis_threshold = np.quantile(
        squared_distance(reference), support.MAHALANOBIS_REFERENCE_QUANTILE
    )
    limits = support.pooled_limits(reference, target)
    histogram, edges = np.histogramdd(
        reference, bins=support.BINS, range=limits.tolist()
    )
    widths = np.array([axis_edges[1] - axis_edges[0] for axis_edges in edges])
    sigma_bins = np.maximum(support.scott_bandwidth(reference) / widths, 0.5)
    density = gaussian_filter(
        histogram.astype(np.float32), sigma=sigma_bins, mode="constant", truncate=4.0
    )
    density /= float(density.sum(dtype=np.float64) * np.prod(widths))
    reference_density = support.density_at_points(reference, density, edges)
    target_density = support.density_at_points(target, density, edges)
    density_threshold = np.quantile(
        reference_density, support.KDE_REFERENCE_QUANTILE
    )
    return {
        "mahalanobis": squared_distance(target) <= mahalanobis_threshold,
        "kde": target_density >= density_threshold,
        "range": np.all(
            (target >= reference.min(axis=0))
            & (target <= reference.max(axis=0)),
            axis=1,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--episode-length", type=int, required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phases = {}
    for value in args.phase:
        name, start, end = value.split(":")
        phases[name] = (int(start), int(end))
    reference = {item.label: item.values for item in support.load_features(args.reference, "insert_peg")}
    target = {item.label: item.values for item in support.load_features(args.target, "insert_peg")}
    step = np.arange(len(next(iter(target.values())))) % args.episode_length
    result = {}
    for label in args.feature:
        masks = support_masks(reference[label], target[label])
        values = target[label][:, 0] if target[label].shape[1] == 1 else np.arctan2(target[label][:, 1], target[label][:, 0])
        result[label] = {}
        for metric, mask in masks.items():
            entry = {"id_fraction": float(mask.mean())}
            if np.any(~mask):
                entry["ood_value_quantiles"] = np.quantile(values[~mask], [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist()
            entry["phase_id_fraction"] = {
                name: float(mask[(step >= start) & (step < end)].mean())
                for name, (start, end) in phases.items()
            }
            result[label][metric] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
