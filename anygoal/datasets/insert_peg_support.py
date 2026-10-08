"""Reference-support helpers for generated Insert-Peg variants."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_SLICES = {
    "arm": slice(0, 16),
    "hand": slice(29, 33),
    "object": slice(33, 37),
    "target": slice(40, 44),
}


def _distribution_support_module():
    path = REPO_ROOT / "scripts/analysis/distribution_support.py"
    spec = importlib.util.spec_from_file_location("peg_distribution_support", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(path: Path, successful_only: bool = False):
    """Return aligned 44-D states and the five support feature arrays."""
    with h5py.File(path, "r") as dataset:
        states = dataset["observation"][:].astype(np.float64)
        actions = dataset["action"][:].astype(np.float64)
        if successful_only:
            required = ("reward", "ep_len", "ep_offset")
            if not all(name in dataset for name in required):
                raise ValueError(f"{path} lacks successful-episode metadata")
            rewards = dataset["reward"][:]
            keep = np.zeros(len(states), dtype=bool)
            for offset, length in zip(dataset["ep_offset"][:], dataset["ep_len"][:]):
                episode = slice(int(offset), int(offset + length))
                if np.max(rewards[episode], initial=0.0) >= 1.0 - 1e-7:
                    keep[episode] = True
            states, actions = states[keep], actions[keep]
            if not len(states):
                raise ValueError(f"{path} contains no successful episodes")
    features = {name: states[:, columns] for name, columns in FIELD_SLICES.items()}
    features["action"] = actions
    return states, features


class ReferenceSupport:
    """Fast reference-only version of distribution_support's three gates."""

    def __init__(self, references: dict[str, np.ndarray], cache_path: Path):
        support = _distribution_support_module()
        self.bins = support.BINS
        self.models = {}
        if cache_path.is_file():
            with np.load(cache_path) as saved:
                for name in references:
                    self.models[name] = {
                        key: saved[f"{name}_{key}"]
                        for key in (
                            "pairs", "edges", "density", "threshold", "mean",
                            "inverse", "mahal", "low", "high",
                        )
                    }
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for name, values in references.items():
            pairs = np.array(list(itertools.combinations(range(values.shape[1]), 2)))
            densities, edges, thresholds = [], [], []
            for pair in pairs:
                points = values[:, pair]
                limits = support.pooled_limits(points, points)
                histogram, pair_edges = np.histogramdd(
                    points, bins=self.bins, range=limits.tolist()
                )
                widths = np.array([edge[1] - edge[0] for edge in pair_edges])
                sigma = np.maximum(support.scott_bandwidth(points) / widths, 0.5)
                density = gaussian_filter(
                    histogram.astype(np.float32), sigma, mode="constant"
                )
                density /= density.sum(dtype=np.float64) * widths.prod()
                point_density = support.density_at_points(points, density, pair_edges)
                thresholds.append(
                    np.quantile(point_density, support.KDE_REFERENCE_QUANTILE)
                )
                densities.append(density)
                edges.append(pair_edges)
            mean = values.mean(axis=0)
            inverse = np.linalg.inv(support.regularized_covariance(values)[0])
            centered = values - mean
            distance = np.einsum(
                "ni,ij,nj->n", centered, inverse, centered, optimize=True
            )
            self.models[name] = {
                "pairs": pairs,
                "edges": np.asarray(edges),
                "density": np.asarray(densities),
                "threshold": np.asarray(thresholds),
                "mean": mean,
                "inverse": inverse,
                "mahal": np.quantile(distance, 0.99),
                "low": values.min(axis=0),
                "high": values.max(axis=0),
            }
            print(f"cached support {name}", flush=True)
        np.savez_compressed(
            cache_path,
            **{
                f"{name}_{key}": value
                for name, model in self.models.items()
                for key, value in model.items()
            },
        )

    def check(self, name: str, value: np.ndarray) -> bool:
        model = self.models[name]
        points = np.asarray(value)[model["pairs"]]
        widths = model["edges"][:, :, 1] - model["edges"][:, :, 0]
        indices = np.floor(
            (points - model["edges"][:, :, 0]) / widths
        ).astype(np.int64)
        if (indices < 0).any() or (indices >= self.bins).any():
            return False
        density = model["density"][
            np.arange(len(indices)), indices[:, 0], indices[:, 1]
        ]
        if np.any(density < model["threshold"]):
            return False
        delta = value - model["mean"]
        return bool(
            np.all(value >= model["low"])
            and np.all(value <= model["high"])
            and delta @ model["inverse"] @ delta <= model["mahal"]
        )

    def state(self, state: np.ndarray) -> bool:
        return all(
            self.check(name, state[columns])
            for name, columns in FIELD_SLICES.items()
        )


def audit_support(references, states, actions):
    support = _distribution_support_module()
    targets = {
        **{name: states[:, columns] for name, columns in FIELD_SLICES.items()},
        "action": actions,
    }
    return {
        name: support.evaluate_feature(
            support.Feature(name, references[name]), support.Feature(name, values)
        )
        for name, values in targets.items()
    }
