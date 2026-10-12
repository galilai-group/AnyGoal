"""Reference-support helpers for generated Insert-Peg variants."""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from scipy.ndimage import gaussian_filter

from anygoal.datasets import support_metrics

FIELD_SLICES = {
    "arm": slice(0, 16),
    "hand": slice(29, 33),
    "object": slice(33, 37),
    "target": slice(40, 44),
}


def _distribution_support_module():
    return support_metrics


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
                            "pairs",
                            "edges",
                            "density",
                            "threshold",
                            "mean",
                            "inverse",
                            "mahal",
                            "low",
                            "high",
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
        indices = np.floor((points - model["edges"][:, :, 0]) / widths).astype(np.int64)
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
            self.check(name, state[columns]) for name, columns in FIELD_SLICES.items()
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


def write_variant_dataset(
    path: Path,
    episodes: list[dict[str, np.ndarray]],
    attributes: dict,
    *,
    include_pixels: bool,
    env,
    overwrite: bool = False,
    render_progress_interval: int | None = None,
) -> None:
    """Write an Insert-Peg variant without changing the canonical HDF5 schema."""
    from anygoal.envs.insert_peg import env as insert_peg_env

    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    lengths = np.asarray([len(item["actions"]) for item in episodes], np.int64)
    observation = np.concatenate([item["states"][:-1] for item in episodes]).astype(
        np.float32
    )
    next_observation = np.concatenate([item["states"][1:] for item in episodes]).astype(
        np.float32
    )
    actions = np.concatenate([item["actions"] for item in episodes]).astype(np.float32)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = np.cumsum(lengths) - 1

    with h5py.File(temporary, "x") as dataset:
        for name, value in attributes.items():
            dataset.attrs[name] = value
        dataset.create_dataset("observation", data=observation)
        dataset.create_dataset("next_observation", data=next_observation)
        dataset.create_dataset("action", data=actions)
        dataset["privilege"] = dataset["observation"]
        dataset["next_privilege"] = dataset["next_observation"]
        for name, columns in insert_peg_env.STATE_FIELDS.items():
            dataset.create_dataset(name, data=observation[:, columns])
            dataset.create_dataset(f"next_{name}", data=next_observation[:, columns])
        dataset.create_dataset("ep_len", data=lengths)
        dataset.create_dataset("ep_offset", data=offsets)
        dataset.create_dataset(
            "ep_idx", data=np.repeat(np.arange(len(lengths)), lengths)
        )
        dataset.create_dataset(
            "step_idx",
            data=np.concatenate([np.arange(length) for length in lengths]),
        )
        starts = np.zeros(len(actions), bool)
        terminal = np.zeros(len(actions), bool)
        starts[offsets] = True
        terminal[ends] = True
        dataset.create_dataset("fragment_start", data=starts)
        dataset.create_dataset("fragment_end", data=terminal)
        dataset.create_dataset("done", data=terminal)
        dataset.create_dataset("success", data=np.repeat(True, len(actions)))
        dataset.create_dataset("discount", data=(~terminal).astype(np.float32))
        dataset.create_dataset("reward", data=np.zeros(len(actions), np.float32))
        next_action = np.zeros_like(actions)
        next_action[:-1] = actions[1:]
        next_action[ends] = 0.0
        dataset.create_dataset("next_action", data=next_action)
        if include_pixels:
            pixels = dataset.create_dataset(
                "pixels",
                shape=(len(observation), 224, 224, 3),
                dtype=np.uint8,
                chunks=(1, 224, 224, 3),
                **hdf5plugin.Blosc(
                    cname="zstd",
                    clevel=5,
                    shuffle=hdf5plugin.Blosc.BITSHUFFLE,
                ),
            )
            for index, state in enumerate(observation):
                env.set_state(state)
                pixels[index] = env.render()
                if render_progress_interval and index % render_progress_interval == 0:
                    print(f"rendered {index}/{len(observation)}", flush=True)
    os.replace(temporary, path)
