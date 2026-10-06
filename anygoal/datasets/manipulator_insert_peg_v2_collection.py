"""HDF5, reference-support, and rendering helpers for peg variant 2."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_SLICES = {
    "arm": slice(0, 16),
    "hand": slice(29, 33),
    "object": slice(33, 37),
    "target": slice(40, 44),
}
STATE_FIELDS = {
    "proprio": slice(0, 33),
    "proprio_arm_pos": slice(0, 16),
    "proprio_arm_vel": slice(16, 24),
    "proprio_touch": slice(24, 29),
    "proprio_hand_pos": slice(29, 33),
    "privileged_object_pos": slice(33, 37),
    "privileged_object_vel": slice(37, 40),
    "privileged_target_pos": slice(40, 44),
}


def _distribution_support_module():
    path = REPO_ROOT / "paper/scripts/distribution_support.py"
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


def load_target_reset_pool(path: Path) -> np.ndarray:
    """Load one fixed slot pose per source episode, excluding |x| <= 0.2."""
    with h5py.File(path, "r") as dataset:
        offsets = dataset["ep_offset"][:].astype(np.int64)
        if "privileged_target_pos" in dataset:
            targets = dataset["privileged_target_pos"][offsets]
        else:
            targets = dataset["observation"][offsets, 40:44]
    targets = np.asarray(targets, dtype=np.float64)
    targets = targets[np.abs(targets[:, 0]) > 0.2]
    if not len(targets):
        raise ValueError(f"{path} has no reset target with |x| > 0.2")
    return targets


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


def save_dataset(path, episodes, reference_path, expert_reference_path, metadata):
    """Write atomically-collected fixed-length trajectories in train schema."""
    with h5py.File(path, "x") as dataset:
        dataset.attrs["max_episode_length"] = 500
        dataset.attrs["reference"] = str(reference_path)
        dataset.attrs["reference_selection"] = "all transitions"
        dataset.attrs["expert_reference"] = str(expert_reference_path)
        dataset.attrs["variant"] = "variant_2_expert"
        dataset.attrs["acceptance_rule"] = json.dumps(metadata)
        lengths = np.asarray([len(actions) for _, actions in episodes], dtype=np.int64)
        for index, (states, actions) in enumerate(episodes):
            group = dataset.create_group(f"episode_{index:03d}")
            group.create_dataset("observation", data=states[:-1])
            group.create_dataset("next_observation", data=states[1:])
            group.create_dataset("action", data=actions)
        observation = np.concatenate([states[:-1] for states, _ in episodes])
        next_observation = np.concatenate([states[1:] for states, _ in episodes])
        action = np.concatenate([actions for _, actions in episodes])
        dataset.create_dataset("observation", data=observation)
        dataset.create_dataset("next_observation", data=next_observation)
        dataset.create_dataset("action", data=action)
        dataset["privilege"] = dataset["observation"]
        dataset["next_privilege"] = dataset["next_observation"]
        for name, columns in STATE_FIELDS.items():
            dataset.create_dataset(name, data=observation[:, columns])
        dataset.create_dataset("ep_len", data=lengths)
        dataset.create_dataset("ep_offset", data=np.r_[0, np.cumsum(lengths)[:-1]])
        dataset.create_dataset("ep_idx", data=np.repeat(np.arange(len(lengths)), lengths))
        dataset.create_dataset(
            "step_idx", data=np.concatenate([np.arange(length) for length in lengths])
        )


def enrich_dataset(path: Path, env_class) -> None:
    """Add 224px observations and remaining standard transition fields."""
    env = env_class(terminate_at_goal=False)
    with h5py.File(path, "r+") as dataset:
        observation = dataset["observation"][:]
        next_observation = dataset["next_observation"][:]
        lengths = dataset["ep_len"][:].astype(np.int64)
        pixels = dataset.create_dataset(
            "pixels", shape=(len(observation), 224, 224, 3), dtype=np.uint8,
            chunks=(1, 224, 224, 3),
            **hdf5plugin.Blosc(
                cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE
            ),
        )
        rewards = np.empty(len(observation), dtype=np.float32)
        for index, state in enumerate(observation):
            env.set_state(state)
            pixels[index] = env.render(width=224, height=224)
            rewards[index] = env.dmc_env.task.get_reward(env.physics)
            if index % 1000 == 0:
                print(f"rendered {index}/{len(observation)}", flush=True)
        dataset.create_dataset("reward", data=rewards)
        dataset.create_dataset("discount", data=np.ones(len(observation), np.float32))
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        ends = np.cumsum(lengths) - 1
        fragment_start = np.zeros(len(observation), dtype=bool)
        fragment_end = np.zeros(len(observation), dtype=bool)
        fragment_start[starts] = True
        fragment_end[ends] = True
        dataset.create_dataset("fragment_start", data=fragment_start)
        dataset.create_dataset("fragment_end", data=fragment_end)
        action = dataset["action"][:]
        next_action = np.empty_like(action)
        next_action[:-1] = action[1:]
        next_action[-1] = 0
        next_action[ends] = 0
        dataset.create_dataset("next_action", data=next_action)
        for name, columns in STATE_FIELDS.items():
            dataset.create_dataset(f"next_{name}", data=next_observation[:, columns])
    env.close()


def render_sample_gifs(figures: Path, episodes, env_class, count: int = 10) -> None:
    """Render at most ten accepted episodes after collection."""
    env = env_class(terminate_at_goal=False)
    videos = []
    for states, _ in episodes[:count]:
        frames = []
        for state in states[:-1:5]:
            env.set_state(state)
            frames.append(env.render(width=160, height=160))
        videos.append(frames)
    env.close()
    figures.mkdir(parents=True, exist_ok=True)
    for index, frames in enumerate(videos):
        imageio.mimsave(
            figures / f"variant_2_expert_episode_{index:02d}.gif",
            frames, duration=50, loop=0,
        )
    grid = []
    for step in range(max(map(len, videos))):
        canvas = Image.new("RGB", (800, 360))
        draw = ImageDraw.Draw(canvas)
        for index, frames in enumerate(videos):
            x, y = (index % 5) * 160, (index // 5) * 180
            canvas.paste(Image.fromarray(frames[min(step, len(frames) - 1)]), (x, y + 20))
            draw.text((x + 3, y + 3), f"V2 ep{index} 500 steps", fill="white")
        grid.append(np.asarray(canvas))
    imageio.mimsave(
        figures / "variant_2_expert_10_episodes.gif", grid, duration=50, loop=0
    )
