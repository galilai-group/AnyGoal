#!/usr/bin/env python
"""Select and plot the widest supported symmetric 3x3 TwoRoom action grid."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path.home() / ".stable_worldmodel" / "datasets" / "tworoom.h5"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "paper" / "results" / "tworoom_counterfactual_sensitivity"
)
SEED = 20260913


def action_grid(magnitude: float) -> np.ndarray:
    coordinates = (-magnitude, 0.0, magnitude)
    return np.asarray(list(itertools.product(coordinates, repeat=2)))


def sampled_episode_actions(
    dataset: h5py.File,
    *,
    episodes: int,
    samples_per_episode: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for offset, length in zip(
        dataset["ep_offset"][:episodes], dataset["ep_len"][:episodes]
    ):
        valid_length = max(int(length) - 1, 1)
        count = min(samples_per_episode, valid_length)
        indices = np.sort(rng.choice(valid_length, size=count, replace=False))
        selected.append(np.asarray(dataset["action"][int(offset) + indices]))
    values = np.concatenate(selected).astype(np.float64)
    return values[np.isfinite(values).all(axis=1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-actions", type=int, default=200_000)
    parser.add_argument("--calibration-actions", type=int, default=20_000)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--plot-episodes", type=int, default=5_000)
    parser.add_argument("--samples-per-episode", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if args.neighbors < 1 or not 0.0 < args.support_quantile < 1.0:
        raise ValueError("neighbors must be positive and support-quantile must be in (0, 1)")

    with h5py.File(dataset_path, "r") as dataset:
        actions = np.asarray(dataset["action"][:], dtype=np.float64)
        episode_count = min(args.plot_episodes, len(dataset["ep_len"]))
        plot_actions = sampled_episode_actions(
            dataset,
            episodes=episode_count,
            samples_per_episode=args.samples_per_episode,
            seed=SEED,
        )
    actions = actions[np.isfinite(actions).all(axis=1)]
    requested = args.reference_actions + args.calibration_actions
    if len(actions) < requested:
        raise ValueError(f"Need {requested:,} finite actions, found {len(actions):,}")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(actions), size=requested, replace=False)
    reference = actions[indices[: args.reference_actions]]
    calibration = actions[indices[args.reference_actions :]]
    tree = cKDTree(reference)

    def neighbor_radius(values: np.ndarray) -> np.ndarray:
        distances = tree.query(values, k=args.neighbors, workers=-1)[0]
        return distances[..., -1]

    calibration_radii = neighbor_radius(calibration)
    support_radius = float(np.quantile(calibration_radii, args.support_quantile))

    candidates = np.linspace(0.0, float(np.abs(actions).max()), 10_001)
    selected_magnitude = None
    for magnitude in candidates[::-1]:
        if np.all(neighbor_radius(action_grid(float(magnitude))) <= support_radius):
            selected_magnitude = float(magnitude)
            break
    if selected_magnitude is None:
        raise RuntimeError("No supported symmetric grid was found, including zero")

    selected_actions = action_grid(selected_magnitude)
    selected_radii = neighbor_radius(selected_actions)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tworoom_counterfactual_actions.json"
    payload = {
        "dataset_path": str(dataset_path),
        "selection": "largest shared magnitude m for {-m, 0, +m}^2",
        "actions": selected_actions.tolist(),
        "magnitude": selected_magnitude,
        "all_points_supported": bool(np.all(selected_radii <= support_radius)),
        "action_20nn_radii": selected_radii.tolist(),
        "support": {
            "method": "raw 2-D Euclidean 20-nearest-neighbor radius",
            "seed": SEED,
            "finite_action_count": len(actions),
            "reference_action_count": args.reference_actions,
            "calibration_action_count": args.calibration_actions,
            "neighbors": args.neighbors,
            "calibration_quantile": args.support_quantile,
            "radius": support_radius,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))

    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.scatter(
        plot_actions[:, 0],
        plot_actions[:, 1],
        s=3,
        alpha=0.16,
        color="royalblue",
        edgecolors="none",
        rasterized=True,
    )
    axis.scatter(
        selected_actions[:, 0],
        selected_actions[:, 1],
        s=320,
        color="yellow",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("action[0]")
    axis.set_ylabel("action[1]")
    axis.set_title(
        f"TwoRoom raw actions: first {episode_count} episodes, "
        f"{args.samples_per_episode} samples each"
    )
    axis.text(
        0.37,
        0.88,
        f"points: {len(plot_actions):,}\nselected magnitude: {selected_magnitude:.4g}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    axis.grid(True, alpha=0.25)
    figure_path = output_dir / "tworoom_action_distribution_with_grid.jpg"
    figure.savefig(figure_path, dpi=args.dpi)
    plt.close(figure)
    print(f"Selected magnitude: {selected_magnitude:g}")
    print(f"Support radius: {support_radius:g}; max grid radius: {selected_radii.max():g}")
    print(f"Saved {output_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
