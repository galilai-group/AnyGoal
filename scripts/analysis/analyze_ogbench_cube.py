"""Summarize and visualize the OGBench cube_single_expert dataset."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import hdf5plugin  # Registers the Blosc filter used by the pixel dataset.
import imageio.v2 as imageio
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anygoal.paths import data_root


DATASET = data_root() / "cube_single_expert.h5"
OUTPUT = Path("outputs/analysis/ogbench_cube_analysis")
REPORT = Path("reports/ogbench_cube_dataset_analysis.md")
RNG_SEED = 20260809


def kde_1d(values: np.ndarray, bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.quantile(values, [0.001, 0.999])
    span = upper - lower
    if span == 0:
        span = 1.0
    lower -= 0.03 * span
    upper += 0.03 * span
    counts, edges = np.histogram(values, bins=bins, range=(lower, upper))
    width = edges[1] - edges[0]
    bandwidth = max(values.std() * len(values) ** (-1 / 5), width * 0.5)
    density = gaussian_filter1d(counts.astype(float), bandwidth / width, mode="constant")
    density /= density.sum() * width
    return (edges[:-1] + edges[1:]) / 2, density


def plot_scalar_kdes(
    values: dict[str, np.ndarray], output_path: Path, title: str
) -> None:
    columns = 4
    rows = int(np.ceil(len(values) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2.25 * rows))
    axes = np.asarray(axes).ravel()
    for axis, (name, data) in zip(axes, values.items()):
        x, density = kde_1d(data)
        axis.fill_between(x, density, color="#ac9cdd", alpha=0.75)
        axis.plot(x, density, color="#7461b5", linewidth=1.1)
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.2)
    for axis in axes[len(values) :]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_joint_kdes(vectors: dict[str, np.ndarray], output_path: Path) -> None:
    """All component-pairs for each named vector, using all-point 2D KDEs."""
    pairs = [(f"{name}[{first},{second}]", data[:, [first, second]])
             for name, data in vectors.items() if data.ndim == 2 and data.shape[1] > 1
             for first in range(data.shape[1]) for second in range(first + 1, data.shape[1])]
    columns = 3
    rows = int(np.ceil(len(pairs) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.6 * rows))
    for axis, (name, points) in zip(np.asarray(axes).ravel(), pairs):
        limits = []
        for values in points.T:
            low, high = np.quantile(values, [0.001, 0.999])
            span = high - low or 1.0
            limits.append((low - 0.03 * span, high + 0.03 * span))
        histogram, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=180, range=limits)
        sigma = np.maximum(points.std(axis=0) * len(points) ** (-1 / 6) /
                           np.array([x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]]), 0.5)
        density = gaussian_filter(histogram, sigma=sigma, mode="constant")
        axis.contourf((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2,
                      density.T, levels=12, cmap="Purples")
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in np.asarray(axes).ravel()[len(pairs) :]:
        axis.set_visible(False)
    figure.suptitle("OGBench Cube: all-frame named state/proprio component-pair KDEs", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def exact_column_names(dataset: h5py.File, sample_indices: np.ndarray) -> list[str]:
    """Label observation columns by exact equality to the named scalar datasets."""
    observation = dataset["observation"][sample_indices]
    candidates: list[tuple[str, np.ndarray]] = []
    for key in dataset:
        if key in {"observation", "target", "pixels", "action", "control", "qpos", "qvel", "prev_qpos", "prev_qvel"}:
            continue
        value = dataset[key]
        if value.shape[0] != observation.shape[0]:
            continue
        if value.ndim == 1:
            candidates.append((key, value[sample_indices][:, None]))
        elif value.ndim == 2:
            candidates.extend((f"{key}[{index}]", value[sample_indices, index : index + 1]) for index in range(value.shape[1]))
    names = []
    for column in range(observation.shape[1]):
        matches = [name for name, value in candidates if np.array_equal(observation[:, column : column + 1], value)]
        names.append(matches[0] if matches else f"observation[{column}] (unmatched)")
    return names


def make_gif(dataset: h5py.File, starts: np.ndarray, lengths: np.ndarray) -> list[int]:
    selected = np.random.default_rng(RNG_SEED).choice(len(starts), size=10, replace=False)
    episode_starts = starts[selected]
    episode_lengths = lengths[selected]
    maximum_length = int(episode_lengths.max())
    gif_path = OUTPUT / "random_10_trajectories_2x5.gif"
    with imageio.get_writer(gif_path, mode="I", duration=0.07, loop=0, palettesize=256, subrectangles=True) as writer:
        for step in range(maximum_length):
            rows = []
            for row in range(2):
                images = []
                for col in range(5):
                    index = row * 5 + col
                    frame = min(step, int(episode_lengths[index]) - 1)
                    image = dataset["pixels"][int(episode_starts[index]) + frame]
                    images.append(image)
                rows.append(np.concatenate(images, axis=1))
            writer.append_data(np.concatenate(rows, axis=0))
    return selected.tolist()


def summarize(values: np.ndarray) -> dict[str, list[float] | float]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def fmt(values: list[float] | float | int) -> str:
    if np.isscalar(values):
        return f"{values:.4g}"
    return ", ".join(f"{value:.4g}" for value in values)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATASET, "r") as dataset:
        lengths = dataset["ep_len"][:].astype(int)
        starts = dataset["ep_offset"][:].astype(int)
        if not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]]):
            raise ValueError("ep_offset does not match cumulative episode lengths")
        initial_indices = starts
        sample_indices = np.linspace(0, len(dataset["observation"]) - 1, 4096, dtype=int)
        observation_names = exact_column_names(dataset, sample_indices)
        observation = dataset["observation"][:]
        initial_observation = dataset["observation"][initial_indices]

        proprio_keys = sorted(key for key in dataset if key.startswith("proprio_"))
        proprio_arrays = {key: dataset[key][:] for key in proprio_keys}
        proprio_values = {
            f"{key}[{column}]": values[:, column] if values.ndim == 2 else values
            for key in proprio_keys
            for values in (proprio_arrays[key],)
            for column in range(values.shape[1] if values.ndim == 2 else 1)
        }
        initial_proprio_values = {name: values[initial_indices] for name, values in proprio_values.items()}
        state_values = {observation_names[index]: observation[:, index] for index in range(observation.shape[1])}
        initial_state_values = {observation_names[index]: initial_observation[:, index] for index in range(observation.shape[1])}
        plot_scalar_kdes({**state_values, **proprio_values}, OUTPUT / "all_state_proprio_scalar_kde.png", "OGBench Cube: all-frame state/observation and proprioceptive KDEs")
        plot_scalar_kdes({**initial_state_values, **initial_proprio_values}, OUTPUT / "initial_state_proprio_scalar_kde.png", "OGBench Cube: initial-frame state/observation and proprioceptive KDEs")
        selected_episodes = make_gif(dataset, starts, lengths)

        fields = {
            key: (proprio_arrays[key] if key in proprio_arrays else dataset[key][:])
            for key in dataset
            if key.startswith("proprio_") or key.startswith("privileged_")
        }
        initial_fields = {key: value[initial_indices] for key, value in fields.items()}

    plot_joint_kdes(fields, OUTPUT / "all_state_proprio_joint_kde.png")

    numeric_initial_fields = {
        key: value for key, value in initial_fields.items()
        if np.issubdtype(value.dtype, np.number)
    }
    initial_summary = {
        "observation": summarize(initial_observation),
        **{key: summarize(value) for key, value in numeric_initial_fields.items()},
    }
    metadata = {
        "dataset": str(DATASET), "episodes": int(len(lengths)), "frames": int(lengths.sum()),
        "length": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())},
        "observation_shape": list(observation.shape), "observation_columns": observation_names,
        "proprio_keys": proprio_keys, "initial_summary": initial_summary,
        "random_episode_indices": selected_episodes, "random_seed": RNG_SEED,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    field_lines = "\n".join(f"- `{key}`: shape {tuple(value.shape[1:]) or '(scalar)'}" for key, value in fields.items())
    initial_lines = "\n".join(
        f"- `{key}` — min/max: [{fmt(stats['min'])}] / [{fmt(stats['max'])}]; 1–99%: [{fmt(stats['q01'])}] / [{fmt(stats['q99'])}]; std: [{fmt(stats['std'])}]"
        for key, stats in initial_summary.items()
    )
    REPORT.write_text(f"""# OGBench Cube dataset analysis

Source: `{DATASET}`. Generated with `scripts/analysis/analyze_ogbench_cube.py` (seed {RNG_SEED}).

## Dataset size

- Trajectories: **{len(lengths):,}**
- Frames: **{lengths.sum():,}**
- Trajectory length: mean **{lengths.mean():.2f}**, min **{lengths.min()}**, max **{lengths.max()}** frames

## State and proprioception contents

The HDF5 file has no `/state` dataset. This report uses `/observation` (shape `{tuple(observation.shape)}`) as the state/agent observation. Its columns were matched by exact equality against named HDF5 arrays on 4,096 frames; unmatched columns remain explicitly marked below. The dataset also exposes the following proprioceptive and privileged state fields:

{field_lines}

Observation-column mapping: `{', '.join(observation_names)}`.

## Visualizations

- [10 random trajectories, 2 × 5 GIF (short trajectories hold their final frame)](../outputs/analysis/ogbench_cube_analysis/random_10_trajectories_2x5.gif)
- [All-frame scalar KDEs for every observation/state and proprioception component](../outputs/analysis/ogbench_cube_analysis/all_state_proprio_scalar_kde.png)
- [All-frame 2D KDEs for every component pair within named state/proprio vectors](../outputs/analysis/ogbench_cube_analysis/all_state_proprio_joint_kde.png)
- [Initial-frame scalar KDEs](../outputs/analysis/ogbench_cube_analysis/initial_state_proprio_scalar_kde.png)
- [Machine-readable statistics](../outputs/analysis/ogbench_cube_analysis/summary.json)

The KDEs use all available frames (or all 10,000 initial frames) with a Scott-rule Gaussian-smoothed histogram, following the all-points KDE approach used for PushT.

## Initial-frame sampling ranges and confidence

The min/max range describes observed support; the 1–99% interval is the more robust estimate of the routine sampling range. Small standard deviation or a nearly identical 1–99% interval indicates a tightly constrained / high-confidence initial component, while a broad interval indicates deliberate variation. With 10,000 independent episode initializations, the reported quantiles are empirically well resolved; these are descriptive support estimates, not formal guarantees beyond the observed dataset.

{initial_lines}

`privileged_target_task` is a non-numeric task-label field, so it is intentionally excluded from numeric range statistics.
""")


if __name__ == "__main__":
    main()
