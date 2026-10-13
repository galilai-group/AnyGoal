"""Measure and visualize the 14 requested Insert-Peg state fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the HDF5 compression filters.
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_DATASET = (
    Path.home() / ".stable_worldmodel/datasets/manipulator_insert_peg_weak_expert_train.h5"
)
DEFAULT_OUTPUT = Path("outputs/analysis/insert_peg_state_kde")
KDE_REFERENCE_QUANTILE = 0.01
GREY = "#d0d0d0"
LINE = "#c96f5d"
FILL = "#e9b5a8"
CMAP = LinearSegmentedColormap.from_list("peg_kde", ["#fff8f5", "#efc0b5", "#d77f6b"])

# The fixed camera is at (0, -16, 0.4), looks along +y, and has a 4 degree
# vertical field of view. The dataset render is square, so horizontal and
# vertical spans agree on the y=0 task plane.
RENDER_HALF_SPAN = 16.0 * np.tan(np.deg2rad(4.0 / 2.0))
RENDER_X_RANGE = (-RENDER_HALF_SPAN, RENDER_HALF_SPAN)
RENDER_Z_RANGE = (0.4 - RENDER_HALF_SPAN, 0.4 + RENDER_HALF_SPAN)

JOINTS = (
    ("root", (-np.pi, np.pi)),
    ("shoulder", np.deg2rad((-160.0, 160.0))),
    ("elbow", np.deg2rad((-160.0, 160.0))),
    ("wrist", np.deg2rad((-140.0, 140.0))),
    ("finger", np.deg2rad((-10.0, 60.0))),
    ("finger_tip", np.deg2rad((-40.0, 20.0))),
    ("thumb", np.deg2rad((-10.0, 60.0))),
    ("thumb_tip", np.deg2rad((-40.0, 20.0))),
)

PAIR_FIELDS = (
    ("hand_position", "proprio_hand_pos", slice(0, 2), "x", "z", None),
    ("hand_rotation", "proprio_hand_pos", slice(2, 4), "qw", "qy", "derived unit circle"),
    ("object_position", "privileged_object_pos", slice(0, 2), "x", "z", "mixture; see report"),
    ("object_rotation", "privileged_object_pos", slice(2, 4), "qw", "qy", "unit circle"),
    ("target_position", "privileged_target_pos", slice(0, 2), "x", "z", "x∈[-0.4,0.4], z∈[0.1,0.4]"),
    ("target_rotation", "privileged_target_pos", slice(2, 4), "qw", "qy", "angle∈[-π/3,π/3]"),
)


def robust_limits(values: np.ndarray, fixed: tuple[float, float] | None = None) -> tuple[float, float]:
    low, high = np.quantile(values, (0.001, 0.999))
    if fixed is not None:
        low, high = min(low, fixed[0]), max(high, fixed[1])
    span = high - low
    if span <= 0:
        span = max(abs(float(low)) * 0.1, 0.1)
    return float(low - 0.05 * span), float(high + 0.05 * span)


def density_1d(
    values: np.ndarray,
    limits: tuple[float, float],
    bins: int = 256,
    mode: str = "constant",
    bandwidth_values: np.ndarray | None = None,
):
    counts, edges = np.histogram(values, bins=bins, range=limits)
    width = edges[1] - edges[0]
    bandwidth_source = values if bandwidth_values is None else bandwidth_values
    bandwidth = max(bandwidth_source.std() * len(values) ** (-1 / 5), width * 0.5)
    density = gaussian_filter1d(counts.astype(float), bandwidth / width, mode=mode, truncate=4.0)
    density /= max(density.sum() * width, np.finfo(float).tiny)
    return (edges[:-1] + edges[1:]) / 2, density


def distribution_limits(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = upper - lower
    fallback = np.maximum(np.maximum(np.abs(lower), np.abs(upper)) * 0.03, 1e-6)
    padding = np.where(span > 0.0, 0.03 * span, fallback)
    return tuple((float(lo - pad), float(hi + pad)) for lo, hi, pad in zip(lower, upper, padding))


def density_at_points(points: np.ndarray, density: np.ndarray, edges: tuple[np.ndarray, ...]) -> np.ndarray:
    indices = [np.searchsorted(edge, points[:, axis], side="right") - 1 for axis, edge in enumerate(edges)]
    inside = np.ones(len(points), dtype=bool)
    for axis, index in enumerate(indices):
        inside &= (index >= 0) & (index < density.shape[axis])
    values = np.zeros(len(points), dtype=np.float64)
    values[inside] = density[tuple(index[inside] for index in indices)]
    return values


def density_2d(points: np.ndarray, bins: int = 256):
    limits = distribution_limits(points)
    histogram, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=limits)
    widths = np.array((x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]))
    bandwidth = np.maximum(points.std(axis=0) * len(points) ** (-1 / 6) / widths, 0.5)
    density = gaussian_filter(histogram.astype(np.float32), bandwidth, mode="constant", truncate=4.0)
    density /= max(float(density.sum(dtype=np.float64) * np.prod(widths)), np.finfo(float).tiny)
    threshold = float(np.quantile(density_at_points(points, density, (x_edges, y_edges)), KDE_REFERENCE_QUANTILE))
    return density, x_edges, y_edges, threshold


def draw_angle(axis, values: np.ndarray, name: str, sampling_range: tuple[float, float]):
    # Match distribution_support.py: periodic [-pi, pi] histogram KDE and the
    # 1st percentile of densities evaluated at reference samples.
    values = (values + np.pi) % (2.0 * np.pi) - np.pi
    mean_angle = np.arctan2(np.sin(values).mean(), np.cos(values).mean())
    bandwidth_values = (values - mean_angle + np.pi) % (2.0 * np.pi) - np.pi
    x_radians, density = density_1d(
        values, (-np.pi, np.pi), mode="wrap", bandwidth_values=bandwidth_values
    )
    edges = np.linspace(-np.pi, np.pi, len(density) + 1)
    threshold = float(np.quantile(density_at_points(values[:, None], density, (edges,)), KDE_REFERENCE_QUANTILE))
    x = np.rad2deg(x_radians)
    sampling_range = tuple(np.rad2deg(sampling_range))
    supported = density >= threshold
    low_density = ~supported
    axis.set_facecolor(GREY)
    axis.fill_between(x, density, where=supported, color=FILL, label="inside KDE support")
    axis.fill_between(x, density, where=low_density, color="#8fa9bd", hatch="///", alpha=0.8, label="below reference-density q=0.01")
    axis.plot(x, density, color=LINE, linewidth=1.7)
    axis.axhline(threshold, color="#41657a", linestyle=":", linewidth=1.0)
    axis.axvline(sampling_range[0], color="#555555", linestyle="--", linewidth=0.8)
    axis.axvline(sampling_range[1], color="#555555", linestyle="--", linewidth=0.8)
    axis.set(xlabel="angle (degrees; −180° to 180°)", ylabel="KDE density (per radian)", title=f"Arm {name} angle")
    axis.set_xlim(-180.0, 180.0)
    axis.grid(color="white", alpha=0.7)


def draw_pair(figure, axes, points: np.ndarray, title: str, labels: tuple[str, str], fixed_limits=None):
    x_fixed = fixed_limits[0] if fixed_limits else None
    y_fixed = fixed_limits[1] if fixed_limits else None
    limits = (
        tuple(x_fixed) if x_fixed is not None else robust_limits(points[:, 0]),
        tuple(y_fixed) if y_fixed is not None else robust_limits(points[:, 1]),
    )
    density, x_edges, y_edges, threshold = density_2d(points)
    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2
    supported = density.T >= threshold
    masked = np.ma.masked_where(~supported, density.T)
    joint, top, right = axes
    for axis in axes:
        axis.set_facecolor(GREY)
    positive = masked.compressed()
    levels = np.linspace(float(positive.min()), float(positive.max()), 13)
    contour = joint.contourf(x, y, masked, levels=levels, cmap=CMAP)
    joint.contour(x, y, masked, levels=levels[::2], colors=LINE, linewidths=0.45)
    joint.set(xlabel=labels[0], ylabel=labels[1], xlim=limits[0], ylim=limits[1])
    joint.grid(color="white", alpha=0.65)
    for axis, values, lim, vertical in ((top, points[:, 0], limits[0], False), (right, points[:, 1], limits[1], True)):
        kde_limits = distribution_limits(values[:, None])[0]
        coordinate, marginal = density_1d(values, kde_limits)
        marginal_edges = np.linspace(kde_limits[0], kde_limits[1], len(marginal) + 1)
        marginal_threshold = float(np.quantile(density_at_points(values[:, None], marginal, (marginal_edges,)), KDE_REFERENCE_QUANTILE))
        mask = marginal >= marginal_threshold
        if vertical:
            axis.fill_betweenx(coordinate, 0, marginal, where=mask, color=FILL)
            axis.plot(marginal[mask], coordinate[mask], color=LINE)
            axis.set_ylim(lim)
        else:
            axis.fill_between(coordinate, marginal, where=mask, color=FILL)
            axis.plot(coordinate[mask], marginal[mask], color=LINE)
            axis.set_xlim(lim)
        axis.grid(color="white", alpha=0.65)
    figure.colorbar(contour, ax=joint, label="KDE density", shrink=0.78)
    figure.suptitle(f"{title} (all {len(points):,} frames; grey = no KDE support)")


def pair_figure(points: np.ndarray, title: str, labels: tuple[str, str], fixed_limits=None):
    figure = plt.figure(figsize=(8.5, 8.0))
    grid = figure.add_gridspec(2, 2, width_ratios=(5, 1.35), height_ratios=(1.35, 5), hspace=0.04, wspace=0.04)
    top = figure.add_subplot(grid[0, 0])
    joint = figure.add_subplot(grid[1, 0], sharex=top)
    right = figure.add_subplot(grid[1, 1], sharey=joint)
    draw_pair(figure, (joint, top, right), points, title, labels, fixed_limits)
    plt.setp(top.get_xticklabels(), visible=False)
    plt.setp(right.get_yticklabels(), visible=False)
    return figure


def summarize(name: str, values: np.ndarray, sampling: str) -> dict:
    return {
        "item": name,
        "sampling_range": sampling,
        "empirical_min": np.min(values, axis=0).tolist(),
        "empirical_max": np.max(values, axis=0).tolist(),
        "q0.1%": np.quantile(values, 0.001, axis=0).tolist(),
        "q99.9%": np.quantile(values, 0.999, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as dataset:
        arm_sincos = dataset["proprio_arm_pos"][:].astype(np.float64)
        pair_values = {
            name: dataset[key][:, columns].astype(np.float64)
            for name, key, columns, _, _, _ in PAIR_FIELDS
        }
        episodes = len(dataset["ep_len"])
    angles = np.arctan2(arm_sincos[:, 0::2], arm_sincos[:, 1::2])
    rows = []

    overview, axes = plt.subplots(4, 2, figsize=(12, 13))
    for index, ((name, sampling_range), axis) in enumerate(zip(JOINTS, axes.ravel()), start=1):
        values = angles[:, index - 1]
        draw_angle(axis, values, name, sampling_range)
        figure, single_axis = plt.subplots(figsize=(7.2, 4.5))
        draw_angle(single_axis, values, name, sampling_range)
        single_axis.legend(fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(args.output / f"{index:02d}_arm_{name}_angle_kde.png", dpi=200)
        plt.close(figure)
        range_degrees = np.rad2deg(sampling_range)
        rows.append(summarize(name, np.rad2deg(values[:, None]), f"[{range_degrees[0]:.8g}, {range_degrees[1]:.8g}] degrees"))
    overview.suptitle(f"Insert-Peg arm angles (all {len(angles):,} frames; dashed = reset limits)")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    overview.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    overview.tight_layout(rect=(0, 0.035, 1, 0.97))
    overview.savefig(args.output / "arm_angles_1d_kde_overview.png", dpi=200)
    plt.close(overview)

    position_fig, position_axes = plt.subplots(1, 3, figsize=(17, 5.2))
    rotation_fig, rotation_axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for offset, (name, _, _, x_label, y_label, sampling) in enumerate(PAIR_FIELDS, start=9):
        points = pair_values[name]
        fixed = None
        if "position" in name:
            fixed = (RENDER_X_RANGE, RENDER_Z_RANGE)
        elif "rotation" in name:
            fixed = ((-1.0, 1.0), (-1.0, 1.0))
        figure = pair_figure(points, name.replace("_", " ").title(), (x_label, y_label), fixed)
        figure.savefig(args.output / f"{offset:02d}_{name}_2d_joint_kde.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

        overview_axis = (position_axes if "position" in name else rotation_axes)[(offset - 9) // 2]
        limits = tuple(tuple(fixed[dim]) if fixed else robust_limits(points[:, dim]) for dim in range(2))
        density, x_edges, y_edges, threshold = density_2d(points)
        masked = np.ma.masked_less(density.T, threshold)
        overview_axis.set_facecolor(GREY)
        overview_axis.contourf((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2, masked, levels=12, cmap=CMAP)
        overview_axis.set(
            xlabel=x_label, ylabel=y_label, title=name.replace("_", " ").title(),
            xlim=limits[0], ylim=limits[1],
        )
        overview_axis.grid(color="white", alpha=0.65)
        if "rotation" in name:
            theta = np.linspace(-np.pi, np.pi, 500)
            overview_axis.plot(np.cos(theta), np.sin(theta), "--", color="#555555", linewidth=0.7, label="unit circle")
            overview_axis.set_aspect("equal", adjustable="box")
        rows.append(summarize(name, points, sampling or "derived; no independent rectangular sampler"))

    for figure, filename, title in (
        (position_fig, "positions_2d_joint_kde_overview.png", "Position joint KDEs"),
        (rotation_fig, "rotations_qw_qy_2d_joint_kde_overview.png", "Planar rotation (qw, qy) joint KDEs"),
    ):
        figure.suptitle(f"Insert-Peg: {title} (grey = below reference-density q=0.01)")
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(args.output / filename, dpi=200)
        plt.close(figure)

    metadata = {
        "dataset": str(args.dataset.resolve()), "frames": len(angles), "episodes": episodes,
        "angle_conversion": "atan2(sin(theta), cos(theta))",
        "angle_display_unit": "degrees in [-180, 180]",
        "support_definition": "density below the 1st percentile of KDE densities evaluated at reference samples",
        "render_plane_ranges": {"x": list(RENDER_X_RANGE), "z": list(RENDER_Z_RANGE)},
        "kde": "Scott-rule Gaussian-smoothed histogram",
        "statistics": rows,
    }
    (args.output / "state_range_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "state_range_summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("item", "sampling_range", "empirical_min", "empirical_max", "q0.1%", "q99.9%", "mean", "std"))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in rows)
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(angles), "episodes": episodes}, indent=2))


if __name__ == "__main__":
    main()
