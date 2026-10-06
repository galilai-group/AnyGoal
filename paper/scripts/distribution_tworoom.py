"""Plot TwoRoom action and position distributions in the paper KDE style."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm


DATASET_PATH = Path.home() / ".stable_worldmodel" / "datasets" / "tworoom.h5"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "assets" / "paper" / "distribution_tworoom"
)
FEATURES = {
    "action": ("Action x", "Action y", np.array(((-1.0, 1.0), (-1.0, 1.0)))),
    "agent_position": (
        "Agent x",
        "Agent y",
        np.array(((0.0, 224.0), (0.0, 224.0))),
    ),
    "agent_initial_position": (
        "Agent x",
        "Agent y",
        np.array(((0.0, 224.0), (0.0, 224.0))),
    ),
    "target_position": (
        "Target x",
        "Target y",
        np.array(((0.0, 224.0), (0.0, 224.0))),
    ),
}
COLORS = ["#fff9f7", "#fcebe5", "#f7d8cf", "#f1c0b3", "#e7a493", "#d98776"]
LINE_COLOR = "#c66f61"
HDR_MASSES = (0.99, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--histogram-only", action="store_true")
    return parser.parse_args()


def finite_rows(values: np.ndarray, name: str) -> np.ndarray:
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        raise ValueError(f"{name} contains no finite rows")
    removed = len(values) - int(valid.sum())
    if removed:
        print(f"Excluded {removed:,} non-finite rows from {name}", flush=True)
    return values[valid].astype(np.float64)


def load_features(path: Path) -> tuple[dict[str, np.ndarray], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as dataset:
        required = {"action", "pos_agent", "pos_target", "ep_offset", "step_idx"}
        missing = required.difference(dataset.keys())
        if missing:
            raise KeyError(f"{path} is missing fields: {sorted(missing)}")
        offsets = dataset["ep_offset"][:].astype(np.int64)
        if not np.all(dataset["step_idx"][offsets] == 0):
            raise ValueError("ep_offset does not point to frame 0 for every episode")
        actions = finite_rows(dataset["action"][:], "action")
        agent_positions = finite_rows(dataset["pos_agent"][:], "pos_agent")
        initial_positions = finite_rows(dataset["pos_agent"][offsets], "initial pos_agent")
        target_positions = finite_rows(dataset["pos_target"][:], "pos_target")
    return {
        "action": actions,
        "agent_position": agent_positions,
        "agent_initial_position": initial_positions,
        "target_position": target_positions,
    }, len(offsets)


def padded_limits(points: np.ndarray) -> np.ndarray:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = upper - lower
    padding = np.where(span > 0.0, 0.03 * span, 1.0)
    return np.column_stack((lower - padding, upper + padding))


def scott_bandwidth(points: np.ndarray, dimensions: int) -> np.ndarray:
    scale = points.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    return scale * len(points) ** (-1.0 / (dimensions + 4.0))


def binned_kde_2d(points: np.ndarray, limits: np.ndarray, bins: int) -> dict:
    histogram, x_edges, y_edges = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, range=limits.tolist()
    )
    widths = np.array((x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]))
    bandwidth = scott_bandwidth(points, dimensions=2)
    density = gaussian_filter(
        histogram,
        sigma=np.maximum(bandwidth / widths, 0.5),
        mode="constant",
        truncate=4.0,
    )
    density /= density.sum() * np.prod(widths)
    return {
        "density": density,
        "x": (x_edges[:-1] + x_edges[1:]) / 2.0,
        "y": (y_edges[:-1] + y_edges[1:]) / 2.0,
        "bandwidth": bandwidth,
    }


def empirical_position_support(
    agent_positions: np.ndarray, limits: np.ndarray, bins: int
) -> np.ndarray:
    """Return observed free-space bins while filling tiny sampling holes."""
    histogram, _, _ = np.histogram2d(
        agent_positions[:, 0],
        agent_positions[:, 1],
        bins=bins,
        range=limits.tolist(),
    )
    return binary_fill_holes(histogram > 0)


def binned_kde_1d(values: np.ndarray, limits: np.ndarray, bins: int) -> tuple:
    histogram, edges = np.histogram(values, bins=bins, range=limits)
    width = edges[1] - edges[0]
    bandwidth = float(values.std(ddof=0) * len(values) ** (-1.0 / 5.0))
    density = gaussian_filter1d(
        histogram.astype(np.float64),
        sigma=max(bandwidth / width, 0.5),
        mode="constant",
        truncate=4.0,
    )
    density /= density.sum() * width
    return (edges[:-1] + edges[1:]) / 2.0, density


def hdr_levels(density: np.ndarray) -> np.ndarray:
    flat = density.ravel()
    positive = flat[flat > 0.0]
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]
    thresholds = [
        flat[order[min(np.searchsorted(cumulative, mass), len(order) - 1)]]
        for mass in HDR_MASSES
    ]
    levels = np.unique(np.array([positive.min(), *sorted(thresholds)]))
    maximum = float(flat.max())
    if levels[-1] >= maximum:
        levels[-1] = np.nextafter(maximum, 0.0)
    return np.append(levels, np.nextafter(maximum, np.inf))


def plot_feature(
    feature_name: str,
    points: np.ndarray,
    episode_count: int,
    output_dir: Path,
    bins: int,
    dpi: int,
    support_points: np.ndarray | None,
) -> None:
    x_label, y_label, display_limits = FEATURES[feature_name]
    kde_limits = padded_limits(points)
    model = binned_kde_2d(points, kde_limits, bins)
    x_marginal, x_density = binned_kde_1d(points[:, 0], kde_limits[0], bins)
    y_marginal, y_density = binned_kde_1d(points[:, 1], kde_limits[1], bins)
    colormap = LinearSegmentedColormap.from_list("tworoom_pastel", COLORS)

    figure = plt.figure(figsize=(8.2, 8.2), facecolor="white")
    grid = figure.add_gridspec(
        4, 4, left=0.16, right=0.95, bottom=0.09, top=0.94, hspace=0.08, wspace=0.08
    )
    colorbar_axis = figure.add_axes((0.035, 0.09, 0.022, 0.64))
    top_axis = figure.add_subplot(grid[0, :3])
    joint_axis = figure.add_subplot(grid[1:, :3], sharex=top_axis)
    right_axis = figure.add_subplot(grid[1:, 3], sharey=joint_axis)

    top_axis.fill_between(x_marginal, x_density, color=COLORS[3], alpha=0.8)
    top_axis.plot(x_marginal, x_density, color=LINE_COLOR, linewidth=1.8)
    right_axis.fill_betweenx(y_marginal, y_density, color=COLORS[3], alpha=0.8)
    right_axis.plot(y_density, y_marginal, color=LINE_COLOR, linewidth=1.8)

    zero_density_color = "#d3d3d3"
    joint_axis.set_facecolor(zero_density_color)
    levels = hdr_levels(model["density"])
    if support_points is None:
        masked_density = np.ma.masked_less_equal(model["density"].T, 0.0)
    else:
        support = empirical_position_support(support_points, kde_limits, bins)
        masked_density = np.ma.masked_where(~support.T, model["density"].T)
    contour = joint_axis.contourf(
        model["x"],
        model["y"],
        masked_density,
        levels=levels,
        cmap=colormap,
        antialiased=True,
        corner_mask=False,
    )
    colorbar = figure.colorbar(contour, cax=colorbar_axis)
    colorbar.set_ticks(np.linspace(levels[0], levels[-1], 5))
    colorbar.set_label("KDE density", fontsize=9)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    joint_axis.scatter(
        *points.mean(axis=0), marker="+", s=65, linewidth=1.6, color=LINE_COLOR, zorder=10
    )

    title = feature_name.replace("_", " ").title()
    if feature_name == "agent_initial_position":
        count_text = f"frame 0 of all {episode_count:,} episodes"
    else:
        count_text = f"all {len(points):,} points"
    top_axis.set_title(f"TwoRoom — {title} ({count_text})", fontsize=11)
    joint_axis.set_xlabel(x_label)
    joint_axis.set_ylabel(y_label)
    joint_axis.set_xlim(display_limits[0])
    joint_axis.set_ylim(display_limits[1])
    joint_axis.grid(color="white", linewidth=0.8, alpha=0.75)
    joint_axis.set_axisbelow(True)

    top_axis.tick_params(axis="x", labelbottom=False)
    top_axis.tick_params(axis="y", left=False, labelleft=False)
    right_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    right_axis.tick_params(axis="y", labelleft=False)
    for axis in (top_axis, right_axis):
        axis.set_facecolor(zero_density_color)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for spine in joint_axis.spines.values():
        spine.set_color("#ddd9e5")

    output_path = output_dir / f"tworoom_{feature_name}_kde.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Saved {output_path} ({len(points):,} points; Scott bandwidth "
        f"{model['bandwidth'].tolist()})",
        flush=True,
    )


def plot_histogram(
    feature_name: str,
    points: np.ndarray,
    episode_count: int,
    output_dir: Path,
    bins: int,
    dpi: int,
) -> None:
    x_label, y_label, display_limits = FEATURES[feature_name]
    limits = padded_limits(points)
    histogram, x_edges, y_edges = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, range=limits.tolist()
    )
    x_counts = histogram.sum(axis=1)
    y_counts = histogram.sum(axis=0)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    positive_counts = histogram[histogram > 0]
    if not len(positive_counts):
        raise ValueError(f"{feature_name} histogram has no occupied bins")
    colormap = LinearSegmentedColormap.from_list("tworoom_histogram", COLORS)

    figure = plt.figure(figsize=(8.2, 8.2), facecolor="white")
    grid = figure.add_gridspec(
        4, 4, left=0.16, right=0.95, bottom=0.09, top=0.94, hspace=0.08, wspace=0.08
    )
    colorbar_axis = figure.add_axes((0.035, 0.09, 0.022, 0.64))
    top_axis = figure.add_subplot(grid[0, :3])
    joint_axis = figure.add_subplot(grid[1:, :3], sharex=top_axis)
    right_axis = figure.add_subplot(grid[1:, 3], sharey=joint_axis)

    top_axis.fill_between(
        x_centers, x_counts, step="mid", color=COLORS[3], alpha=0.8
    )
    top_axis.step(x_centers, x_counts, where="mid", color=LINE_COLOR, linewidth=1.2)
    right_axis.fill_betweenx(
        y_centers, y_counts, step="mid", color=COLORS[3], alpha=0.8
    )
    right_axis.step(y_counts, y_centers, where="mid", color=LINE_COLOR, linewidth=1.2)

    zero_count_color = "#d3d3d3"
    joint_axis.set_facecolor(zero_count_color)
    image = joint_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_less_equal(histogram.T, 0),
        cmap=colormap,
        norm=LogNorm(vmin=1.0, vmax=float(positive_counts.max())),
        shading="flat",
        antialiased=False,
        rasterized=True,
    )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Instance count per bin", fontsize=9)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")

    title = feature_name.replace("_", " ").title()
    if feature_name == "agent_initial_position":
        count_text = f"frame 0 of all {episode_count:,} episodes"
    else:
        count_text = f"all {len(points):,} points"
    top_axis.set_title(f"TwoRoom — {title} Histogram ({count_text})", fontsize=11)
    joint_axis.set_xlabel(x_label)
    joint_axis.set_ylabel(y_label)
    joint_axis.set_xlim(display_limits[0])
    joint_axis.set_ylim(display_limits[1])
    joint_axis.grid(color="white", linewidth=0.8, alpha=0.75)
    joint_axis.set_axisbelow(True)

    top_axis.tick_params(axis="x", labelbottom=False)
    top_axis.tick_params(axis="y", left=False, labelleft=False)
    right_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    right_axis.tick_params(axis="y", labelleft=False)
    for axis in (top_axis, right_axis):
        axis.set_facecolor(zero_count_color)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for spine in joint_axis.spines.values():
        spine.set_color("#ddd9e5")

    output_path = output_dir / f"tworoom_{feature_name}_histogram.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Saved {output_path} ({len(points):,} points; "
        f"{int(np.count_nonzero(histogram)):,}/{histogram.size:,} occupied bins)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.bins < 64:
        raise ValueError("bins must be at least 64")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, episode_count = load_features(args.dataset)
    for feature_name, points in features.items():
        if args.histogram_only:
            plot_histogram(
                feature_name,
                points,
                episode_count,
                args.output_dir,
                args.bins,
                args.dpi,
            )
            continue
        support_points = None if feature_name == "action" else features["agent_position"]
        plot_feature(
            feature_name,
            points,
            episode_count,
            args.output_dir,
            args.bins,
            args.dpi,
            support_points,
        )


if __name__ == "__main__":
    main()
