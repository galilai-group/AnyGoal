"""Visualize all PushT agent/action pairs with discrete action-angle colors."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    PowerNorm,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
ANGLE_BIN_WIDTH = 45
ANGLE_BIN_COUNT = 8


def default_dataset_path(filename: str) -> Path:
    path = DATASET_DIR / filename
    legacy_path = DATASET_DIR.parent / filename
    return path if path.exists() or not legacy_path.exists() else legacy_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path("pusht_expert_train.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "agent_position_action_angle.jpg",
    )
    parser.add_argument(
        "--continuous-colormap-output",
        type=Path,
        default=RESULTS_DIR / "action_angle_colormap_continuous.jpg",
    )
    parser.add_argument(
        "--discrete-colormap-output",
        type=Path,
        default=RESULTS_DIR / "action_angle_colormap_discrete.jpg",
    )
    parser.add_argument("--gridsize", type=int, default=65)
    parser.add_argument("--kde-bins", type=int, default=180)
    parser.add_argument("--kde-sigma", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--position-key", default="proprio")
    parser.add_argument("--task-label", default="PushT")
    parser.add_argument("--coordinate-limit", type=float, default=500.0)
    parser.add_argument("--skip-angle-wheels", action="store_true")
    return parser.parse_args()


def continuous_color(angle_degrees: np.ndarray) -> np.ndarray:
    hues = ((np.asarray(angle_degrees) + 180.0) % 360.0) / 360.0
    return plt.get_cmap("hsv")(hues)


def discrete_angle_colors() -> np.ndarray:
    middle_angles = np.arange(ANGLE_BIN_COUNT) * ANGLE_BIN_WIDTH + 22.5
    return continuous_color(middle_angles)


def modal_category(values: np.ndarray) -> int:
    categories = np.asarray(values, dtype=np.int64)
    return int(np.argmax(np.bincount(categories, minlength=ANGLE_BIN_COUNT)))


def render_angle_wheel(
    path: Path,
    *,
    discrete: bool,
    colors: np.ndarray,
    dpi: int,
) -> None:
    grid_size = 900
    axis_values = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis_values, axis_values)
    radius = np.sqrt(xx * xx + yy * yy)
    angles = (np.degrees(np.arctan2(yy, xx)) + 360.0) % 360.0
    if discrete:
        bins = (angles // ANGLE_BIN_WIDTH).astype(np.int64) % ANGLE_BIN_COUNT
        wheel = colors[bins]
        title = "Discrete Action Angle Colormap (8 bins)"
    else:
        wheel = continuous_color(angles)
        title = "Continuous Action Angle Colormap"
    alpha = (radius <= 1.0).astype(np.float64)[..., None]
    wheel_rgb = wheel[..., :3] * alpha + (1.0 - alpha)

    figure, plot_axis = plt.subplots(figsize=(5.6, 5.6), dpi=dpi)
    plot_axis.imshow(wheel_rgb, extent=[-1, 1, -1, 1], origin="lower")
    plot_axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.75)
    plot_axis.axvline(0.0, color="#111827", linewidth=0.8, alpha=0.75)
    plot_axis.set_title(title)
    plot_axis.set_xlabel("action[0]")
    plot_axis.set_ylabel("action[1]")
    plot_axis.set_aspect("equal", adjustable="box")
    plot_axis.set_xticks([-1, 0, 1])
    plot_axis.set_yticks([-1, 0, 1])
    figure.tight_layout()
    figure.savefig(path, format="jpg", pil_kwargs={"quality": 95})
    plt.close(figure)


def render_modal_hexbin(
    path: Path,
    points: np.ndarray,
    angle_bins: np.ndarray,
    colors: np.ndarray,
    gridsize: int,
    dpi: int,
    coordinate_limit: float,
    task_label: str,
    position_key: str,
) -> None:
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, ANGLE_BIN_COUNT + 0.5), ANGLE_BIN_COUNT)
    figure, axis = plt.subplots(figsize=(8, 8), dpi=dpi)
    hexbins = axis.hexbin(
        points[:, 0],
        points[:, 1],
        C=angle_bins,
        reduce_C_function=modal_category,
        gridsize=gridsize,
        extent=(0.0, coordinate_limit, 0.0, coordinate_limit),
        mincnt=1,
        cmap=cmap,
        norm=norm,
    )
    colorbar = figure.colorbar(
        hexbins,
        ax=axis,
        ticks=np.arange(ANGLE_BIN_COUNT),
        shrink=0.82,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(
        [
            f"{start:03d}-{start + ANGLE_BIN_WIDTH:03d}°"
            for start in range(0, 360, ANGLE_BIN_WIDTH)
        ]
    )
    colorbar.set_label("Most frequent action-angle range in hexbin")
    axis.set_title(
        f"{task_label} Agent Position by Modal Action Angle\n"
        f"all {len(points):,} state-action pairs"
    )
    axis.set_xlabel(f"{position_key}[0] / agent x")
    axis.set_ylabel(f"{position_key}[1] / agent y")
    axis.set_xlim(0.0, coordinate_limit)
    axis.set_ylim(0.0, coordinate_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.5)
    figure.tight_layout()
    figure.savefig(path, format="jpg", pil_kwargs={"quality": 95})
    plt.close(figure)


def render_angle_kde(
    path: Path,
    points: np.ndarray,
    color: np.ndarray,
    start_angle: int,
    bins: int,
    sigma: float,
    dpi: int,
    coordinate_limit: float,
    task_label: str,
    position_key: str,
) -> None:
    histogram, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=((0.0, coordinate_limit), (0.0, coordinate_limit)),
    )
    density = gaussian_filter(histogram.T, sigma=sigma)
    positive_density = density[density > 0.0]
    upper = (
        float(np.quantile(positive_density, 0.995))
        if len(positive_density)
        else 1.0
    )
    color_hex = matplotlib.colors.to_hex(color)
    cmap = LinearSegmentedColormap.from_list(
        f"angle_{start_angle:03d}", ["#ffffff", color_hex]
    )

    figure, axis = plt.subplots(figsize=(8, 8), dpi=dpi)
    image = axis.imshow(
        density,
        origin="lower",
        extent=(0.0, coordinate_limit, 0.0, coordinate_limit),
        cmap=cmap,
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=max(upper, 1e-12)),
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axis, label="Gaussian-smoothed bin count")
    axis.set_title(
        f"{task_label} Agent-Position KDE\n"
        f"action angle [{start_angle}, {start_angle + ANGLE_BIN_WIDTH})°, "
        f"n={len(points):,}"
    )
    axis.set_xlabel(f"{position_key}[0] / agent x")
    axis.set_ylabel(f"{position_key}[1] / agent y")
    axis.set_xlim(0.0, coordinate_limit)
    axis.set_ylim(0.0, coordinate_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.45)
    figure.tight_layout()
    figure.savefig(path, format="jpg", pil_kwargs={"quality": 95})
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.gridsize < 2 or args.kde_bins < 2:
        raise ValueError("gridsize and kde-bins must be at least 2.")
    if args.kde_sigma <= 0.0 or args.dpi < 1:
        raise ValueError("kde-sigma and dpi must be positive.")
    if args.coordinate_limit <= 0.0:
        raise ValueError("coordinate-limit must be positive.")

    dataset_path = args.dataset.expanduser().resolve()
    with h5py.File(dataset_path, "r") as dataset:
        if args.position_key not in dataset or "action" not in dataset:
            raise KeyError(
                f"{args.position_key}/action columns not found in {dataset_path}"
            )
        points = dataset[args.position_key][:, :2].astype(np.float32)
        actions = dataset["action"][:, :2].astype(np.float32)
    if len(points) != len(actions):
        raise ValueError("proprio/action length mismatch.")
    valid = np.isfinite(points).all(axis=1) & np.isfinite(actions).all(axis=1)
    points = points[valid]
    actions = actions[valid]
    if not len(points):
        raise RuntimeError(f"No finite state-action pairs in {dataset_path}")

    angles = (np.degrees(np.arctan2(actions[:, 1], actions[:, 0])) + 360.0) % 360.0
    angle_bins = (
        (angles // ANGLE_BIN_WIDTH).astype(np.int64) % ANGLE_BIN_COUNT
    )
    colors = discrete_angle_colors()
    output_path = args.output.expanduser().resolve()
    continuous_path = args.continuous_colormap_output.expanduser().resolve()
    discrete_path = args.discrete_colormap_output.expanduser().resolve()
    paths = [output_path]
    if not args.skip_angle_wheels:
        paths.extend((continuous_path, discrete_path))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    render_modal_hexbin(
        output_path,
        points,
        angle_bins,
        colors,
        args.gridsize,
        args.dpi,
        args.coordinate_limit,
        args.task_label,
        args.position_key,
    )
    if not args.skip_angle_wheels:
        render_angle_wheel(
            continuous_path, discrete=False, colors=colors, dpi=args.dpi
        )
        render_angle_wheel(
            discrete_path, discrete=True, colors=colors, dpi=args.dpi
        )

    bin_paths = []
    for bin_index, start_angle in enumerate(range(0, 360, ANGLE_BIN_WIDTH)):
        mask = angle_bins == bin_index
        bin_path = output_path.with_name(
            f"{output_path.stem}_angle_{start_angle:03d}_"
            f"{start_angle + ANGLE_BIN_WIDTH:03d}_kde{output_path.suffix}"
        )
        render_angle_kde(
            bin_path,
            points[mask],
            colors[bin_index],
            start_angle,
            args.kde_bins,
            args.kde_sigma,
            args.dpi,
            args.coordinate_limit,
            args.task_label,
            args.position_key,
        )
        bin_paths.append(bin_path)
        print(
            f"angle [{start_angle}, {start_angle + ANGLE_BIN_WIDTH})°: "
            f"{int(mask.sum()):,} points -> {bin_path}"
        )

    print(f"dataset: {dataset_path}")
    print(f"all finite state-action pairs: {len(points):,}")
    print(f"saved modal-angle hexbin: {output_path}")
    if not args.skip_angle_wheels:
        print(f"saved continuous colormap: {continuous_path}")
        print(f"saved discrete colormap: {discrete_path}")
    print(f"saved per-angle KDE figures: {len(bin_paths)}")


if __name__ == "__main__":
    main()
