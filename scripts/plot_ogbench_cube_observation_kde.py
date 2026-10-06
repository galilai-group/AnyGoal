"""Semantic KDE plots for the 28-D OGBench Cube state observation."""

from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from anygoal.paths import data_root


DATASET = data_root() / "cube_single_expert.h5"
OUTPUT = Path("assets/paper/ogbench_cube_analysis/semantic_observation_kde")
GREY = "#d3d3d3"
PURPLE = LinearSegmentedColormap.from_list("cube_purple", ["#faf9ff", "#ded7f7", "#8e7bc8"])
TEAL = LinearSegmentedColormap.from_list("cube_teal", ["#f7fffc", "#b8e1db", "#287b70"])

SCALAR_FIELDS = {
    **{index: f"UR5e joint position {index} (rad)" for index in range(6)},
    **{index: f"UR5e joint velocity {index - 6} (rad/s)" for index in range(6, 12)},
    17: "Gripper opening × 3",
    18: "Gripper contact force norm",
    22: "Cube quaternion w",
    23: "Cube quaternion x",
    24: "Cube quaternion y",
    25: "Cube quaternion z",
}
ACTION_FIELDS = {
    0: "Relative end-effector Δx (normalized; ±0.05 m)",
    1: "Relative end-effector Δy (normalized; ±0.05 m)",
    2: "Relative end-effector Δz (normalized; ±0.05 m)",
    3: "Relative end-effector Δyaw (normalized; ±0.3 rad)",
    4: "Relative gripper opening Δ (normalized)",
}


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.001, 0.999))
    span = high - low
    if span == 0:
        span = max(abs(float(low)) * 0.08, 0.1)
    return float(low - 0.04 * span), float(high + 0.04 * span)


def density_1d(values: np.ndarray, bins: int = 250):
    limits = robust_limits(values)
    counts, edges = np.histogram(values, bins=bins, range=limits)
    width = edges[1] - edges[0]
    bandwidth = max(values.std() * len(values) ** (-1 / 5), width * 0.5)
    density = gaussian_filter1d(counts.astype(float), bandwidth / width, mode="constant")
    density /= density.sum() * width
    return (edges[:-1] + edges[1:]) / 2, density


def plot_scalar_kdes(observation: np.ndarray) -> None:
    figure, axes = plt.subplots(5, 4, figsize=(15, 13))
    for axis, (index, label) in zip(axes.ravel(), SCALAR_FIELDS.items()):
        x, density = density_1d(observation[:, index])
        axis.set_facecolor(GREY)
        supported = density > density.max() * 1e-5
        axis.fill_between(x, density, where=supported, color="#ac9cdd", alpha=0.9)
        axis.plot(x[supported], density[supported], color="#7461b5", linewidth=1.25)
        axis.set_title(f"observation[{index}] — {label}", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(color="white", linewidth=0.7, alpha=0.75)
    for axis in axes.ravel()[len(SCALAR_FIELDS):]:
        axis.set_visible(False)
    figure.suptitle("OGBench Cube: 1-D observation KDEs (grey = no KDE support)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(OUTPUT / "observation_scalar_1d_kde.png", dpi=200)
    plt.close(figure)


def plot_action_scalar_kdes(actions: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(18, 3.7))
    for axis, (index, label) in zip(axes, ACTION_FIELDS.items()):
        x, density = density_1d(actions[:, index])
        axis.set_facecolor(GREY)
        supported = density > density.max() * 1e-5
        axis.fill_between(x, density, where=supported, color="#ac9cdd", alpha=0.9)
        axis.plot(x[supported], density[supported], color="#7461b5", linewidth=1.25)
        axis.set_title(f"action[{index}]\n{label}", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(color="white", linewidth=0.7, alpha=0.75)
    figure.suptitle("OGBench Cube: 1-D action KDEs (grey = no KDE support)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(OUTPUT / "action_scalar_1d_kde.png", dpi=200)
    plt.close(figure)


def joint_density(points: np.ndarray, bins: int = 260):
    limits = np.asarray([robust_limits(points[:, dimension]) for dimension in range(2)])
    histogram, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=limits)
    widths = np.array([x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]])
    bandwidth = np.maximum(points.std(axis=0) * len(points) ** (-1 / 6) / widths, 0.5)
    density = gaussian_filter(histogram, bandwidth, mode="constant")
    density[density < density.max() * 1e-5] = 0.0
    return density, x_edges, y_edges


def plot_yaw_pairs(observation: np.ndarray) -> None:
    pairs = [
        (observation[:, 15:17], "End-effector yaw: cos / sin", "effector"),
        (observation[:, 26:28], "Cube yaw: cos / sin", "cube"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for axis, (points, title, name) in zip(axes, pairs):
        density, x_edges, y_edges = joint_density(points)
        x = (x_edges[:-1] + x_edges[1:]) / 2
        y = (y_edges[:-1] + y_edges[1:]) / 2
        axis.set_facecolor(GREY)
        masked = np.ma.masked_less_equal(density.T, 0)
        contour = axis.contourf(x, y, masked, levels=14, cmap=PURPLE)
        axis.contour(x, y, masked, levels=8, colors="#665092", linewidths=0.55)
        axis.set(xlabel=f"cos({name}_yaw)", ylabel=f"sin({name}_yaw)", title=title, aspect="equal")
        axis.grid(color="white", linewidth=0.7, alpha=0.7)
        figure.colorbar(contour, ax=axis, label="KDE density", shrink=0.78)
    figure.suptitle("OGBench Cube: yaw joint KDEs (grey = no KDE support)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT / "observation_yaw_2d_joint_kde.png", dpi=200)
    plt.close(figure)


def plot_action_xyz_pairs(actions: np.ndarray) -> None:
    """Plot exact 2-D marginals of the displayed 3-D xyz KDE.

    This intentionally does not fit independent 2-D KDEs: the support masks and
    contours must agree exactly with the corresponding 3-D projection planes.
    """
    xyz_density, centers, _ = density_3d(actions[:, :3])
    pairs = [
        (0, 1, xyz_density.sum(axis=2)),
        (0, 2, xyz_density.sum(axis=1)),
        (1, 2, xyz_density.sum(axis=0)),
    ]
    labels = ("Δx", "Δy", "Δz")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for axis, (first, second, density) in zip(axes, pairs):
        x, y = centers[first], centers[second]
        axis.set_facecolor(GREY)
        masked = np.ma.masked_less_equal(density.T, density.max() * 0.01)
        contour = axis.contourf(x, y, masked, levels=14, cmap=PURPLE)
        axis.contour(x, y, masked, levels=8, colors="#665092", linewidths=0.55)
        axis.set(xlabel=f"action[{first}] ({labels[first]})", ylabel=f"action[{second}] ({labels[second]})", title=f"{labels[first]} / {labels[second]} relative action", aspect="equal")
        axis.grid(color="white", linewidth=0.7, alpha=0.7)
        figure.colorbar(contour, ax=axis, label="KDE density", shrink=0.76)
    figure.suptitle("OGBench Cube: xyz action joint KDEs (grey = no KDE support)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(OUTPUT / "action_xyz_2d_joint_kde.png", dpi=200)
    plt.close(figure)


def density_3d(points: np.ndarray, bins: int = 72):
    limits = np.asarray([robust_limits(points[:, dimension]) for dimension in range(3)])
    histogram, edges = np.histogramdd(points, bins=bins, range=limits.tolist())
    widths = np.array([edge[1] - edge[0] for edge in edges])
    bandwidth = np.maximum(points.std(axis=0) * len(points) ** (-1 / 7) / widths, 0.5)
    density = gaussian_filter(histogram, bandwidth, mode="constant")
    density[density < density.max() * 2e-4] = 0.0
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    return density, centers, limits


def plot_3d_kde(points: np.ndarray, title: str, labels: tuple[str, str, str], filename: str) -> None:
    density, centers, limits = density_3d(points)
    grid = np.meshgrid(*centers, indexing="ij")
    positive = np.flatnonzero(density.ravel() > density.max() * 0.025)
    if len(positive) > 5000:
        positive = positive[np.linspace(0, len(positive) - 1, 5000, dtype=int)]
    figure = plt.figure(figsize=(9.2, 8.1))
    axis = figure.add_subplot(111, projection="3d")
    axis.xaxis.pane.set_facecolor(GREY)
    axis.yaxis.pane.set_facecolor(GREY)
    axis.zaxis.pane.set_facecolor(GREY)
    axis.xaxis.pane.set_alpha(0.35)
    axis.yaxis.pane.set_alpha(0.35)
    axis.zaxis.pane.set_alpha(0.35)
    max_density = density.max()
    cloud = axis.scatter(
        grid[0].ravel()[positive], grid[1].ravel()[positive], grid[2].ravel()[positive],
        c=density.ravel()[positive], cmap=TEAL, s=8 + 42 * density.ravel()[positive] / max_density,
        alpha=0.38, linewidths=0,
    )
    x_grid, y_grid = np.meshgrid(centers[0], centers[1], indexing="ij")
    xy = density.sum(axis=2)
    xy = np.ma.masked_less_equal(xy, xy.max() * 0.01)
    axis.contourf(x_grid, y_grid, xy, zdir="z", offset=limits[2, 0], levels=12, cmap=TEAL, alpha=0.9)
    x_grid, z_grid = np.meshgrid(centers[0], centers[2], indexing="ij")
    xz = density.sum(axis=1)
    xz = np.ma.masked_less_equal(xz, xz.max() * 0.01)
    axis.contourf(x_grid, xz * 0 + limits[1, 1], z_grid, zdir="y", offset=limits[1, 1], levels=12, cmap=TEAL, alpha=0.9)
    y_grid, z_grid = np.meshgrid(centers[1], centers[2], indexing="ij")
    yz = density.sum(axis=0)
    yz = np.ma.masked_less_equal(yz, yz.max() * 0.01)
    axis.contourf(yz * 0 + limits[0, 0], y_grid, z_grid, zdir="x", offset=limits[0, 0], levels=12, cmap=TEAL, alpha=0.9)
    axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2], title=title)
    axis.view_init(elev=27, azim=-58)
    figure.colorbar(cloud, ax=axis, shrink=0.62, pad=0.08, label="3-D KDE density")
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATASET, "r") as dataset:
        observation = dataset["observation"][:].astype(np.float64)
        actions = dataset["action"][:].astype(np.float64)
    actions = actions[np.isfinite(actions).all(axis=1)]
    plot_scalar_kdes(observation)
    plot_yaw_pairs(observation)
    plot_3d_kde(observation[:, 12:15], "End-effector position: 3-D KDE", ("x (normalized)", "y (normalized)", "z (normalized)"), "effector_position_3d_kde.png")
    plot_3d_kde(observation[:, 19:22], "Cube position: 3-D KDE", ("x (normalized)", "y (normalized)", "z (normalized)"), "cube_position_3d_kde.png")
    plot_action_scalar_kdes(actions)
    plot_action_xyz_pairs(actions)
    plot_3d_kde(actions[:, :3], "Relative end-effector xyz action: 3-D KDE", ("Δx (normalized)", "Δy (normalized)", "Δz (normalized)"), "action_xyz_3d_kde.png")


if __name__ == "__main__":
    main()
