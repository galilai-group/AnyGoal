"""Research visualization helpers for counterfactual-sensitivity bias axes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


GRID_ENVIRONMENTS = (
    ("tworoom", "default", "corridor_center_distance", "TwoRoom"),
    ("pusht", "default", "agent_object_distance", "Push-T"),
    (
        "insert_peg",
        "default",
        "peg_target_distance_after_grip",
        "Insert Peg",
    ),
    ("cube", "horizontal_action_by_height", "effector_z", "Cube"),
)


def plot_counterfactual_biases(
    sensitivities: np.ndarray,
    bias_axes: Mapping[str, np.ndarray],
    metrics: Mapping[str, dict],
    output_directory: str | Path,
    *,
    model_name: str,
    environment: str,
    gridsize: int = 90,
    dpi: int = 220,
) -> list[Path]:
    """Save one sensitivity hexbin with CSU bin medians per bias axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensitivities = np.asarray(sensitivities)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, bias_values in bias_axes.items():
        values = np.asarray(bias_values)
        axis_metrics = metrics[name]
        edges = np.asarray(axis_metrics["bin_edges"])
        populated = np.asarray(axis_metrics["populated_bin_indices"], dtype=int)
        medians = np.asarray(axis_metrics["bin_median_sensitivities"])
        centers = (edges[:-1] + edges[1:]) / 2.0

        figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        density = axis.hexbin(
            values,
            sensitivities,
            gridsize=gridsize,
            bins="log",
            cmap="managua",
            mincnt=1,
        )
        figure.colorbar(density, ax=axis, label="log10(count)")
        axis.plot(
            centers[populated],
            medians,
            color="white",
            marker="o",
            markeredgecolor="black",
            linewidth=2,
            label="Per-bin median $U_b$",
        )
        axis.set_xlabel(name.replace("_", " "))
        axis.set_ylabel("Counterfactual sensitivity $S_i$")
        axis.set_title(
            f"{environment} · {model_name}\n"
            f"RAS={metrics['ras']:.4f}, CSU={axis_metrics['value']:.4f}"
        )
        axis.grid(True, alpha=0.2)
        axis.legend()
        path = output_directory / f"counterfactual_bias_{name}.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_counterfactual_bias_grid(
    panels: list[Mapping[str, object]],
    output_path: str | Path,
    *,
    model_name: str,
    representation: str,
    gridsize: int = 90,
    dpi: int = 220,
) -> Path:
    """Plot four equally sized dataset panels with one shared count scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns

    if len(panels) != 4:
        raise ValueError(f"Expected four dataset panels, got {len(panels)}")

    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    figure = plt.figure(figsize=(17.5, 4.4), constrained_layout=False)
    grid = figure.add_gridspec(
        1, 5, width_ratios=(1, 1, 1, 1, 0.045), wspace=0.30
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    colorbar_axis = figure.add_subplot(grid[0, 4])
    densities = []
    for axis, panel in zip(axes, panels, strict=True):
        density = axis.hexbin(
            np.asarray(panel["bias"]),
            np.asarray(panel["sensitivity"]),
            gridsize=gridsize,
            mincnt=1,
            cmap=cmap,
        )
        densities.append(density)
        axis.set_box_aspect(1)
        axis.set_xlabel(str(panel["axis_label"]))
        axis.set_title(
            f"{panel['title']}\nRAS={panel['ras']:.3f}, CSU={panel['csu']:.3f}"
        )
        axis.grid(True, alpha=0.15)
    axes[0].set_ylabel("Counterfactual sensitivity $S_i$")

    maximum_count = max(float(density.get_array().max()) for density in densities)
    shared_norm = LogNorm(vmin=1, vmax=max(maximum_count, 1.0))
    for density in densities:
        density.set_norm(shared_norm)
    figure.colorbar(densities[-1], cax=colorbar_axis, label="Count (log scale)")
    figure.suptitle(f"{model_name} · {representation}", y=0.995)
    figure.subplots_adjust(left=0.055, right=0.96, bottom=0.16, top=0.82)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


__all__ = [
    "GRID_ENVIRONMENTS",
    "plot_counterfactual_bias_grid",
    "plot_counterfactual_biases",
]
