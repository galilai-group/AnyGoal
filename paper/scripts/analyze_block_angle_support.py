"""Diagnose PushT-reference block-angle support failures by episode and time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BINS = 256
REFERENCE_KDE_QUANTILE = 0.01
REFERENCE_MAHALANOBIS_QUANTILE = 0.99
REQUIRED_TARGET_FRACTION = 0.95
PUSHT_GOAL_ANGLE = np.pi / 4.0
TARGET_GOAL_ANGLE = -3.0 * np.pi / 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument(
        "--pusht",
        type=Path,
        default=Path.home() / ".stable_worldmodel" / "pusht_expert_train.h5",
    )
    parser.add_argument(
        "--stackt",
        type=Path,
        default=(
            Path.home()
            / ".stable_worldmodel"
            / "datasets"
            / "stackt_test.h5"
        ),
    )
    parser.add_argument(
        "--antigoal",
        type=Path,
        default=Path("pusht_antigoal_test.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "results"
            / "distribution_support"
            / "block_angle_analysis"
        ),
    )
    return parser.parse_args()


def wrap_angle(values: np.ndarray, period: float = 2.0 * np.pi) -> np.ndarray:
    return (values + period / 2.0) % period - period / 2.0


def load_angles(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as dataset:
        result = {"angle": wrap_angle(dataset["state"][:, 4].astype(np.float64))}
        for field in ("episode_idx", "step_idx"):
            if field in dataset:
                result[field] = dataset[field][:].astype(np.int64)
    return result


def fit_periodic_kde(
    reference: np.ndarray,
    *,
    period: float = 2.0 * np.pi,
    bandwidth_multiplier: float = 1.0,
) -> dict[str, np.ndarray | float]:
    reference = wrap_angle(reference, period)
    edges = np.linspace(-period / 2.0, period / 2.0, BINS + 1)
    width = float(edges[1] - edges[0])
    histogram, _ = np.histogram(reference, bins=edges)
    phase_scale = 2.0 * np.pi / period
    mean_phase = np.arctan2(
        np.sin(reference * phase_scale).mean(),
        np.cos(reference * phase_scale).mean(),
    )
    mean_angle = mean_phase / phase_scale
    centered = wrap_angle(reference - mean_angle, period)
    bandwidth = float(
        centered.std(ddof=0)
        * len(reference) ** (-1.0 / 5.0)
        * bandwidth_multiplier
    )
    sigma_bins = max(bandwidth / width, 0.5)
    density = gaussian_filter1d(
        histogram.astype(np.float64),
        sigma=sigma_bins,
        mode="wrap",
        truncate=4.0,
    )
    density /= density.sum() * width
    return {
        "edges": edges,
        "centers": (edges[:-1] + edges[1:]) / 2.0,
        "density": density,
        "bandwidth": bandwidth,
        "sigma_bins": sigma_bins,
    }


def lookup_density(
    values: np.ndarray, model: dict[str, np.ndarray | float], period: float
) -> np.ndarray:
    wrapped = wrap_angle(values, period)
    edges = model["edges"]
    indices = np.searchsorted(edges, wrapped, side="right") - 1
    indices %= BINS
    return model["density"][indices]


def kde_result(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    period: float = 2.0 * np.pi,
    reference_quantile: float = REFERENCE_KDE_QUANTILE,
    bandwidth_multiplier: float = 1.0,
) -> dict:
    model = fit_periodic_kde(
        reference,
        period=period,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    reference_density = lookup_density(reference, model, period)
    target_density = lookup_density(target, model, period)
    threshold = float(np.quantile(reference_density, reference_quantile))
    supported = target_density >= threshold
    return {
        "model": model,
        "reference_density_at_points": reference_density,
        "target_density_at_points": target_density,
        "threshold": threshold,
        "supported": supported,
        "target_fraction": float(supported.mean()),
    }


def mahalanobis_result(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    period: float = 2.0 * np.pi,
) -> dict:
    phase_scale = 2.0 * np.pi / period
    reference_phase = wrap_angle(reference, period) * phase_scale
    target_phase = wrap_angle(target, period) * phase_scale
    reference_embedding = np.column_stack(
        (np.cos(reference_phase), np.sin(reference_phase))
    )
    target_embedding = np.column_stack(
        (np.cos(target_phase), np.sin(target_phase))
    )
    mean = reference_embedding.mean(axis=0)
    covariance = np.cov(reference_embedding, rowvar=False, bias=True)
    regularizer = max(float(np.trace(covariance) / 2.0) * 1e-8, 1e-12)
    inverse = np.linalg.inv(covariance + np.eye(2) * regularizer)

    def squared_distance(points: np.ndarray) -> np.ndarray:
        centered = points - mean
        return np.einsum("ni,ij,nj->n", centered, inverse, centered)

    reference_distance = squared_distance(reference_embedding)
    target_distance = squared_distance(target_embedding)
    threshold = float(
        np.quantile(reference_distance, REFERENCE_MAHALANOBIS_QUANTILE)
    )
    return {
        "reference_distance": reference_distance,
        "target_distance": target_distance,
        "threshold": threshold,
        "target_fraction": float(np.mean(target_distance <= threshold)),
    }


def low_density_intervals(model: dict, threshold: float) -> list[list[float]]:
    low = np.asarray(model["density"]) < threshold
    edges = np.asarray(model["edges"])
    changes = np.diff(np.r_[False, low, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        [float(np.rad2deg(edges[start])), float(np.rad2deg(edges[stop]))]
        for start, stop in zip(starts, stops)
    ]


def episode_diagnostics(
    angles: np.ndarray,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    supported: np.ndarray,
    progress_bins: int = 40,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    rows = []
    progress = np.empty(len(angles), dtype=np.float64)
    episode_ids = np.unique(episode_idx)
    heatmap = np.full((len(episode_ids), progress_bins), np.nan)
    for row_index, episode in enumerate(episode_ids):
        indices = np.flatnonzero(episode_idx == episode)
        indices = indices[np.argsort(step_idx[indices])]
        count = len(indices)
        episode_progress = np.arange(count) / max(count - 1, 1)
        progress[indices] = episode_progress
        bin_indices = np.minimum((episode_progress * progress_bins).astype(int), progress_bins - 1)
        for bin_index in range(progress_bins):
            selected = indices[bin_indices == bin_index]
            if len(selected):
                heatmap[row_index, bin_index] = supported[selected].mean()
        rows.append(
            {
                "episode": int(episode),
                "length": count,
                "support_fraction": float(supported[indices].mean()),
                "unsupported_points": int((~supported[indices]).sum()),
                "initial_angle_degrees": float(np.rad2deg(angles[indices[0]])),
                "terminal_angle_degrees": float(np.rad2deg(angles[indices[-1]])),
                "initial_supported": bool(supported[indices[0]]),
                "terminal_supported": bool(supported[indices[-1]]),
            }
        )

    progress_support = np.array(
        [
            supported[
                (progress >= bin_index / progress_bins)
                & (progress < (bin_index + 1) / progress_bins)
            ].mean()
            for bin_index in range(progress_bins)
        ]
    )
    fractions = np.array([row["support_fraction"] for row in rows])
    unsupported_counts = np.array([row["unsupported_points"] for row in rows])
    order = np.argsort(fractions)
    heatmap = heatmap[order]
    contribution_order = np.argsort(unsupported_counts)[::-1]
    cumulative_contribution = np.cumsum(unsupported_counts[contribution_order])
    cumulative_contribution = cumulative_contribution / cumulative_contribution[-1]

    keep = np.ones(len(angles), dtype=bool)
    removed_episodes = 0
    for row_index in order:
        if supported[keep].mean() >= REQUIRED_TARGET_FRACTION:
            break
        keep[episode_idx == rows[row_index]["episode"]] = False
        removed_episodes += 1

    largest_passing_cutoff = None
    for cutoff in np.linspace(0.01, 1.0, 100):
        selected = progress <= cutoff
        if supported[selected].mean() >= REQUIRED_TARGET_FRACTION:
            largest_passing_cutoff = float(cutoff)

    summary = {
        "episode_count": int(len(rows)),
        "episodes_with_any_unsupported_point": int(
            sum(row["unsupported_points"] > 0 for row in rows)
        ),
        "episodes_below_0.95_support": int(
            sum(row["support_fraction"] < REQUIRED_TARGET_FRACTION for row in rows)
        ),
        "episode_support_quantiles": {
            str(quantile): float(np.quantile(fractions, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "mean_episode_support": float(fractions.mean()),
        "episodes_accounting_for_50pct_of_unsupported_points": int(
            np.searchsorted(cumulative_contribution, 0.5) + 1
        ),
        "episodes_accounting_for_80pct_of_unsupported_points": int(
            np.searchsorted(cumulative_contribution, 0.8) + 1
        ),
        "top_10_episode_share_of_unsupported_points": float(
            cumulative_contribution[min(9, len(cumulative_contribution) - 1)]
        ),
        "initial_state_support": float(
            np.mean([row["initial_supported"] for row in rows])
        ),
        "terminal_state_support": float(
            np.mean([row["terminal_supported"] for row in rows])
        ),
        "episodes_removed_from_worst_up_to_reach_0.95": removed_episodes,
        "fraction_of_points_removed_with_those_episodes": float(1.0 - keep.mean()),
        "remaining_support_after_removal": float(supported[keep].mean()),
        "largest_normalized_progress_cutoff_that_passes": largest_passing_cutoff,
        "fraction_of_points_kept_at_progress_cutoff": float(
            np.mean(progress <= largest_passing_cutoff)
        ),
        "support_by_normalized_progress_bin": progress_support.tolist(),
    }
    return rows, heatmap, progress_support, summary


def alternative_results(reference: np.ndarray, target: np.ndarray) -> dict:
    raw_kde = kde_result(reference, target)
    raw_mahalanobis = mahalanobis_result(reference, target)
    goal_relative_target = wrap_angle(target - TARGET_GOAL_ANGLE)
    goal_relative_reference = wrap_angle(reference - PUSHT_GOAL_ANGLE)
    relative_kde = kde_result(goal_relative_reference, goal_relative_target)
    relative_mahalanobis = mahalanobis_result(
        goal_relative_reference, goal_relative_target
    )
    axial_kde = kde_result(reference, target, period=np.pi)
    axial_mahalanobis = mahalanobis_result(reference, target, period=np.pi)

    q_density_fifth_percentile = float(
        np.quantile(raw_kde["target_density_at_points"], 0.05, method="lower")
    )
    critical_reference_kde_quantile = float(
        np.mean(
            raw_kde["reference_density_at_points"]
            <= q_density_fifth_percentile
        )
    )
    target_distance_95 = float(
        np.quantile(raw_mahalanobis["target_distance"], 0.95)
    )
    critical_reference_mahalanobis_quantile = float(
        np.mean(raw_mahalanobis["reference_distance"] <= target_distance_95)
    )
    bandwidth_scan = {}
    for multiplier in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        bandwidth_scan[str(multiplier)] = kde_result(
            reference, target, bandwidth_multiplier=multiplier
        )["target_fraction"]
    return {
        "raw_world_angle": {
            "kde": raw_kde["target_fraction"],
            "mahalanobis": raw_mahalanobis["target_fraction"],
        },
        "goal_relative_angle": {
            "kde": relative_kde["target_fraction"],
            "mahalanobis": relative_mahalanobis["target_fraction"],
        },
        "orientation_modulo_pi": {
            "kde": axial_kde["target_fraction"],
            "mahalanobis": axial_mahalanobis["target_fraction"],
        },
        "kde_with_reference_tail_0.005": kde_result(
            reference, target, reference_quantile=0.005
        )["target_fraction"],
        "reference_kde_quantile_needed_for_0.95_target_support": (
            critical_reference_kde_quantile
        ),
        "reference_mahalanobis_quantile_needed_for_0.95_target_support": (
            critical_reference_mahalanobis_quantile
        ),
        "kde_bandwidth_multiplier_scan": bandwidth_scan,
    }


def plot_kde_comparison(
    reference: np.ndarray,
    targets: dict[str, np.ndarray],
    raw_results: dict[str, dict],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), sharex=True, sharey=True)
    reference_model = raw_results["StackT"]["kde"]["model"]
    x = np.rad2deg(reference_model["centers"])
    for axis, (name, target) in zip(axes, targets.items()):
        target_model = fit_periodic_kde(target)
        threshold = raw_results[name]["kde"]["threshold"]
        low = reference_model["density"] < threshold
        axis.fill_between(
            x,
            0.0,
            np.maximum(reference_model["density"], target_model["density"]),
            where=low,
            color="#d95f59",
            alpha=0.16,
            step="mid",
            label="Below PushT 1% density threshold",
        )
        axis.plot(x, reference_model["density"], color="#355c9a", lw=2.2, label="PushT reference KDE")
        axis.plot(x, target_model["density"], color="#d9782d", lw=2.0, label=f"{name} KDE")
        axis.axhline(threshold, color="#a33a35", ls="--", lw=1.3, label="PushT density threshold")
        axis.axvline(45.0, color="#355c9a", ls=":", lw=1.4, label="PushT goal: 45°")
        axis.axvline(-135.0, color="#d9782d", ls=":", lw=1.4, label="Target goal: -135°")
        fraction = raw_results[name]["kde"]["target_fraction"]
        axis.set_title(f"PushT → {name}: support = {fraction:.3f}")
        axis.set_xlabel("Wrapped block angle (degrees)")
        axis.set_xlim(-180.0, 180.0)
        axis.set_xticks(np.arange(-180, 181, 45))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Periodic KDE density")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"block_angle_kde_comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_episode_diagnostics(
    diagnostics: dict[str, dict], output_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))
    for row_index, (name, values) in enumerate(diagnostics.items()):
        episode_rows = values["episode_rows"]
        fractions = np.sort([row["support_fraction"] for row in episode_rows])
        axes[row_index, 0].plot(fractions, color="#4c78a8", lw=2)
        axes[row_index, 0].axhline(0.95, color="#c43c39", ls="--")
        axes[row_index, 0].set_ylim(0.0, 1.02)
        axes[row_index, 0].set_ylabel(f"{name}\nKDE support fraction")
        axes[row_index, 0].set_xlabel("Episodes sorted worst → best")
        axes[row_index, 0].grid(alpha=0.2)

        image = axes[row_index, 1].imshow(
            values["heatmap"],
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
            extent=(0.0, 1.0, 0.0, len(episode_rows)),
        )
        axes[row_index, 1].set_xlabel("Normalized episode progress")
        axes[row_index, 1].set_ylabel("Episodes sorted worst → best")
        axes[row_index, 1].set_title("Local support over time (red = unsupported)")

        progress_support = values["progress_support"]
        centers = (np.arange(len(progress_support)) + 0.5) / len(progress_support)
        axes[row_index, 2].plot(centers, progress_support, color="#4c78a8", lw=2.2)
        axes[row_index, 2].axhline(0.95, color="#c43c39", ls="--")
        axes[row_index, 2].set_ylim(0.0, 1.02)
        axes[row_index, 2].set_xlabel("Normalized episode progress")
        axes[row_index, 2].set_ylabel("Point-weighted KDE support")
        axes[row_index, 2].grid(alpha=0.2)
    figure.suptitle("Block-angle support is distributed across episodes and worsens late", fontsize=14)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.91, wspace=0.3, hspace=0.34)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"block_angle_episode_diagnostics.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_alternatives(alternatives: dict[str, dict], output_dir: Path) -> None:
    representations = ("raw_world_angle", "goal_relative_angle", "orientation_modulo_pi")
    labels = ("Raw world angle", "Goal-relative angle", "Orientation modulo π")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    x = np.arange(len(representations))
    width = 0.34
    for axis, (name, values) in zip(axes, alternatives.items()):
        kde = [values[key]["kde"] for key in representations]
        mahalanobis = [values[key]["mahalanobis"] for key in representations]
        axis.bar(x - width / 2.0, kde, width, color="#d9782d", label="KDE")
        axis.bar(x + width / 2.0, mahalanobis, width, color="#4c78a8", label="Mahalanobis")
        axis.axhline(0.95, color="#c43c39", ls="--", lw=1.5, label="Pass threshold")
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_ylim(0.85, 1.005)
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Target support fraction")
    axes[0].legend(frameon=False)
    figure.suptitle("Passing without recollection requires changing angle semantics")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"block_angle_alternative_representations.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if key != "model"}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    global BINS
    args = parse_args()
    if args.bins < 4:
        raise ValueError("bins must be at least 4")
    BINS = args.bins
    for path in (args.pusht, args.stackt, args.antigoal):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_data = load_angles(args.pusht)
    target_data = {
        "StackT": load_angles(args.stackt),
        "PushT-Antigoal": load_angles(args.antigoal),
    }
    raw_results = {}
    diagnostics = {}
    alternatives = {}
    episode_csv_rows = []
    for name, data in target_data.items():
        kde = kde_result(reference_data["angle"], data["angle"])
        mahalanobis = mahalanobis_result(reference_data["angle"], data["angle"])
        episode_rows, heatmap, progress_support, episode_summary = episode_diagnostics(
            data["angle"], data["episode_idx"], data["step_idx"], kde["supported"]
        )
        raw_results[name] = {"kde": kde, "mahalanobis": mahalanobis}
        diagnostics[name] = {
            "episode_rows": episode_rows,
            "heatmap": heatmap,
            "progress_support": progress_support,
            "summary": episode_summary,
        }
        alternatives[name] = alternative_results(reference_data["angle"], data["angle"])
        episode_csv_rows.extend({"target": name, **row} for row in episode_rows)
        print(
            f"{name}: KDE={kde['target_fraction']:.6f}, "
            f"Mahalanobis={mahalanobis['target_fraction']:.6f}, "
            f"episodes below 0.95={episode_summary['episodes_below_0.95_support']}/400",
            flush=True,
        )

    reference_model = raw_results["StackT"]["kde"]["model"]
    threshold = raw_results["StackT"]["kde"]["threshold"]
    summary = {
        "definition": {
            "bins": BINS,
            "reference_kde_quantile": REFERENCE_KDE_QUANTILE,
            "reference_mahalanobis_quantile": REFERENCE_MAHALANOBIS_QUANTILE,
            "required_target_fraction": REQUIRED_TARGET_FRACTION,
            "reference_bandwidth": reference_model["bandwidth"],
            "reference_sigma_bins": reference_model["sigma_bins"],
            "reference_density_threshold": threshold,
            "low_density_intervals_degrees": low_density_intervals(
                reference_model, threshold
            ),
        },
        "paths": {
            "PushT": str(args.pusht.resolve()),
            "StackT": str(args.stackt.resolve()),
            "PushT-Antigoal": str(args.antigoal.resolve()),
        },
        "targets": {
            name: {
                "rows": int(len(target_data[name]["angle"])),
                "unsupported_points": int((~values["kde"]["supported"]).sum()),
                "kde_target_fraction": values["kde"]["target_fraction"],
                "mahalanobis_target_fraction": values["mahalanobis"]["target_fraction"],
                "episodes": diagnostics[name]["summary"],
                "alternatives": alternatives[name],
            }
            for name, values in raw_results.items()
        },
    }
    (args.output_dir / "block_angle_analysis.json").write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "block_angle_episode_metrics.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_csv_rows[0]))
        writer.writeheader()
        writer.writerows(episode_csv_rows)

    plot_kde_comparison(
        reference_data["angle"],
        {name: data["angle"] for name, data in target_data.items()},
        raw_results,
        args.output_dir,
    )
    plot_episode_diagnostics(diagnostics, args.output_dir)
    plot_alternatives(alternatives, args.output_dir)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
