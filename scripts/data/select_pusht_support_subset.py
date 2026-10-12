"""Select a trajectory-level, support-preserving PushT training subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import matplotlib
import numpy as np
import yaml
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from anygoal.paths import data_root


DEFAULT_INPUT = data_root() / "pusht_expert_train.h5"
DEFAULT_OUTPUT = data_root() / "pusht_mini_expert_train.h5"
GOAL_POSITION = np.asarray((256.0, 256.0), dtype=np.float64)
GOAL_ANGLE = np.pi / 4


@dataclass
class SupportView:
    name: str
    shape: tuple[int, ...]
    edges: tuple[np.ndarray, ...]
    bin_ids: np.ndarray
    frame_counts: np.ndarray | None = None
    trajectory_counts: np.ndarray | None = None
    trajectory_bin_counts: sparse.csr_matrix | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--target-trajectories", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position-bins", type=int, default=40)
    parser.add_argument("--action-bins", type=int, default=40)
    parser.add_argument("--rotation-bins", type=int, default=36)
    parser.add_argument("--distance-bins", type=int, default=32)
    parser.add_argument("--joint-distance-bins", type=int, default=16)
    parser.add_argument("--joint-rotation-bins", type=int, default=18)
    parser.add_argument("--coarse-position-bins", type=int, default=16)
    parser.add_argument("--coarse-rotation-bins", type=int, default=12)
    parser.add_argument("--contact-threshold", type=float, default=100.0)
    parser.add_argument(
        "--target-bin-trajectories",
        type=int,
        default=250,
        help="Density cap: target_c=min(full trajectory occupancy, this value).",
    )
    parser.add_argument("--rare-quantile", type=float, default=0.25)
    parser.add_argument("--score-epsilon", type=float, default=1e-12)
    parser.add_argument("--copy-batch-frames", type=int, default=1_000)
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument(
        "--h5-only",
        action="store_true",
        help="Reuse selected IDs in diagnostics-dir and only write the subset HDF5.",
    )
    return parser.parse_args()


def validate_schema(dataset: h5py.File) -> dict[str, str]:
    required = {
        "state",
        "action",
        "ep_len",
        "ep_offset",
        "episode_idx",
        "step_idx",
    }
    missing = sorted(required.difference(dataset.keys()))
    if missing:
        raise ValueError(f"Input dataset is missing required fields: {missing}")
    if dataset["state"].ndim != 2 or dataset["state"].shape[1] < 5:
        raise ValueError(f"Expected state shape (frames, >=5), got {dataset['state'].shape}")
    if dataset["action"].ndim != 2 or dataset["action"].shape[1] < 2:
        raise ValueError(
            f"Expected action shape (frames, >=2), got {dataset['action'].shape}"
        )
    return {
        "p_agent_x": "state[:, 0]",
        "p_agent_y": "state[:, 1]",
        "p_block_x": "state[:, 2]",
        "p_block_y": "state[:, 3]",
        "theta_block": "state[:, 4]",
        "a_x": "action[:, 0]",
        "a_y": "action[:, 1]",
        "p_goal": "fixed (256, 256)",
        "theta_goal": "fixed pi / 4",
        "d_agent_block": "L2(agent position, block position)",
        "d_block_goal": "L2(block position, fixed goal position)",
        "dtheta_goal": "wrap(theta_block - pi / 4) to [-pi, pi)",
        "contact": "1[d_agent_block < contact_threshold]",
    }


def fixed_edges(values: np.ndarray, bins: int) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if not len(finite_values):
        raise ValueError("Cannot bin a factor with no finite values.")
    lower = float(finite_values.min())
    upper = float(finite_values.max())
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    padding = max((upper - lower) * 1e-9, 1e-9)
    return np.linspace(lower - padding, upper + padding, bins + 1)


def fixed_bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.searchsorted(edges, values, side="right") - 1
    return np.clip(result, 0, len(edges) - 2).astype(np.int32)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def flattened_ids(
    values: tuple[np.ndarray, ...], edges: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, tuple[int, ...]]:
    shape = tuple(len(axis_edges) - 1 for axis_edges in edges)
    axis_ids = tuple(
        fixed_bin_ids(axis_values, axis_edges)
        for axis_values, axis_edges in zip(values, edges, strict=True)
    )
    return np.ravel_multi_index(axis_ids, shape).astype(np.int32), shape


def create_views(factors: dict[str, np.ndarray], args: argparse.Namespace) -> list[SupportView]:
    position_edges = {
        name: fixed_edges(factors[name], args.position_bins)
        for name in ("p_agent_x", "p_agent_y", "p_block_x", "p_block_y")
    }
    action_edges = {
        name: fixed_edges(factors[name], args.action_bins) for name in ("a_x", "a_y")
    }
    theta_edges = np.linspace(-np.pi, np.pi, args.rotation_bins + 1)
    distance_edges = {
        name: fixed_edges(factors[name], args.distance_bins)
        for name in ("d_agent_block", "d_block_goal")
    }

    definitions: list[tuple[str, tuple[str, ...], tuple[np.ndarray, ...]]] = [
        (
            "block_position",
            ("p_block_x", "p_block_y"),
            (position_edges["p_block_x"], position_edges["p_block_y"]),
        ),
        (
            "agent_position",
            ("p_agent_x", "p_agent_y"),
            (position_edges["p_agent_x"], position_edges["p_agent_y"]),
        ),
        ("action", ("a_x", "a_y"), (action_edges["a_x"], action_edges["a_y"])),
        ("block_rotation", ("theta_block_wrapped",), (theta_edges,)),
        (
            "agent_block_distance",
            ("d_agent_block",),
            (distance_edges["d_agent_block"],),
        ),
        (
            "block_goal_distance",
            ("d_block_goal",),
            (distance_edges["d_block_goal"],),
        ),
        (
            "contact",
            ("contact",),
            (np.asarray((-0.5, 0.5, 1.5)),),
        ),
    ]
    views: list[SupportView] = []
    for name, factor_names, edges in definitions:
        ids, shape = flattened_ids(tuple(factors[key] for key in factor_names), edges)
        views.append(SupportView(name, shape, edges, ids))

    joint_distance_edges = fixed_edges(
        factors["d_agent_block"], args.joint_distance_bins
    )
    joint_theta_edges = np.linspace(-np.pi, np.pi, args.joint_rotation_bins + 1)
    ids, shape = flattened_ids(
        (factors["d_agent_block"], factors["theta_block_wrapped"]),
        (joint_distance_edges, joint_theta_edges),
    )
    views.append(
        SupportView(
            "agent_block_distance_x_rotation",
            shape,
            (joint_distance_edges, joint_theta_edges),
            ids,
        )
    )

    coarse_edges = (
        fixed_edges(factors["p_block_x"], args.coarse_position_bins),
        fixed_edges(factors["p_block_y"], args.coarse_position_bins),
        np.linspace(-np.pi, np.pi, args.coarse_rotation_bins + 1),
    )
    ids, shape = flattened_ids(
        (
            factors["p_block_x"],
            factors["p_block_y"],
            factors["theta_block_wrapped"],
        ),
        coarse_edges,
    )
    views.append(
        SupportView("block_position_x_rotation_coarse", shape, coarse_edges, ids)
    )
    return views


def attach_trajectory_counts(
    views: list[SupportView], frame_episode_ids: np.ndarray, trajectory_count: int
) -> None:
    for view in views:
        bin_count = math.prod(view.shape)
        frame_counts = np.bincount(view.bin_ids, minlength=bin_count).astype(np.int64)
        matrix = sparse.coo_matrix(
            (
                np.ones(len(view.bin_ids), dtype=np.int32),
                (frame_episode_ids, view.bin_ids),
            ),
            shape=(trajectory_count, bin_count),
            dtype=np.int32,
        ).tocsr()
        matrix.sort_indices()
        visited = matrix.copy()
        visited.data[:] = 1
        view.frame_counts = frame_counts
        view.trajectory_counts = np.asarray(visited.sum(axis=0)).ravel().astype(np.int64)
        view.trajectory_bin_counts = matrix
        view.bin_ids = np.empty(0, dtype=np.int32)


def select_trajectories(
    views: list[SupportView], lengths: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    frame_matrices = [view.trajectory_bin_counts for view in views]
    if any(matrix is None for matrix in frame_matrices):
        raise RuntimeError("Trajectory/bin matrices were not initialized.")
    count_matrix = sparse.hstack(frame_matrices, format="csr", dtype=np.int32)
    visited = count_matrix.copy()
    visited.data[:] = 1
    full_counts = np.asarray(visited.sum(axis=0)).ravel().astype(np.int64)
    targets = np.minimum(full_counts, args.target_bin_trajectories).astype(np.int64)
    weights = 1.0 / np.sqrt(full_counts.astype(np.float64) + args.score_epsilon)
    length_scale = np.sqrt(lengths.astype(np.float64))
    initial_coefficients = weights * targets
    initial_scores = np.asarray(visited @ initial_coefficients).ravel() / length_scale

    rng = np.random.default_rng(args.seed)
    tie_keys = rng.random(len(lengths))
    current_scores = initial_scores.copy()
    visited_by_bin = visited.tocsc()
    selected_counts = np.zeros(visited.shape[1], dtype=np.int32)
    selected_ids: list[int] = []
    selection_scores: list[float] = []

    while len(selected_ids) < args.target_trajectories:
        maximum_score = float(np.max(current_scores))
        tied_ids = np.flatnonzero(
            np.abs(current_scores - maximum_score) <= args.score_epsilon
        )
        trajectory_id = int(tied_ids[np.argmin(tie_keys[tied_ids])])
        row_start = visited.indptr[trajectory_id]
        row_end = visited.indptr[trajectory_id + 1]
        columns = visited.indices[row_start:row_end]
        score_at_selection = current_scores[trajectory_id]
        old_coefficients = (
            weights[columns]
            * np.maximum(0, targets[columns] - selected_counts[columns])
            / np.sqrt(1.0 + selected_counts[columns])
        )
        selected_ids.append(trajectory_id)
        selection_scores.append(float(score_at_selection))
        selected_counts[columns] += 1
        new_coefficients = (
            weights[columns]
            * np.maximum(0, targets[columns] - selected_counts[columns])
            / np.sqrt(1.0 + selected_counts[columns])
        )
        coefficient_change = new_coefficients - old_coefficients
        score_change = np.asarray(
            visited_by_bin[:, columns] @ coefficient_change
        ).ravel()
        current_scores += score_change / length_scale
        current_scores[trajectory_id] = -np.inf
        if len(selected_ids) % 250 == 0:
            print(
                f"Selected {len(selected_ids):,}/{args.target_trajectories:,}; "
                f"score={score_at_selection:.8g}",
                flush=True,
            )

    return (
        np.asarray(selected_ids, dtype=np.int64),
        np.asarray(selection_scores, dtype=np.float64),
        count_matrix,
    )


def selected_frame_indices(
    selected_ids: np.ndarray, offsets: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    ordered = np.sort(selected_ids)
    return np.concatenate(
        [
            np.arange(offsets[index], offsets[index] + lengths[index], dtype=np.int64)
            for index in ordered
        ]
    )


def save_id_outputs(
    diagnostics_dir: Path,
    selected_ids: np.ndarray,
    frame_indices: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    scores: np.ndarray,
) -> None:
    with (diagnostics_dir / "selected_trajectory_ids.json").open("w") as handle:
        json.dump(selected_ids.tolist(), handle, indent=2)
        handle.write("\n")
    np.save(diagnostics_dir / "selected_trajectory_ids.npy", selected_ids)
    np.save(diagnostics_dir / "selected_frame_indices.npy", frame_indices)
    selection_order = np.empty(len(lengths), dtype=np.int64)
    selection_order.fill(-1)
    selection_order[selected_ids] = np.arange(len(selected_ids))
    score_by_id = np.full(len(lengths), np.nan, dtype=np.float64)
    score_by_id[selected_ids] = scores
    with (diagnostics_dir / "subset_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trajectory_id",
                "start_idx",
                "end_idx",
                "length",
                "selection_order",
                "final_score_at_selection",
            ],
        )
        writer.writeheader()
        for trajectory_id in selected_ids:
            trajectory_id = int(trajectory_id)
            writer.writerow(
                {
                    "trajectory_id": trajectory_id,
                    "start_idx": int(offsets[trajectory_id]),
                    "end_idx": int(offsets[trajectory_id] + lengths[trajectory_id]),
                    "length": int(lengths[trajectory_id]),
                    "selection_order": int(selection_order[trajectory_id]),
                    "final_score_at_selection": f"{score_by_id[trajectory_id]:.12g}",
                }
            )


def occupied_counts(matrix: sparse.csr_matrix, ids: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[ids].sum(axis=0)).ravel().astype(np.int64)


def view_slices(views: list[SupportView]) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for view in views:
        size = math.prod(view.shape)
        result[view.name] = slice(offset, offset + size)
        offset += size
    return result


def save_view_coverage(
    path: Path,
    views: list[SupportView],
    count_matrix: sparse.csr_matrix,
    selected_ids: np.ndarray,
    random_ids: np.ndarray,
    rare_quantile: float,
) -> None:
    slices = view_slices(views)
    subset_counts = occupied_counts(count_matrix, selected_ids)
    random_counts = occupied_counts(count_matrix, random_ids)
    fields = [
        "view_name",
        "num_full_occupied_bins",
        "num_subset_occupied_bins",
        "coverage_ratio",
        "rare_bin_coverage_ratio",
        "dense_bin_downsample_ratio",
    ]
    rows_by_kind: dict[str, list[dict[str, Any]]] = {"subset": [], "random": []}
    for kind, all_selected_frame_counts in (
        ("subset", subset_counts),
        ("random", random_counts),
    ):
        for view in views:
            if view.frame_counts is None or view.trajectory_counts is None:
                raise RuntimeError("View counts are missing.")
            local = slices[view.name]
            selected_frame_counts = all_selected_frame_counts[local]
            occupied = view.trajectory_counts > 0
            positive_counts = view.trajectory_counts[occupied]
            rare_threshold = np.quantile(positive_counts, rare_quantile)
            dense_threshold = np.quantile(positive_counts, 1.0 - rare_quantile)
            rare = occupied & (view.trajectory_counts <= rare_threshold)
            dense = occupied & (view.trajectory_counts >= dense_threshold)
            full_occupied = int(occupied.sum())
            subset_occupied = int(np.count_nonzero(selected_frame_counts))
            rare_covered = int(np.count_nonzero(selected_frame_counts[rare]))
            dense_full_frames = int(view.frame_counts[dense].sum())
            dense_selected_frames = int(selected_frame_counts[dense].sum())
            rows_by_kind[kind].append(
                {
                    "view_name": view.name,
                    "num_full_occupied_bins": full_occupied,
                    "num_subset_occupied_bins": subset_occupied,
                    "coverage_ratio": subset_occupied / full_occupied,
                    "rare_bin_coverage_ratio": rare_covered / max(1, int(rare.sum())),
                    "dense_bin_downsample_ratio": dense_selected_frames
                    / max(1, dense_full_frames),
                }
            )
    for output_path, rows in (
        (path, rows_by_kind["subset"]),
        (path.with_name("random_baseline_bin_coverage_by_view.csv"), rows_by_kind["random"]),
    ):
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def bin_range(view: SupportView, flat_id: int) -> str:
    coordinates = np.unravel_index(flat_id, view.shape)
    intervals = [
        [float(view.edges[axis][coordinate]), float(view.edges[axis][coordinate + 1])]
        for axis, coordinate in enumerate(coordinates)
    ]
    return json.dumps(intervals, separators=(",", ":"))


def save_uncovered_bins(
    path: Path,
    views: list[SupportView],
    count_matrix: sparse.csr_matrix,
    selected_ids: np.ndarray,
) -> None:
    subset_counts = occupied_counts(count_matrix, selected_ids)
    slices = view_slices(views)
    with path.open("w", newline="") as handle:
        fields = ["view_name", "bin_id", "full_count", "bin_range"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for view in views:
            if view.frame_counts is None:
                raise RuntimeError("View frame counts are missing.")
            local_subset = subset_counts[slices[view.name]]
            missing = np.flatnonzero((view.frame_counts > 0) & (local_subset == 0))
            for bin_id in missing:
                writer.writerow(
                    {
                        "view_name": view.name,
                        "bin_id": int(bin_id),
                        "full_count": int(view.frame_counts[bin_id]),
                        "bin_range": bin_range(view, int(bin_id)),
                    }
                )


def one_dimensional_edges(
    factors: dict[str, np.ndarray], args: argparse.Namespace
) -> dict[str, np.ndarray]:
    edges = {
        name: fixed_edges(factors[name], args.position_bins)
        for name in ("p_agent_x", "p_agent_y", "p_block_x", "p_block_y")
    }
    edges.update(
        {
            name: fixed_edges(factors[name], args.action_bins)
            for name in ("a_x", "a_y")
        }
    )
    edges["theta_block"] = np.linspace(-np.pi, np.pi, args.rotation_bins + 1)
    edges["dtheta_goal"] = np.linspace(-np.pi, np.pi, args.rotation_bins + 1)
    edges["d_agent_block"] = fixed_edges(
        factors["d_agent_block"], args.distance_bins
    )
    edges["d_block_goal"] = fixed_edges(factors["d_block_goal"], args.distance_bins)
    edges["contact"] = np.asarray((-0.5, 0.5, 1.5))
    return edges


def save_support_summary(
    path: Path,
    factors: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    args: argparse.Namespace,
) -> None:
    edges = one_dimensional_edges(factors, args)
    fields = [
        "factor",
        "full_min",
        "full_max",
        "subset_min",
        "subset_max",
        "full_p01",
        "full_p99",
        "subset_p01",
        "subset_p99",
        "number_of_full_bins",
        "number_of_subset_bins",
        "bin_coverage_ratio",
    ]
    factor_order = [
        "p_agent_x",
        "p_agent_y",
        "p_block_x",
        "p_block_y",
        "theta_block",
        "a_x",
        "a_y",
        "d_agent_block",
        "d_block_goal",
        "dtheta_goal",
        "contact",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in factor_order:
            full = factors["theta_block_wrapped"] if name == "theta_block" else factors[name]
            subset = full[frame_indices]
            full_bins = np.bincount(
                fixed_bin_ids(full, edges[name]), minlength=len(edges[name]) - 1
            )
            subset_bins = np.bincount(
                fixed_bin_ids(subset, edges[name]), minlength=len(edges[name]) - 1
            )
            full_occupied = int(np.count_nonzero(full_bins))
            subset_occupied = int(np.count_nonzero(subset_bins))
            writer.writerow(
                {
                    "factor": name,
                    "full_min": float(np.min(full)),
                    "full_max": float(np.max(full)),
                    "subset_min": float(np.min(subset)),
                    "subset_max": float(np.max(subset)),
                    "full_p01": float(np.quantile(full, 0.01)),
                    "full_p99": float(np.quantile(full, 0.99)),
                    "subset_p01": float(np.quantile(subset, 0.01)),
                    "subset_p99": float(np.quantile(subset, 0.99)),
                    "number_of_full_bins": full_occupied,
                    "number_of_subset_bins": subset_occupied,
                    "bin_coverage_ratio": subset_occupied / full_occupied,
                }
            )


def plot_2d_comparison(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray],
    title: str,
    dpi: int,
) -> None:
    full_hist, _, _ = np.histogram2d(x, y, bins=edges)
    subset_hist, _, _ = np.histogram2d(x[selected], y[selected], bins=edges)
    positive = full_hist[full_hist > 0]
    norm = colors.LogNorm(vmin=1, vmax=max(1, float(positive.max())))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, histogram, label in zip(
        axes, (full_hist, subset_hist), ("Full", "Support-preserving 5K"), strict=True
    ):
        image = axis.pcolormesh(edges[0], edges[1], histogram.T, norm=norm, cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, label="frame count (shared log scale)")
    figure.suptitle(title)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def plot_histogram_comparison(
    path: Path,
    values: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
    title: str,
    xlabel: str,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.hist(values, bins=edges, density=True, histtype="step", linewidth=2, label="Full")
    axis.hist(
        values[selected],
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2,
        label="Support-preserving 5K",
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("normalized density")
    axis.legend()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def save_plots(
    plot_dir: Path,
    factors: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    views: list[SupportView],
    dpi: int,
) -> None:
    plot_dir.mkdir()
    view_by_name = {view.name: view for view in views}
    plot_2d_comparison(
        plot_dir / "block_position_full_vs_subset.png",
        factors["p_block_x"],
        factors["p_block_y"],
        frame_indices,
        view_by_name["block_position"].edges,
        "Physical T position",
        dpi,
    )
    plot_2d_comparison(
        plot_dir / "agent_position_full_vs_subset.png",
        factors["p_agent_x"],
        factors["p_agent_y"],
        frame_indices,
        view_by_name["agent_position"].edges,
        "Agent position",
        dpi,
    )
    plot_2d_comparison(
        plot_dir / "action_full_vs_subset.png",
        factors["a_x"],
        factors["a_y"],
        frame_indices,
        view_by_name["action"].edges,
        "Action",
        dpi,
    )
    plot_histogram_comparison(
        plot_dir / "block_rotation_full_vs_subset.png",
        factors["theta_block_wrapped"],
        frame_indices,
        view_by_name["block_rotation"].edges[0],
        "Physical T rotation",
        "wrapped angle (rad)",
        dpi,
    )
    plot_histogram_comparison(
        plot_dir / "agent_block_distance_full_vs_subset.png",
        factors["d_agent_block"],
        frame_indices,
        view_by_name["agent_block_distance"].edges[0],
        "Agent–block distance",
        "distance",
        dpi,
    )
    plot_histogram_comparison(
        plot_dir / "block_goal_distance_full_vs_subset.png",
        factors["d_block_goal"],
        frame_indices,
        view_by_name["block_goal_distance"].edges[0],
        "Block–goal distance",
        "distance",
        dpi,
    )

    block_view = view_by_name["block_position"]
    if block_view.frame_counts is None:
        raise RuntimeError("Block view counts are missing.")
    full = block_view.frame_counts.reshape(block_view.shape)
    subset = np.histogram2d(
        factors["p_block_x"][frame_indices],
        factors["p_block_y"][frame_indices],
        bins=block_view.edges,
    )[0]
    occupied_images = {
        "block_position_full_occupied_bins.png": (full > 0).astype(float),
        "block_position_subset_occupied_bins.png": (subset > 0).astype(float),
        "block_position_missing_bins.png": ((full > 0) & (subset == 0)).astype(float),
    }
    for filename, image_values in occupied_images.items():
        figure, axis = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
        image = axis.pcolormesh(
            block_view.edges[0], block_view.edges[1], image_values.T, cmap="magma"
        )
        axis.set_xlabel("block x")
        axis.set_ylabel("block y")
        axis.set_title(filename.removesuffix(".png").replace("_", " ").title())
        figure.colorbar(image, ax=axis)
        figure.savefig(plot_dir / filename, dpi=dpi)
        plt.close(figure)

    ratio = np.divide(subset, full, out=np.full_like(subset, np.nan), where=full > 0)
    figure, axis = plt.subplots(figsize=(5.8, 5), constrained_layout=True)
    image = axis.pcolormesh(
        block_view.edges[0], block_view.edges[1], ratio.T, cmap="viridis", vmin=0
    )
    axis.set_xlabel("block x")
    axis.set_ylabel("block y")
    axis.set_title("Selected frame density / full frame density")
    figure.colorbar(image, ax=axis, label="retained frame fraction")
    figure.savefig(plot_dir / "block_position_selected_full_density_ratio.png", dpi=dpi)
    plt.close(figure)


def create_output_h5(
    input_path: Path,
    output_path: Path,
    selected_ids: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    copy_batch_frames: int,
    seed: int,
) -> None:
    partial_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path} or {partial_path}")
    ordered_ids = np.sort(selected_ids)
    selected_lengths = lengths[ordered_ids]
    selected_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(selected_lengths[:-1], dtype=np.int64))
    )
    output_frames = int(selected_lengths.sum())
    try:
        with h5py.File(input_path, "r") as source, h5py.File(
            partial_path, "w", libver="latest"
        ) as destination:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            destination.attrs["subset_method"] = "trajectory_support_preserving_density_cap"
            destination.attrs["source_dataset"] = str(input_path.resolve())
            destination.attrs["source_trajectory_ids_file"] = "selected_trajectory_ids.npy"
            destination.attrs["selection_seed"] = seed

            for name, source_dataset in source.items():
                if not isinstance(source_dataset, h5py.Dataset):
                    continue
                if source_dataset.shape[0] == len(lengths):
                    shape = (len(ordered_ids), *source_dataset.shape[1:])
                elif source_dataset.shape[0] == source["state"].shape[0]:
                    shape = (output_frames, *source_dataset.shape[1:])
                else:
                    raise ValueError(
                        f"Cannot classify dataset {name!r} with shape {source_dataset.shape}"
                    )
                creation: dict[str, Any] = {}
                if source_dataset.chunks is not None:
                    creation["chunks"] = source_dataset.chunks
                creation_properties = source_dataset.id.get_create_plist()
                filter_ids = {
                    creation_properties.get_filter(index)[0]
                    for index in range(creation_properties.get_nfilters())
                }
                if 32001 in filter_ids:
                    creation.update(
                        hdf5plugin.Blosc(
                            cname="lz4",
                            clevel=5,
                            shuffle=hdf5plugin.Blosc.SHUFFLE,
                        )
                    )
                destination_dataset = destination.create_dataset(
                    name,
                    shape=shape,
                    maxshape=(None, *shape[1:]),
                    dtype=source_dataset.dtype,
                    **creation,
                )
                for key, value in source_dataset.attrs.items():
                    destination_dataset.attrs[key] = value

            destination["ep_len"][:] = selected_lengths
            destination["ep_offset"][:] = selected_offsets
            output_cursor = 0
            for new_episode_id, source_episode_id in enumerate(ordered_ids):
                source_episode_id = int(source_episode_id)
                source_start = int(offsets[source_episode_id])
                episode_length = int(lengths[source_episode_id])
                output_end = output_cursor + episode_length
                for name, source_dataset in source.items():
                    if not isinstance(source_dataset, h5py.Dataset):
                        continue
                    if source_dataset.shape[0] != source["state"].shape[0]:
                        continue
                    if name == "episode_idx":
                        destination[name][output_cursor:output_end] = new_episode_id
                    elif name == "step_idx":
                        destination[name][output_cursor:output_end] = np.arange(
                            episode_length, dtype=source_dataset.dtype
                        )
                    else:
                        for local_start in range(0, episode_length, copy_batch_frames):
                            local_end = min(episode_length, local_start + copy_batch_frames)
                            destination[name][
                                output_cursor + local_start : output_cursor + local_end
                            ] = source_dataset[
                                source_start + local_start : source_start + local_end
                            ]
                output_cursor = output_end
                if (new_episode_id + 1) % 250 == 0:
                    print(
                        f"Copied {new_episode_id + 1:,}/{len(ordered_ids):,} trajectories",
                        flush=True,
                    )
            destination.flush()
        os.replace(partial_path, output_path)
    except Exception:
        print(f"Incomplete HDF5 retained for inspection: {partial_path}", flush=True)
        raise


def save_config(
    path: Path,
    args: argparse.Namespace,
    factor_mapping: dict[str, str],
    views: list[SupportView],
    input_frames: int,
    input_trajectories: int,
) -> None:
    config = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "diagnostics_dir": str(path.parent.resolve()),
        "seed": args.seed,
        "target_trajectory_count": args.target_trajectories,
        "input_trajectory_count": input_trajectories,
        "input_frame_count": input_frames,
        "goal_position": GOAL_POSITION.tolist(),
        "goal_angle_radians": float(GOAL_ANGLE),
        "contact_threshold": args.contact_threshold,
        "selection": {
            "algorithm": "incrementally updated greedy weighted set cover",
            "full_bin_count_unit": "number of trajectories visiting the bin",
            "selected_bin_count_unit": "number of selected trajectories visiting the bin",
            "within_trajectory_representation": ["binary visits", "frame counts"],
            "rare_bin_weight": "1 / sqrt(N_c + epsilon)",
            "target": "min(N_c, target_bin_trajectories)",
            "target_bin_trajectories": args.target_bin_trajectories,
            "score": (
                "sum_c w_c * max(0, target_c - M_c) / sqrt(1 + M_c) "
                "/ sqrt(trajectory_length)"
            ),
            "tie_breaking": "seeded random key; no random primary selection",
            "epsilon": args.score_epsilon,
        },
        "diagnostics": {
            "rare_bin_definition": f"bottom occupied-bin q={args.rare_quantile}",
            "dense_bin_definition": f"top occupied-bin q={1 - args.rare_quantile}",
            "density_ratio_unit": "selected frames / full frames",
            "bin_edges": "equal-width full-data min/max, except circular angles/contact",
        },
        "factor_mapping": factor_mapping,
        "views": [
            {
                "name": view.name,
                "shape": list(view.shape),
                "edges": [axis.tolist() for axis in view.edges],
            }
            for view in views
        ],
    }
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_report(path: Path, args: argparse.Namespace, selected_frames: int) -> None:
    diagnostics_dir = path.parent
    text = f"""# PushT 5K support-preserving trajectory subset

## Selection method

The subset contains exactly {args.target_trajectories:,} complete trajectories ({selected_frames:,} frames). Each full-data trajectory was represented by the support bins it visited across several marginal and coarse joint views. A deterministic greedy weighted set-cover procedure selected the trajectory with the largest remaining support gain at each step; sparse incremental updates change only scores for trajectories sharing an affected bin. Bin weights are `1 / sqrt(N_c + epsilon)`, where `N_c` is the number of full-data trajectories visiting the bin. Coverage targets are capped at {args.target_bin_trajectories} trajectories per bin, so high-density bins stop dominating after the cap while lower-occupancy bins retain a larger fraction of their available coverage. Scores are divided by the square root of trajectory length. Seed {args.seed} is used only to break score ties.

This is support-preserving rather than distribution-preserving: it deliberately over-retains rare occupied bins and down-samples repeated central coverage. It does not attempt to reproduce full-data bin probabilities.

## Factors and support views

The raw PushT mapping is agent position = `state[:, 0:2]`, physical T position = `state[:, 2:4]`, physical T rotation = `state[:, 4]`, and action = `action[:, 0:2]`. Derived factors are agent–block distance, block–goal distance for goal `(256, 256)`, wrapped rotation difference to goal angle `pi/4`, and contact defined by agent–block distance < {args.contact_threshold:g}.

Selection views cover block position, agent position, action, circular block rotation, both distances, contact, distance × rotation, and a coarse block-position × rotation grid. Exact equal-width/circular bin edges and the field mapping are in `config_used.yaml`.

## Diagnostics

- `support_summary.csv` reports marginal ranges, 1st/99th percentiles, and occupied-bin coverage.
- `bin_coverage_by_view.csv` reports full/subset support coverage, rare-bin coverage, and retained dense-bin frame density.
- `uncovered_bins.csv` explicitly records any full-data occupied bins not visited by the subset.
- `plots/` compares full and subset marginals and visualizes block-position occupancy, missing bins, and retained-density ratios.
- `random_baseline_bin_coverage_by_view.csv` and `random_baseline_trajectory_ids.*` provide a seeded random 5K trajectory baseline for coverage comparison only.

## Loading for DriftWorld

The subset HDF5 is `{args.output}` and preserves the full StableWM schema with trajectory-relative `episode_idx`, `step_idx`, `ep_len`, and `ep_offset` rebuilt consistently. Point DriftWorld at the file by overriding both the dataset directory and stem:

```bash
python scripts/train/train_driftworld.py \\
  data.dataset_root={args.output.parent.parent} \\
  data.dataset_name={args.output.stem} \\
  data.num_train_demos={args.target_trajectories}
```

Original source trajectory IDs are in `{diagnostics_dir / 'selected_trajectory_ids.npy'}` in greedy selection order. The subset HDF5 itself stores episodes in ascending original trajectory-ID order for efficient sequential access; use `subset_manifest.csv` to map selection order back to source ranges. `end_idx` is exclusive.
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    if args.target_trajectories < 1:
        raise ValueError("--target-trajectories must be positive.")
    if args.target_bin_trajectories < 1:
        raise ValueError("--target-bin-trajectories must be positive.")
    diagnostics_dir = args.diagnostics_dir or args.output.with_suffix("").with_name(
        args.output.stem + "_support"
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output HDF5: {args.output}")
    if diagnostics_dir.exists() and not args.h5_only:
        raise FileExistsError(f"Refusing to overwrite diagnostics: {diagnostics_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.h5_only:
        ids_path = diagnostics_dir / "selected_trajectory_ids.npy"
        if not ids_path.is_file():
            raise FileNotFoundError(f"Cannot reuse selection; missing {ids_path}")
        selected_ids = np.load(ids_path).astype(np.int64)
        if (
            len(selected_ids) != args.target_trajectories
            or len(np.unique(selected_ids)) != args.target_trajectories
        ):
            raise ValueError(f"Invalid reusable selected-ID file: {ids_path}")
        with h5py.File(args.input, "r") as dataset:
            validate_schema(dataset)
            lengths = np.asarray(dataset["ep_len"][:], dtype=np.int64)
            offsets = np.asarray(dataset["ep_offset"][:], dtype=np.int64)
        print(f"Reusing {len(selected_ids):,} selected IDs from {ids_path}", flush=True)
        create_output_h5(
            args.input,
            args.output,
            selected_ids,
            offsets,
            lengths,
            args.copy_batch_frames,
            args.seed,
        )
        print(f"Done writing reusable selection to {args.output}", flush=True)
        return
    diagnostics_dir.mkdir(parents=True)

    print(f"Loading state/action metadata from {args.input}", flush=True)
    with h5py.File(args.input, "r") as dataset:
        factor_mapping = validate_schema(dataset)
        state = np.asarray(dataset["state"][:], dtype=np.float64)
        action = np.asarray(dataset["action"][:], dtype=np.float64)
        lengths = np.asarray(dataset["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(dataset["ep_offset"][:], dtype=np.int64)
        frame_episode_ids = np.asarray(dataset["episode_idx"][:], dtype=np.int64)
        input_frames = len(state)
    if len(lengths) < args.target_trajectories:
        raise ValueError(
            f"Requested {args.target_trajectories} trajectories from only {len(lengths)}."
        )
    expected_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
    )
    if not np.array_equal(offsets, expected_offsets) or int(lengths.sum()) != input_frames:
        raise ValueError("ep_len/ep_offset do not define a contiguous frame partition.")
    if (
        len(frame_episode_ids) != input_frames
        or frame_episode_ids.min() != 0
        or frame_episode_ids.max() != len(lengths) - 1
    ):
        raise ValueError("episode_idx is inconsistent with the episode table.")

    factors = {
        "p_agent_x": state[:, 0],
        "p_agent_y": state[:, 1],
        "p_block_x": state[:, 2],
        "p_block_y": state[:, 3],
        "theta_block": state[:, 4],
        "theta_block_wrapped": wrap_angle(state[:, 4]),
        "a_x": action[:, 0],
        "a_y": action[:, 1],
    }
    factors["d_agent_block"] = np.hypot(
        factors["p_agent_x"] - factors["p_block_x"],
        factors["p_agent_y"] - factors["p_block_y"],
    )
    factors["d_block_goal"] = np.hypot(
        factors["p_block_x"] - GOAL_POSITION[0],
        factors["p_block_y"] - GOAL_POSITION[1],
    )
    factors["dtheta_goal"] = wrap_angle(state[:, 4] - GOAL_ANGLE)
    factors["contact"] = (
        factors["d_agent_block"] < args.contact_threshold
    ).astype(np.int8)
    if any(not np.isfinite(values).all() for values in factors.values()):
        raise ValueError("State/action factors contain non-finite values.")

    views = create_views(factors, args)
    print("Building per-trajectory support representations", flush=True)
    attach_trajectory_counts(views, frame_episode_ids, len(lengths))
    selected_ids, scores, count_matrix = select_trajectories(views, lengths, args)
    if len(np.unique(selected_ids)) != args.target_trajectories:
        raise RuntimeError("Selection did not produce exactly the requested unique IDs.")
    frame_indices = selected_frame_indices(selected_ids, offsets, lengths)

    rng = np.random.default_rng(args.seed)
    random_ids = np.sort(
        rng.choice(len(lengths), size=args.target_trajectories, replace=False)
    ).astype(np.int64)
    with (diagnostics_dir / "random_baseline_trajectory_ids.json").open("w") as handle:
        json.dump(random_ids.tolist(), handle, indent=2)
        handle.write("\n")
    np.save(diagnostics_dir / "random_baseline_trajectory_ids.npy", random_ids)

    save_id_outputs(
        diagnostics_dir, selected_ids, frame_indices, offsets, lengths, scores
    )
    save_support_summary(
        diagnostics_dir / "support_summary.csv", factors, frame_indices, args
    )
    save_view_coverage(
        diagnostics_dir / "bin_coverage_by_view.csv",
        views,
        count_matrix,
        selected_ids,
        random_ids,
        args.rare_quantile,
    )
    save_uncovered_bins(
        diagnostics_dir / "uncovered_bins.csv", views, count_matrix, selected_ids
    )
    save_plots(
        diagnostics_dir / "plots", factors, frame_indices, views, args.plot_dpi
    )
    save_config(
        diagnostics_dir / "config_used.yaml",
        args,
        factor_mapping,
        views,
        input_frames,
        len(lengths),
    )
    save_report(
        diagnostics_dir / "README.md", args, selected_frames=len(frame_indices)
    )
    print(f"Writing self-contained HDF5 subset to {args.output}", flush=True)
    create_output_h5(
        args.input,
        args.output,
        selected_ids,
        offsets,
        lengths,
        args.copy_batch_frames,
        args.seed,
    )
    print(
        f"Done: {len(selected_ids):,} trajectories, {len(frame_indices):,} frames; "
        f"diagnostics={diagnostics_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
