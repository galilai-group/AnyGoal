#!/usr/bin/env python
"""Select the widest supported 3x3 hand-endpoint grid for Insert Peg."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets.insert_peg_true_state import (  # noqa: E402
    InsertPegStateAdapter,
)
from anygoal.envs.insert_peg import (  # noqa: E402
    HAND_POSITION_SLICE,
    InsertPegEnv,
)


DATASET_PATH = (
    Path.home()
    / ".stable_worldmodel"
    / "datasets"
    / "insert_peg_weak_expert_train.h5"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "insert_peg_counterfactual_sensitivity"
)
SEED = 20260914


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-actions", type=int, default=200_000)
    parser.add_argument("--calibration-actions", type=int, default=20_000)
    parser.add_argument("--candidate-actions", type=int, default=50_000)
    parser.add_argument("--anchor-candidates", type=int, default=100_000)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--relative-grid-error", type=float, default=0.10)
    parser.add_argument("--spacing-steps", type=int, default=4_000)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def select_anchor(
    observations: np.ndarray,
    rng: np.random.Generator,
    candidate_count: int,
) -> tuple[int, np.ndarray]:
    adapter = InsertPegStateAdapter(include_velocities=False)
    indices = rng.choice(
        len(observations), size=min(candidate_count, len(observations)), replace=False
    )
    states = observations[indices][:, adapter._essential_indices].astype(np.float64)
    finite = np.isfinite(states).all(axis=1)
    states = states[finite]
    indices = indices[finite]
    center = np.mean(states, axis=0)
    scale = np.std(states, axis=0)
    scale[scale < 1e-8] = 1.0
    distances = np.mean(((states - center) / scale) ** 2, axis=1)
    local_index = int(np.argmin(distances))
    return int(indices[local_index]), observations[indices[local_index]].copy()


def simulate_endpoints(
    anchor_state: np.ndarray,
    actions: np.ndarray,
    frameskip: int,
) -> np.ndarray:
    environment = InsertPegEnv(terminate_at_goal=False)
    environment.reset()
    endpoints = np.empty((len(actions), 2), dtype=np.float64)
    for index, action in enumerate(actions):
        environment.set_state(anchor_state)
        state = anchor_state
        for _ in range(frameskip):
            state, _, _, _, _ = environment.step(action)
        endpoints[index] = state[HAND_POSITION_SLICE]
        if (index + 1) % 5_000 == 0:
            print(f"Simulated {index + 1:,}/{len(actions):,} candidate actions", flush=True)
    environment.close()
    return endpoints


def grid_offsets(spacing: float) -> np.ndarray:
    coordinates = (-spacing, 0.0, spacing)
    return np.asarray(list(itertools.product(coordinates, repeat=2)))


def select_grid(
    endpoints: np.ndarray,
    zero_endpoint: np.ndarray,
    relative_error: float,
    spacing_steps: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(endpoints)
    displacement = endpoints - zero_endpoint
    positive = displacement.max(axis=0)
    negative = -displacement.min(axis=0)
    upper = float(np.min(np.concatenate((positive, negative))))
    center_grid_index = 4
    for spacing in np.linspace(upper, 0.0, spacing_steps, endpoint=False):
        if spacing <= 0.0:
            continue
        targets = zero_endpoint + grid_offsets(float(spacing))
        errors, indices = tree.query(targets)
        indices[center_grid_index] = -1
        errors[center_grid_index] = 0.0
        noncentral = np.delete(indices, center_grid_index)
        if len(np.unique(noncentral)) != 8:
            continue
        if float(errors.max()) <= relative_error * spacing:
            return float(spacing), targets, indices, errors
    raise RuntimeError(
        "No 3x3 grid met the requested endpoint-error tolerance; increase "
        "--candidate-actions or --relative-grid-error."
    )


def main() -> None:
    args = parse_args()
    if args.frameskip < 1 or args.spacing_steps < 2:
        raise ValueError("frameskip must be positive and spacing-steps at least two")
    if not 0.0 < args.relative_grid_error < 0.5:
        raise ValueError("relative-grid-error must be in (0, 0.5)")
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    with h5py.File(dataset_path, "r") as dataset:
        actions = np.asarray(dataset["action"][:], dtype=np.float64)
        observations = np.asarray(dataset["observation"][:], dtype=np.float64)
    valid = np.isfinite(actions).all(axis=1) & np.isfinite(observations).all(axis=1)
    actions = actions[valid]
    observations = observations[valid]
    requested = (
        args.reference_actions + args.calibration_actions + args.candidate_actions
    )
    if len(actions) < requested:
        raise ValueError(f"Need {requested:,} finite rows, found {len(actions):,}")

    rng = np.random.default_rng(SEED)
    selected = rng.choice(len(actions), size=requested, replace=False)
    reference = actions[selected[: args.reference_actions]]
    calibration_start = args.reference_actions
    calibration_end = calibration_start + args.calibration_actions
    calibration = actions[selected[calibration_start:calibration_end]]
    candidates = actions[selected[calibration_end:]]
    action_tree = cKDTree(reference)

    def support_radius(values: np.ndarray) -> np.ndarray:
        return action_tree.query(values, k=args.neighbors, workers=-1)[0][..., -1]

    calibration_radii = support_radius(calibration)
    maximum_supported_radius = float(
        np.quantile(calibration_radii, args.support_quantile)
    )
    candidate_radii = support_radius(candidates)
    supported = candidate_radii <= maximum_supported_radius
    candidates = candidates[supported]
    candidate_radii = candidate_radii[supported]
    zero_action = np.zeros(5, dtype=np.float64)
    zero_radius = float(support_radius(zero_action[None])[0])
    # Zero is required as the grid center; it lies numerically on the calibrated
    # 95% boundary, so include it in the effective threshold explicitly.
    maximum_supported_radius = max(maximum_supported_radius, zero_radius)

    anchor_index, anchor_state = select_anchor(
        observations, rng, args.anchor_candidates
    )
    simulated_actions = np.vstack((zero_action, candidates))
    endpoints = simulate_endpoints(anchor_state, simulated_actions, args.frameskip)
    zero_endpoint = endpoints[0]
    candidate_endpoints = endpoints[1:]
    spacing, targets, candidate_indices, endpoint_errors = select_grid(
        candidate_endpoints,
        zero_endpoint,
        args.relative_grid_error,
        args.spacing_steps,
    )

    selected_actions = np.empty((9, 5), dtype=np.float64)
    selected_endpoints = np.empty((9, 2), dtype=np.float64)
    selected_radii = np.empty(9, dtype=np.float64)
    for grid_index, candidate_index in enumerate(candidate_indices):
        if grid_index == 4:
            selected_actions[grid_index] = zero_action
            selected_endpoints[grid_index] = zero_endpoint
            selected_radii[grid_index] = zero_radius
        else:
            selected_actions[grid_index] = candidates[candidate_index]
            selected_endpoints[grid_index] = candidate_endpoints[candidate_index]
            selected_radii[grid_index] = candidate_radii[candidate_index]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "insert_peg_counterfactual_actions.json"
    payload = {
        "dataset_path": str(dataset_path),
        "selection": (
            "maximum axis-aligned 3x3 hand (x,z) endpoint spacing around the "
            "zero-action endpoint at a representative expert anchor state"
        ),
        "seed": SEED,
        "frameskip": args.frameskip,
        "action_intervention": "each selected raw 5D action is repeated for five controls",
        "grid_spacing": spacing,
        "relative_grid_error_limit": args.relative_grid_error,
        "anchor": {
            "filtered_dataset_index": anchor_index,
            "hand_position": anchor_state[HAND_POSITION_SLICE].tolist(),
            "object_position": anchor_state[33:35].tolist(),
            "target_position": anchor_state[40:42].tolist(),
        },
        "zero_action_endpoint": zero_endpoint.tolist(),
        "actions": selected_actions.tolist(),
        "target_hand_endpoints": targets.tolist(),
        "realized_hand_endpoints": selected_endpoints.tolist(),
        "endpoint_errors": endpoint_errors.tolist(),
        "maximum_relative_endpoint_error": float(endpoint_errors.max() / spacing),
        "action_20nn_radii": selected_radii.tolist(),
        "all_actions_supported": bool(
            np.all(selected_radii <= maximum_supported_radius)
        ),
        "support": {
            "method": "raw 5-D Euclidean 20-nearest-neighbor radius",
            "reference_action_count": args.reference_actions,
            "calibration_action_count": args.calibration_actions,
            "candidate_action_count": args.candidate_actions,
            "supported_candidate_count": len(candidates),
            "calibration_quantile": args.support_quantile,
            "radius": maximum_supported_radius,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    display_count = min(20_000, len(candidate_endpoints))
    display_indices = rng.choice(
        len(candidate_endpoints), size=display_count, replace=False
    )
    axis.scatter(
        candidate_endpoints[display_indices, 0],
        candidate_endpoints[display_indices, 1],
        s=4,
        alpha=0.12,
        color="royalblue",
        edgecolors="none",
        rasterized=True,
        label="Supported expert actions",
    )
    axis.scatter(
        targets[:, 0],
        targets[:, 1],
        s=260,
        marker="s",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Ideal grid",
        zorder=4,
    )
    axis.scatter(
        selected_endpoints[:, 0],
        selected_endpoints[:, 1],
        s=180,
        color="yellow",
        edgecolors="black",
        linewidths=1.2,
        label="Selected endpoints",
        zorder=5,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Hand endpoint x")
    axis.set_ylabel("Hand endpoint z")
    axis.set_title(
        f"Insert Peg supported counterfactual grid (spacing={spacing:.5f})"
    )
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)
    figure_path = output_dir / "insert_peg_hand_endpoint_grid.jpg"
    figure.savefig(figure_path, dpi=args.dpi)
    plt.close(figure)

    print(f"Selected grid spacing: {spacing:.8f}")
    print(f"Maximum relative endpoint error: {endpoint_errors.max() / spacing:.5f}")
    print(
        f"Support radius: {maximum_supported_radius:.6f}; selected max: "
        f"{selected_radii.max():.6f}"
    )
    print("Selected actions:")
    print(np.array2string(selected_actions, precision=7, suppress_small=True))
    print(f"Saved {output_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
