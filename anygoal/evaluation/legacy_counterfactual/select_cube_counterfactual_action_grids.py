#!/usr/bin/env python
"""Select maximum-volume Cube counterfactual action grids inside support."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets.cube_collection import (
    CALIBRATION_ACTIONS,
    REFERENCE_ACTIONS,
    REFERENCE_ACTION_SEED,
    SUPPORT_NEIGHBORS,
    SUPPORT_QUANTILE,
    CubeActionSupport,
)


DEFAULT_DATASET = (
    Path.home() / ".stable_worldmodel" / "datasets" / "cube_single_expert.h5"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "paper"
    / "results"
    / "cube_counterfactual_sensitivity"
    / "action_grids.json"
)
DEFAULT_FIXED_Z_OUTPUT = (
    REPO_ROOT
    / "paper"
    / "results"
    / "cube_counterfactual_sensitivity"
    / "fixed_negative_z_8point_grids.json"
)
OPTIMIZER_SEED = 20260820


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--popsize", type=int, default=40)
    parser.add_argument(
        "--fixed-negative-z",
        type=float,
        nargs="+",
        help=(
            "Evaluate 8-point xy grids at these fixed z values. Each value must "
            "be in [-1, -0.25)."
        ),
    )
    return parser.parse_args()


def cartesian_action_grid(magnitudes: np.ndarray) -> np.ndarray:
    """Return {-m, 0, m} products in xyz with yaw=gripper=0."""
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    if magnitudes.shape not in {(2,), (3,)}:
        raise ValueError("magnitudes must contain either two or three values")
    coordinates = [(-value, 0.0, value) for value in magnitudes]
    if len(magnitudes) == 2:
        coordinates.append((0.0,))
    xyz = np.asarray(list(itertools.product(*coordinates)), dtype=np.float64)
    actions = np.zeros((len(xyz), 5), dtype=np.float64)
    actions[:, :3] = xyz
    return actions


def select_grid(
    support: CubeActionSupport,
    dimensions: int,
    *,
    maxiter: int,
    popsize: int,
) -> dict:
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")

    def radii(magnitudes: np.ndarray) -> np.ndarray:
        actions = cartesian_action_grid(magnitudes)
        return np.asarray(support._neighbor_radius(actions), dtype=np.float64)

    def maximum_radius(magnitudes: np.ndarray) -> float:
        return float(radii(magnitudes).max())

    radius_scale = max(support.support_radius, np.finfo(np.float64).eps)

    def objective(magnitudes: np.ndarray) -> float:
        volume_cost = -float(np.log(np.maximum(magnitudes, 1e-12)).sum())
        excess = max(0.0, maximum_radius(magnitudes) - support.support_radius)
        return volume_cost + 1e6 * excess / radius_scale

    result = differential_evolution(
        objective,
        bounds=[(1e-6, 1.0)] * dimensions,
        seed=OPTIMIZER_SEED + dimensions,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-10,
        atol=1e-12,
        polish=False,
        updating="immediate",
        workers=1,
    )
    magnitudes = np.asarray(result.x, dtype=np.float64)
    if maximum_radius(magnitudes) > support.support_radius:
        lower = 0.0
        upper = 1.0
        for _ in range(64):
            scale = (lower + upper) / 2.0
            if maximum_radius(magnitudes * scale) <= support.support_radius:
                lower = scale
            else:
                upper = scale
        magnitudes *= lower

    # Make the DE solution coordinate-wise maximal while preserving feasibility.
    for _ in range(8):
        previous = magnitudes.copy()
        for axis in range(dimensions):
            lower = float(magnitudes[axis])
            upper = 1.0
            for _ in range(52):
                candidate = magnitudes.copy()
                candidate[axis] = (lower + upper) / 2.0
                if maximum_radius(candidate) <= support.support_radius:
                    lower = float(candidate[axis])
                else:
                    upper = float(candidate[axis])
            magnitudes[axis] = lower
        if np.max(np.abs(magnitudes - previous)) < 1e-12:
            break

    actions = cartesian_action_grid(magnitudes)
    action_radii = radii(magnitudes)
    if np.any(action_radii > support.support_radius):
        raise RuntimeError("Selected action grid does not pass CubeActionSupport")
    return {
        "dimensions": dimensions,
        "objective": "maximum Cartesian grid volume (product of symmetric magnitudes)",
        "magnitudes": magnitudes.tolist(),
        "volume": float(np.prod(magnitudes)),
        "actions": actions.tolist(),
        "action_20nn_radii": action_radii.tolist(),
        "maximum_20nn_radius": float(action_radii.max()),
        "all_points_supported": True,
        "optimizer": {
            "method": "seeded scipy differential_evolution followed by coordinate-wise binary refinement",
            "seed": OPTIMIZER_SEED + dimensions,
            "maxiter": maxiter,
            "popsize": popsize,
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
        },
    }


def fixed_negative_z_action_grid(
    magnitudes: np.ndarray, fixed_z: float
) -> np.ndarray:
    """Return the eight non-central xy points at a fixed negative z."""
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    if magnitudes.shape != (2,):
        raise ValueError("fixed-z grid requires x and y magnitudes")
    xy = np.asarray(
        list(itertools.product((-magnitudes[0], 0.0, magnitudes[0]),
                               (-magnitudes[1], 0.0, magnitudes[1]))),
        dtype=np.float64,
    )
    xy = xy[np.any(xy != 0.0, axis=1)]
    actions = np.zeros((len(xy), 5), dtype=np.float64)
    actions[:, :2] = xy
    actions[:, 2] = fixed_z
    return actions


def select_fixed_negative_z_grid(
    support: CubeActionSupport,
    fixed_z: float,
    *,
    maxiter: int,
    popsize: int,
) -> dict:
    """Maximize xy grid area at fixed z while requiring all eight points."""
    if not -1.0 <= fixed_z < -0.25:
        raise ValueError("fixed z must be in [-1, -0.25)")

    def radii(magnitudes: np.ndarray) -> np.ndarray:
        return np.asarray(
            support._neighbor_radius(
                fixed_negative_z_action_grid(magnitudes, fixed_z)
            ),
            dtype=np.float64,
        )

    def maximum_radius(magnitudes: np.ndarray) -> float:
        return float(radii(magnitudes).max())

    radius_scale = max(support.support_radius, np.finfo(np.float64).eps)

    def objective(magnitudes: np.ndarray) -> float:
        area_cost = -float(np.log(magnitudes).sum())
        excess = max(0.0, maximum_radius(magnitudes) - support.support_radius)
        return area_cost + 1e6 * excess / radius_scale

    result = differential_evolution(
        objective,
        bounds=[(0.25, 1.0)] * 2,
        seed=OPTIMIZER_SEED + int(round(abs(fixed_z) * 10_000)),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-10,
        atol=1e-12,
        polish=False,
        updating="immediate",
        workers=1,
    )
    magnitudes = np.asarray(result.x, dtype=np.float64)
    if maximum_radius(magnitudes) > support.support_radius:
        lower, upper = 0.0, 1.0
        for _ in range(64):
            scale = (lower + upper) / 2.0
            if maximum_radius(magnitudes * scale) <= support.support_radius:
                lower = scale
            else:
                upper = scale
        magnitudes = np.maximum(magnitudes * lower, 0.25)

    for _ in range(8):
        previous = magnitudes.copy()
        for axis in range(2):
            lower, upper = float(magnitudes[axis]), 1.0
            for _ in range(52):
                candidate = magnitudes.copy()
                candidate[axis] = (lower + upper) / 2.0
                if maximum_radius(candidate) <= support.support_radius:
                    lower = float(candidate[axis])
                else:
                    upper = float(candidate[axis])
            magnitudes[axis] = lower
        if np.max(np.abs(magnitudes - previous)) < 1e-12:
            break

    actions = fixed_negative_z_action_grid(magnitudes, fixed_z)
    action_radii = radii(magnitudes)
    if len(actions) != 8 or np.any(action_radii > support.support_radius):
        raise RuntimeError("Selected fixed-z grid is not fully supported")
    return {
        "fixed_delta_z": float(fixed_z),
        "objective": "maximum 8-point xy-grid area (a*b) at fixed delta z",
        "magnitudes": {"a": float(magnitudes[0]), "b": float(magnitudes[1])},
        "area": float(np.prod(magnitudes)),
        "actions": actions.tolist(),
        "action_20nn_radii": action_radii.tolist(),
        "maximum_20nn_radius": float(action_radii.max()),
        "all_points_supported": True,
        "optimizer": {
            "method": "seeded scipy differential_evolution followed by coordinate-wise binary refinement",
            "seed": OPTIMIZER_SEED + int(round(abs(fixed_z) * 10_000)),
            "maxiter": maxiter,
            "popsize": popsize,
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
        },
    }


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if args.maxiter < 1 or args.popsize < 1:
        raise ValueError("maxiter and popsize must be positive")

    support = CubeActionSupport(dataset_path)
    result = {
        "dataset_path": str(dataset_path),
        "action_coordinates": "raw stored normalized 5-D Cube action",
        "fixed_action_components": {"delta_yaw": 0.0, "delta_gripper": 0.0},
        "support": {
            "method": "raw normalized 5-D Euclidean 20-nearest-neighbor radius",
            "reference_subset": "finite actions with proprio_gripper_opening < 0.05",
            "seed": REFERENCE_ACTION_SEED,
            "reference_action_count": REFERENCE_ACTIONS,
            "calibration_action_count": CALIBRATION_ACTIONS,
            "neighbors": SUPPORT_NEIGHBORS,
            "calibration_quantile": SUPPORT_QUANTILE,
            "support_radius": support.support_radius,
            "open_gripper_action_count": support.open_action_count,
        },
    }
    if args.fixed_negative_z is None:
        result["grid_3d_27"] = select_grid(
            support, 3, maxiter=args.maxiter, popsize=args.popsize
        )
        result["grid_xy_9"] = select_grid(
            support, 2, maxiter=args.maxiter, popsize=args.popsize
        )
    else:
        result["fixed_negative_z_8point_grids"] = [
            select_fixed_negative_z_grid(
                support, fixed_z, maxiter=args.maxiter, popsize=args.popsize
            )
            for fixed_z in args.fixed_negative_z
        ]
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_FIXED_Z_OUTPUT
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
