"""Evaluate unified directional distribution support metrics for HDF5 datasets.

The reference dataset is P and the target dataset is Q. Feature definitions are
inferred from the HDF5 schema. Use --fill-table with reports/stat_fullresult.tex
to evaluate exactly the pairs and factors summarized in reports/stat_check.tex
and replace the full-result table's metric cells.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from anygoal.paths import cache_root
from scipy.ndimage import gaussian_filter


BINS = 256
MAHALANOBIS_REFERENCE_QUANTILE = 0.99
MAHALANOBIS_TARGET_FRACTION = 0.95
KDE_REFERENCE_QUANTILE = 0.01
KDE_TARGET_FRACTION = 0.95
RANGE_TARGET_FRACTION = 0.99
DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
STABLE_WORLDMODEL_DIR = DATASET_DIR.parent
FALLBACK_DATA_DIR = cache_root() / "data"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "analysis"
    / "distribution_support"
)


@dataclass
class Feature:
    label: str
    values: np.ndarray
    periodic: bool = False


TWOROOM_TABLE_FEATURES = ("agent pos (2D)", "action (2D)")
PUSHT_TABLE_FEATURES = (
    "block angle",
    "agent pos (2D)",
    "block pos (2D)",
    "action (2D)",
)
INSERT_PEG_TABLE_FEATURES = (
    "arm root angle (cos,sin)",
    "arm shoulder angle (cos,sin)",
    "arm elbow angle (cos,sin)",
    "arm wrist angle (cos,sin)",
    "finger angle (cos,sin)",
    "fingertip angle (cos,sin)",
    "thumb angle (cos,sin)",
    "thumbtip angle (cos,sin)",
    "hand position (x,z)",
    "hand orientation (qw,qy)",
    "object position (x,z)",
    "object orientation (qw,qy)",
    "target position (x,z)",
    "target orientation (qw,qy)",
    "action dimension 0",
    "action dimension 1",
    "action dimension 2",
    "action dimension 3",
    "action dimension 4",
)
CUBE_TABLE_FEATURES = (
    "joint pos 0",
    "joint pos 1",
    "joint pos 2",
    "joint pos 3",
    "joint pos 4",
    "joint pos 5",
    "gripper opening",
    "gripper contact",
    "EE yaw",
    "delta yaw",
    "delta gripper",
    "EE pos (3D)",
    "block pos (3D)",
    "block quat (4D)",
    "delta pos (3D)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Reference dataset P.")
    parser.add_argument("--target", type=Path, help="Target dataset Q.")
    parser.add_argument("--reference-label")
    parser.add_argument("--target-label")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fill-table",
        type=Path,
        help="Run all predefined pairs and fill this TeX table in place.",
    )
    args = parser.parse_args()
    if args.fill_table is None and (args.reference is None or args.target is None):
        parser.error("provide both --reference and --target, or use --fill-table")
    if args.fill_table is not None and (args.reference is not None or args.target is not None):
        parser.error("--fill-table cannot be combined with --reference or --target")
    return args


def find_dataset(filename: str) -> Path:
    candidates = (
        DATASET_DIR / filename,
        STABLE_WORLDMODEL_DIR / filename,
        FALLBACK_DATA_DIR / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in: "
        + ", ".join(str(path.parent) for path in candidates)
    )


def finite_rows(values: np.ndarray, path: Path, feature: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    valid = np.isfinite(values).all(axis=1)
    values = values[valid]
    if not len(values):
        raise ValueError(f"{path}:{feature} contains no finite rows")
    return values


def read_field(dataset: h5py.File, path: Path, name: str) -> np.ndarray:
    if name not in dataset:
        raise KeyError(f"{path} is missing required field {name!r}")
    return finite_rows(dataset[name][:], path, name)


def detect_schema(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as dataset:
        fields = set(dataset.keys())
    if {
        "proprio_arm_pos",
        "proprio_hand_pos",
        "privileged_object_pos",
        "privileged_target_pos",
        "action",
    } <= fields:
        return "insert_peg"
    if {"proprio_joint_pos", "privileged_block_0_quat", "action"} <= fields:
        return "cube"
    if {"state", "proprio", "action"} <= fields:
        return "pusht"
    if {"proprio", "action"} <= fields:
        return "tworoom"
    raise ValueError(f"Could not infer a supported dataset schema from {path}")


def load_cube_features(path: Path) -> list[Feature]:
    with h5py.File(path, "r") as dataset:
        joint = read_field(dataset, path, "proprio_joint_pos")
        effector = read_field(dataset, path, "proprio_effector_pos")
        yaw = read_field(dataset, path, "proprio_effector_yaw")
        opening = read_field(dataset, path, "proprio_gripper_opening")
        contact = read_field(dataset, path, "proprio_gripper_contact")
        block_position = read_field(dataset, path, "privileged_block_0_pos")
        block_quaternion = read_field(dataset, path, "privileged_block_0_quat")
        action = read_field(dataset, path, "action")

    expected_dimensions = {
        "proprio_joint_pos": (joint, 6),
        "proprio_effector_pos": (effector, 3),
        "proprio_effector_yaw": (yaw, 1),
        "proprio_gripper_opening": (opening, 1),
        "proprio_gripper_contact": (contact, 1),
        "privileged_block_0_pos": (block_position, 3),
        "privileged_block_0_quat": (block_quaternion, 4),
        "action": (action, 5),
    }
    for name, (values, dimensions) in expected_dimensions.items():
        if values.shape[1] != dimensions:
            raise ValueError(
                f"{path}:{name} has {values.shape[1]} columns; expected {dimensions}"
            )

    features = [
        Feature(f"joint pos {axis}", joint[:, axis : axis + 1]) for axis in range(6)
    ]
    features.extend(
        Feature(f"EE pos {axis_name}", effector[:, axis : axis + 1])
        for axis, axis_name in enumerate("xyz")
    )
    features.extend(
        (
            Feature("EE yaw", yaw, periodic=True),
            Feature("gripper opening", opening),
            Feature("gripper contact", contact),
        )
    )
    features.extend(
        Feature(f"block pos {axis_name}", block_position[:, axis : axis + 1])
        for axis, axis_name in enumerate("xyz")
    )
    features.extend(
        Feature(f"block quat {axis_name}", block_quaternion[:, axis : axis + 1])
        for axis, axis_name in enumerate("wxyz")
    )
    action_names = ("delta x", "delta y", "delta z", "delta yaw", "delta gripper")
    features.extend(
        Feature(name, action[:, axis : axis + 1])
        for axis, name in enumerate(action_names)
    )
    features.extend(
        (
            Feature("EE pos (3D)", effector),
            Feature("block pos (3D)", block_position),
            Feature("block quat (4D)", block_quaternion),
            Feature("delta pos (3D)", action[:, :3]),
        )
    )
    return features


def load_pusht_features(path: Path) -> list[Feature]:
    with h5py.File(path, "r") as dataset:
        proprio = read_field(dataset, path, "proprio")
        state = read_field(dataset, path, "state")
        action = read_field(dataset, path, "action")
    if proprio.shape[1] != 4:
        raise ValueError(f"{path}:proprio has {proprio.shape[1]} columns; expected 4")
    if state.shape[1] != 7:
        raise ValueError(f"{path}:state has {state.shape[1]} columns; expected 7")
    if action.shape[1] != 2:
        raise ValueError(f"{path}:action has {action.shape[1]} columns; expected 2")
    agent_position = proprio[:, :2]
    block_position = state[:, 2:4]
    block_angle = state[:, 4:5]
    return [
        Feature("agent pos x", agent_position[:, :1]),
        Feature("agent pos y", agent_position[:, 1:2]),
        Feature("block pos x", block_position[:, :1]),
        Feature("block pos y", block_position[:, 1:2]),
        Feature("block angle", block_angle, periodic=True),
        Feature("x axis", action[:, :1]),
        Feature("y axis", action[:, 1:2]),
        Feature("agent pos (2D)", agent_position),
        Feature("block pos (2D)", block_position),
        Feature("action (2D)", action),
    ]


def load_tworoom_features(path: Path) -> list[Feature]:
    with h5py.File(path, "r") as dataset:
        agent_position = read_field(dataset, path, "proprio")
        action = read_field(dataset, path, "action")
    if agent_position.shape[1] != 2:
        raise ValueError(
            f"{path}:proprio has {agent_position.shape[1]} columns; expected 2"
        )
    if action.shape[1] != 2:
        raise ValueError(
            f"{path}:action has {action.shape[1]} columns; expected 2"
        )
    return [
        Feature("agent pos x", agent_position[:, :1]),
        Feature("agent pos y", agent_position[:, 1:2]),
        Feature("action x", action[:, :1]),
        Feature("action y", action[:, 1:2]),
        Feature("agent pos (2D)", agent_position),
        Feature("action (2D)", action),
    ]


def load_insert_peg_features(path: Path) -> list[Feature]:
    """Load independently controlled Insert-Peg coordinates as separate features."""
    with h5py.File(path, "r") as dataset:
        arm = read_field(dataset, path, "proprio_arm_pos")
        hand = read_field(dataset, path, "proprio_hand_pos")
        object_pose = read_field(dataset, path, "privileged_object_pos")
        target_pose = read_field(dataset, path, "privileged_target_pos")
        action = read_field(dataset, path, "action")
    expected = {"arm": (arm, 16), "hand": (hand, 4), "object": (object_pose, 4),
                "target": (target_pose, 4), "action": (action, 5)}
    for label, (feature_values, dimensions) in expected.items():
        if feature_values.shape[1] != dimensions:
            raise ValueError(
                f"{path}:{label} has {feature_values.shape[1]} columns; "
                f"expected {dimensions}"
            )

    joint_names = (
        "arm root", "arm shoulder", "arm elbow", "arm wrist",
        "finger", "fingertip", "thumb", "thumbtip",
    )
    features = [
        Feature(f"{name} angle (cos,sin)", arm[:, 2 * index : 2 * index + 2])
        for index, name in enumerate(joint_names)
    ]
    features.extend(
        (
            Feature("hand position (x,z)", hand[:, :2]),
            Feature("hand orientation (qw,qy)", hand[:, 2:4]),
            Feature("object position (x,z)", object_pose[:, :2]),
            Feature("object orientation (qw,qy)", object_pose[:, 2:4]),
            Feature("target position (x,z)", target_pose[:, :2]),
            Feature("target orientation (qw,qy)", target_pose[:, 2:4]),
        )
    )
    features.extend(
        Feature(f"action dimension {index}", action[:, index : index + 1])
        for index in range(action.shape[1])
    )
    return features


def load_features(path: Path, schema: str) -> list[Feature]:
    if schema == "cube":
        return load_cube_features(path)
    if schema == "pusht":
        return load_pusht_features(path)
    if schema == "tworoom":
        return load_tworoom_features(path)
    if schema == "insert_peg":
        return load_insert_peg_features(path)
    raise ValueError(f"Unsupported schema: {schema}")


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def pooled_limits(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    lower = np.minimum(reference.min(axis=0), target.min(axis=0))
    upper = np.maximum(reference.max(axis=0), target.max(axis=0))
    span = upper - lower
    fallback = np.maximum(np.maximum(np.abs(lower), np.abs(upper)) * 0.03, 1e-6)
    padding = np.where(span > 0.0, 0.03 * span, fallback)
    return np.column_stack((lower - padding, upper + padding))


def scott_bandwidth(points: np.ndarray) -> np.ndarray:
    dimensions = points.shape[1]
    return points.std(axis=0, ddof=0) * len(points) ** (-1.0 / (dimensions + 4.0))


def density_at_points(
    points: np.ndarray, density: np.ndarray, edges: list[np.ndarray]
) -> np.ndarray:
    indices = [
        np.searchsorted(axis_edges, points[:, axis], side="right") - 1
        for axis, axis_edges in enumerate(edges)
    ]
    inside = np.ones(len(points), dtype=bool)
    for axis, axis_indices in enumerate(indices):
        inside &= (axis_indices >= 0) & (axis_indices < density.shape[axis])
    values = np.zeros(len(points), dtype=np.float64)
    values[inside] = density[tuple(axis_indices[inside] for axis_indices in indices)]
    return values


def direct_kde_support(
    reference: np.ndarray, target: np.ndarray, *, periodic: bool
) -> dict:
    dimensions = reference.shape[1]
    if periodic:
        if dimensions != 1:
            raise ValueError("Periodic KDE is only implemented for scalar angles")
        reference = wrap_radians(reference)
        target = wrap_radians(target)
        limits = np.array([[-np.pi, np.pi]])
        mean_angle = np.arctan2(
            np.sin(reference[:, 0]).mean(), np.cos(reference[:, 0]).mean()
        )
        bandwidth_points = wrap_radians(reference - mean_angle)
        mode = "wrap"
    else:
        limits = pooled_limits(reference, target)
        bandwidth_points = reference
        mode = "constant"

    histogram, edges = np.histogramdd(
        reference, bins=BINS, range=limits.tolist()
    )
    widths = np.array([axis_edges[1] - axis_edges[0] for axis_edges in edges])
    bandwidth = scott_bandwidth(bandwidth_points)
    sigma_bins = np.maximum(bandwidth / widths, 0.5)
    density = gaussian_filter(
        histogram.astype(np.float32),
        sigma=sigma_bins,
        mode=mode,
        truncate=4.0,
    )
    normalization = float(density.sum(dtype=np.float64) * np.prod(widths))
    if normalization <= 0.0:
        raise ValueError("KDE has zero mass")
    density /= normalization
    reference_density = density_at_points(reference, density, edges)
    target_density = density_at_points(target, density, edges)
    threshold = float(np.quantile(reference_density, KDE_REFERENCE_QUANTILE))
    fraction = float(np.mean(target_density >= threshold))
    return {
        "mode": f"{dimensions}D histogram KDE",
        "bandwidth": bandwidth.tolist(),
        "sigma_bins": sigma_bins.tolist(),
        "density_threshold": threshold,
        "target_fraction": fraction,
    }


def kde_support(reference: np.ndarray, target: np.ndarray, periodic: bool) -> dict:
    dimensions = reference.shape[1]
    if dimensions <= 3:
        return direct_kde_support(reference, target, periodic=periodic)
    if periodic:
        raise ValueError("Periodic features with more than one dimension are unsupported")
    return full_gaussian_kde_support(reference, target)


def quantize_to_reference_bins(
    values: np.ndarray, limits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    widths = (limits[:, 1] - limits[:, 0]) / BINS
    inside = ((values >= limits[:, 0]) & (values <= limits[:, 1])).all(axis=1)
    indices = np.floor((values - limits[:, 0]) / widths).astype(np.int64)
    indices = np.clip(indices, 0, BINS - 1)
    centers = limits[:, 0] + (indices + 0.5) * widths
    return centers, inside


def weighted_log_gaussian_kde(
    reference: np.ndarray,
    log_weights: np.ndarray,
    query: np.ndarray,
    *,
    total_weight: int,
    bandwidth: float,
    log_determinant: float,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    reference_tensor = torch.from_numpy(reference.astype(np.float32, copy=False)).to(device)
    reference_norm = (reference_tensor * reference_tensor).sum(dim=1)
    log_weight_tensor = torch.from_numpy(log_weights.astype(np.float32, copy=False)).to(device)
    dimensions = reference.shape[1]
    log_normalizer = (
        np.log(total_weight)
        + 0.5 * dimensions * np.log(2.0 * np.pi)
        + dimensions * np.log(bandwidth)
        + log_determinant
    )
    inverse_two_bandwidth_squared = 0.5 / bandwidth**2
    output = np.empty(len(query), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(query), batch_size):
            stop = min(start + batch_size, len(query))
            query_tensor = torch.from_numpy(
                query[start:stop].astype(np.float32, copy=False)
            ).to(device)
            squared_distance = (
                (query_tensor * query_tensor).sum(dim=1)[:, None]
                + reference_norm[None, :]
                - 2.0 * query_tensor @ reference_tensor.T
            ).clamp_min_(0.0)
            output[start:stop] = (
                torch.logsumexp(
                    log_weight_tensor[None, :]
                    - squared_distance * inverse_two_bandwidth_squared,
                    dim=1,
                ).cpu().numpy().astype(np.float64)
                - log_normalizer
            )
    return output


def full_gaussian_kde_support(reference: np.ndarray, target: np.ndarray) -> dict:
    """Evaluate every dimension jointly without a pairwise marginal fallback."""
    limits = pooled_limits(reference, reference)
    reference_quantized, reference_inside = quantize_to_reference_bins(reference, limits)
    if not reference_inside.all():
        raise RuntimeError("Reference bin limits do not contain every reference row")
    unique_reference, inverse, counts = np.unique(
        reference_quantized, axis=0, return_inverse=True, return_counts=True
    )

    mean = reference_quantized.mean(axis=0)
    covariance = np.atleast_2d(np.cov(reference_quantized, rowvar=False, bias=False))
    covariance_scale = float(np.trace(covariance) / covariance.shape[0])
    regularizer = max(covariance_scale * 1e-10, 1e-12)
    covariance = covariance + np.eye(covariance.shape[0]) * regularizer
    cholesky = np.linalg.cholesky(covariance)
    log_determinant = float(np.log(np.diag(cholesky)).sum())

    def whiten(values: np.ndarray) -> np.ndarray:
        return np.linalg.solve(cholesky, (values - mean).T).T.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    bandwidth = len(reference_quantized) ** (-1.0 / (reference.shape[1] + 4.0))
    unique_reference_white = whiten(unique_reference)
    unique_reference_density = weighted_log_gaussian_kde(
        unique_reference_white,
        np.log(counts),
        unique_reference_white,
        total_weight=len(reference_quantized),
        bandwidth=bandwidth,
        log_determinant=log_determinant,
        device=device,
    )
    threshold = float(
        np.quantile(unique_reference_density[inverse], KDE_REFERENCE_QUANTILE)
    )

    target_quantized, target_inside = quantize_to_reference_bins(target, limits)
    target_density = np.full(len(target), -np.inf, dtype=np.float64)
    inside_indices = np.flatnonzero(target_inside)
    unique_target, target_inverse = np.unique(
        target_quantized[target_inside], axis=0, return_inverse=True
    )
    if len(unique_target):
        unique_target_density = weighted_log_gaussian_kde(
            unique_reference_white,
            np.log(counts),
            whiten(unique_target),
            total_weight=len(reference_quantized),
            bandwidth=bandwidth,
            log_determinant=log_determinant,
            device=device,
        )
        target_density[inside_indices] = unique_target_density[target_inverse]
    return {
        "mode": f"full-covariance {reference.shape[1]}D Gaussian KDE after 256-bin quantization",
        "target_fraction": float(np.mean(target_density >= threshold)),
        "log_density_threshold": threshold,
        "bandwidth": bandwidth,
        "covariance_regularizer": regularizer,
        "reference_unique_centers": int(len(unique_reference)),
        "target_unique_centers": int(len(unique_target)),
        "device": str(device),
    }


def regularized_covariance(points: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = np.atleast_2d(np.cov(points, rowvar=False, bias=True))
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularizer = max(scale * 1e-8, 1e-12)
    return covariance + np.eye(covariance.shape[0]) * regularizer, regularizer


def mahalanobis_support(
    reference: np.ndarray, target: np.ndarray, periodic: bool
) -> dict:
    if periodic:
        reference_angles = wrap_radians(reference)[:, 0]
        target_angles = wrap_radians(target)[:, 0]
        reference = np.column_stack(
            (np.cos(reference_angles), np.sin(reference_angles))
        )
        target = np.column_stack((np.cos(target_angles), np.sin(target_angles)))
        representation = "cosine/sine circular embedding"
    else:
        representation = "stored feature coordinates"
    mean = reference.mean(axis=0)
    covariance, regularizer = regularized_covariance(reference)
    inverse = np.linalg.inv(covariance)

    def squared_distance(points: np.ndarray) -> np.ndarray:
        centered = points - mean
        return np.einsum(
            "ni,ij,nj->n", centered, inverse, centered, optimize=True
        )

    reference_distance = squared_distance(reference)
    target_distance = squared_distance(target)
    threshold = float(
        np.quantile(reference_distance, MAHALANOBIS_REFERENCE_QUANTILE)
    )
    return {
        "representation": representation,
        "covariance_regularizer": regularizer,
        "squared_distance_threshold": threshold,
        "target_fraction": float(np.mean(target_distance <= threshold)),
    }


def evaluate_feature(reference: Feature, target: Feature) -> dict:
    if reference.label != target.label:
        raise ValueError(f"Feature mismatch: {reference.label} != {target.label}")
    if reference.values.shape[1] != target.values.shape[1]:
        raise ValueError(f"Dimension mismatch for {reference.label}")
    if reference.periodic != target.periodic:
        raise ValueError(f"Periodicity mismatch for {reference.label}")

    mahalanobis = mahalanobis_support(
        reference.values, target.values, reference.periodic
    )
    kde = kde_support(reference.values, target.values, reference.periodic)
    if reference.periodic:
        range_result = {"ignored_for_periodic_variable": True, "target_fraction": None}
        range_pass = None
    else:
        lower = reference.values.min(axis=0)
        upper = reference.values.max(axis=0)
        inside = np.all(
            (target.values >= lower) & (target.values <= upper), axis=1
        )
        range_fraction = float(np.mean(inside))
        range_result = {
            "ignored_for_periodic_variable": False,
            "reference_min": lower.tolist(),
            "reference_max": upper.tolist(),
            "target_fraction": range_fraction,
        }
        range_pass = range_fraction >= RANGE_TARGET_FRACTION

    criteria = {
        "mahalanobis": mahalanobis["target_fraction"]
        >= MAHALANOBIS_TARGET_FRACTION,
        "kde": kde["target_fraction"] >= KDE_TARGET_FRACTION,
        "range": range_pass,
    }
    verdict = all(value for value in criteria.values() if value is not None)
    return {
        "dimensions": int(reference.values.shape[1]),
        "periodic": reference.periodic,
        "reference_rows": int(len(reference.values)),
        "target_rows": int(len(target.values)),
        "mahalanobis": mahalanobis,
        "kde": kde,
        "range": range_result,
        "criteria": criteria,
        "final_verdict": bool(verdict),
    }


def compare_pair(
    reference_path: Path,
    target_path: Path,
    reference_label: str,
    target_label: str,
    selected_features: tuple[str, ...] | None = None,
) -> dict:
    reference_schema = detect_schema(reference_path)
    target_schema = detect_schema(target_path)
    if reference_schema != target_schema:
        raise ValueError(
            f"Schema mismatch: {reference_path} is {reference_schema}, "
            f"but {target_path} is {target_schema}"
        )
    reference_features = load_features(reference_path, reference_schema)
    target_features = load_features(target_path, target_schema)
    if [feature.label for feature in reference_features] != [
        feature.label for feature in target_features
    ]:
        raise ValueError("Reference and target feature lists differ")
    if selected_features is not None:
        reference_by_label = {feature.label: feature for feature in reference_features}
        target_by_label = {feature.label: feature for feature in target_features}
        missing = set(selected_features).difference(reference_by_label).union(
            set(selected_features).difference(target_by_label)
        )
        if missing:
            raise KeyError(f"Missing requested features: {sorted(missing)}")
        reference_features = [reference_by_label[label] for label in selected_features]
        target_features = [target_by_label[label] for label in selected_features]

    results = {}
    for reference_feature, target_feature in zip(reference_features, target_features):
        result = evaluate_feature(reference_feature, target_feature)
        results[reference_feature.label] = result
        range_fraction = result["range"]["target_fraction"]
        range_text = "ignored" if range_fraction is None else f"{range_fraction:.6f}"
        print(
            f"{reference_label} -> {target_label} | {reference_feature.label} | "
            f"mahalanobis={result['mahalanobis']['target_fraction']:.6f} "
            f"kde={result['kde']['target_fraction']:.6f} "
            f"range={range_text} verdict={result['final_verdict']}",
            flush=True,
        )
    return {
        "reference_label": reference_label,
        "target_label": target_label,
        "reference_path": str(reference_path.resolve()),
        "target_path": str(target_path.resolve()),
        "schema": reference_schema,
        "features": results,
    }


def metric_rows(pair: dict) -> list[dict]:
    rows = []
    for feature, result in pair["features"].items():
        rows.append(
            {
                "reference": pair["reference_label"],
                "target": pair["target_label"],
                "feature": feature,
                "dimensions": result["dimensions"],
                "periodic": result["periodic"],
                "mahalanobis_fraction": result["mahalanobis"]["target_fraction"],
                "mahalanobis_pass": result["criteria"]["mahalanobis"],
                "kde_mode": result["kde"]["mode"],
                "kde_fraction": result["kde"]["target_fraction"],
                "kde_pass": result["criteria"]["kde"],
                "range_fraction": result["range"]["target_fraction"],
                "range_pass": result["criteria"]["range"],
                "final_verdict": result["final_verdict"],
            }
        )
    return rows


def write_outputs(pairs: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": {
            "bins": BINS,
            "pooled_nonperiodic_padding_fraction": 0.03,
            "mahalanobis_reference_quantile": MAHALANOBIS_REFERENCE_QUANTILE,
            "mahalanobis_required_target_fraction": MAHALANOBIS_TARGET_FRACTION,
            "kde_reference_quantile": KDE_REFERENCE_QUANTILE,
            "kde_required_target_fraction": KDE_TARGET_FRACTION,
            "range_required_target_fraction": RANGE_TARGET_FRACTION,
            "kde_up_to_three_dimensions": "direct histogram KDE",
            "kde_four_or_more_dimensions": (
                "full-covariance Gaussian KDE after 256-bin quantization"
            ),
        },
        "pairs": pairs,
    }
    (output_dir / "support_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    rows = [row for pair in pairs for row in metric_rows(pair)]
    with (output_dir / "support_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_fraction(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def table_feature_label(feature: str) -> str:
    if feature.endswith(" angle (cos,sin)"):
        return feature.removesuffix(" (cos,sin)")
    if feature.startswith("object "):
        return feature.replace("object ", "peg ", 1)
    if feature.startswith("action dimension "):
        return feature.replace("action dimension ", "motor command ", 1)
    return feature


def fill_tex_table(table_path: Path, pairs: list[dict]) -> None:
    expected = [
        (feature, result)
        for pair in pairs
        for feature, result in pair["features"].items()
    ]
    lines = table_path.read_text().splitlines()
    data_indices = [
        index
        for index, line in enumerate(lines)
        if line.count("&") == 8 and "factor of variants" not in line
    ]
    if len(data_indices) != len(expected):
        raise ValueError(
            f"{table_path} has {len(data_indices)} data rows; expected {len(expected)}"
        )
    for line_index, (expected_feature, result) in zip(data_indices, expected):
        columns = lines[line_index].removesuffix(r" \\").split("&")
        table_feature = columns[4].strip()
        expected_table_feature = table_feature_label(expected_feature)
        if table_feature != expected_table_feature:
            raise ValueError(
                f"Table row {line_index + 1} is {table_feature!r}; "
                f"expected {expected_table_feature!r}"
            )
        metric_columns = [
            format_fraction(result["mahalanobis"]["target_fraction"]),
            format_fraction(result["kde"]["target_fraction"]),
            format_fraction(result["range"]["target_fraction"]),
            "pass" if result["final_verdict"] else "fail",
        ]
        lines[line_index] = " & ".join(
            [column.rstrip() for column in columns[:5]] + metric_columns
        ) + r" \\"
    table_path.write_text("\n".join(lines) + "\n")


def batch_pairs() -> list[tuple[Path, Path, str, str, tuple[str, ...]]]:
    return [
        (
            find_dataset("tworoom.h5"),
            find_dataset("tworoom_route_test.h5"),
            "TwoRoom",
            "TR-Route",
            TWOROOM_TABLE_FEATURES,
        ),
        (
            find_dataset("pusht_expert_train.h5"),
            find_dataset("avoidt_test.h5"),
            "PushT",
            "Avoid-T",
            PUSHT_TABLE_FEATURES,
        ),
        (
            find_dataset("pusht_expert_train.h5"),
            find_dataset("stackt_test.h5"),
            "PushT",
            "Stack-T",
            PUSHT_TABLE_FEATURES,
        ),
        (
            find_dataset("pusht_expert_train.h5"),
            find_dataset("pusht_antigoal_test.h5"),
            "PushT",
            "Push-T Antigoal",
            PUSHT_TABLE_FEATURES,
        ),
        (
            find_dataset("manipulator_insert_peg_weak_expert_train.h5"),
            find_dataset("avoid_peg_test.h5"),
            "Insert-Peg",
            "Avoid-Peg",
            INSERT_PEG_TABLE_FEATURES,
        ),
        (
            find_dataset("manipulator_insert_peg_weak_expert_train.h5"),
            find_dataset("extract_peg_test.h5"),
            "Insert-Peg",
            "Extract-Peg",
            INSERT_PEG_TABLE_FEATURES,
        ),
        (
            find_dataset("cube_single_expert.h5"),
            find_dataset("cube_hop_test.h5"),
            "Cube",
            "Cube-Hop",
            CUBE_TABLE_FEATURES,
        ),
        (
            find_dataset("cube_single_expert.h5"),
            find_dataset("cube_diagonal_test.h5"),
            "Cube",
            "Cube-Diagonal",
            CUBE_TABLE_FEATURES,
        ),
    ]


def main() -> None:
    args = parse_args()
    if args.fill_table is not None:
        pairs = [compare_pair(*configuration) for configuration in batch_pairs()]
        write_outputs(pairs, args.output_dir)
        fill_tex_table(args.fill_table, pairs)
        print(f"Filled {args.fill_table} and wrote metrics to {args.output_dir}")
        return

    reference_label = args.reference_label or args.reference.stem
    target_label = args.target_label or args.target.stem
    pairs = [
        compare_pair(args.reference, args.target, reference_label, target_label)
    ]
    write_outputs(pairs, args.output_dir)
    print(f"Wrote metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
