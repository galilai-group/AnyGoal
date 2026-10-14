"""Report full-dimensional KDE support for joint state/action distributions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

import distribution_support as support
from anygoal.paths import cache_root, output_root


OUTPUT_DIR = output_root() / "analysis" / "joint_distribution_support"
BASE_CACHE_DIR = cache_root() / "joint_kde_base_cache_v1"
CACHE_VERSION = 1


@dataclass(frozen=True)
class EnvironmentSpec:
    base_environment: str
    reference_filename: str
    target_filename: str
    schema: str


ENVIRONMENTS = {
    "avoidt": EnvironmentSpec(
        "pusht", "pusht_expert_train.h5", "avoidt_test.h5", "pusht"
    ),
    "stackt": EnvironmentSpec(
        "pusht", "pusht_expert_train.h5", "stackt_test.h5", "pusht"
    ),
    "pusht_antigoal": EnvironmentSpec(
        "pusht",
        "pusht_expert_train.h5",
        "pusht_antigoal_test.h5",
        "pusht",
    ),
    "cube_hop": EnvironmentSpec(
        "cube", "cube_single_expert.h5", "cube_hop_test.h5", "cube"
    ),
    "tworoom_route": EnvironmentSpec(
        "tworoom", "tworoom.h5", "tworoom_route_test.h5", "tworoom"
    ),
}

CASE_ORDER = (
    "agent_state",
    "all_state",
    "action_agent_state",
    "action_all_state",
)

BASE_ENVIRONMENTS = {
    "pusht": ("pusht_expert_train.h5", "pusht"),
    "cube": ("cube_single_expert.h5", "cube"),
    "tworoom": ("tworoom.h5", "tworoom"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=("all", *ENVIRONMENTS),
        default="all",
        help="Target environment to analyze (default: all).",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-cache-dir", type=Path, default=BASE_CACHE_DIR)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument(
        "--schema", choices=("cube", "pusht", "tworoom", "insert_peg")
    )
    parser.add_argument("--target-label", default="custom_target")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kde-batch-size", type=int, default=1024)
    return parser.parse_args()


def read_columns(dataset: h5py.File, path: Path, name: str, width: int) -> np.ndarray:
    if name not in dataset:
        raise KeyError(f"{path} is missing required field {name!r}")
    values = np.asarray(dataset[name][:], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(
            f"{path}:{name} has shape {values.shape}; expected [N, {width}]"
        )
    return values


def load_components(path: Path, schema: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as dataset:
        if schema == "pusht":
            state = read_columns(dataset, path, "state", 7)
            components = {
                "agent_state": state[:, :2],
                "other_state": state[:, 2:5],
                "action": read_columns(dataset, path, "action", 2),
            }
        elif schema == "cube":
            components = {
                "agent_state": np.column_stack(
                    (
                        read_columns(dataset, path, "proprio_effector_pos", 3),
                        read_columns(dataset, path, "proprio_gripper_opening", 1),
                        read_columns(dataset, path, "proprio_gripper_contact", 1),
                        read_columns(dataset, path, "proprio_effector_yaw", 1),
                    )
                ),
                "other_state": np.column_stack(
                    (
                        read_columns(dataset, path, "privileged_block_0_pos", 3),
                        read_columns(dataset, path, "privileged_block_0_quat", 4),
                    )
                ),
                "action": read_columns(dataset, path, "action", 5),
            }
        elif schema == "tworoom":
            components = {
                "agent_state": read_columns(dataset, path, "proprio", 2),
                "other_state": np.empty((len(dataset["proprio"]), 0)),
                "action": read_columns(dataset, path, "action", 2),
            }
        elif schema == "insert_peg":
            components = {
                "agent_state": np.column_stack(
                    (
                        read_columns(dataset, path, "proprio_arm_pos", 16),
                        read_columns(dataset, path, "proprio_hand_pos", 4),
                    )
                ),
                "other_state": np.column_stack(
                    (
                        read_columns(dataset, path, "privileged_object_pos", 4),
                        read_columns(dataset, path, "privileged_target_pos", 4),
                    )
                ),
                "action": read_columns(dataset, path, "action", 5),
            }
        else:
            raise ValueError(f"Unsupported schema: {schema}")

    row_counts = {len(values) for values in components.values()}
    if len(row_counts) != 1:
        raise ValueError(f"Fields in {path} have inconsistent row counts")
    return components


def make_case_vectors(
    components: dict[str, np.ndarray], case_name: str
) -> np.ndarray:
    agent_state = components["agent_state"]
    other_state = components["other_state"]
    action = components["action"]
    if case_name == "agent_state":
        values = agent_state
    elif case_name == "all_state":
        values = np.column_stack((agent_state, other_state))
    elif case_name == "action_agent_state":
        values = np.column_stack((action, agent_state))
    elif case_name == "action_all_state":
        values = np.column_stack((action, agent_state, other_state))
    else:
        raise KeyError(case_name)

    # Cube action has one terminal NaN per episode. State-only cases retain
    # those valid terminal states; action cases exclude them by construction.
    return values[np.isfinite(values).all(axis=1)]


def reference_bin_limits(reference: np.ndarray) -> np.ndarray:
    return support.pooled_limits(reference, reference)


def quantize_to_bin_centers(
    values: np.ndarray, limits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates inside fixed base limits to 256-bin centers."""
    widths = (limits[:, 1] - limits[:, 0]) / support.BINS
    inside = ((values >= limits[:, 0]) & (values <= limits[:, 1])).all(axis=1)
    indices = np.floor((values - limits[:, 0]) / widths).astype(np.int64)
    indices = np.clip(indices, 0, support.BINS - 1)
    quantized = limits[:, 0] + (indices + 0.5) * widths
    return quantized, inside


def fit_reference_whitening(
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Apply the full reference covariance used by scipy gaussian_kde."""
    mean = reference.mean(axis=0)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False, bias=False))
    covariance_scale = float(np.trace(covariance) / covariance.shape[0])
    regularizer = max(covariance_scale * 1e-10, 1e-12)
    covariance = covariance + np.eye(covariance.shape[0]) * regularizer
    cholesky = np.linalg.cholesky(covariance)
    log_abs_determinant = float(np.log(np.diag(cholesky)).sum())
    return mean, cholesky, regularizer, log_abs_determinant


def apply_whitening(
    values: np.ndarray, mean: np.ndarray, cholesky: np.ndarray
) -> np.ndarray:
    return np.linalg.solve(cholesky, (values - mean).T).T


def evaluate_full_kde(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    bandwidth: float,
    log_abs_determinant: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate the full Gaussian KDE without dropping any dimensions or rows."""
    reference_tensor = torch.from_numpy(
        reference.astype(np.float32, copy=False)
    ).to(device)
    reference_norm = (reference_tensor * reference_tensor).sum(dim=1)
    dimensions = reference.shape[1]
    log_normalizer = (
        np.log(len(reference))
        + 0.5 * dimensions * np.log(2.0 * np.pi)
        + dimensions * np.log(bandwidth)
        + log_abs_determinant
    )
    log_density = np.empty(len(query), dtype=np.float64)
    inverse_two_bandwidth_squared = 0.5 / bandwidth**2

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
            batch_log_density = torch.logsumexp(
                -squared_distance * inverse_two_bandwidth_squared, dim=1
            )
            log_density[start:stop] = (
                batch_log_density.cpu().numpy().astype(np.float64) - log_normalizer
            )
    return log_density


def full_kde_support(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict:
    cache = prepare_reference_cache(
        reference, device=device, batch_size=batch_size
    )
    return score_target_from_cache(
        cache, target, device=device, batch_size=batch_size
    )


def prepare_reference_cache(
    reference: np.ndarray, *, device: torch.device, batch_size: int
) -> dict:
    limits = reference_bin_limits(reference)
    reference, inside = quantize_to_bin_centers(reference, limits)
    if not inside.all():
        raise RuntimeError("Base bin limits do not contain every reference row")
    mean, cholesky, regularizer, log_determinant = fit_reference_whitening(reference)
    reference = apply_whitening(reference, mean, cholesky).astype(np.float32)
    dimensions = reference.shape[1]
    bandwidth = len(reference) ** (-1.0 / (dimensions + 4.0))
    reference_log_density = evaluate_full_kde(
        reference,
        reference,
        bandwidth=bandwidth,
        log_abs_determinant=log_determinant,
        device=device,
        batch_size=batch_size,
    )
    return {
        "reference": reference,
        "limits": limits,
        "mean": mean,
        "cholesky": cholesky,
        "regularizer": regularizer,
        "log_determinant": log_determinant,
        "bandwidth": bandwidth,
        "threshold": float(
            np.quantile(reference_log_density, support.KDE_REFERENCE_QUANTILE)
        ),
    }


def score_target_from_cache(
    cache: dict,
    target: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict:
    target, inside = quantize_to_bin_centers(target, cache["limits"])
    target = apply_whitening(target, cache["mean"], cache["cholesky"])
    target_log_density = np.full(len(target), -np.inf, dtype=np.float64)
    target_log_density[inside] = evaluate_full_kde(
        cache["reference"],
        target[inside],
        bandwidth=cache["bandwidth"],
        log_abs_determinant=cache["log_determinant"],
        device=device,
        batch_size=batch_size,
    )
    return {
        "target_fraction": float(
            np.mean(target_log_density >= cache["threshold"])
        ),
        "log_density_threshold": cache["threshold"],
        "scott_factor": cache["bandwidth"],
        "covariance_regularizer": cache["regularizer"],
    }


def base_cache_path(cache_dir: Path, base_environment: str, case_name: str) -> Path:
    return cache_dir / f"{base_environment}_{case_name}_full_kde.npz"


def save_reference_cache(
    path: Path,
    cache: dict,
    *,
    base_environment: str,
    case_name: str,
    reference_path: Path,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.npz")
    reference_stat = reference_path.stat()
    np.savez(
        temporary_path,
        cache_version=CACHE_VERSION,
        base_environment=base_environment,
        case_name=case_name,
        reference_path=str(reference_path.resolve()),
        reference_size=reference_stat.st_size,
        reference_mtime_ns=reference_stat.st_mtime_ns,
        bins=support.BINS,
        reference_density_quantile=support.KDE_REFERENCE_QUANTILE,
        reference=cache["reference"],
        limits=cache["limits"],
        mean=cache["mean"],
        cholesky=cache["cholesky"],
        regularizer=cache["regularizer"],
        log_determinant=cache["log_determinant"],
        bandwidth=cache["bandwidth"],
        threshold=cache["threshold"],
    )
    temporary_path.replace(path)


def load_reference_cache(path: Path, reference_path: Path) -> dict:
    reference_stat = reference_path.stat()
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["cache_version"]) != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version in {path}")
        if int(saved["bins"]) != support.BINS:
            raise ValueError(f"Bin count mismatch in {path}")
        if str(saved["reference_path"]) != str(reference_path.resolve()):
            raise ValueError(f"Reference path mismatch in {path}")
        if int(saved["reference_size"]) != reference_stat.st_size or int(
            saved["reference_mtime_ns"]
        ) != reference_stat.st_mtime_ns:
            raise ValueError(f"Reference dataset changed after creating {path}")
        return {
            "reference": saved["reference"].copy(),
            "limits": saved["limits"].copy(),
            "mean": saved["mean"].copy(),
            "cholesky": saved["cholesky"].copy(),
            "regularizer": float(saved["regularizer"]),
            "log_determinant": float(saved["log_determinant"]),
            "bandwidth": float(saved["bandwidth"]),
            "threshold": float(saved["threshold"]),
        }


def analyze_environment(
    environment: str,
    spec: EnvironmentSpec,
    output_dir: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    reference_path = Path(spec.reference_filename)
    if not reference_path.is_file():
        reference_path = support.find_dataset(spec.reference_filename)
    target_path = Path(spec.target_filename)
    if not target_path.is_file():
        target_path = support.find_dataset(spec.target_filename)
    print(f"[{environment}] reference: {reference_path}", flush=True)
    print(f"[{environment}] target:    {target_path}", flush=True)
    target_components = load_components(target_path, spec.schema)
    reference_components = None

    rows = []
    for case_name in CASE_ORDER:
        target = make_case_vectors(target_components, case_name)
        cache_path = base_cache_path(cache_dir, spec.base_environment, case_name)
        if cache_path.is_file():
            cache = load_reference_cache(cache_path, reference_path)
            result = score_target_from_cache(
                cache, target, device=device, batch_size=batch_size
            )
            reference_row_count = len(cache["reference"])
            reference_dimensions = cache["reference"].shape[1]
            used_cache = True
        else:
            if reference_components is None:
                reference_components = load_components(reference_path, spec.schema)
            reference = make_case_vectors(reference_components, case_name)
            cache = prepare_reference_cache(
                reference, device=device, batch_size=batch_size
            )
            save_reference_cache(
                cache_path,
                cache,
                base_environment=spec.base_environment,
                case_name=case_name,
                reference_path=reference_path,
            )
            result = score_target_from_cache(
                cache, target, device=device, batch_size=batch_size
            )
            reference_row_count = len(reference)
            reference_dimensions = reference.shape[1]
            used_cache = False
        if not reference_row_count or not len(target):
            raise ValueError(f"{environment}:{case_name} has no finite rows")
        if reference_dimensions != target.shape[1]:
            raise ValueError(f"{environment}:{case_name} dimension mismatch")
        id_fraction = float(result["target_fraction"])
        rows.append(
            {
                "environment": environment,
                "base_environment": spec.base_environment,
                "distribution": case_name,
                "dimensions": target.shape[1],
                "reference_rows": reference_row_count,
                "target_rows": len(target),
                "id_fraction": f"{id_fraction:.12g}",
                "id_percent": f"{100.0 * id_fraction:.10g}",
                "kde_mode": "full-dimensional Gaussian KDE",
                "bins_per_axis": support.BINS,
                "reference_density_quantile": support.KDE_REFERENCE_QUANTILE,
                "bandwidth_rule": "Scott",
                "scott_factor": f'{result["scott_factor"]:.12g}',
                "log_density_threshold": f'{result["log_density_threshold"]:.12g}',
                "covariance_regularizer": f'{result["covariance_regularizer"]:.12g}',
                "base_cache": str(cache_path.resolve()) if used_cache else "",
            }
        )
        print(
            f"  {case_name} ({target.shape[1]}D): "
            f"ID {100.0 * id_fraction:.4f}%",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{environment}_id_percentages.csv"
    write_csv(output_path, rows)
    print(f"  saved {output_path}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.kde_batch_size < 1:
        raise ValueError("--kde-batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    custom_values = (args.reference, args.target, args.schema)
    if any(value is not None for value in custom_values):
        if not all(value is not None for value in custom_values):
            raise ValueError("--reference, --target, and --schema must be used together")
        environments = {
            args.target_label: EnvironmentSpec(
                args.schema, str(args.reference), str(args.target), args.schema
            )
        }
    else:
        environments = (
            ENVIRONMENTS
            if args.environment == "all"
            else {args.environment: ENVIRONMENTS[args.environment]}
        )
    all_rows = []
    for environment, spec in environments.items():
        all_rows.extend(
            analyze_environment(
                environment,
                spec,
                args.output_dir,
                args.base_cache_dir,
                device,
                args.kde_batch_size,
            )
        )
    if args.environment == "all":
        write_csv(args.output_dir / "all_environments_id_percentages.csv", all_rows)


if __name__ == "__main__":
    main()
