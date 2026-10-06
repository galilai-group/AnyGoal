"""Evaluate Cube block quaternions with a full four-dimensional Gaussian KDE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import distribution_support as support
import joint_distribution_support as joint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--target-label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if len(args.target) != len(args.target_label):
        parser.error("use one --target-label for every --target")
    return args


def read_quaternion(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as dataset:
        values = np.asarray(dataset["privileged_block_0_quat"][:], dtype=np.float64)
    return values[np.isfinite(values).all(axis=1)]


def weighted_log_kde(
    reference: np.ndarray,
    log_weights: np.ndarray,
    query: np.ndarray,
    *,
    total_weight: int,
    bandwidth: float,
    log_determinant: float,
    device: torch.device,
    batch_size: int,
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision("highest")

    reference_raw = read_quaternion(args.reference)
    limits = joint.reference_bin_limits(reference_raw)
    reference_quantized, inside = joint.quantize_to_bin_centers(reference_raw, limits)
    if not inside.all():
        raise RuntimeError("reference limits do not contain every reference row")
    unique_reference, inverse, counts = np.unique(
        reference_quantized, axis=0, return_inverse=True, return_counts=True
    )
    mean, cholesky, regularizer, log_determinant = joint.fit_reference_whitening(
        reference_quantized
    )
    unique_reference_white = joint.apply_whitening(
        unique_reference, mean, cholesky
    ).astype(np.float32)
    bandwidth = len(reference_quantized) ** (-1.0 / 8.0)
    unique_reference_density = weighted_log_kde(
        unique_reference_white,
        np.log(counts),
        unique_reference_white,
        total_weight=len(reference_quantized),
        bandwidth=bandwidth,
        log_determinant=log_determinant,
        device=device,
        batch_size=args.batch_size,
    )
    reference_density = unique_reference_density[inverse]
    threshold = float(np.quantile(reference_density, support.KDE_REFERENCE_QUANTILE))
    print(
        f"reference rows={len(reference_quantized)} unique_centers={len(unique_reference)} "
        f"threshold={threshold:.9g}",
        flush=True,
    )

    results = {}
    for label, path in zip(args.target_label, args.target):
        target_raw = read_quaternion(path)
        target_quantized, target_inside = joint.quantize_to_bin_centers(target_raw, limits)
        target_density = np.full(len(target_raw), -np.inf, dtype=np.float64)
        inside_indices = np.flatnonzero(target_inside)
        unique_target, target_inverse = np.unique(
            target_quantized[target_inside], axis=0, return_inverse=True
        )
        unique_target_white = joint.apply_whitening(
            unique_target, mean, cholesky
        ).astype(np.float32)
        unique_target_density = weighted_log_kde(
            unique_reference_white,
            np.log(counts),
            unique_target_white,
            total_weight=len(reference_quantized),
            bandwidth=bandwidth,
            log_determinant=log_determinant,
            device=device,
            batch_size=args.batch_size,
        )
        target_density[inside_indices] = unique_target_density[target_inverse]
        fraction = float(np.mean(target_density >= threshold))
        results[label] = {
            "path": str(path.resolve()),
            "rows": len(target_raw),
            "unique_quantized_centers": len(unique_target),
            "inside_reference_limits_fraction": float(np.mean(target_inside)),
            "full_4d_kde_support_fraction": fraction,
            "pass_at_0.95": fraction >= support.KDE_TARGET_FRACTION,
        }
        print(f"{label}: full-4D KDE support={fraction:.9f}", flush=True)

    payload = {
        "method": "weighted full-covariance 4D Gaussian KDE after 256-bin quantization",
        "dimensions": 4,
        "reference_path": str(args.reference.resolve()),
        "reference_rows": len(reference_quantized),
        "unique_reference_centers": len(unique_reference),
        "scott_factor": bandwidth,
        "covariance_regularizer": regularizer,
        "reference_log_density_quantile": support.KDE_REFERENCE_QUANTILE,
        "log_density_threshold": threshold,
        "required_target_fraction": support.KDE_TARGET_FRACTION,
        "targets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
