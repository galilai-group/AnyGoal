"""Precompute reusable full-dimensional KDE statistics for base datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import distribution_support as support
import joint_distribution_support as joint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", choices=("all", *joint.BASE_ENVIRONMENTS), default="all"
    )
    parser.add_argument("--cache-dir", type=Path, default=joint.BASE_CACHE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kde-batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kde_batch_size < 1:
        raise ValueError("--kde-batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision("highest")
    bases = (
        joint.BASE_ENVIRONMENTS
        if args.base == "all"
        else {args.base: joint.BASE_ENVIRONMENTS[args.base]}
    )
    for base_environment, (filename, schema) in bases.items():
        reference_path = support.find_dataset(filename)
        components = joint.load_components(reference_path, schema)
        for case_name in joint.CASE_ORDER:
            output_path = joint.base_cache_path(
                args.cache_dir, base_environment, case_name
            )
            print(
                f"[{base_environment}] building {case_name}: {output_path}",
                flush=True,
            )
            reference = joint.make_case_vectors(components, case_name)
            cache = joint.prepare_reference_cache(
                reference, device=device, batch_size=args.kde_batch_size
            )
            joint.save_reference_cache(
                output_path,
                cache,
                base_environment=base_environment,
                case_name=case_name,
                reference_path=reference_path,
            )
            print(
                f"  rows={len(reference):,}, dimensions={reference.shape[1]}, "
                f"threshold={cache['threshold']:.8g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
