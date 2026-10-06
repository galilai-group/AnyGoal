"""Run distribution_support metrics for several targets with one reference load."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from distribution_support import detect_schema, evaluate_feature, load_features, write_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("targets", nargs="+", type=Path)
    args = parser.parse_args()

    schema = detect_schema(args.reference)
    reference_features = load_features(args.reference, schema)
    pairs = []
    for target_path in args.targets:
        if detect_schema(target_path) != schema:
            raise ValueError(f"Schema mismatch: {target_path}")
        target_features = load_features(target_path, schema)
        results = {}
        for reference, target in zip(reference_features, target_features):
            result = evaluate_feature(reference, target)
            results[reference.label] = result
            print(
                f"{target_path.stem} | {reference.label} | "
                f"mahalanobis={result['mahalanobis']['target_fraction']:.6f} "
                f"kde={result['kde']['target_fraction']:.6f} "
                f"verdict={result['final_verdict']}",
                flush=True,
            )
        pairs.append(
            {
                "reference_label": args.reference.stem,
                "target_label": target_path.stem,
                "reference_path": str(args.reference.resolve()),
                "target_path": str(target_path.resolve()),
                "schema": schema,
                "features": results,
            }
        )
    write_outputs(pairs, args.output_dir)


if __name__ == "__main__":
    main()
