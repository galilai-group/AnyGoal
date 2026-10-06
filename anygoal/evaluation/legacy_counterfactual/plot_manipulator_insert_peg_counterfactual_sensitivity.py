#!/usr/bin/env python
"""Plot Insert Peg sensitivity against four task-space distances."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _counterfactual_sensitivity import aggregate_sensitivities, safe_layer_token


RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results/manipulator_insert_peg_counterfactual_sensitivity"
)
METRICS = {
    "hand_peg_distance": "L2 distance: hand position to peg at time t",
    "target_peg_distance": "L2 distance: target position to peg at time t",
    "hand_target_distance": "L2 distance: hand position to target at time t",
    "min_hand_distance": "min(hand-target distance, hand-peg distance) at time t",
}


def latest_metadata(model_family: str) -> Path:
    paths = list((RESULTS_DIR / model_family).glob("**/*.metadata.json"))
    if not paths:
        raise FileNotFoundError(f"No Insert Peg {model_family} metadata found")
    return max(paths, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=("lewm", "adaln"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    metadata_path = (
        latest_metadata(args.model_family)
        if args.metadata is None
        else args.metadata.expanduser().resolve()
    )
    metadata, common, sensitivities = aggregate_sensitivities(metadata_path)
    output_dir = (
        metadata_path.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, x_label in METRICS.items():
        if metric not in common:
            raise KeyError(f"Missing {metric!r} in sensitivity CSVs")
        for layer, sensitivity in sensitivities.items():
            figure, axis = plt.subplots(figsize=(8, 6))
            figure.subplots_adjust(left=0.14, right=0.87, bottom=0.15, top=0.88)
            density = axis.hexbin(
                common[metric],
                sensitivity,
                gridsize=args.gridsize,
                bins="log",
                cmap="managua",
            )
            figure.colorbar(density, ax=axis, label="log10(count)")
            axis.set_xlabel(x_label)
            axis.set_ylabel("Sum counterfactual action sensitivity")
            token = safe_layer_token(layer)
            axis.set_title(
                f"Insert Peg {args.model_family.upper()} sensitivity: "
                f"{metric}, {token}"
            )
            axis.grid(True, alpha=0.2)
            output_path = output_dir / (
                f"{metadata['output_prefix']}_{token}_{metric}_hexbin.jpg"
            )
            figure.savefig(output_path, dpi=args.dpi)
            plt.close(figure)
            print(f"Saved {output_path}")
    print(f"Rows plotted per figure: {len(common['distance'])}")


if __name__ == "__main__":
    main()
