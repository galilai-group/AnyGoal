#!/usr/bin/env python
"""Plot TwoRoom sensitivity against distance to the corridor center."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _counterfactual_sensitivity import aggregate_sensitivities, safe_layer_token


RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "tworoom_counterfactual_sensitivity"
)


def latest_metadata(model_family: str) -> Path:
    candidates = list((RESULTS_DIR / model_family).glob("**/*.metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No TwoRoom {model_family} sensitivity metadata found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=("lewm", "adaln"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--corridor-center-x", type=float, default=112.0)
    parser.add_argument("--corridor-center-y", type=float, default=52.5)
    args = parser.parse_args()
    metadata_path = (
        latest_metadata(args.model_family)
        if args.metadata is None
        else args.metadata.expanduser().resolve()
    )
    metadata, common, sensitivities = aggregate_sensitivities(metadata_path)
    if metadata.get("model_family") != args.model_family:
        raise ValueError("Requested model family does not match metadata")
    output_dir = (
        metadata_path.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    corridor_distance = np.hypot(
        common["ball_x"] - args.corridor_center_x,
        common["ball_y"] - args.corridor_center_y,
    )

    for layer, sensitivity in sensitivities.items():
        figure, axis = plt.subplots(figsize=(8, 6))
        figure.subplots_adjust(left=0.14, right=0.87, bottom=0.15, top=0.88)
        density = axis.hexbin(
            corridor_distance,
            sensitivity,
            gridsize=args.gridsize,
            bins="log",
            cmap="managua",
        )
        figure.colorbar(density, ax=axis, label="log10(count)")
        axis.set_xlabel(
            "L2 distance: agent position to corridor center (112, 52.5) at time t"
        )
        axis.set_ylabel("Sum counterfactual action sensitivity")
        token = safe_layer_token(layer)
        axis.set_title(
            f"TwoRoom {args.model_family.upper()} sensitivity by corridor distance: "
            f"{token}"
        )
        axis.grid(True, alpha=0.2)
        output_path = output_dir / (
            f"{metadata['output_prefix']}_{token}_corridor_distance_hexbin.jpg"
        )
        figure.savefig(output_path, dpi=args.dpi)
        plt.close(figure)
        print(f"Saved {output_path}")
    print(f"Rows plotted per figure: {len(common['ball_x'])}")


if __name__ == "__main__":
    main()
