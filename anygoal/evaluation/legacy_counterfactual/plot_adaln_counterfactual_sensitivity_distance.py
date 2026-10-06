#!/usr/bin/env python
"""Plot PushT oracle-state AdaLN sensitivity against agent-to-T distance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _counterfactual_sensitivity import (
    aggregate_sensitivities,
    safe_layer_token,
)


RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "counterfactual_sensitivity"
    / "adaln"
)


def latest_metadata() -> Path:
    candidates = list(RESULTS_DIR.glob("**/*.metadata.json"))
    if not candidates:
        raise FileNotFoundError(
            "No AdaLN metadata found; run save_adaln_counterfactual_sensitivity.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    metadata_path = (
        latest_metadata() if args.metadata is None else args.metadata.expanduser().resolve()
    )
    metadata, common, sensitivities = aggregate_sensitivities(metadata_path)
    if metadata.get("model_family") != "adaln_true_state":
        raise ValueError(f"Metadata is not from the oracle-state AdaLN model: {metadata_path}")
    output_dir = (
        metadata_path.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer, sensitivity in sensitivities.items():
        figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        density = axis.hexbin(
            common["distance"],
            sensitivity,
            gridsize=args.gridsize,
            bins="log",
            cmap="managua",
        )
        figure.colorbar(density, ax=axis, label="log10(count)")
        axis.set_xlabel("L2 distance: agent position to physical T position at time t")
        axis.set_ylabel("Sum counterfactual action sensitivity")
        axis.set_title(
            "Oracle-state AdaLN counterfactual sensitivity by distance: "
            f"{safe_layer_token(layer)}"
        )
        axis.grid(True, alpha=0.25)
        output_path = output_dir / (
            f"{metadata['output_prefix']}_{safe_layer_token(layer)}_distance_hexbin.jpg"
        )
        figure.savefig(output_path, dpi=args.dpi)
        plt.close(figure)
        print(f"Saved {output_path}")
    print(f"Rows plotted per figure: {len(common['distance'])}")


if __name__ == "__main__":
    main()
