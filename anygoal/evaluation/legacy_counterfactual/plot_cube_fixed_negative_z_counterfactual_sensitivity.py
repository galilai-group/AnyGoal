#!/usr/bin/env python
"""Plot Cube delta-z-action sensitivity for three fixed-negative-z grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-effector-z", type=float, default=0.04)
    parser.add_argument("--maximum-effector-z", type=float, default=0.35)
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def layer_token(layer: str) -> str:
    return "layeraverage" if layer == "average" else f"layer{layer}"


def main() -> None:
    args = parse_args()
    if args.gridsize < 2 or args.dpi < 1:
        raise ValueError("gridsize must be at least 2 and dpi must be positive")
    metadata_path = args.metadata.expanduser().resolve()
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise ValueError(f"Sensitivity run is not complete: {metadata_path}")
    fixed_grids = metadata["action_grids"].get("fixed_negative_z_8point_grids")
    if fixed_grids is None or len(fixed_grids) != 3:
        raise ValueError("Metadata must contain exactly three fixed-negative-z grids")
    output_dir = metadata_path.parent if args.output_dir is None else args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(Path(metadata["results_h5"]), "r") as results:
        action_z = results["common/action_z"][:]
        effector_z = results["common/effector_z"][:]
        selected_layers = [str(layer) for layer in metadata["selected_layers"]]
        sensitivity = {
            f"fixed_negative_z_set{index}": {
                layer: results[
                    f"sensitivity/fixed_negative_z_set{index}/layer_{layer}"
                ][:]
                for layer in selected_layers
            }
            for index in range(1, 4)
        }

    first_mask = (
        np.isfinite(action_z)
        & np.isfinite(effector_z)
        & (effector_z >= args.minimum_effector_z)
        & (effector_z <= args.maximum_effector_z)
    )
    layers_to_plot = [*selected_layers, "average"]
    for index, grid in enumerate(fixed_grids, start=1):
        grid_name = f"fixed_negative_z_set{index}"
        sensitivity[grid_name]["average"] = np.mean(
            np.stack([sensitivity[grid_name][layer] for layer in selected_layers]),
            axis=0,
        ).astype(np.float32)
        fixed_z = grid["fixed_delta_z"]
        for layer in layers_to_plot:
            figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
            density = axis.hexbin(
                action_z[first_mask],
                sensitivity[grid_name][layer][first_mask],
                gridsize=args.gridsize,
                bins="log",
                cmap="managua",
            )
            figure.colorbar(density, ax=axis, label="log10(count)")
            axis.set_xlabel("Expert action delta z (normalized)")
            axis.set_ylabel("Sum counterfactual action sensitivity")
            axis.set_title(
                "Cube counterfactual sensitivity by vertical action: "
                f"set {index}, fixed counterfactual delta z={fixed_z:g}, "
                f"{layer_token(layer)}"
            )
            axis.set_xlim(-1.0, 1.0)
            axis.grid(True, alpha=0.25)
            output_path = output_dir / (
                f"{metadata['output_prefix']}_set{index}_{layer_token(layer)}_"
                "delta_z_hexbin.jpg"
            )
            figure.savefig(output_path, dpi=args.dpi)
            plt.close(figure)
            print(f"Saved {output_path}")

    summary_path = output_dir / f"{metadata['output_prefix']}.fixed_z_plot_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata_path": str(metadata_path),
                "first_plot_rows": int(first_mask.sum()),
                "fixed_delta_z_values": [grid["fixed_delta_z"] for grid in fixed_grids],
                "layers": layers_to_plot,
                "gridsize": args.gridsize,
                "dpi": args.dpi,
                "colormap": "managua",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
