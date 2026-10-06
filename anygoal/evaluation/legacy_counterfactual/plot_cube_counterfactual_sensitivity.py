#!/usr/bin/env python
"""Plot the two OGBench Cube counterfactual-sensitivity biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "cube_counterfactual_sensitivity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-effector-z", type=float, default=0.04)
    parser.add_argument("--maximum-effector-z", type=float, default=0.35)
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def latest_metadata() -> Path:
    candidates = list(RESULTS_DIR.glob("**/*.metadata.json"))
    complete = []
    for path in candidates:
        try:
            if json.loads(path.read_text()).get("status") == "complete":
                complete.append(path)
        except (json.JSONDecodeError, OSError):
            continue
    if not complete:
        raise FileNotFoundError(
            "No complete Cube sensitivity metadata found; run "
            "save_cube_counterfactual_sensitivity.py first or pass --metadata."
        )
    return max(complete, key=lambda path: path.stat().st_mtime)


def safe_layer_token(layer: int | str) -> str:
    return "layeraverage" if str(layer) == "average" else f"layer{layer}"


def save_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    *,
    output_path: Path,
    xlabel: str,
    title: str,
    gridsize: int,
    dpi: int,
    xlim: tuple[float, float] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    density = axis.hexbin(x, y, gridsize=gridsize, bins="log", cmap="managua")
    figure.colorbar(density, ax=axis, label="log10(count)")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Sum counterfactual action sensitivity")
    axis.set_title(title)
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.grid(True, alpha=0.25)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    if args.gridsize < 2 or args.dpi < 1:
        raise ValueError("gridsize must be at least 2 and dpi must be positive")
    if args.minimum_effector_z > args.maximum_effector_z:
        raise ValueError("minimum-effector-z cannot exceed maximum-effector-z")
    metadata_path = (
        latest_metadata()
        if args.metadata is None
        else args.metadata.expanduser().resolve()
    )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise ValueError(f"Sensitivity run is not complete: {metadata_path}")
    results_h5 = Path(metadata["results_h5"]).expanduser().resolve()
    output_dir = (
        metadata_path.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(results_h5, "r") as results:
        action_z = results["common/action_z"][:]
        effector_z = results["common/effector_z"][:]
        selected_layers = [str(layer) for layer in metadata["selected_layers"]]
        sensitivity = {
            grid_name: {
                layer: results[f"sensitivity/{grid_name}/layer_{layer}"][:]
                for layer in selected_layers
            }
            for grid_name in ("grid_3d_27", "grid_xy_9")
        }

    for grid_sensitivity in sensitivity.values():
        grid_sensitivity["average"] = np.mean(
            np.stack([grid_sensitivity[layer] for layer in selected_layers]),
            axis=0,
        ).astype(np.float32)

    first_mask = (
        np.isfinite(action_z)
        & np.isfinite(effector_z)
        & (effector_z >= args.minimum_effector_z)
        & (effector_z <= args.maximum_effector_z)
    )
    second_mask = np.isfinite(effector_z)
    layers_to_plot = [*selected_layers, "average"]
    for layer in layers_to_plot:
        token = safe_layer_token(layer)
        save_hexbin(
            action_z[first_mask],
            sensitivity["grid_3d_27"][layer][first_mask],
            output_path=output_dir
            / f"{metadata['output_prefix']}_{token}_delta_z_hexbin.jpg",
            xlabel="Expert action delta z (normalized)",
            title=f"Cube counterfactual sensitivity by vertical action: {token}",
            gridsize=args.gridsize,
            dpi=args.dpi,
            xlim=(-1.0, 1.0),
        )
        save_hexbin(
            effector_z[second_mask],
            sensitivity["grid_xy_9"][layer][second_mask],
            output_path=output_dir
            / f"{metadata['output_prefix']}_{token}_effector_z_hexbin.jpg",
            xlabel="End-effector z position (m)",
            title=f"Cube horizontal-action sensitivity by height: {token}",
            gridsize=args.gridsize,
            dpi=args.dpi,
        )

    plot_summary = {
        "metadata_path": str(metadata_path),
        "results_h5": str(results_h5),
        "first_plot_rows": int(first_mask.sum()),
        "first_plot_effector_z_range_m": [
            args.minimum_effector_z,
            args.maximum_effector_z,
        ],
        "second_plot_rows": int(second_mask.sum()),
        "layers": layers_to_plot,
        "gridsize": args.gridsize,
        "dpi": args.dpi,
        "colormap": "managua",
    }
    summary_path = output_dir / f"{metadata['output_prefix']}.plot_summary.json"
    summary_path.write_text(json.dumps(plot_summary, indent=2) + "\n")
    print(f"Saved {summary_path}")
    print(
        f"Rows plotted: first={first_mask.sum():,}, second={second_mask.sum():,}"
    )


if __name__ == "__main__":
    main()
