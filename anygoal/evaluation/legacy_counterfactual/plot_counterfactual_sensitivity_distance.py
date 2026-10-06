"""Plot saved sensitivity against agent-to-physical-T distance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _counterfactual_sensitivity import (
    aggregate_sensitivities,
    metadata_path_or_latest,
    safe_layer_token,
)


DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"


def required_dataset_path() -> Path:
    dataset_path = DATASET_DIR / "pusht_expert_train.h5"
    legacy_path = DATASET_DIR.parent / "pusht_expert_train.h5"
    selected = (
        dataset_path
        if dataset_path.exists() or not legacy_path.exists()
        else legacy_path
    )
    return selected.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gridsize < 2 or args.dpi < 1:
        raise ValueError("gridsize must be at least 2 and dpi must be positive.")
    metadata_path = metadata_path_or_latest(args.metadata)
    metadata, common, sensitivities = aggregate_sensitivities(metadata_path)
    metadata_dataset_path = Path(metadata["dataset_path"]).expanduser().resolve()
    expected_dataset_path = required_dataset_path()
    if metadata_dataset_path != expected_dataset_path:
        raise ValueError(
            "Counterfactual sensitivity must come from "
            f"pusht_expert_train.h5 ({expected_dataset_path}); metadata uses "
            f"{metadata_dataset_path}. Re-run save_counterfactual_sensitivity.py."
        )
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
        axis.set_xlabel(
            "L2 distance: agent position to physical T position at time t"
        )
        axis.set_ylabel("Sum counterfactual action sensitivity")
        axis.set_title(
            f"Counterfactual sensitivity by distance: {safe_layer_token(layer)}"
        )
        axis.grid(True, alpha=0.25)
        output_path = output_dir / (
            f'{metadata["output_prefix"]}_{safe_layer_token(layer)}_'
            "distance_hexbin.jpg"
        )
        figure.savefig(output_path, dpi=args.dpi)
        plt.close(figure)
        print(f"Saved {output_path}")
    print(f"Rows plotted per figure: {len(common['distance'])}")


if __name__ == "__main__":
    main()
