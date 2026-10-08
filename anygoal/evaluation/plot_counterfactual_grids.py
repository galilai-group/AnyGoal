"""Render model-wise counterfactual-bias grids from saved evaluation caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .counterfactual_plotting import (
    GRID_ENVIRONMENTS,
    plot_counterfactual_bias_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("lewm", "sensorimotor", "adaln_predictor")
REPRESENTATIONS = ("average", "final")
LAYER_NAMES = tuple(str(index) for index in range(6)) + ("final",)


def _sensitivity(cache: np.lib.npyio.NpzFile, protocol: str, representation: str):
    if representation == "final":
        return cache[f"sensitivity__{protocol}__final"]
    result = np.zeros_like(cache[f"sensitivity__{protocol}__0"])
    for layer in LAYER_NAMES:
        result += cache[f"sensitivity__{protocol}__{layer}"]
    result /= len(LAYER_NAMES)
    return result


def render_counterfactual_grids(
    input_root: str | Path,
    output_directory: str | Path,
    *,
    insert_peg_root: str | Path | None = None,
    gridsize: int = 90,
    dpi: int = 220,
) -> list[Path]:
    """Create the six model-by-representation 1x4 comparison figures."""
    input_root = Path(input_root)
    insert_peg_root = input_root if insert_peg_root is None else Path(insert_peg_root)
    output_directory = Path(output_directory)
    paths = []
    for model in MODELS:
        for representation in REPRESENTATIONS:
            panels = []
            for environment, protocol, bias_axis, title in GRID_ENVIRONMENTS:
                root = (
                    insert_peg_root
                    if environment == "insert_peg"
                    else input_root
                )
                environment_directory = root / model / environment
                metrics = json.loads(
                    (environment_directory / "metrics.json").read_text()
                )
                representation_metrics = metrics["protocols"][protocol][
                    "representations"
                ][representation]
                with np.load(
                    environment_directory / "counterfactual_cache.npz"
                ) as cache:
                    sensitivity = _sensitivity(
                        cache, protocol, representation
                    ).copy()
                    bias = cache[f"bias__{bias_axis}"].copy()
                panels.append(
                    {
                        "title": title,
                        "axis_label": bias_axis.replace("_", " "),
                        "bias": bias,
                        "sensitivity": sensitivity,
                        "ras": representation_metrics["ras"],
                        "csu": representation_metrics["csu"][bias_axis]["value"],
                    }
                )
            path = output_directory / f"{model}_{representation}.png"
            plot_counterfactual_bias_grid(
                panels,
                path,
                model_name=model,
                representation=representation,
                gridsize=gridsize,
                dpi=dpi,
            )
            paths.append(path)
            print(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "outputs/counterfactual_metrics_layerwise_20260918",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--insert-peg-root",
        type=Path,
        help="Optional results root supplying only the Insert Peg panels.",
    )
    parser.add_argument("--gridsize", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = args.output_directory or args.input_root / "comparison_figures"
    render_counterfactual_grids(
        args.input_root,
        output_directory,
        insert_peg_root=args.insert_peg_root,
        gridsize=args.gridsize,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
