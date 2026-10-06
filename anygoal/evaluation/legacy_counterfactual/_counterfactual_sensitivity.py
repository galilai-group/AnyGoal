"""Shared readers for saved counterfactual-sensitivity CSV files."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def find_latest_metadata() -> Path:
    candidates = list(
        (RESULTS_DIR / "counterfactual_sensitivity").glob(
            "**/*.metadata.json"
        )
    )
    candidates.extend(
        (RESULTS_DIR / "counterfactual_action_effect").glob(
            "**/*.metadata.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No counterfactual metadata found under paper/results. Run "
            "save_counterfactual_sensitivity.py first or pass --metadata."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def metadata_path_or_latest(path: Path | None) -> Path:
    result = find_latest_metadata() if path is None else path.expanduser()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result.resolve()


def load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    required = {
        "dataset_path",
        "output_prefix",
        "selected_layers",
        "atomic_files",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Missing metadata fields: {sorted(missing)}")
    return metadata


def resolve_atomic_path(metadata_path: Path, saved_path: str) -> Path:
    path = Path(saved_path).expanduser()
    if path.is_file():
        return path.resolve()
    sibling = metadata_path.parent / path.name
    if sibling.is_file():
        return sibling.resolve()
    raise FileNotFoundError(path)


def layer_token(layer: int | str) -> str:
    return str(layer)


def safe_layer_token(layer: str) -> str:
    return "layeraverage" if layer == "average" else f"layer{layer}"


def aggregate_sensitivities(
    metadata_path: Path,
) -> tuple[dict, dict[str, np.ndarray], OrderedDict[str, np.ndarray]]:
    """Sum interventions per layer and average the resulting layer arrays."""
    metadata = load_metadata(metadata_path)
    selected_layers = [layer_token(layer) for layer in metadata["selected_layers"]]
    files_by_layer: dict[str, list[Path]] = {
        layer: [] for layer in selected_layers
    }
    for item in metadata["atomic_files"]:
        layer = layer_token(item["layer"])
        if layer in files_by_layer:
            files_by_layer[layer].append(
                resolve_atomic_path(metadata_path, item["path"])
            )
    missing = [layer for layer, paths in files_by_layer.items() if not paths]
    if missing:
        raise ValueError(f"Missing atomic CSV files for layers: {missing}")

    first_path = files_by_layer[selected_layers[0]][0]
    available_columns = set(pd.read_csv(first_path, nrows=0).columns)
    common_columns = [
        "dataset_index",
        "episode",
        "frame_t",
        "global_t_index",
        "distance",
    ]
    common_columns.extend(
        column for column in ("ball_x", "ball_y") if column in available_columns
    )
    common_columns.extend(
        column
        for column in (
            "hand_peg_distance",
            "target_peg_distance",
            "hand_target_distance",
            "min_hand_distance",
        )
        if column in available_columns
    )
    common_frame = pd.read_csv(
        first_path,
        usecols=common_columns,
        dtype={
            "dataset_index": "int64",
            "episode": "int64",
            "frame_t": "int64",
            "global_t_index": "int64",
            "distance": "float32",
            "ball_x": "float32",
            "ball_y": "float32",
            "hand_peg_distance": "float32",
            "target_peg_distance": "float32",
            "hand_target_distance": "float32",
            "min_hand_distance": "float32",
        },
    )
    common = {
        column: common_frame[column].to_numpy()
        for column in common_frame.columns
    }
    row_count = len(common_frame)
    del common_frame

    layer_sensitivities: OrderedDict[str, np.ndarray] = OrderedDict()
    for layer in selected_layers:
        layer_sum = np.zeros(row_count, dtype=np.float32)
        for csv_path in files_by_layer[layer]:
            effect = pd.read_csv(
                csv_path,
                usecols=["counterfactual_effect"],
                dtype={"counterfactual_effect": "float32"},
            )["counterfactual_effect"].to_numpy(dtype=np.float32)
            if len(effect) != row_count:
                raise ValueError(
                    f"Row count mismatch in {csv_path}: "
                    f"{len(effect)} != {row_count}"
                )
            layer_sum += effect
        layer_sensitivities[layer] = layer_sum

    layer_sensitivities["average"] = np.mean(
        np.stack(tuple(layer_sensitivities.values()), axis=0), axis=0
    ).astype(np.float32)
    return metadata, common, layer_sensitivities
