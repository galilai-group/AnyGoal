"""Replace the after-grip Insert Peg bias axis without rerunning inference."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from anygoal.paths import data_root
from scripts.analysis.counterfactual_metrics import (
    counterfactual_sensitivity_uniformity,
)
from scripts.analysis.run_counterfactual_evaluation import (
    INSERT_PEG_GRIP_DISTANCE,
    INSERT_PEG_GRIP_FRAMES,
    INSERT_PEG_SYNC_TOLERANCE,
    MODEL_NAMES,
    REPO_ROOT,
    _bin_edges,
    _insert_peg_after_grip_mask,
)


BIAS_AXIS = "peg_target_distance_after_grip"


def collect_selected_bias(dataset_path: Path, frame_skip: int = 5) -> np.ndarray:
    """Collect peg-target distances in the runner's deterministic window order."""
    selected = []
    span = 4 * frame_skip
    current_offset = 2 * frame_skip
    with h5py.File(dataset_path, "r") as dataset:
        observations = dataset["observation"]
        for offset, length in zip(dataset["ep_offset"][:], dataset["ep_len"][:]):
            episode = observations[offset : offset + length]
            mask = _insert_peg_after_grip_mask(episode)
            starts = np.arange(max(int(length) - span + 1, 0))
            current = starts + current_offset
            current = current[mask[current]]
            peg = episode[current, 33:35]
            target = episode[current, 40:42]
            selected.append(np.linalg.norm(peg - target, axis=1))
    return np.concatenate(selected).astype(np.float32)


def update_result(result_directory: Path, bias: np.ndarray) -> None:
    cache_path = result_directory / "counterfactual_cache.npz"
    with np.load(cache_path) as source:
        cache = {
            key: source[key]
            for key in source.files
            if not key.startswith("bias__")
        }
    sensitivity_count = len(cache["sensitivity__default__final"])
    if sensitivity_count != len(bias):
        raise ValueError(
            f"Bias has {len(bias)} rows but {cache_path} has {sensitivity_count}"
        )
    cache[f"bias__{BIAS_AXIS}"] = bias
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=result_directory, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temporary_path, **cache)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    metrics_path = result_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["frame_selection"] = {
        "name": "after_grip_close_synchronized",
        "definition": (
            "current frame ends 10 consecutive raw frames with hand-peg "
            "distance <= 0.08 m and each intervening "
            "||(delta hand) - (delta peg)|| <= 0.01 m"
        ),
        "consecutive_frames": INSERT_PEG_GRIP_FRAMES,
        "hand_peg_distance_max_m": INSERT_PEG_GRIP_DISTANCE,
        "displacement_mismatch_max_m_per_frame": INSERT_PEG_SYNC_TOLERANCE,
        "bias_axis": BIAS_AXIS,
    }
    representations = metrics["protocols"]["default"]["representations"]
    edges = _bin_edges(bias, metrics["num_bins"])
    for representation, representation_metrics in representations.items():
        sensitivity = cache[f"sensitivity__default__{representation}"]
        score, medians, populated = counterfactual_sensitivity_uniformity(
            sensitivity, bias, edges
        )
        representation_metrics["csu"] = {
            BIAS_AXIS: {
                "value": score,
                "bin_edges": edges.tolist(),
                "populated_bin_indices": populated.tolist(),
                "bin_median_sensitivities": medians.tolist(),
            }
        }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=data_root() / "manipulator_insert_peg_weak_expert_train.h5",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/counterfactual_metrics_insert_peg_after_grip_20260918",
    )
    args = parser.parse_args()
    bias = collect_selected_bias(args.dataset)
    for model in MODEL_NAMES:
        result_directory = args.results_root / model / "insert_peg"
        update_result(result_directory, bias)
        print(f"updated {result_directory}")


if __name__ == "__main__":
    main()
