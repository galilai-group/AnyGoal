"""Scale-normalized counterfactual action-sensitivity analysis metrics.

The definitions in this module follow ``reports/counterfactual_analysis.tex``.
They operate on model outputs and are independent of any particular world
model, dataset, or caching format, but remain part of the research analysis.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial.distance import pdist


def counterfactual_sensitivity(
    factual_outputs: np.ndarray,
    counterfactual_outputs: np.ndarray,
) -> np.ndarray:
    """Return per-sample ``S_i``, averaging L2 effects over interventions.

    ``factual_outputs`` has shape ``(samples, ...)`` and
    ``counterfactual_outputs`` has shape ``(samples, actions, ...)``. All
    non-sample dimensions are treated as one representation vector.
    """
    factual = np.asarray(factual_outputs, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_outputs, dtype=np.float64)
    if counterfactual.ndim < 3 or factual.shape[0] != counterfactual.shape[0]:
        raise ValueError("Outputs must have matching sample axes and an action axis.")
    if factual.shape[1:] != counterfactual.shape[2:]:
        raise ValueError(
            f"Output shapes do not match: {factual.shape} and {counterfactual.shape}."
        )
    differences = counterfactual - factual[:, None]
    effects = np.linalg.norm(differences.reshape(*differences.shape[:2], -1), axis=-1)
    return effects.mean(axis=1)


def factual_output_variation(factual_outputs: np.ndarray) -> float:
    """Return the exact median pairwise factual-output distance ``D_f``."""
    outputs = np.asarray(factual_outputs, dtype=np.float64)
    if outputs.shape[0] < 2:
        raise ValueError("At least two factual outputs are required for D_f.")
    distances = pdist(outputs.reshape(outputs.shape[0], -1), metric="euclidean")
    return float(np.median(distances))


def relative_action_sensitivity(
    sensitivities: np.ndarray,
    factual_variation: float,
    *,
    epsilon: float = np.finfo(np.float64).eps,
) -> tuple[float, np.ndarray]:
    """Return RAS and the per-sample relative sensitivities ``R_i``."""
    values = np.asarray(sensitivities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Sensitivities must be a finite non-negative vector.")
    if not np.isfinite(factual_variation) or factual_variation < 0:
        raise ValueError("Factual variation must be finite and non-negative.")
    relative = values / (values + float(factual_variation) + float(epsilon))
    return float(np.median(relative)), relative


def counterfactual_sensitivity_uniformity(
    sensitivities: np.ndarray,
    bias_values: np.ndarray,
    bin_edges: np.ndarray,
    *,
    epsilon: float = np.finfo(np.float64).eps,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return CSU, populated-bin medians ``U_b``, and their bin indices.

    Empty bins are omitted, as the report defines equal weighting over
    populated bins. Values on the final edge are included in the final bin.
    """
    sensitivities = np.asarray(sensitivities, dtype=np.float64)
    bias_values = np.asarray(bias_values, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if sensitivities.ndim != 1 or bias_values.shape != sensitivities.shape:
        raise ValueError("Sensitivities and bias values must be matching vectors.")
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be a strictly increasing vector.")
    finite = np.isfinite(sensitivities) & np.isfinite(bias_values)
    bin_indices = np.searchsorted(edges, bias_values[finite], side="right") - 1
    bin_indices[bias_values[finite] == edges[-1]] = len(edges) - 2
    valid = (bin_indices >= 0) & (bin_indices < len(edges) - 1)
    finite_sensitivities = sensitivities[finite][valid]
    bin_indices = bin_indices[valid]
    populated = np.unique(bin_indices)
    if not len(populated):
        raise ValueError("No finite samples fall inside the supplied bins.")
    medians = np.asarray(
        [np.median(finite_sensitivities[bin_indices == index]) for index in populated]
    )
    mean_median = float(medians.mean())
    normalized_rms = np.sqrt(
        np.mean((medians / (mean_median + float(epsilon)) - 1.0) ** 2)
    )
    return float(1.0 / (1.0 + normalized_rms)), medians, populated


def evaluate_counterfactual_metrics(
    factual_outputs: np.ndarray,
    counterfactual_outputs: np.ndarray,
    bias_axes: Mapping[str, np.ndarray],
    bin_edges: Mapping[str, np.ndarray],
    *,
    epsilon: float = np.finfo(np.float64).eps,
) -> dict:
    """Evaluate RAS and CSU for one representation and several bias axes."""
    sensitivities = counterfactual_sensitivity(
        factual_outputs, counterfactual_outputs
    )
    factual_variation = factual_output_variation(factual_outputs)
    ras, relative = relative_action_sensitivity(
        sensitivities, factual_variation, epsilon=epsilon
    )
    uniformity = {}
    for name, values in bias_axes.items():
        score, medians, populated = counterfactual_sensitivity_uniformity(
            sensitivities, values, bin_edges[name], epsilon=epsilon
        )
        uniformity[name] = {
            "csu": score,
            "bin_edges": np.asarray(bin_edges[name]).tolist(),
            "populated_bin_indices": populated.tolist(),
            "bin_median_sensitivities": medians.tolist(),
        }
    return {
        "ras": ras,
        "factual_output_variation": factual_variation,
        "median_counterfactual_sensitivity": float(np.median(sensitivities)),
        "csu": uniformity,
        "sensitivities": sensitivities,
        "relative_sensitivities": relative,
    }


__all__ = [
    "counterfactual_sensitivity",
    "counterfactual_sensitivity_uniformity",
    "evaluate_counterfactual_metrics",
    "factual_output_variation",
    "relative_action_sensitivity",
]
