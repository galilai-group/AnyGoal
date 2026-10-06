"""Reusable world-model evaluation utilities."""

from .counterfactual import (
    counterfactual_sensitivity,
    counterfactual_sensitivity_uniformity,
    evaluate_counterfactual_metrics,
    factual_output_variation,
    relative_action_sensitivity,
)
from .prediction import PredictionProtocol, evaluate_prediction
from .counterfactual_plotting import (
    plot_counterfactual_bias_grid,
    plot_counterfactual_biases,
)

__all__ = [
    "PredictionProtocol",
    "counterfactual_sensitivity",
    "counterfactual_sensitivity_uniformity",
    "evaluate_counterfactual_metrics",
    "evaluate_prediction",
    "factual_output_variation",
    "plot_counterfactual_bias_grid",
    "plot_counterfactual_biases",
    "relative_action_sensitivity",
]
