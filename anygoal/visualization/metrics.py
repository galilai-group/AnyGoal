"""Visualization for common JSON result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix or "value": float(value)}
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_leaves(item, child))
        return result
    return {}


def plot_metrics(
    metrics: str | Path,
    output: str | Path,
    *,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Plot numeric leaves from prediction, planning, or analysis metrics JSON."""

    source = Path(metrics).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    values = _numeric_leaves(json.loads(source.read_text(encoding="utf-8")))
    if not values:
        raise ValueError(f"No numeric metrics found in {source}.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("metrics-plot requires `pip install anygoal[analysis]`.") from error

    height = max(3.0, 0.3 * len(values))
    figure, axis = plt.subplots(figsize=(9, height))
    labels = list(values)
    axis.barh(labels, [values[label] for label in labels])
    axis.set_title(title or source.stem)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
