"""Write the Cube report from the completed analysis summary."""
import json
from pathlib import Path

import h5py
import numpy as np
from anygoal.paths import data_root

DATASET = data_root() / "cube_single_expert.h5"
SUMMARY = Path("outputs/analysis/ogbench_cube_analysis/summary.json")
REPORT = Path("reports/ogbench_cube_dataset_analysis.md")


def fmt(values):
    if np.isscalar(values):
        return f"{values:.4g}"
    return ", ".join(f"{value:.4g}" for value in values)


def main():
    summary = json.loads(SUMMARY.read_text())
    with h5py.File(DATASET, "r") as dataset:
        fields = {key: tuple(dataset[key].shape[1:]) or "(scalar)" for key in dataset if key.startswith("proprio_") or key.startswith("privileged_")}
    initial_lines = "\n".join(
        f"- `{key}` — min/max: [{fmt(stats['min'])}] / [{fmt(stats['max'])}]; 1–99%: [{fmt(stats['q01'])}] / [{fmt(stats['q99'])}]; std: [{fmt(stats['std'])}]"
        for key, stats in summary["initial_summary"].items()
    )
    field_lines = "\n".join(f"- `{key}`: shape {shape}" for key, shape in fields.items())
    length = summary["length"]
    REPORT.write_text(f"""# OGBench Cube dataset analysis

Source: `{DATASET}`. Generated with `scripts/analysis/analyze_ogbench_cube.py` (seed {summary['random_seed']}).

## Dataset size

- Trajectories: **{summary['episodes']:,}**
- Frames: **{summary['frames']:,}**
- Trajectory length: mean **{length['mean']:.2f}**, min **{length['min']}**, max **{length['max']}** frames

## State and proprioception contents

There is no `/state` dataset; `/observation` is the 28-dimensional state/agent observation. Its exact-match named-column mapping is: `{', '.join(summary['observation_columns'])}`.

Named proprioceptive and privileged fields:

{field_lines}

## Visualizations

- [10 random trajectories, 2 × 5 GIF (short trajectories hold their final frame)](../outputs/analysis/ogbench_cube_analysis/random_10_trajectories_2x5.gif)
- [All-frame scalar KDEs](../outputs/analysis/ogbench_cube_analysis/all_state_proprio_scalar_kde.png)
- [All-frame 2D component-pair KDEs](../outputs/analysis/ogbench_cube_analysis/all_state_proprio_joint_kde.png)
- [Initial-frame scalar KDEs](../outputs/analysis/ogbench_cube_analysis/initial_state_proprio_scalar_kde.png)
- [Machine-readable statistics](../outputs/analysis/ogbench_cube_analysis/summary.json)

KDEs use all available frames (or all 10,000 initial frames) and Scott-rule Gaussian-smoothed histograms, matching the prior PushT all-points KDE approach.

## Initial-frame sampling ranges and confidence

Min/max is the observed support. The 1–99% interval is the robust routine sampling range. Near-zero standard deviation and a collapsed 1–99% interval indicate tightly constrained initial sampling; broad intervals indicate deliberate variation. With 10,000 episode starts, these empirical quantiles are well resolved, though they are descriptive—not guarantees beyond this dataset.

{initial_lines}

`privileged_target_task` is a non-numeric task label, and is intentionally excluded from numeric range statistics.
""")


if __name__ == "__main__":
    main()
