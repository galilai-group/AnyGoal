"""Summarize 15D Cube-everything planning in CubeHop using Cube data."""

from pathlib import Path

import summarize_cube_toy_predictor_planning as summary
from anygoal.paths import checkpoint_root


ROOT = Path(__file__).resolve().parents[1]
summary.RESULTS_ROOT = (
    checkpoint_root() / "toy_planning_20260827_cube_everything_from_cube_fast_toy_predictors"
)
summary.RAW_OUTPUT = ROOT / "outputs/cube_everything_from_cube_fast_planning_raw.csv"
summary.AVERAGE_OUTPUT = (
    ROOT / "outputs/cube_everything_from_cube_fast_planning_seed_average.csv"
)
summary.SCENARIOS = {
    "cube_dataset_cube_hop_env": ("anygoal/CubeHop-v0", "cube_single_expert"),
}
summary.REPRESENTATIONS = {"cube_everything": 15}


if __name__ == "__main__":
    summary.main()
