"""Summarize Cube/CubeHop planning for the everything predictors."""

from pathlib import Path

import summarize_cube_toy_predictor_planning as summary
from anygoal.paths import checkpoint_root


ROOT = Path(__file__).resolve().parents[1]
summary.RESULTS_ROOT = checkpoint_root() / "toy_planning_20260827_cube_everything_toy_predictors"
summary.RAW_OUTPUT = ROOT / "outputs/cube_everything_planning_raw.csv"
summary.AVERAGE_OUTPUT = ROOT / "outputs/cube_everything_planning_seed_average.csv"
summary.SCENARIOS = {
    "cube_dataset_cube_env": ("swm/OGBCube-v0", "cube_single_expert"),
    "cube_hop_dataset_cube_hop_env": (
        "anygoal/CubeHop-v0",
        "cube_hop_test",
    ),
}
summary.REPRESENTATIONS = {
    "cube_everything": 15,
    "cube_everything_extreme": 22,
}


if __name__ == "__main__":
    summary.main()
