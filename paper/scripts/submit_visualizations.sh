#!/bin/bash
set -euo pipefail
mkdir -p log_slurm
rollout_job=$(sbatch --parsable paper/scripts/rollout_visualizations.sbatch)
planning_job=$(sbatch --parsable --dependency="afterok:${rollout_job}" paper/scripts/planning_visualizations.sbatch)
echo "rollout_job=${rollout_job}"
echo "planning_job=${planning_job} dependency=afterok:${rollout_job}"
