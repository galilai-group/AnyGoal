#!/bin/bash
#SBATCH --job-name=dataset_first10_gifs
#SBATCH --partition=cpuq
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=log_slurm/%j.out
#SBATCH --error=log_slurm/%j.err
set -euo pipefail
source /grid/klindt/home/nam/le-wm/.venv/bin/activate
datasets=(pusht avoidt stackt pusht-antigoal cube cubehop cubediagonal insertpeg hug_box extractpeg tworoom tworoom_route)
pids=()
for dataset in "${datasets[@]}"; do
  python paper/scripts/render_first_ten_dataset_episodes.py --dataset-id "$dataset" >"log_slurm/${SLURM_JOB_ID}_${dataset}.out" 2>"log_slurm/${SLURM_JOB_ID}_${dataset}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
