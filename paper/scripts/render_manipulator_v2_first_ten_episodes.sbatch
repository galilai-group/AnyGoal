#!/bin/bash
#SBATCH --job-name=manip_v2_first10_gifs
#SBATCH --partition=cpuq
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=log_slurm/%A_%a.out
#SBATCH --error=log_slurm/%A_%a.err
set -euo pipefail
source /grid/klindt/home/nam/le-wm/.venv/bin/activate
datasets=(hug_box extractpeg)
python paper/scripts/render_first_ten_dataset_episodes.py \
  --dataset-id "${datasets[$SLURM_ARRAY_TASK_ID]}" \
  --output-root paper/results/manipulator_v2_first_10_episodes
