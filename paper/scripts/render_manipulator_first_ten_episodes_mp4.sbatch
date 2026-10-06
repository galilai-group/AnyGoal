#!/bin/bash
#SBATCH --job-name=manip_first10_mp4
#SBATCH --partition=cpuq
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=log_slurm/%A_%a.out
#SBATCH --error=log_slurm/%A_%a.err
set -euo pipefail
source /grid/klindt/home/nam/le-wm/.venv/bin/activate
datasets=(hug_box extractpeg insertpeg)
python paper/scripts/render_first_ten_dataset_episodes.py \
  --dataset-id "${datasets[$SLURM_ARRAY_TASK_ID]}" \
  --format mp4 \
  --output-root paper/results/manipulator_first_10_episodes_mp4
