#!/bin/bash
#SBATCH --job-name=extractpeg_v3_gifs
#SBATCH --partition=cpuq
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=log_slurm/%j.out
#SBATCH --error=log_slurm/%j.err
set -euo pipefail
source /grid/klindt/home/nam/le-wm/.venv/bin/activate
python paper/scripts/render_first_ten_dataset_episodes.py \
  --dataset-id extractpeg-v3 \
  --output-root paper/results/extractpeg_v3_first_10_episodes
