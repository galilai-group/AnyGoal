#!/bin/bash
#SBATCH --job-name=planning_gifs
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:8
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --exclude=bamgpu02,bamgpu07,bamgpu17,bamgpu20,bamgpu25
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=log_slurm/%j.out
#SBATCH --error=log_slurm/%j.err
set -euo pipefail
source /grid/klindt/home/nam/le-wm/.venv/bin/activate
pids=()
for model_index in $(seq 0 7); do
  (
    export CUDA_VISIBLE_DEVICES="$model_index"
    for dataset_index in $(seq 0 11); do
      python paper/scripts/run_planning_visualization.py --task-id "$((dataset_index*8+model_index))"
    done
  ) >"log_slurm/${SLURM_JOB_ID}_model_${model_index}.out" 2>"log_slurm/${SLURM_JOB_ID}_model_${model_index}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
