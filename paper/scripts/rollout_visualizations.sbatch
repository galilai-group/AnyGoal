#!/bin/bash
#SBATCH --job-name=rollout_gifs
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
datasets=(pusht avoidt stackt pusht_antigoal cube cube_hop cube_diagonal manipulator_insert_peg manipulator_hug_box manipulator_extract_peg tworoom tworoom_route)
models=(lewm sensorimotor prejepa driftworld irasim toy-adaln toy-action-duplicate toy-driftworld)
pids=()
for model_index in $(seq 0 7); do
  (
    export CUDA_VISIBLE_DEVICES="$model_index"
    for dataset in "${datasets[@]}"; do
      python paper/scripts/visualize_model_rollouts.py --environment "$dataset" --model "${models[$model_index]}"
    done
  ) >"log_slurm/${SLURM_JOB_ID}_model_${model_index}.out" 2>"log_slurm/${SLURM_JOB_ID}_model_${model_index}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
