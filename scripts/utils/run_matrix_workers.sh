#!/bin/bash
set -euo pipefail

workers=$1
status=0
pids=()
for ((worker = 0; worker < workers; worker++)); do
    shard=$((ANYGOAL_SHARD_OFFSET + SLURM_ARRAY_TASK_ID * workers + worker))
    CUDA_VISIBLE_DEVICES="$worker" MASTER_PORT="$((29500 + shard))" \
        ANYGOAL_MATRIX_SHARD="$shard" \
        scripts/utils/run_matrix_shard.sh \
        >"log_slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_shard${shard}.out" \
        2>"log_slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_shard${shard}.err" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
