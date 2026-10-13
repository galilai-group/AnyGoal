#!/bin/bash
set -euo pipefail

mapfile -t cases < <(pytest --collect-only -q -m matrix_e2e | rg '^tests/.+::')
selected=()
shard=${ANYGOAL_MATRIX_SHARD:-${SLURM_ARRAY_TASK_ID:?}}
for index in "${!cases[@]}"; do
    if ((index % ANYGOAL_MATRIX_SHARDS == shard)); then
        selected+=("${cases[$index]}")
    fi
done
pytest -q -m matrix_e2e "${selected[@]}"
