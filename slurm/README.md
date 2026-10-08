# Slurm submission

`run.sh` is the only scheduler-independent worker. Cluster policy lives in an
optional site YAML file rather than in the worker. Generate an initial file
with `anygoal config show > ~/.config/anygoal/site.yaml`, then adapt its Slurm keys.

```bash
anygoal submit --site ~/.config/anygoal/site.yaml --preset gpu-train \
  --job-name driftworld-pusht --dry-run -- \
  anygoal train driftworld \
    --data ~/.stable_worldmodel/datasets/pusht_expert_train.h5 --task pusht
```

Remove `--dry-run` to submit. Experiment matrices should use `--array` and read
`SLURM_ARRAY_TASK_ID`; GPU counts and seed counts do not require another sbatch
file.
