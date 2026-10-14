# Analysis and visualization recipes

Run commands from the repository root. Datasets use `ANYGOAL_DATA_ROOT`, or
the directory resolved by Stable Worldmodel when it is unset. Generated
artifacts default to `outputs/analysis`.

## Distribution support

`distribution_support.py` computes marginal directional support metrics for a
reference/target HDF5 pair. `distribution_support_sweep.py` reuses one loaded
reference across several targets. `joint_distribution_support.py` evaluates
the joint state/action cases, and `build_joint_kde_base_cache.py` prepares the
reusable base KDE caches.

```bash
python scripts/analysis/distribution_support.py --help
python scripts/analysis/distribution_support_sweep.py --help
python scripts/analysis/joint_distribution_support.py --help
python scripts/analysis/build_joint_kde_base_cache.py --help
```

## Result tables and model visualizations

`build_stat_fullresult.py` evaluates the pairs represented in
`reports/stat_fullresult.tex` and fills that table. The two visualization
recipes generate planning and open-loop rollout artifacts for the supported
model/dataset matrix.

```bash
python scripts/analysis/build_stat_fullresult.py --help
python scripts/analysis/run_planning_visualization.py --help
python scripts/analysis/visualize_model_rollouts.py --help
```

The public rollout/planning evaluation matrix is retained in
[`manifest.yaml`](manifest.yaml) and can be dispatched with:

```bash
anygoal reproduce scripts/analysis/manifest.yaml
```
