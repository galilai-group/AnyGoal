# Analysis and visualization recipes

Run commands from the repository root. Datasets default to
`~/.stable_worldmodel/datasets`; generated artifacts default to
`outputs/analysis`.

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

## Counterfactual analysis

The counterfactual sensitivity protocol is a research analysis for LeWM,
Sensorimotor, and the toy AdaLN predictor rather than a public AnyGoal evaluation
API. Its fixed intervention actions live in
`anygoal/resources/counterfactual/`.

```bash
python -m scripts.analysis.run_counterfactual_evaluation --help
python -m scripts.analysis.plot_counterfactual_grids --help
python -m scripts.analysis.summarize_counterfactual_metrics --help
python -m scripts.analysis.update_insert_peg_after_grip_bias --help
```

The public rollout/planning evaluation matrix is retained in
[`manifest.yaml`](manifest.yaml) and can be dispatched with:

```bash
anygoal reproduce scripts/analysis/manifest.yaml
```
