# Paper reproduction

Run every command from the repository root. Datasets are read from
`~/.stable_worldmodel/datasets`, and generated files are written under
`paper/results`. Submit expensive commands through the generic `anygoal submit`
interface and the site profile; analysis parameters remain in the Python
recipe rather than scheduler files.

## AvoidT and PushT distributions

```bash
python paper/scripts/distribution_avoidt_to_pusht.py
```

This produces all-point action, agent-position, and physical-T-position KDEs,
summary statistics, and support metrics.

![AvoidT action KDE](../assets/paper/distribution_avoidt_to_pusht/avoidt_action_all_points_kde.png)

## StackT and PushT distributions

```bash
python paper/scripts/distribution_stackt_to_pusht.py
```

The same all-point analysis is applied to the collected StackT expert data.

![StackT action KDE](../assets/paper/distribution_stackt_to_pusht/stackt_action_all_points_kde.png)

## PushT-Antigoal and PushT distributions

```bash
python paper/scripts/distribution_push_t_antigoal_to_pusht.py
```

This applies the same all-point KDE and support analysis to the OOD-only
PushT-Antigoal dataset under the configured AnyGoal data root.

![PushT-Antigoal action KDE](../assets/paper/distribution_push_t_antigoal_to_pusht/push_t_antigoal_action_all_points_kde.png)

## Joint ID/OOD statistics

```bash
python paper/scripts/pusht_avoidt_statistic_repro.py
python paper/scripts/pusht_stackt_statistic_repro.py
python paper/scripts/pusht_antigoal_statistic_repro.py
```

This fits the fixed-seed PushT KDE subsample and assigns row-level ID/OOD
labels to the AvoidT, StackT, and PushT-Antigoal datasets for three joint
feature definitions.

![AvoidT joint ID/OOD statistics](../assets/paper/pusht_avoidt_statistic_repro/joint_pair_id_ood_score_distributions.png)

![StackT joint ID/OOD statistics](../assets/paper/pusht_stackt_statistic_repro/joint_pair_id_ood_score_distributions.png)

![PushT-Antigoal joint ID/OOD statistics](../assets/paper/pusht_antigoal_statistic_repro/joint_pair_id_ood_score_distributions.png)

## Save LeWM counterfactual sensitivity

```bash
python paper/scripts/save_counterfactual_sensitivity.py --adaln-layers 0,1,2,3,4,5,final
```

This is the only expensive model-inference step. It saves one atomic CSV per
layer and action intervention plus metadata. The plotting scripts below reuse
those files without loading the model. Both saving and plotting require
`pusht_expert_train.h5`; other datasets are rejected.

## Counterfactual sensitivity by distance

```bash
python paper/scripts/plot_counterfactual_sensitivity_distance.py
```

The plot uses the `managua` colormap and sums all action interventions for
each layer before plotting sensitivity against agent-to-physical-T distance.

![Average counterfactual sensitivity by distance](../assets/paper/counterfactual_sensitivity/distance_layer_average.jpg)

## Cube counterfactual action sensitivity

```bash
anygoal submit --preset gpu-eval --job-name cube-counterfactual --dry-run -- \
  python paper/scripts/joint_distribution_support.py --help
```

This selects pointwise-supported 27-action xyz and 9-action xy grids, sums
Cube LeWM counterfactual sensitivities for every valid dataset window, and
plots sensitivity against expert delta-z action and end-effector height. The
complete method and selected action magnitudes are documented in
`reports/CUBE_COUNTERFACTUAL_SENSITIVITY.md`.

## Agent position by action angle

```bash
python paper/scripts/visualize_state_with_actioncolor.py
```

Every PushT state-action pair contributes to a spatial hexbin. Each hexagon is
colored by its most frequent 45-degree action bin. Eight per-angle figures use
Gaussian-smoothed KDEs and the corresponding discrete angle color.

![Modal action-angle hexbin](../assets/paper/agent_position_action_angle.jpg)

![Continuous action-angle colormap](../assets/paper/action_angle_colormap_continuous.jpg)

![Discrete action-angle colormap](../assets/paper/action_angle_colormap_discrete.jpg)

## Sampled task episodes

```bash
python paper/scripts/visualize_task_episodes.py --task avoidt
python paper/scripts/visualize_task_episodes.py --task stackt
```

Ten random episodes from the task's catalog-registered dataset are arranged in
a 2-by-5 animated GIF. Shorter episodes hold their final frame until the longest
sampled episode ends. Use `--dataset` to override the registered path.
