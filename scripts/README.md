# Script layout

Scripts are grouped by workflow, following the layout used by
stable-worldmodel. Workflow-specific configuration lives beside its entry
points under `config/`.

| Directory | Purpose |
| --- | --- |
| `data/` | Dataset collection, conversion, validation, and preparation |
| `train/` | Model training and training config |
| `plan/` | MPC/planning evaluation and planning config |
| `eval/` | Prediction evaluation, checks, and result summaries |
| `analysis/` | Statistical analysis and publication visualizations |
| `release/` | Hugging Face publication and migration utilities |
| `utils/` | Reusable artifact-generation and maintenance utilities |

The supported installed entry points remain `anygoal collect`,
`anygoal train`, `anygoal eval`, and `anygoal visualize`. Registry entries in
`anygoal/resources/catalog.yaml` map those commands to the scripts above.
