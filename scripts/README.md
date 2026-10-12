# Script layout

Scripts are grouped by research workflow. Supported API/CLI implementations
and runtime configuration live in `anygoal/runners/` and
`anygoal/resources/`; the train, plan, and eval files here are compatibility
wrappers for direct source-checkout use.

| Directory | Purpose |
| --- | --- |
| `data/` | Dataset collection, conversion, validation, and preparation |
| `train/` | Thin wrappers for installed model training runners |
| `plan/` | Thin wrappers for installed MPC/planning runners |
| `eval/` | Thin wrappers for installed rollout evaluation runners |
| `analysis/` | Statistical analysis and publication visualizations |
| `release/` | Hugging Face publication and migration utilities |
| `utils/` | Reusable artifact-generation and maintenance utilities |

The supported installed entry points are `anygoal collect`, `anygoal train`,
`anygoal eval`, and `anygoal models`. Registry entries in
`anygoal/resources/catalog.yaml` map those commands to package modules, never
to this directory.
