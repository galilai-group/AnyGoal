# Script layout

The supported workflow has four entry points:

| Operation | Installed command | Source wrapper |
| --- | --- | --- |
| Data collection/import | `anygoal data collect` | `scripts/collect.py` |
| Training | `anygoal train` | `scripts/train.py` |
| Evaluation | `anygoal eval` | `scripts/eval.py` and the model adapters it dispatches |
| Visualization | `anygoal visualize` | `scripts/visualize.py` |

The remaining files are compatibility implementations used by the registry or
paper analysis recipes. New model/dataset/seed/GPU combinations must not add a
new matrix or worker script: add a catalog entry, config, or paper-manifest row
instead. Once a compatibility implementation has moved under `anygoal/`, leave
only a thin wrapper here.
