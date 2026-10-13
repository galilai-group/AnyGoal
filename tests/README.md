# Test suite

The suite is organized by public workflow. Shared model and dataset IDs live in
`support/catalog.py`, so every matrix test is generated from one explicit
support definition rather than hand-picked cases.

## Test levels

The default suite is deterministic and does not require downloaded model
sources, checkpoints, or a GPU:

```bash
pytest -q
```

Integration tests exercise real simulation environments and native components
with reduced inputs:

```bash
pytest -q -m integration
```

Artifact-backed matrix tests execute every supported combination at reduced
scale: 8 generated collection tasks, 32 model/base-dataset training runs, and
96 model/dataset combinations for both planning and rollout. They are not
paper-reproduction runs. Submit them on an allocated GPU and provide the
canonical artifacts:

```bash
ANYGOAL_MATRIX_DATA_ROOT=/path/to/datasets pytest -q -m matrix_e2e
```

`ANYGOAL_MATRIX_DEVICE` defaults to `cuda`. The packaged rollout configuration
defines checkpoint and auxiliary config paths; original upstream model sources
must have been installed with `anygoal models install --all`.
`ANYGOAL_MATRIX_EVAL_CONFIG` may point to a site-specific copy of the rollout
configuration when artifacts are provisioned outside those default paths. A
preflight test reports every missing dataset, checkpoint, auxiliary config, and
source checkout together.

## Layout

- `contract/`: public IDs, imports, API/CLI parity, and complete mocked matrices.
- `collection/`, `training/`, `planning/`, `rollout/`: workflow behavior and
  reduced end-to-end matrices.
- `envs/`: real simulator behavior; automatically marked `integration`.
- `models/`: model and adapter numerical contracts.
- `analysis/`: tests for research-only scripts outside the public API.
- `support/`: shared constants, artifact resolution, and tiny fixture builders.
