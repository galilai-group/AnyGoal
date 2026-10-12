# Contributing to AnyGoal

AnyGoal keeps benchmark semantics stable while allowing new datasets, models,
and visualizers to be added without editing evaluator conditionals. Changes to
random seeds, temporal alignment, normalization, action conventions, loss
functions, or default hyperparameters must be called out explicitly and must
include before/after numerical evidence.

## Development setup

Install an editable development build and run the contract suite:

```bash
pip install -e '.[dev,analysis]'
pytest -q tests/contract
anygoal list datasets
anygoal list models
anygoal list visualizers
```

Do not commit data, checkpoints, credentials, personal paths, W&B run IDs, or
cluster policy. Dataset artifacts belong outside the repository and checkpoint
artifacts must never be overwritten.

## Adding a dataset

1. Choose a stable ID and add one entry to
   `anygoal/resources/catalog.yaml`. Declare aliases, family, canonical
   filename, environment, evaluator ID, config stem, physical schema,
   publication seed, and whether the data is imported or generated.
2. Expose the logical fields `action`, `pixels`, optional `state`/`proprio`,
   episode boundaries, and a valid-action mask. Document action alignment,
   dimensions, units, frame skip, image order, and the terminal-row rule.
3. Imported data must document source URL, version, license, checksum, and any
   deterministic conversion. Generated data must refuse overwrite and provide
   deterministic `smoke` and publication presets.
4. Implement the `anygoal.types.DatasetAdapter` contract when the common HDF5
   validator is insufficient. Heavy dependencies must be imported lazily.
5. Add a tiny fixture testing schema, episode boundaries, deterministic seeds,
   overwrite refusal, aliases, and CLI help. Run `anygoal data validate` on a
   complete artifact.

External packages may publish a `DatasetSpec` or compatible adapter through
the `anygoal.datasets` entry-point group. A generated external collector is a
Python module name (or a repository-relative `.py` file for in-tree changes)
whose CLI accepts `--output`, `--seed`, and `--overwrite`. The object returned
by the entry point must have a globally unique `id` and the fields of
`DatasetSpec`.

## Adding a model

1. Add a catalog entry with a stable ID, aliases, evaluator ID, capabilities
   (`train`, `rollout`, `plan`), supported families, lazy trainer/evaluator
   module or script paths, config directories/styles, seed style, and
   checkpoint override key.
2. Implement `ModelAdapter`, including `validate_dataset`. Rollout adapters
   implement `rollout_errors`; planning adapters implement candidate costs.
3. State observation keys, action convention, history length, frame skip,
   preprocessing, normalization ownership, and output/error space.
4. Add load round-trip, output-shape, seeded determinism, CPU/GPU smoke, and at
   least one tiny end-to-end test. Existing checkpoint outputs must remain
   numerically unchanged.

External models use the `anygoal.models` entry-point group. Optional dependency
errors must name the install extra rather than failing during `import anygoal`.
The trainer, rollout runner, and planner fields may name installed Python modules;
in-tree integrations may instead use repository-relative `.py` files. A
rollout runner must accept `--model`, `--environment`, `--seed`, and optional
`--checkpoint`. Planner config and seed arguments are generated from the
catalog metadata, so adding an integration must not add a model-name branch to
`anygoal.cli`.

## Adding a visualization

Register its ID, description, required inputs, optional inputs, accepted schema
versions, and output extension. Rendering must work headlessly, select samples
deterministically, require an explicit output path, and refuse overwrite by
default. Add a tiny rendering test and ensure the requirements appear in
`anygoal list visualizers` and `anygoal visualize --help`.

An external runner is an installed Python module recorded as `runner`. It must
accept the declared `input_option` and `output_option`; AnyGoal forwards those
options plus arguments after `--` without adding a visualizer-name branch.

External visualizers use the `anygoal.visualizers` entry-point group.

## Pull-request checklist

- Public IDs and aliases do not collide.
- No personal absolute paths, credentials, or cluster settings were added.
- Imports do not eagerly load optional model dependencies.
- Seeds, tensor shapes, temporal alignment, and hyperparameters are unchanged,
  or the intentional scientific change is documented and benchmarked.
- Outputs are versioned, written atomically where practical, and not silently
  overwritten.
- Public symbols have concise docstrings and type annotations.
- Contract tests, CLI `--help`, wheel build, and wheel-only import pass.
- Dataset/model licenses and citations are documented.
