# AnyGoal

AnyGoal is an installable benchmark API for collecting out-of-distribution
control datasets and evaluating world models through rollout error,
and planning. It extends
[`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel) and
keeps the original model implementations and numerical protocols intact.

## Installation

Python 3.10, Cuda 12.8

For a source checkout that can run the full research code without inheriting
packages from another Python or Conda environment, create an isolated virtual
environment and install the pinned repository requirements:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` includes the CUDA 12.8 PyTorch package index and the exact
package versions used by this repository. It also installs the current AnyGoal
source checkout in editable mode after resolving the pinned environment.

The default install includes dataset I/O, collection, validation, and the
simulation environments. The explicit `data` extra is an equivalent alias:

```bash
pip install anygoal
pip install 'anygoal[data]'
```

Add rollout/planning dependencies for evaluation, or install the complete
training stack:

```bash
pip install 'anygoal[eval]'
pip install 'anygoal[all]'
```

`eval` includes the default data functionality, and `all` includes data,
evaluation, and training. The wheel contains the public API, CLI, runner
implementations, and their runtime configurations. It does not redistribute
third-party model repositories.

## Discover the API

```bash
anygoal --help
anygoal list datasets
anygoal list models
anygoal list visualizers
anygoal visualize --help
```

The machine-readable support list is
[`anygoal/resources/catalog.yaml`](anygoal/resources/catalog.yaml). Its public
dataset IDs are:

```text
pusht, avoidt, stackt, pusht-antigoal,
cube, cubehop, cubediagonal,
insertpeg, avoid_peg, extractpeg,
tworoom, tworoom_route
```

The public model IDs are:

```text
lewm, sensorimotor, prejepa, driftworld, irasim,
toy-adaln, toy-action-duplicate, toy-driftworld
```


## Download the original model sources

Evaluation and training integrations use the original upstream repositories.
Download only the model sources you need; AnyGoal checks out the revisions
recorded in `anygoal/resources/sources.yaml` under
`~/.cache/anygoal/sources` and does not run a hidden `pip install`:

```bash
anygoal models list
anygoal models install driftworld
anygoal models install irasim
anygoal models install lewm
anygoal models install sensorimotor
# Or download all four:
anygoal models install --all
```

For development, an initialized `worldmodels/` submodule at the pinned commit
is also recognized. Explicit `DRIFTWORLD_SOURCE_ROOT`, `IRASIM_SOURCE_ROOT`,
`LEWM_SOURCE_ROOT`, and `SENSORIMOTOR_SOURCE_ROOT` overrides take precedence.

## Data

Generated variants use their registered in-package collector. Omitted options
use that dataset's declared defaults, while Python keyword names map directly
to the corresponding hyphenated CLI options.
Generated datasets default to `$STABLEWM_HOME/datasets` (or
`~/.stable_worldmodel/datasets` when `STABLEWM_HOME` is unset).

The Python and terminal interfaces call the same in-package collector:

```python
import anygoal

dataset = anygoal.collect(
    "avoidt",
    num_episodes=400,
    num_parallel=8,
    seed=0,
)
```

```bash
anygoal collect avoidt --num-episodes 400 --num-parallel 8
anygoal collect cubediagonal --output /path/to/cubediagonal.h5
```

Base datasets (`pusht`, `cube`, `insertpeg`, and `tworoom`) are distributed as
HDF5 files rather than reconstructed by AnyGoal collectors. Download them from
the published Hugging Face dataset repository, then validate them in place:

```bash
anygoal data validate ~/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  --dataset pusht
```

Collectors refuse to overwrite by default. Canonical filenames are
recorded in [`anygoal/resources/catalog.yaml`](anygoal/resources/catalog.yaml);
the shared workspace inventory remains in `../unstable-worldmodel/datasets.md`.

## Train and evaluate

The public CLI resolves an ID and calls the installed AnyGoal runner around the
original model source without changing model math, hyperparameters, seeds,
data alignment, or normalization:

The exact canonical spellings, accepted aliases, and HDF5 filenames are listed
in [`scripts/train/training_api_ids.csv`](scripts/train/training_api_ids.csv).

```bash
anygoal train driftworld \
  --data ~/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  --task pusht --dry-run

anygoal train irasim \
  --data ~/.stable_worldmodel/datasets/insert_peg_weak_expert_train.h5 \
  --task insert_peg --dry-run

anygoal eval rollout \
  --model driftworld --dataset avoidt --seed 0 --dry-run

anygoal eval planning \
  --model irasim --dataset cubehop --seed 0 --dry-run
```

Remove `--dry-run` after checking resolved paths. Training accepts common
overrides such as `--learning-rate`, `--batch-size`, `--seed`, and
`--num-workers`; omitted values remain exactly as defined by the selected
model/task profile. `--task` is always required; when the HDF5 filename or
metadata identifies the task, AnyGoal also checks that identity against the
explicit value. Verified profiles cover all four base tasks for LeWM,
Sensorimotor, PreJEPA, DriftWorld, IRASim, and the three toy predictors. LeWM
and PreJEPA training on PushT, Cube, or TwoRoom emits a warning because the
paper used the official released checkpoints for those combinations.
Evaluation accepts
`--checkpoint` and `--model-config`.
The current research
artifact paths are recorded in the shared workspace inventory at
`../unstable-worldmodel/checkpoints.md`; public URLs, licenses, and hashes still
need to be supplied before release.

## Site and W&B configuration

An optional site configuration can be generated from the package defaults:

```bash
mkdir -p ~/.config/anygoal
anygoal config show > ~/.config/anygoal/site.yaml
```

Configuration precedence is:

```text
CLI override > experiment config > user/site config > package defaults
```

The site file centralizes output roots and W&B entity/project/mode.

## Evaluation matrix

[`scripts/analysis/manifest.yaml`](scripts/analysis/manifest.yaml) records the rollout and planning
matrix without one launcher per combination:

```bash
anygoal reproduce scripts/analysis/manifest.yaml
```

The command prints each evaluation command in the matrix for execution by the
workflow or scheduler chosen by the user.

## Python API and extensions

```python
import anygoal
from anygoal import get_registry
from anygoal.datasets import validate_hdf5

registry = get_registry()
dataset = registry.dataset("cube_diagonal")
report = validate_hdf5("/path/to/cube_diagonal_test.h5", dataset_id=dataset.id)

model, result = anygoal.train(
    "driftworld",
    data="/path/to/pusht_expert_train.h5",
    task="pusht",
)
```

Planning keeps Stable Worldmodel's model, cost, solver, and policy APIs intact;
AnyGoal only supplies the registered task evaluation protocol:

```python
from stable_worldmodel.planning import CEMSolver, GoalMSE, ShootingCostEvaluator
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

cost = ShootingCostEvaluator(model=model, objective=GoalMSE())
solver = CEMSolver(cost=cost, num_samples=300, n_steps=30, device="cuda")
policy = WorldModelPolicy(
    solver=solver,
    config=PlanConfig(horizon=5, receding_horizon=5, action_block=5),
)
result = anygoal.plan(
    policy,
    data="/path/to/avoidt_test.h5",
    seed=0,
)
```

The planning defaults reproduce the shared evaluation protocol: 50 trials, a
25-step dataset goal offset, and a 50-step rollout budget. Pass an already
constructed `world` only when custom Stable Worldmodel environment setup is
needed. The preserved `anygoal eval planning` commands remain the paper's exact
model-specific reproduction entry points. `task` is inferred from a canonical
HDF5 filename or metadata; pass it explicitly for custom or ambiguous data.
Passing `seed=[0, 1, 2]` returns a dictionary of `PlanningResult` objects keyed
by evaluation-start sampling seed. Solver randomness remains controlled by the
caller-built policy.

Open-loop rollout error uses the preserved AnyGoal episode sampling,
action-block alignment, and model-native error spaces:

```python
rollout = anygoal.evaluate_rollout(
    "prejepa",
    data="/path/to/avoidt_test.h5",
    checkpoint="/path/to/prejepa_pusht_object.ckpt",
    seed=[0, 1, 2],
)
five_step_mse = rollout[0].five_step_mse
terminal_25_step_mse = rollout[0].horizons[25].terminal_mse
```

The default rollout protocol uses three history frames, five environment
steps per action block, and reports the existing 5-step short error plus
15- and 25-step autoregressive metrics. Each result records whether its MSE is
in latent, normalized-state, or normalized-pixel space.

`data` may be a local Stable World-Model HDF5 path or an HDF5 dataset object.
The returned `model` is the reloaded, evaluation-mode model, while `result`
records the checkpoint, resolved config, command, and run directory. Common
arguments use the signature above; model-specific nested options use double
underscores in Python (for example, `visualization__enabled=False`) or dotted
`key=value` arguments in the CLI.

Third-party packages can register datasets, models, and visualizers through the
`anygoal.datasets`, `anygoal.models`, and `anygoal.visualizers` entry-point
groups. Required I/O contracts, artifact metadata, and tests are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Development

```bash
pytest -q tests/contract
python -m build
```

Model/data integrations may add compatibility tests, but scientific changes to
losses, preprocessing, action conventions, temporal alignment, solver defaults,
or seeds require explicit numerical before/after evidence.
