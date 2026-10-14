# AnyGoal

> **Beta Release**
>
> AnyGoal Pypi is currently in beta.
>
> We welcome feedback, bug reports, and feature requests through GitHub Issues.

AnyGoal is an installable benchmark API for collecting out-of-distribution
control datasets and evaluating world models through rollout error,
and planning. It extends
[`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel) and
keeps the original model implementations and numerical protocols intact.

## Installation

AnyGoal supports Python 3.10 and newer. The basic API and CLI do not require a
GPU. Training and GPU-backed evaluation require a PyTorch installation that is
compatible with the host's accelerator and driver.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
For a source checkout that can run the full research code without inheriting
packages from another Python or Conda environment, create an isolated virtual
environment and install the pinned repository requirements:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` includes the CUDA 12.8 PyTorch package index and the exact
package versions used by this repository. It also installs the current AnyGoal
source checkout in editable mode after resolving the pinned environment.

The default install includes dataset I/O, collection, validation, and the
simulation environments. The explicit `data` extra is an equivalent alias:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install anygoal
# Equivalent explicit alias for the default data functionality:
uv pip install 'anygoal[data]'
```

A minimal installation check that does not download datasets or checkpoints is:

```python
import anygoal

registry = anygoal.get_registry()
print(anygoal.__version__)
print(registry.dataset("pusht").filename)
```

```bash
anygoal --version
anygoal list datasets
```

Add rollout/planning dependencies for evaluation, or install the complete
training stack:

```bash
uv pip install 'anygoal[eval]'
uv pip install 'anygoal[all]'
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
`$ANYGOAL_SOURCE_ROOT` (by default `$ANYGOAL_HOME/sources`) and does not run a
hidden `pip install`:

```bash
anygoal models list
anygoal models install driftworld
anygoal models install irasim
anygoal models install lewm
anygoal models install sensorimotor
# Or download all four:
anygoal models install --all
```

The IRASim install also fetches its pinned `pytorch-fid` and `stylegan-v`
evaluation sources. Upstream stores them as gitlinks without a `.gitmodules`
file, so AnyGoal resolves them explicitly.

For development, an initialized `worldmodels/` submodule at the pinned commit
is also recognized. Explicit `DRIFTWORLD_SOURCE_ROOT`, `IRASIM_SOURCE_ROOT`,
`LEWM_SOURCE_ROOT`, and `SENSORIMOTOR_SOURCE_ROOT` overrides take precedence.

## Data

### Download the published artifacts

The AnyGoal datasets and checkpoints are published in separate Hugging Face
repositories with the same ID. After authenticating for private repositories,
download each complete repository directly into the paths used by the bundled
evaluation configs:

```bash
hf auth login
hf download HazelNam/anygoal --repo-type model \
  --local-dir "$(anygoal config path checkpoints)"
hf download HazelNam/anygoal --repo-type dataset \
  --local-dir "$(anygoal config path data)"
```

These commands download all artifacts published in the AnyGoal repositories.
The dataset repository intentionally excludes the upstream PushT, Cube, and
TwoRoom base datasets, and the model repository intentionally excludes their
official LeWM and PreJEPA checkpoints. Those upstream artifacts must still be
downloaded from their original publishers.

The AnyGoal CLI can instead download only the artifacts required by one
evaluation, download missing artifacts immediately before evaluation, or
download one complete artifact category:

```bash
anygoal artifacts download --model driftworld --dataset cubehop

anygoal eval planning \
  --model driftworld --dataset cubehop --download

anygoal artifacts download --checkpoints --all
anygoal artifacts download --datasets --all
```

An evaluation dataset uses the checkpoint trained on its base family; for
example, `cubehop` resolves to `driftworld_cube.pt`. Existing local files are
reused.

Generated variants use their registered in-package collector. Omitted options
use that dataset's declared defaults, while Python keyword names map directly
to the corresponding hyphenated CLI options.
Generated datasets use `ANYGOAL_DATA_ROOT` when set. Otherwise AnyGoal asks
Stable Worldmodel for its dataset directory, respecting `STABLEWM_HOME` and
Stable Worldmodel's own default without duplicating that default here.

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
anygoal data validate "$(anygoal config path data)/pusht_expert_train.h5" \
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
  --data "$(anygoal config path data)/pusht_expert_train.h5" \
  --task pusht --dry-run

anygoal train irasim \
  --data "$(anygoal config path data)/manipulator_insert_peg_weak_expert_train.h5" \
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
The downloadable AnyGoal artifact inventory is recorded in
[`anygoal/resources/artifacts.yaml`](anygoal/resources/artifacts.yaml).

## Site and W&B configuration

AnyGoal-owned files share one root while compatible datasets continue to use
Stable Worldmodel's storage policy:

| Variable | Default |
| --- | --- |
| `ANYGOAL_HOME` | `~/.anygoal` |
| `ANYGOAL_DATA_ROOT` | Stable Worldmodel's `datasets` directory |
| `ANYGOAL_CHECKPOINT_ROOT` | `$ANYGOAL_HOME/checkpoints` |
| `ANYGOAL_OUTPUT_ROOT` | `./outputs` |
| `ANYGOAL_SOURCE_ROOT` | `$ANYGOAL_HOME/sources` |
| `ANYGOAL_CACHE_ROOT` | `$ANYGOAL_HOME/cache` |

Each specific root overrides the path derived from `ANYGOAL_HOME`. Inspect the
effective paths without relying on shell-variable expansion:

```bash
anygoal config path data
anygoal config path checkpoints
anygoal config path sources
```

An optional site configuration can be generated from the package defaults:

```bash
mkdir -p ~/.config/anygoal
anygoal config show > ~/.config/anygoal/site.yaml
```

Configuration precedence is:

```text
CLI override > experiment config > user/site config > package defaults
```

The site file centralizes W&B entity/project/mode. Filesystem roots come from
the environment variables above and are reported as read-only effective values
by `anygoal config show`.

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
groups. Integrations must preserve upstream licenses and attribution, and must
not redistribute models, datasets, or checkpoints without permission.

## License and attribution

AnyGoal-authored source code and AnyGoal-generated datasets and checkpoints are
Copyright 2026 Heejeong Nam and licensed under
[Apache-2.0](LICENSE), unless an artifact is accompanied by different terms.

AnyGoal's wheel does not contain the third-party model repositories or their
model and dataset artifacts. Those materials remain governed by their original
licenses, terms, and citation requirements. At the revisions recorded in
[`anygoal/resources/sources.yaml`](anygoal/resources/sources.yaml), IRASim uses
Apache-2.0, while LeWM and Sensorimotor World Model use MIT. No repository-level
license was found for DriftWorld, so its source must not be redistributed or
relicensed without explicit permission from its copyright holders. Nested
projects retain their own license notices.

The AnyGoal InsertPeg dataset is a modified derivative of the Apache-2.0
[OpenDILabCommunity RL Unplugged DM Control Suite dataset](https://huggingface.co/datasets/OpenDILabCommunity/rl_unplugged_dm_control_suite).
The modification and attribution details are recorded in
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md). Other external
datasets and official checkpoints are not redistributed by AnyGoal and remain
available from their original authors.

## Development

```bash
pytest -q
pytest -q -m integration
# On an allocated GPU with canonical datasets/checkpoints/sources:
pytest -q -m matrix_e2e
python -m build
```

The default suite is artifact-free. `integration` uses real environments and
native components with reduced inputs, while `matrix_e2e` runs all supported
model/dataset combinations at reduced scale. It is not a paper-reproduction
suite; details and required environment variables are in
[`tests/README.md`](tests/README.md).

Model/data integrations may add compatibility tests, but scientific changes to
losses, preprocessing, action conventions, temporal alignment, solver defaults,
or seeds require explicit numerical before/after evidence.
