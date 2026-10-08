# AnyGoal

AnyGoal is an installable benchmark API for collecting out-of-distribution
control datasets and evaluating world models through prediction error,
planning, and counterfactual analysis. It extends
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

For the smaller published package API, install from PyPI:

```bash
pip install anygoal
```

Install optional environment, training, or analysis dependencies as needed:

```bash
pip install 'anygoal[env]'
pip install 'anygoal[train]'
pip install 'anygoal[analysis]'
```

The wheel contains the public API, registries, schemas, validation, generic
visualization, and CLI. Research training backends that are not redistributable
are loaded lazily from a source checkout configured by environment variables.

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


## Download and install the third-party research code


```bash
git clone git@github.com:galilai-group/AnyGoal.git
cd AnyGoal

git submodule update --init \
  worldmodels/driftworld \
  worldmodels/IRASim \
  worldmodels/lewm \
  worldmodels/sensorimotor-world-model

git clone https://github.com/mseitzer/pytorch-fid.git \
  worldmodels/IRASim/pytorch-fid
git -C worldmodels/IRASim/pytorch-fid \
  checkout --detach b9c18118d082cbd263c1b8963fc4221dc1cbb659

git clone https://github.com/universome/stylegan-v.git \
  worldmodels/IRASim/stylegan-v
git -C worldmodels/IRASim/stylegan-v \
  checkout --detach 3fecd69c602e1cda204357201461c0fb0a634909
```

## Data

Generated variants use their registered in-package collector. Omitted options
use that dataset's declared defaults, while Python keyword names map directly
to the corresponding hyphenated CLI options.

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

The public CLI resolves an ID and calls the existing model-specific
implementation without changing its model code, hyperparameters, seeds, data
alignment, or normalization:

```bash
anygoal train driftworld \
  --data ~/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  --task pusht --dry-run

anygoal train irasim \
  --data ~/.stable_worldmodel/datasets/insert_peg_weak_expert_train.h5 \
  --task insert_peg --dry-run

anygoal eval prediction \
  --model driftworld --dataset avoidt --seed 0 --dry-run

anygoal eval planning \
  --model irasim --dataset cubehop --seed 0 --dry-run

anygoal eval counterfactual \
  --model lewm --dataset pusht --seed 0 --dry-run
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

## Visualize

Generic visualization operates on logical artifact fields, so newly registered
datasets and models can use it without task-specific code:

```bash
anygoal visualize dataset-gif \
  --input /path/to/dataset.h5 --output /tmp/episode.gif

anygoal visualize metrics-plot \
  --input /path/to/metrics.json --output /tmp/metrics.png
```

`anygoal visualize --help` describes the required fields, optional inputs, and
output format for every registered visualizer.

## Site, W&B, and Slurm configuration

An optional site configuration can be generated from the package defaults:

```bash
mkdir -p ~/.config/anygoal
anygoal config show > ~/.config/anygoal/site.yaml
```

Configuration precedence is:

```text
CLI override > experiment config > user/site config > package defaults
```

The site file centralizes scheduler resources, environment activation, output
roots, and W&B entity/project/mode. A single generic Slurm worker replaces the
former model/GPU/seed-specific launchers:

```bash
anygoal submit --preset gpu-train --job-name driftworld-pusht --dry-run -- \
  anygoal train driftworld \
    --data ~/.stable_worldmodel/datasets/pusht_expert_train.h5 --task pusht
```

Remove `--dry-run` to call `sbatch`. See [`slurm/README.md`](slurm/README.md).

## Evaluation matrix

[`scripts/analysis/manifest.yaml`](scripts/analysis/manifest.yaml) records the prediction and planning
matrix without one launcher per combination:

```bash
anygoal reproduce scripts/analysis/manifest.yaml
anygoal reproduce scripts/analysis/manifest.yaml --submit --site ~/.config/anygoal/site.yaml
```

The first command only prints commands. Submission is also a dry run unless
`--no-dry-run` is supplied.

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
