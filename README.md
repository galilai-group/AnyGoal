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
pip install 'anygoal[paper]'
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
insertpeg, hug_box, extractpeg,
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

Generated variants use their registered collector. The publication preset
preserves the recorded publication recipe and seeds; the smoke preset only
reduces the number of episodes/environments. CubeDiagonal's former 50+350
launcher sequence is reproduced and merged by the one command below.

```bash
anygoal data collect avoidt --preset smoke --dry-run
anygoal data collect cubediagonal --output /path/to/cubediagonal.h5
```

Base datasets whose canonical source is upstream are imported explicitly and
validated before copying:

```bash
anygoal data collect pusht \
  --source /download/pusht_expert_train.h5 \
  --output ~/.stable_worldmodel/datasets/pusht_expert_train.h5

anygoal data validate ~/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  --dataset pusht
```

Collectors and imports refuse to overwrite by default. See
[`datasets.md`](datasets.md) for canonical filenames.

## Train and evaluate

The public CLI resolves an ID and calls the existing model-specific
implementation without changing its model code, hyperparameters, seeds, data
alignment, or normalization:

```bash
anygoal train driftworld --dataset pusht --dry-run

anygoal train sensorimotor --dataset insertpeg --dry-run

anygoal eval prediction \
  --model driftworld --dataset avoidt --seed 0 --dry-run

anygoal eval planning \
  --model irasim --dataset cubehop --seed 0 --dry-run

anygoal eval counterfactual \
  --model lewm --dataset pusht --seed 0 --dry-run
```

Remove `--dry-run` after checking resolved paths. Use `--checkpoint` and
`--model-config` to override the catalog/config defaults. The current research
artifact paths are recorded in [`checkpoints.md`](checkpoints.md); public URLs,
licenses, and hashes still need to be supplied before release.

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

Copy [`configs/site/example.yaml`](configs/site/example.yaml) to
`~/.config/anygoal/site.yaml`. Configuration precedence is:

```text
CLI override > experiment config > user/site config > package defaults
```

The site file centralizes scheduler resources, environment activation, output
roots, and W&B entity/project/mode. A single generic Slurm worker replaces the
former model/GPU/seed-specific launchers:

```bash
anygoal submit --preset gpu-train --job-name driftworld-pusht --dry-run -- \
  anygoal train driftworld --dataset pusht
```

Remove `--dry-run` to call `sbatch`. See [`slurm/README.md`](slurm/README.md).

## Paper matrix

[`paper/manifest.yaml`](paper/manifest.yaml) records the prediction and planning
matrix without one launcher per combination:

```bash
anygoal reproduce paper/manifest.yaml
anygoal reproduce paper/manifest.yaml --submit --site ~/.config/anygoal/site.yaml
```

The first command only prints commands. Submission is also a dry run unless
`--no-dry-run` is supplied.

## Python API and extensions

```python
from anygoal import get_registry
from anygoal.datasets import validate_hdf5

registry = get_registry()
dataset = registry.dataset("cube_diagonal")
report = validate_hdf5("/path/to/cube_diagonal_test.h5", dataset_id=dataset.id)
```

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
