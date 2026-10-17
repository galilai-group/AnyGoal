# AnyGoal
## AnyGoal: On World Model Generalization Beyond Expert Rollouts
by [Heejeong Nam](https://hazel-heejeong-nam.github.io/), [Divyat Mahajan](https://divyat09.github.io/), [Quentin Le Lidec](https://quentinll.github.io/), [Randall Balestriero](https://randallbalestriero.github.io/), [David Klindt](https://scholar.google.de/citations?user=EpT-nUAAAAAJ&hl=en).



AnyGoal is a benchmark for testing compositional generalization of world models.
It extends
[`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel) and
keeps the original model implementations and numerical protocols intact.

> `anygoal` PyPI is currently in beta.
> We welcome feedback, bug reports, and feature requests through GitHub Issues.

## Installation

AnyGoal supports Python 3.10 and 3.11.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
For a source checkout that can run the full research code without inheriting
packages from another Python or Conda environment, create an isolated virtual
environment and install the pinned repository requirements:

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install "anygoal[eval]==0.1.0b3"
anygoal --version
```


>The default install `uv pip install anygoal` includes dataset I/O, collection, validation, and the simulation environments, ***without the rollout and planning evaluation dependencies***. Add them with `anygoal[eval]` or install the complete training stack with `anygoal[all]`.

We provided `requirements.txt` for reference, which includes the CUDA 12.8 PyTorch package index and the exact
package versions used by this repository.


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


## Quick start with the CLI

### 1. Check default paths

Command below prints the default paths for the AnyGoal home directory, data, checkpoints, outputs, sources, and cache:
```bash
anygoal config path home
anygoal config path data
anygoal config path checkpoints
anygoal config path outputs
anygoal config path sources
anygoal config path cache
```

### 2. Check the available datasets, models, and visualizers
You can check the available datasets, models, and visualizers with the CLI:
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


### 3. Download the original model sources (code only, no weights)

Evaluation and training integrations use the original upstream repositories.

```bash
anygoal models list
```
will print the pinned commit for each model. Status `missing` indicates that the source is not yet downloaded. Download the sources for each model with:

```bash
anygoal models install driftworld
anygoal models install irasim
anygoal models install lewm
anygoal models install sensorimotor
# Or download all four:
anygoal models install --all
```

### 4. Download datasets and checkpoints from Hugging Face

Let's download `pusht` dataset, `avoidt` dataset, and `lewm` checkpoint for a quick start:

```bash
anygoal artifacts download --model lewm
anygoal artifacts download --dataset pusht
anygoal artifacts download --dataset avoidt
```

You can also download everything with:

```bash
anygoal artifacts download --checkpoints --all
anygoal artifacts download --datasets --all
```

### 5. Run a planning evaluation with the downloaded artifacts

```bash
# plan with a base dataset
anygoal eval planning --model lewm --dataset pusht --seed 0

# plan with an anygoal dataset
anygoal eval planning --model lewm --dataset avoidt --seed 0

# rollout with an anygoal dataset
anygoal eval rollout --model lewm --dataset avoidt --seed 0
```

Both `eval planning` and `eval rollout` resolve the default local checkpoint from
the shared artifact catalog using the selected model and the dataset's base family.
Use `--checkpoint /path/to/weights` to select your own weights, or `--download`
to fetch missing artifacts. An explicit `--model-config` or Hydra checkpoint
override keeps control of its own checkpoint. Pass Hydra overrides after `--`,
for example `-- policy=lewm`; `--policy=lewm` is not an AnyGoal CLI option.


## Dataset collection

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
recorded in [`anygoal/resources/catalog.yaml`](anygoal/resources/catalog.yaml).

## Train

Training needs `uv pip install 'anygoal[all]'`. Make sure you installed models by `anygoal models install --all`. Currently, we support training on the eight models described in the paper.

The exact canonical spellings, accepted aliases, and HDF5 filenames are listed
in [`scripts/train/training_api_ids.csv`](scripts/train/training_api_ids.csv).

```bash
# check the resolved paths with --dry-run before training
anygoal train driftworld \
  --data "$(anygoal config path data)/pusht_expert_train.h5" \
  --task pusht --dry-run

# remove --dry-run after checking resolved paths
anygoal train driftworld \
  --data "$(anygoal config path data)/pusht_expert_train.h5" \
  --task pusht

```

Training accepts common
overrides such as `--learning-rate`, `--batch-size`, `--seed`, and
`--num-workers`; omitted values remain exactly as defined by the selected
model/task profile. `--task` is always required; when the HDF5 filename or
metadata identifies the task, AnyGoal also checks that identity against the
explicit value.

LeWM and PreJEPA training on PushT, Cube, or TwoRoom emits a warning because the
paper used the official released checkpoints for those combinations.



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
