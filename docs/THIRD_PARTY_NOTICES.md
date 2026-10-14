# Third-Party Notices

AnyGoal-authored source code and AnyGoal-generated datasets and checkpoints are
Copyright 2026 Heejeong Nam and are licensed under Apache-2.0 unless an artifact
is accompanied by different terms.

## Modified InsertPeg dataset

AnyGoal's InsertPeg dataset is a modified derivative of
`manipulator_insert_peg.npy` from the
[OpenDILabCommunity RL Unplugged DM Control Suite dataset](https://huggingface.co/datasets/OpenDILabCommunity/rl_unplugged_dm_control_suite).
The source dataset is licensed under Apache-2.0. The derivative was modified
and packaged for the AnyGoal benchmark and is redistributed under Apache-2.0.
Users should also cite [RL Unplugged: A Suite of Benchmarks for Offline
Reinforcement Learning](https://arxiv.org/abs/2006.13888).

## External model implementations and artifacts

The AnyGoal wheel does not redistribute third-party model repositories. It can
download pinned upstream sources listed in `anygoal/resources/sources.yaml`:

- IRASim: Apache-2.0 at the pinned revision.
- LeWM: MIT at the pinned revision.
- Sensorimotor World Model: MIT at the pinned revision.
- DriftWorld: no repository-level license was found at the pinned revision;
  do not redistribute or relicense it without permission from its copyright
  holders.

Nested projects and externally supplied models, datasets, and checkpoints
retain their own copyright, license, citation, and redistribution terms.
AnyGoal's Apache-2.0 license does not replace those terms.
