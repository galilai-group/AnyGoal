"""Collect AvoidT with corner-sampled ball goals and a fixed goal T."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import stable_worldmodel as swm

from anygoal.datasets.pusht.support import (
    PushTTrainSchemaWriter,
    collect_accepted_episodes,
)
from anygoal.envs.avoidt import env as avoidt_env
from anygoal.envs.avoidt import policy as avoidt_policy
from anygoal.paths import data_root

GOALS = ("top-left", "top-right", "bottom-left", "bottom-right")
COLLECT_COLUMNS = {"action", "pixels", "proprio", "state"}
DEFAULT_OUTPUT = data_root() / "avoidt_test.h5"


def _validate_options(
    episodes_per_goal: int,
    min_frames: int,
    num_parallel: int,
    max_steps: int,
    resolution: int,
    output: Path,
    overwrite: bool,
) -> None:
    if episodes_per_goal <= 0:
        raise ValueError("episodes-per-goal must be positive.")
    if num_parallel <= 0 or num_parallel > episodes_per_goal:
        raise ValueError("num-parallel must be in [1, episodes-per-goal].")
    if min_frames <= 0 or min_frames > max_steps:
        raise ValueError("min-frames must be in [1, max-steps].")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")


def _collect_goal(
    goal_name: str,
    writer: PushTTrainSchemaWriter,
    *,
    episodes_per_goal: int,
    min_frames: int,
    num_parallel: int,
    max_steps: int,
    resolution: int,
    seed: int,
) -> dict[str, int]:
    randomization_wrapper = partial(
        avoidt_env.AvoidTEpisodeRandomizationWrapper,
        corner=goal_name,
        clearance=20.0,
    )
    world = swm.World(
        "swm/PushT-v1",
        num_envs=num_parallel,
        image_shape=(resolution, resolution),
        max_episode_steps=max_steps,
        resolution=resolution,
        pre_wrappers=[randomization_wrapper],
    )
    world.set_policy(avoidt_policy.SampledGoalAvoidTPolicy(goal=(75.0, 75.0), seed=seed))
    try:
        return collect_accepted_episodes(
            world,
            writer,
            episodes=episodes_per_goal,
            min_frames=min_frames,
            seed=seed,
            columns=COLLECT_COLUMNS,
        )
    finally:
        world.close()


def collect_dataset(
    *,
    episodes_per_goal: int = 100,
    min_frames: int = 26,
    num_parallel: int = 8,
    max_steps: int = 200,
    resolution: int = 224,
    seed: int = 0,
    output: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> Path:
    _validate_options(
        episodes_per_goal,
        min_frames,
        num_parallel,
        max_steps,
        resolution,
        output,
        overwrite,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )
    writer = PushTTrainSchemaWriter(temporary_output, resolution)
    for goal_name in GOALS:
        stats = _collect_goal(
            goal_name,
            writer,
            episodes_per_goal=episodes_per_goal,
            min_frames=min_frames,
            num_parallel=num_parallel,
            max_steps=max_steps,
            resolution=resolution,
            seed=seed,
        )
        print(
            f"{goal_name}: {stats['accepted']} accepted / "
            f"{stats['attempted']} attempted "
            f"({stats['rejected_short']} short, "
            f"{stats['rejected_failed']} failed)"
        )
    os.replace(temporary_output, output)
    print(f"Dataset: {output}")
    return output
