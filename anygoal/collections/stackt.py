"""Collect StackT-v2 trajectories with precise, PushT-supported rotation."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import stable_worldmodel as swm

from anygoal.datasets.pusht.support import (
    BlockAngleSupportFilter,
    PushTTrainSchemaWriter,
    collect_accepted_episodes,
    complete_on_block_goal,
)
from anygoal.envs.stackt import policy as stackt_policy
from anygoal.paths import data_root

COLLECT_COLUMNS = {"action", "pixels", "proprio", "state"}
DEFAULT_OUTPUT = data_root() / "stackt_test.h5"
DEFAULT_REFERENCE = data_root() / "pusht_expert_train.h5"


def _validate_options(
    episodes: int,
    min_frames: int,
    num_parallel: int,
    max_steps: int,
    speed: float,
    minimum_angle_support: float,
    reference: Path,
    output: Path,
    overwrite: bool,
) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if num_parallel <= 0 or num_parallel > episodes:
        raise ValueError("num-parallel must be in [1, episodes]")
    if min_frames <= 0 or min_frames > max_steps:
        raise ValueError("min-frames must be in [1, max-steps]")
    if not 0.0 < speed <= 1.0:
        raise ValueError("speed must be in (0, 1]")
    if not 0.95 <= minimum_angle_support <= 1.0:
        raise ValueError("minimum-angle-support must be in [0.95, 1]")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")


def collect_dataset(
    *,
    episodes: int = 400,
    min_frames: int = 20,
    num_parallel: int = 8,
    max_steps: int = 500,
    resolution: int = 224,
    speed: float = 0.7,
    clearance: float = 20.0,
    seed: int = 0,
    minimum_angle_support: float = 0.955,
    reference: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> Path:
    _validate_options(
        episodes,
        min_frames,
        num_parallel,
        max_steps,
        speed,
        minimum_angle_support,
        reference,
        output,
        overwrite,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )
    angle_filter = BlockAngleSupportFilter(
        reference, minimum_fraction=minimum_angle_support
    )
    rejected_angle_support = 0

    def accept_supported_episode(episode: dict) -> bool:
        nonlocal rejected_angle_support
        if angle_filter.accepts(episode):
            return True
        rejected_angle_support += 1
        return False

    world = swm.World(
        "anygoal/StackT-v2",
        num_envs=num_parallel,
        image_shape=(resolution, resolution),
        max_episode_steps=max_steps,
        resolution=resolution,
        clearance=clearance,
    )
    world.set_policy(stackt_policy.StackTV2Policy(speed=speed, seed=seed))
    writer = PushTTrainSchemaWriter(temporary_output, resolution)
    try:
        stats = collect_accepted_episodes(
            world,
            writer,
            episodes=episodes,
            min_frames=min_frames,
            seed=seed,
            columns=COLLECT_COLUMNS,
            accept_episode=accept_supported_episode,
            complete_episode=complete_on_block_goal,
        )
    finally:
        world.close()

    os.replace(temporary_output, output)
    with h5py.File(output, "a") as dataset:
        dataset.attrs["environment"] = "anygoal/StackT-v2"
        dataset.attrs["pusht_angle_reference"] = str(reference.resolve())
        dataset.attrs["minimum_episode_angle_support"] = minimum_angle_support
        dataset.attrs["angle_kde_bins"] = 256
    print(
        f"{stats['accepted']} accepted / {stats['attempted']} attempted "
        f"({stats['rejected_short']} short, "
        f"{stats['rejected_failed']} failed, "
        f"{rejected_angle_support} angle-support rejected)"
    )
    print(f"Dataset: {output}")
    return output
