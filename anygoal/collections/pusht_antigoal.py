"""Collect balanced PushT-Antigoal-v3 trajectories that pass all support gates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import h5py
import stable_worldmodel as swm

from anygoal.datasets.pusht.support import (
    BlockAngleSupportFilter,
    PushTJointSupportFilter,
    PushTTrainSchemaWriter,
    collect_accepted_episodes,
    complete_on_block_goal,
)
from anygoal.envs.pusht_antigoal import env as pusht_antigoal_env
from anygoal.envs.pusht_antigoal import policy as pusht_antigoal_policy
from anygoal.paths import data_root

COLLECT_COLUMNS = {"action", "pixels", "proprio", "state"}
DEFAULT_OUTPUT = data_root() / "pusht_antigoal_test.h5"
DEFAULT_REFERENCE = data_root() / "pusht_expert_train.h5"


def _validate_options(
    episodes_per_goal: int,
    min_frames: int,
    num_parallel: int,
    max_steps: int,
    speed: float,
    minimum_support: float,
    reference: Path,
    output: Path,
    overwrite: bool,
) -> None:
    if episodes_per_goal <= 0:
        raise ValueError("episodes-per-goal must be positive")
    if num_parallel <= 0:
        raise ValueError("num-parallel must be positive")
    if min_frames <= 0 or min_frames > max_steps:
        raise ValueError("min-frames must be in [1, max-steps]")
    if not 0.0 < speed <= 1.0:
        raise ValueError("speed must be in (0, 1]")
    if not 0.95 <= minimum_support <= 1.0:
        raise ValueError("minimum-support must be in [0.95, 1]")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")


def collect_dataset(
    *,
    episodes_per_goal: int = 100,
    min_frames: int = 26,
    num_parallel: int = 8,
    max_steps: int = 500,
    resolution: int = 224,
    speed: float = 0.65,
    seed: int = 0,
    target_corner: Literal["top-left", "top-right", "bottom-left", "bottom-right"]
    | None = None,
    minimum_support: float = 0.951,
    skip_support_filter: bool = False,
    reference: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> Path:
    _validate_options(
        episodes_per_goal,
        min_frames,
        num_parallel,
        max_steps,
        speed,
        minimum_support,
        reference,
        output,
        overwrite,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )
    writer = PushTTrainSchemaWriter(temporary_output, resolution)
    angle_filter = BlockAngleSupportFilter(reference, minimum_fraction=minimum_support)
    joint_filter = PushTJointSupportFilter(reference, minimum_fraction=minimum_support)
    totals = {
        "attempted": 0,
        "accepted": 0,
        "rejected_short": 0,
        "rejected_failed": 0,
        "rejected_filtered": 0,
    }
    rejected_angle = 0
    rejected_joint = 0

    def accept_supported_episode(episode: dict) -> bool:
        nonlocal rejected_angle, rejected_joint
        if skip_support_filter:
            return True
        if not angle_filter.accepts(episode):
            rejected_angle += 1
            return False
        if not joint_filter.accepts(episode):
            rejected_joint += 1
            return False
        return True

    try:
        selected_corners = (
            pusht_antigoal_env.ANTIGOAL_NAMES
            if target_corner is None
            else (target_corner,)
        )
        for corner_name in selected_corners:
            corner_index = pusht_antigoal_env.ANTIGOAL_NAMES.index(corner_name)
            world = swm.World(
                "anygoal/PushTAntigoal-v3",
                num_envs=num_parallel,
                image_shape=(resolution, resolution),
                max_episode_steps=max_steps,
                resolution=resolution,
                target_corner=corner_name,
            )
            world.set_policy(
                pusht_antigoal_policy.PushTAntigoalV3Policy(
                    speed=speed, seed=seed + corner_index
                )
            )
            try:
                stats = collect_accepted_episodes(
                    world,
                    writer,
                    episodes=episodes_per_goal,
                    min_frames=min_frames,
                    seed=seed + corner_index,
                    columns=COLLECT_COLUMNS,
                    accept_episode=accept_supported_episode,
                    complete_episode=complete_on_block_goal,
                )
            finally:
                world.close()
            for name in totals:
                totals[name] += stats[name]
            print(f"Completed {corner_name}: {episodes_per_goal}", flush=True)
    except BaseException:
        print(
            f"Rejected so far: angle={rejected_angle}, joint={rejected_joint}",
            flush=True,
        )
        if temporary_output.exists():
            print(f"Partial dataset retained at {temporary_output}", flush=True)
        raise

    os.replace(temporary_output, output)
    with h5py.File(output, "a") as dataset:
        dataset.attrs["environment"] = "anygoal/PushTAntigoal-v3"
        dataset.attrs["pusht_support_reference"] = str(reference.resolve())
        dataset.attrs["minimum_episode_support"] = minimum_support
        dataset.attrs["kde_bins"] = 256
        dataset.attrs["corner_balance"] = "balanced"
        dataset.attrs["corner_quota_per_goal"] = episodes_per_goal
        dataset.attrs["antigoal_positions"] = (
            pusht_antigoal_env.ANTIGOAL_V3_POSITION_ARRAY
        )
    print(
        f"{totals['accepted']} accepted / {totals['attempted']} attempted "
        f"({totals['rejected_short']} short, "
        f"{totals['rejected_failed']} failed, "
        f"{rejected_angle} angle-support rejected, "
        f"{rejected_joint} joint-support rejected)"
    )
    print(f"Dataset: {output}")
    return output
