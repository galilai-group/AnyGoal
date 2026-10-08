"""Collect balanced PushT-Antigoal-v3 trajectories that pass all support gates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import stable_worldmodel as swm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anygoal  # noqa: F401
from anygoal.datasets.pusht_support import (
    BlockAngleSupportFilter,
    PushTJointSupportFilter,
    PushTTrainSchemaWriter,
    collect_accepted_episodes,
    complete_on_block_goal,
)
from anygoal.envs.push_t_antigoal import (
    ANTIGOAL_NAMES,
    ANTIGOAL_V3_POSITION_ARRAY,
    PushTAntigoalV3Policy,
)
from anygoal.paths import data_root


COLLECT_COLUMNS = {"action", "pixels", "proprio", "state"}
DEFAULT_OUTPUT = data_root() / "pusht_antigoal_test.h5"
DEFAULT_REFERENCE = data_root() / "pusht_expert_train.h5"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-goal", type=int, default=100)
    parser.add_argument("--min-frames", type=int, default=26)
    parser.add_argument("--num-parallel", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--speed", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-corner", choices=ANTIGOAL_NAMES)
    parser.add_argument("--minimum-support", type=float, default=0.951)
    parser.add_argument("--skip-support-filter", action="store_true")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes_per_goal <= 0:
        raise ValueError("episodes-per-goal must be positive")
    if args.num_parallel <= 0:
        raise ValueError("num-parallel must be positive")
    if args.min_frames <= 0 or args.min_frames > args.max_steps:
        raise ValueError("min-frames must be in [1, max-steps]")
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("speed must be in (0, 1]")
    if not 0.95 <= args.minimum_support <= 1.0:
        raise ValueError("minimum-support must be in [0.95, 1]")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )
    writer = PushTTrainSchemaWriter(temporary_output, args.resolution)
    angle_filter = BlockAngleSupportFilter(
        args.reference, minimum_fraction=args.minimum_support
    )
    joint_filter = PushTJointSupportFilter(
        args.reference, minimum_fraction=args.minimum_support
    )
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
        if args.skip_support_filter:
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
            ANTIGOAL_NAMES
            if args.target_corner is None
            else (args.target_corner,)
        )
        for corner_name in selected_corners:
            corner_index = ANTIGOAL_NAMES.index(corner_name)
            world = swm.World(
                "anygoal/PushTAntigoal-v3",
                num_envs=args.num_parallel,
                image_shape=(args.resolution, args.resolution),
                max_episode_steps=args.max_steps,
                resolution=args.resolution,
                target_corner=corner_name,
            )
            world.set_policy(
                PushTAntigoalV3Policy(
                    speed=args.speed, seed=args.seed + corner_index
                )
            )
            try:
                stats = collect_accepted_episodes(
                    world,
                    writer,
                    episodes=args.episodes_per_goal,
                    min_frames=args.min_frames,
                    seed=args.seed + corner_index,
                    columns=COLLECT_COLUMNS,
                    accept_episode=accept_supported_episode,
                    complete_episode=complete_on_block_goal,
                )
            finally:
                world.close()
            for name in totals:
                totals[name] += stats[name]
            print(f"Completed {corner_name}: {args.episodes_per_goal}", flush=True)
    except BaseException:
        print(
            f"Rejected so far: angle={rejected_angle}, "
            f"joint={rejected_joint}",
            flush=True,
        )
        if temporary_output.exists():
            print(f"Partial dataset retained at {temporary_output}", flush=True)
        raise

    os.replace(temporary_output, args.output)
    with h5py.File(args.output, "a") as dataset:
        dataset.attrs["environment"] = "anygoal/PushTAntigoal-v3"
        dataset.attrs["pusht_support_reference"] = str(args.reference.resolve())
        dataset.attrs["minimum_episode_support"] = args.minimum_support
        dataset.attrs["kde_bins"] = 256
        dataset.attrs["corner_balance"] = "balanced"
        dataset.attrs["corner_quota_per_goal"] = args.episodes_per_goal
        dataset.attrs["antigoal_positions"] = ANTIGOAL_V3_POSITION_ARRAY
    print(
        f'{totals["accepted"]} accepted / {totals["attempted"]} attempted '
        f'({totals["rejected_short"]} short, '
        f'{totals["rejected_failed"]} failed, '
        f"{rejected_angle} angle-support rejected, "
        f"{rejected_joint} joint-support rejected)"
    )
    print(f"Dataset: {args.output}")
    return args.output
