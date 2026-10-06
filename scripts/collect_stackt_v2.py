"""Collect StackT-v2 trajectories with precise, PushT-supported rotation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import stable_worldmodel as swm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anygoal  # noqa: F401
from anygoal.datasets import (
    BlockAngleSupportFilter,
    PushTTrainSchemaWriter,
    collect_accepted_episodes,
    complete_on_block_goal,
)
from anygoal.envs import StackTV2Policy
from anygoal.paths import data_root


COLLECT_COLUMNS = {"action", "pixels", "proprio", "state"}
DEFAULT_OUTPUT = data_root() / "stackt_test.h5"
DEFAULT_REFERENCE = data_root() / "pusht_expert_train.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--speed", type=float, default=0.7)
    parser.add_argument("--clearance", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-angle-support", type=float, default=0.955)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.num_envs <= 0 or args.num_envs > args.episodes:
        raise ValueError("num-envs must be in [1, episodes]")
    if args.min_frames <= 0 or args.min_frames > args.max_steps:
        raise ValueError("min-frames must be in [1, max-steps]")
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("speed must be in (0, 1]")
    if not 0.95 <= args.minimum_angle_support <= 1.0:
        raise ValueError("minimum-angle-support must be in [0.95, 1]")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )
    angle_filter = BlockAngleSupportFilter(
        args.reference, minimum_fraction=args.minimum_angle_support
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
        num_envs=args.num_envs,
        image_shape=(args.resolution, args.resolution),
        max_episode_steps=args.max_steps,
        resolution=args.resolution,
        clearance=args.clearance,
    )
    world.set_policy(StackTV2Policy(speed=args.speed, seed=args.seed))
    writer = PushTTrainSchemaWriter(temporary_output, args.resolution)
    try:
        stats = collect_accepted_episodes(
            world,
            writer,
            episodes=args.episodes,
            min_frames=args.min_frames,
            seed=args.seed,
            columns=COLLECT_COLUMNS,
            accept_episode=accept_supported_episode,
            complete_episode=complete_on_block_goal,
        )
    finally:
        world.close()

    os.replace(temporary_output, args.output)
    with h5py.File(args.output, "a") as dataset:
        dataset.attrs["environment"] = "anygoal/StackT-v2"
        dataset.attrs["pusht_angle_reference"] = str(args.reference.resolve())
        dataset.attrs["minimum_episode_angle_support"] = (
            args.minimum_angle_support
        )
        dataset.attrs["angle_kde_bins"] = 256
    print(
        f'{stats["accepted"]} accepted / {stats["attempted"]} attempted '
        f'({stats["rejected_short"]} short, '
        f'{stats["rejected_failed"]} failed, '
        f"{rejected_angle_support} angle-support rejected)"
    )
    print(f"Dataset: {args.output}")


if __name__ == "__main__":
    main()
