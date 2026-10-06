"""Collect AvoidT with corner-sampled ball goals and a fixed goal T."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import stable_worldmodel as swm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets import (
    PushTTrainSchemaWriter,
    collect_accepted_episodes,
)
from anygoal.envs.avoid_t import (
    AvoidTEpisodeRandomizationWrapper,
    SampledGoalAvoidTPolicy,
)


GOALS = ("top-left", "top-right", "bottom-left", "bottom-right")
COLLECT_COLUMNS = {"action", "pixels", "proprio", "state"}
DEFAULT_OUTPUT = (
    Path.home()
    / ".stable_worldmodel"
    / "datasets"
    / "avoidt_test.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-goal", type=int, default=100)
    parser.add_argument("--min-frames", type=int, default=26)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes_per_goal <= 0:
        raise ValueError("episodes-per-goal must be positive.")
    if args.num_envs <= 0 or args.num_envs > args.episodes_per_goal:
        raise ValueError("num-envs must be in [1, episodes-per-goal].")
    if args.min_frames <= 0 or args.min_frames > args.max_steps:
        raise ValueError("min-frames must be in [1, max-steps].")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive.")
    if args.output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")


def collect_goal(
    goal_name: str,
    writer: PushTTrainSchemaWriter,
    args: argparse.Namespace,
) -> dict[str, int]:
    randomization_wrapper = partial(
        AvoidTEpisodeRandomizationWrapper,
        corner=goal_name,
        clearance=20.0,
    )
    world = swm.World(
        "swm/PushT-v1",
        num_envs=args.num_envs,
        image_shape=(args.resolution, args.resolution),
        max_episode_steps=args.max_steps,
        resolution=args.resolution,
        pre_wrappers=[randomization_wrapper],
    )
    world.set_policy(
        SampledGoalAvoidTPolicy(goal=(75.0, 75.0), seed=args.seed)
    )
    try:
        return collect_accepted_episodes(
            world,
            writer,
            episodes=args.episodes_per_goal,
            min_frames=args.min_frames,
            seed=args.seed,
            columns=COLLECT_COLUMNS,
        )
    finally:
        world.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )
    writer = PushTTrainSchemaWriter(temporary_output, args.resolution)
    for goal_name in GOALS:
        stats = collect_goal(goal_name, writer, args)
        print(
            f'{goal_name}: {stats["accepted"]} accepted / '
            f'{stats["attempted"]} attempted '
            f'({stats["rejected_short"]} short, '
            f'{stats["rejected_failed"]} failed)'
        )
    os.replace(temporary_output, args.output)
    print(f"Dataset: {args.output}")


if __name__ == "__main__":
    main()
