"""Collect fixed-length OGBench CubeHop expert trajectories."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anygoal  # noqa: F401
from anygoal.datasets import (
    CubeActionSupport,
    CubeHDF5Writer,
    collect_fixed_length_episodes,
    reference_episode_length,
)
from anygoal.datasets.cube_collection import CUBE_H5_COLUMNS
from anygoal.envs import CubeHopPolicy
from anygoal.envs.cube_hop import (
    CUBE_HOP_SIDES,
    EE_START_HIGH,
    EE_START_LOW,
    MAX_TRANSLATION_ACTION_NORM,
    MAX_YAW_ACTION,
    RIGHT_YAW_DELTA,
    SIDE_SAMPLING_BOUNDS,
)


DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
DEFAULT_REFERENCE = DATASET_DIR / "cube_single_expert.h5"
DEFAULT_OUTPUT = DATASET_DIR / "cube_hop_test.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.episodes % len(CUBE_HOP_SIDES) != 0:
        raise ValueError(
            f"episodes must be positive and divisible by {len(CUBE_HOP_SIDES)} "
            "to collect equal right/left halves."
        )
    if args.num_envs <= 0 or args.num_envs > args.episodes:
        raise ValueError("num-envs must be in [1, episodes].")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive.")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5.")
    metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
    for path in (args.output, metadata_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")


def summarize_points(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(len(points)),
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "mean": points.mean(axis=0).tolist(),
    }


def summarize_counts(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    if not len(counts):
        return {"episodes": 0, "total": 0, "min": None, "max": None, "mean": None}
    return {
        "episodes": int(len(counts)),
        "total": int(counts.sum()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
    }


def validate_output(path: Path, episodes: int, episode_length: int) -> dict:
    with h5py.File(path, "r") as dataset:
        expected_keys = set(CUBE_H5_COLUMNS) | {"ep_len", "ep_offset"}
        if set(dataset) != expected_keys:
            raise ValueError(
                f"Output schema differs from Cube: {sorted(set(dataset) ^ expected_keys)}"
            )
        lengths = dataset["ep_len"][:]
        offsets = dataset["ep_offset"][:]
        if not np.array_equal(
            lengths, np.full(episodes, episode_length, dtype=np.int32)
        ):
            raise ValueError("Output episode lengths are incorrect.")
        expected_offsets = np.arange(episodes, dtype=np.int64) * episode_length
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("Output episode offsets are incorrect.")
        terminal_indices = offsets + lengths - 1
        if not np.isnan(dataset["action"][terminal_indices]).all():
            raise ValueError("Every terminal CubeHop action must be NaN.")
        if not dataset["truncated"][terminal_indices].all():
            raise ValueError("Every CubeHop episode must end by time-limit truncation.")
        gripper_opening = dataset["proprio_gripper_opening"][:, 0]
        if float(gripper_opening.max()) >= 0.05:
            raise ValueError(
                "CubeHop gripper left the reference open-state range: "
                f"max opening={gripper_opening.max():.6g}."
            )
        return {
            "root_columns": sorted(dataset.keys()),
            "action_shape": list(dataset["action"].shape),
            "observation_shape": list(dataset["observation"].shape),
            "pixels_shape": list(dataset["pixels"].shape),
            "pixel_chunks": list(dataset["pixels"].chunks),
            "pixel_compression_filter": (
                32001 if "32001" in dataset["pixels"]._filters else None
            ),
            "gripper_opening_min": float(gripper_opening.min()),
            "gripper_opening_max": float(gripper_opening.max()),
            "gripper_opening_q99": float(np.quantile(gripper_opening, 0.99)),
        }


def main() -> None:
    args = parse_args()
    validate_args(args)
    episode_length = reference_episode_length(args.reference)
    action_steps = episode_length - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )

    action_support = CubeActionSupport(args.reference)
    world = swm.World(
        "anygoal/CubeHop-v0",
        num_envs=args.num_envs,
        image_shape=(args.resolution, args.resolution),
        max_episode_steps=action_steps,
        height=args.resolution,
        width=args.resolution,
    )
    policy = CubeHopPolicy(action_support=action_support, seed=args.seed)
    world.set_policy(policy)
    writer = CubeHDF5Writer(temporary_output)
    try:
        collection = collect_fixed_length_episodes(
            world,
            writer,
            episodes=args.episodes,
            episode_length=episode_length,
            seed=args.seed,
        )
    finally:
        world.close()

    initial_effector_positions = collection.pop("initial_effector_positions")
    initial_cube_positions = collection.pop("initial_cube_positions")
    sampled_gripper_targets = collection.pop("sampled_gripper_targets")
    sampled_waypoints = collection.pop("sampled_waypoints")
    completed_gripper_targets = collection.pop("completed_gripper_targets")
    completed_waypoints = collection.pop("completed_waypoints")
    episode_sides = collection.pop("episode_sides")
    initial_effector_positions_by_side = collection.pop(
        "initial_effector_positions_by_side"
    )
    initial_cube_positions_by_side = collection.pop("initial_cube_positions_by_side")
    sampled_gripper_targets_by_side = collection.pop("sampled_gripper_targets_by_side")
    sampled_waypoints_by_side = collection.pop("sampled_waypoints_by_side")
    completed_gripper_targets_by_side = collection.pop(
        "completed_gripper_targets_by_side"
    )
    completed_waypoints_by_side = collection.pop("completed_waypoints_by_side")
    expected_side_count = args.episodes // len(CUBE_HOP_SIDES)
    observed_side_counts = {side: episode_sides.count(side) for side in CUBE_HOP_SIDES}
    if any(count != expected_side_count for count in observed_side_counts.values()):
        raise RuntimeError(f"Unbalanced accepted episodes: {observed_side_counts}.")
    os.replace(temporary_output, args.output)
    output_validation = validate_output(args.output, args.episodes, episode_length)

    side_sampling = {}
    arrivals_by_side = {}
    for side in CUBE_HOP_SIDES:
        bounds = SIDE_SAMPLING_BOUNDS[side]
        side_sampling[side] = {
            "episodes": observed_side_counts[side],
            "initial_effector_observed": summarize_points(
                initial_effector_positions_by_side[side]
            ),
            "cube_bounds": [bounds["cube"][0].tolist(), bounds["cube"][1].tolist()],
            "cube_observed": summarize_points(initial_cube_positions_by_side[side]),
            "gripper_target_bounds": [
                bounds["gripper_target"][0].tolist(),
                bounds["gripper_target"][1].tolist(),
            ],
            "sampled_gripper_targets": summarize_points(
                sampled_gripper_targets_by_side[side]
            ),
            "waypoint_bounds": [
                bounds["waypoint"][0].tolist(),
                bounds["waypoint"][1].tolist(),
            ],
            "sampled_waypoints": summarize_points(sampled_waypoints_by_side[side]),
        }
        arrivals_by_side[side] = {
            "gripper_targets": summarize_counts(
                completed_gripper_targets_by_side[side]
            ),
            "waypoints": summarize_counts(completed_waypoints_by_side[side]),
        }

    metadata = {
        "dataset": str(args.output.resolve()),
        "reference_dataset": str(args.reference.resolve()),
        "seed": args.seed,
        "collection": collection,
        "uniform_sampling": {
            "initial_effector_bounds": [
                EE_START_LOW.tolist(),
                EE_START_HIGH.tolist(),
            ],
            "initial_effector_observed": summarize_points(initial_effector_positions),
            "cube_observed": summarize_points(initial_cube_positions),
            "sampled_gripper_targets": summarize_points(sampled_gripper_targets),
            "sampled_waypoints": summarize_points(sampled_waypoints),
            "side_sampling": side_sampling,
            "sampling_note": (
                "Every candidate point is sampled independently and uniformly; "
                "candidate episodes with robot-cube contact or measurable cube "
                "motion are rejected to keep accepted trajectories cube-free."
            ),
        },
        "completed_arrivals": {
            "gripper_targets": summarize_counts(completed_gripper_targets),
            "waypoints": summarize_counts(completed_waypoints),
            "by_side": arrivals_by_side,
        },
        "side_assignment": {
            "policy": "exactly episodes // 2 accepted episodes per side",
            "accepted_counts": observed_side_counts,
            "episode_sides": episode_sides,
        },
        "action_policy": {
            "translation_xyz_l2_cap_normalized": MAX_TRANSLATION_ACTION_NORM,
            "translation_l2_cap_meters": 0.05 * MAX_TRANSLATION_ACTION_NORM,
            "clockwise_yaw_delta_radians": RIGHT_YAW_DELTA,
            "per_step_yaw_action_cap_normalized": MAX_YAW_ACTION,
            "per_step_yaw_cap_radians": 0.3 * MAX_YAW_ACTION,
            "gripper_action": (
                "-proprio_gripper_opening each step, which tracks absolute "
                "fully-open position 0"
            ),
            "training_support": action_support.metadata(),
        },
        "output_validation": output_validation,
    }
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.stem}.tmp-{os.getpid()}{metadata_output.suffix}"
    )
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_output)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Dataset: {args.output}", flush=True)
    print(f"Metadata: {metadata_output}", flush=True)


if __name__ == "__main__":
    main()
