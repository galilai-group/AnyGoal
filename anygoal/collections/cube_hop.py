"""Collect fixed-length OGBench CubeHop expert trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm

from anygoal.datasets.cube.support import (
    CUBE_H5_COLUMNS,
    CubeActionSupport,
    CubeHDF5Writer,
    collect_fixed_length_episodes,
    reference_episode_length,
)
from anygoal.envs.cube_hop import env as cube_hop_env
from anygoal.envs.cube_hop import policy as cube_hop_policy
from anygoal.paths import data_root

DATASET_DIR = data_root()
DEFAULT_REFERENCE = DATASET_DIR / "cube_single_expert.h5"
DEFAULT_OUTPUT = DATASET_DIR / "cube_hop_test.h5"


def _validate_options(
    episodes: int,
    num_parallel: int,
    resolution: int,
    reference: Path,
    output: Path,
    metadata_output: Path,
    overwrite: bool,
) -> None:
    if episodes <= 0 or episodes % len(cube_hop_env.CUBE_HOP_SIDES) != 0:
        raise ValueError(
            "episodes must be positive and divisible by "
            f"{len(cube_hop_env.CUBE_HOP_SIDES)} "
            "to collect equal right/left halves."
        )
    if num_parallel <= 0 or num_parallel > episodes:
        raise ValueError("num-parallel must be in [1, episodes].")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5.")
    for path in (output, metadata_output):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")


def _summarize_points(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(points),
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "mean": points.mean(axis=0).tolist(),
    }


def _summarize_counts(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    if not len(counts):
        return {"episodes": 0, "total": 0, "min": None, "max": None, "mean": None}
    return {
        "episodes": len(counts),
        "total": int(counts.sum()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
    }


def _validate_dataset(path: Path, episodes: int, episode_length: int) -> dict:
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


def collect_dataset(
    *,
    episodes: int = 400,
    num_parallel: int = 8,
    resolution: int = 224,
    seed: int = 0,
    reference: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_OUTPUT,
    metadata_output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    metadata_output = metadata_output or output.with_suffix(".metadata.json")
    _validate_options(
        episodes,
        num_parallel,
        resolution,
        reference,
        output,
        metadata_output,
        overwrite,
    )
    episode_length = reference_episode_length(reference)
    action_steps = episode_length - 1
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )

    action_support = CubeActionSupport(reference)
    world = swm.World(
        "anygoal/CubeHop-v0",
        num_envs=num_parallel,
        image_shape=(resolution, resolution),
        max_episode_steps=action_steps,
        height=resolution,
        width=resolution,
    )
    policy = cube_hop_policy.CubeHopPolicy(action_support=action_support, seed=seed)
    world.set_policy(policy)
    writer = CubeHDF5Writer(temporary_output)
    try:
        collection = collect_fixed_length_episodes(
            world,
            writer,
            episodes=episodes,
            episode_length=episode_length,
            seed=seed,
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
    expected_side_count = episodes // len(cube_hop_env.CUBE_HOP_SIDES)
    observed_side_counts = {
        side: episode_sides.count(side) for side in cube_hop_env.CUBE_HOP_SIDES
    }
    if any(count != expected_side_count for count in observed_side_counts.values()):
        raise RuntimeError(f"Unbalanced accepted episodes: {observed_side_counts}.")
    os.replace(temporary_output, output)
    output_validation = _validate_dataset(output, episodes, episode_length)

    side_sampling = {}
    arrivals_by_side = {}
    for side in cube_hop_env.CUBE_HOP_SIDES:
        bounds = cube_hop_env.SIDE_SAMPLING_BOUNDS[side]
        side_sampling[side] = {
            "episodes": observed_side_counts[side],
            "initial_effector_observed": _summarize_points(
                initial_effector_positions_by_side[side]
            ),
            "cube_bounds": [bounds["cube"][0].tolist(), bounds["cube"][1].tolist()],
            "cube_observed": _summarize_points(initial_cube_positions_by_side[side]),
            "gripper_target_bounds": [
                bounds["gripper_target"][0].tolist(),
                bounds["gripper_target"][1].tolist(),
            ],
            "sampled_gripper_targets": _summarize_points(
                sampled_gripper_targets_by_side[side]
            ),
            "waypoint_bounds": [
                bounds["waypoint"][0].tolist(),
                bounds["waypoint"][1].tolist(),
            ],
            "sampled_waypoints": _summarize_points(sampled_waypoints_by_side[side]),
        }
        arrivals_by_side[side] = {
            "gripper_targets": _summarize_counts(
                completed_gripper_targets_by_side[side]
            ),
            "waypoints": _summarize_counts(completed_waypoints_by_side[side]),
        }

    metadata = {
        "dataset": str(output.resolve()),
        "reference_dataset": str(reference.resolve()),
        "seed": seed,
        "collection": collection,
        "uniform_sampling": {
            "initial_effector_bounds": [
                cube_hop_env.EE_START_LOW.tolist(),
                cube_hop_env.EE_START_HIGH.tolist(),
            ],
            "initial_effector_observed": _summarize_points(initial_effector_positions),
            "cube_observed": _summarize_points(initial_cube_positions),
            "sampled_gripper_targets": _summarize_points(sampled_gripper_targets),
            "sampled_waypoints": _summarize_points(sampled_waypoints),
            "side_sampling": side_sampling,
            "sampling_note": (
                "Every candidate point is sampled independently and uniformly; "
                "candidate episodes with robot-cube contact or measurable cube "
                "motion are rejected to keep accepted trajectories cube-free."
            ),
        },
        "completed_arrivals": {
            "gripper_targets": _summarize_counts(completed_gripper_targets),
            "waypoints": _summarize_counts(completed_waypoints),
            "by_side": arrivals_by_side,
        },
        "side_assignment": {
            "policy": "exactly episodes // 2 accepted episodes per side",
            "accepted_counts": observed_side_counts,
            "episode_sides": episode_sides,
        },
        "action_policy": {
            "translation_xyz_l2_cap_normalized": cube_hop_env.MAX_TRANSLATION_ACTION_NORM,
            "translation_l2_cap_meters": 0.05
            * cube_hop_env.MAX_TRANSLATION_ACTION_NORM,
            "clockwise_yaw_delta_radians": cube_hop_env.RIGHT_YAW_DELTA,
            "per_step_yaw_action_cap_normalized": cube_hop_env.MAX_YAW_ACTION,
            "per_step_yaw_cap_radians": 0.3 * cube_hop_env.MAX_YAW_ACTION,
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
    print(f"Dataset: {output}", flush=True)
    print(f"Metadata: {metadata_output}", flush=True)
    return output
