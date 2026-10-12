"""Collect original-schema Cube trajectories with diagonal edge grasps."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from tqdm import tqdm

from anygoal.datasets.cube.support import (
    CUBE_H5_COLUMNS,
    SOURCE_COLUMNS,
    CubeHDF5Writer,
    flattened_column,
    reference_episode_length,
)
from anygoal.envs.cube_diagonal import env as cube_diagonal_env
from anygoal.envs.cube_diagonal import policy as cube_diagonal_policy
from anygoal.paths import data_root

DEFAULT_REFERENCE = data_root() / "cube_single_expert.h5"
DEFAULT_OUTPUT = data_root() / "cube_diagonal_test.h5"
MAX_CANDIDATE_MULTIPLIER = 20


def _validate_options(
    episodes: int,
    num_parallel: int,
    resolution: int,
    reference: Path,
    output: Path,
    metadata_output: Path,
    cube_yaw_half_width_deg: float,
    min_lifted_targets: int,
    min_completed_tasks: int,
    max_abs_block_quat_xy: float | None,
    overwrite: bool,
) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if num_parallel <= 0 or num_parallel > episodes:
        raise ValueError("num-parallel must be in [1, episodes].")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if min_completed_tasks < 0:
        raise ValueError("min-completed-tasks must be nonnegative.")
    if min_lifted_targets < 0:
        raise ValueError("min-lifted-targets must be nonnegative.")
    if max_abs_block_quat_xy is not None and max_abs_block_quat_xy <= 0.0:
        raise ValueError("max-abs-block-quat-xy must be positive.")
    if cube_yaw_half_width_deg < 0.0:
        raise ValueError("cube-yaw-half-width-deg must be nonnegative.")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5.")
    for path in (output, metadata_output):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")


def _info_value(info_dict: dict, source: str, env_idx: int):
    value = info_dict[source]
    if isinstance(value, torch.Tensor):
        return value[env_idx, 0].detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        selected = value[env_idx, 0]
        return selected.copy() if isinstance(selected, np.ndarray) else selected
    return value[env_idx][0]


def _append_frame(
    buffers: list[defaultdict[str, list]],
    info_dict: dict,
    actions: np.ndarray,
) -> None:
    for env_idx, buffer in enumerate(buffers):
        buffer["action"].append(actions[env_idx].copy())
        for source in SOURCE_COLUMNS:
            buffer[flattened_column(source)].append(
                _info_value(info_dict, source, env_idx)
            )


def _collect_episodes(
    world,
    writer: CubeHDF5Writer,
    episodes: int,
    episode_length: int,
    seed: int,
    require_diagonal_lift: bool,
    min_lifted_targets: int,
    min_completed_tasks: int,
    max_abs_block_quat_xy: float | None,
) -> tuple[list[dict], list[dict], int]:
    action_steps = episode_length - 1
    summaries: list[dict] = []
    candidate_summaries: list[dict] = []
    accepted = 0
    simulated = 0
    max_candidates = episodes * MAX_CANDIDATE_MULTIPLIER

    with (
        writer as active_writer,
        tqdm(total=episodes, desc="CubeDiagonal episodes") as progress,
    ):
        while accepted < episodes:
            if simulated >= max_candidates:
                raise RuntimeError(
                    f"Accepted only {accepted}/{episodes} episodes after "
                    f"{simulated} candidate episodes."
                )
            active_envs = min(world.num_envs, max_candidates - simulated)
            batch_seeds = [seed + simulated + idx for idx in range(world.num_envs)]
            simulated += active_envs
            world.reset(seed=batch_seeds)
            buffers = [defaultdict(list) for _ in range(world.num_envs)]

            for step in range(action_steps):
                actions = world._get_actions()
                _append_frame(buffers, world.infos, actions)
                _, world.rewards, world.terminateds, world.truncateds, world.infos = (
                    world.envs.step(actions)
                )
                if step < action_steps - 1 and (
                    world.terminateds.any() or world.truncateds.any()
                ):
                    raise RuntimeError(
                        f"Episode ended early at action step {step + 1}."
                    )

            if not np.all(world.truncateds):
                raise RuntimeError("Batch did not end at the reference time limit.")
            _append_frame(
                buffers,
                world.infos,
                np.full((world.num_envs, 5), np.nan, dtype=np.float32),
            )

            for env_idx, buffer in enumerate(buffers[:active_envs]):
                summary = world.policy.episode_summary(env_idx)
                block_quaternion = np.asarray(
                    buffer["privileged_block_0_quat"], dtype=np.float64
                )
                summary["max_abs_block_quat_xy"] = float(
                    np.abs(block_quaternion[:, 1:3]).max()
                )
                candidate_summaries.append(summary)
                if (
                    require_diagonal_lift
                    and not summary["lifted_with_opposite_edge_grasp"]
                ):
                    continue
                if len(summary["targets_with_diagonal_lift"]) < min_lifted_targets:
                    continue
                if summary["completed_diagonal_tasks"] < min_completed_tasks:
                    continue
                if (
                    max_abs_block_quat_xy is not None
                    and summary["max_abs_block_quat_xy"] > max_abs_block_quat_xy
                ):
                    continue
                if accepted >= episodes:
                    break
                steps = np.asarray(buffer["step_idx"], dtype=np.int64)
                if not np.array_equal(steps, np.arange(episode_length)):
                    raise RuntimeError(f"Invalid step alignment: {steps.tolist()}")
                buffer["ep_idx"] = [np.int32(accepted)] * episode_length
                active_writer.write_episode(dict(buffer))
                summaries.append(summary)
                accepted += 1
                progress.update(1)
            progress.set_postfix(simulated=simulated)
    return summaries, candidate_summaries, simulated


def _validate_dataset(path: Path, episodes: int, episode_length: int) -> dict:
    with h5py.File(path, "r") as dataset:
        expected_keys = set(CUBE_H5_COLUMNS) | {"ep_len", "ep_offset"}
        if set(dataset) != expected_keys:
            raise ValueError("Output schema differs from cube_single_expert.")
        lengths = dataset["ep_len"][:]
        offsets = dataset["ep_offset"][:]
        terminal = offsets + lengths - 1
        if not np.array_equal(lengths, np.full(episodes, episode_length)):
            raise ValueError("Output episode lengths are incorrect.")
        if not np.isnan(dataset["action"][terminal]).all():
            raise ValueError("Terminal actions must be NaN.")
        if not dataset["truncated"][terminal].all():
            raise ValueError("Episodes must end by time-limit truncation.")
        tasks = dataset["privileged_target_task"][:].astype(str)
        if not np.all(tasks == "cube_diagonal"):
            raise ValueError("Task labels are not cube_diagonal.")
        return {
            "root_columns": sorted(dataset.keys()),
            "episode_lengths": np.unique(lengths).astype(int).tolist(),
            "frames": int(lengths.sum()),
            "action_shape": list(dataset["action"].shape),
            "observation_shape": list(dataset["observation"].shape),
            "pixels_shape": list(dataset["pixels"].shape),
            "pixels_blosc_filter": "32001" in dataset["pixels"]._filters,
        }


def collect_dataset(
    *,
    episodes: int = 400,
    num_parallel: int = 5,
    resolution: int = 224,
    seed: int = 10065,
    reference: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_OUTPUT,
    metadata_output: Path | None = None,
    cube_yaw_center_deg: float = 85.0,
    cube_yaw_half_width_deg: float = 5.0,
    require_diagonal_lift: bool = True,
    min_lifted_targets: int = 2,
    min_completed_tasks: int = 0,
    max_abs_block_quat_xy: float | None = 0.2,
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
        cube_yaw_half_width_deg,
        min_lifted_targets,
        min_completed_tasks,
        max_abs_block_quat_xy,
        overwrite,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    episode_length = reference_episode_length(reference)
    action_steps = episode_length - 1
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )

    world = swm.World(
        "anygoal/CubeDiagonal-v0",
        num_envs=num_parallel,
        image_shape=(resolution, resolution),
        max_episode_steps=action_steps,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
        height=resolution,
        width=resolution,
        cube_yaw_center=np.deg2rad(cube_yaw_center_deg),
        cube_yaw_half_width=np.deg2rad(cube_yaw_half_width_deg),
    )
    policy = cube_diagonal_policy.CubeDiagonalPolicy(seed=seed)
    world.set_policy(policy)
    try:
        summaries, candidate_summaries, simulated = _collect_episodes(
            world,
            CubeHDF5Writer(temporary_output),
            episodes=episodes,
            episode_length=episode_length,
            seed=seed,
            require_diagonal_lift=require_diagonal_lift,
            min_lifted_targets=min_lifted_targets,
            min_completed_tasks=min_completed_tasks,
            max_abs_block_quat_xy=max_abs_block_quat_xy,
        )
    finally:
        world.close()

    os.replace(temporary_output, output)
    metadata = {
        "dataset": str(output.resolve()),
        "reference_dataset": str(reference.resolve()),
        "environment_id": "anygoal/CubeDiagonal-v0",
        "episodes": episodes,
        "simulated_candidate_episodes": simulated,
        "evaluated_candidate_episodes": len(candidate_summaries),
        "required_diagonal_lift": require_diagonal_lift,
        "minimum_lifted_targets": min_lifted_targets,
        "minimum_completed_tasks": min_completed_tasks,
        "maximum_abs_block_quat_xy": max_abs_block_quat_xy,
        "seed": seed,
        "seed_scope": (
            "Python, NumPy global, Torch, policy, variation space, and per-episode env reset"
        ),
        "num_parallel": num_parallel,
        "episode_length": episode_length,
        "cube_side_length_m": cube_diagonal_env.CUBE_SIDE_LENGTH,
        "grasp_yaw_offset_rad": cube_diagonal_env.DIAGONAL_GRASP_OFFSET,
        "closed_grasp_target": cube_diagonal_env.DIAGONAL_CLOSED_GRASP,
        "expected_pad_gap_m": cube_diagonal_env.EXPECTED_PAD_GAP,
        "cube_yaw_center_deg": cube_yaw_center_deg,
        "cube_yaw_half_width_deg": cube_yaw_half_width_deg,
        "policy_change": (
            "CubePlanOracle pick/place yaw is cube yaw + pi/4. The closed "
            "grasp target is calibrated to 0.83 for the larger diagonal span; "
            "cube start/goal yaw uses the configured support-preserving band. "
            "Keyframes, waypoint, timing, fixed target, physics, action noise, "
            "and other task settings are retained."
        ),
        "episode_summaries": summaries,
        "candidate_episode_summaries": candidate_summaries,
        "candidate_diagonal_lifts": int(
            sum(item["lifted_with_opposite_edge_grasp"] for item in candidate_summaries)
        ),
        "candidate_diagonal_lift_rate": float(
            np.mean(
                [
                    item["lifted_with_opposite_edge_grasp"]
                    for item in candidate_summaries
                ]
            )
        ),
        "successful_diagonal_lifts": int(
            sum(item["lifted_with_opposite_edge_grasp"] for item in summaries)
        ),
        "output_validation": _validate_dataset(output, episodes, episode_length),
    }
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.stem}.tmp-{os.getpid()}{metadata_output.suffix}"
    )
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_output)
    print(json.dumps(metadata, indent=2), flush=True)

    return output
