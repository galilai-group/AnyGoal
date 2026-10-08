"""Collect original-schema Cube trajectories with diagonal edge grasps."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anygoal  # noqa: F401, E402
from anygoal.envs.cube_diagonal import (  # noqa: E402
    CUBE_SIDE_LENGTH,
    DIAGONAL_CLOSED_GRASP,
    DIAGONAL_GRASP_OFFSET,
    EXPECTED_PAD_GAP,
    CubeDiagonalPolicy,
)
from anygoal.datasets.cube_support import (  # noqa: E402
    CUBE_H5_COLUMNS,
    SOURCE_COLUMNS,
    CubeHDF5Writer,
    flattened_column,
    reference_episode_length,
)
from anygoal.paths import data_root  # noqa: E402


DEFAULT_REFERENCE = data_root() / "cube_single_expert.h5"
DEFAULT_OUTPUT = data_root() / "cube_diagonal_test.h5"
DEFAULT_GIF_DIR = REPO_ROOT / "outputs" / "cube_diagonal"
MAX_CANDIDATE_MULTIPLIER = 20


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--num-parallel", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=10065)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--gif-dir", type=Path, default=DEFAULT_GIF_DIR)
    parser.add_argument("--gif-episodes", type=int, default=0)
    parser.add_argument("--gif-prefix", default="cube_diagonal")
    parser.add_argument("--gif-duration-ms", type=int, default=50)
    parser.add_argument("--cube-yaw-center-deg", type=float, default=85.0)
    parser.add_argument("--cube-yaw-half-width-deg", type=float, default=5.0)
    parser.add_argument(
        "--require-diagonal-lift",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-lifted-targets", type=int, default=2)
    parser.add_argument("--min-completed-tasks", type=int, default=0)
    parser.add_argument("--max-abs-block-quat-xy", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def validate_args(args: argparse.Namespace, metadata_output: Path) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if args.num_parallel <= 0 or args.num_parallel > args.episodes:
        raise ValueError("num-parallel must be in [1, episodes].")
    if args.resolution <= 0 or args.gif_duration_ms <= 0:
        raise ValueError("resolution and gif-duration-ms must be positive.")
    if args.gif_episodes < 0 or args.gif_episodes > args.episodes:
        raise ValueError("gif-episodes must be in [0, episodes].")
    if args.min_completed_tasks < 0:
        raise ValueError("min-completed-tasks must be nonnegative.")
    if args.min_lifted_targets < 0:
        raise ValueError("min-lifted-targets must be nonnegative.")
    if (
        args.max_abs_block_quat_xy is not None
        and args.max_abs_block_quat_xy <= 0.0
    ):
        raise ValueError("max-abs-block-quat-xy must be positive.")
    if args.cube_yaw_half_width_deg < 0.0:
        raise ValueError("cube-yaw-half-width-deg must be nonnegative.")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.output.suffix not in (".h5", ".hdf5"):
        raise ValueError("output must end in .h5 or .hdf5.")
    expected_gifs = [
        args.gif_dir / f"{args.gif_prefix}_episode_{index}.gif"
        for index in range(args.gif_episodes)
    ]
    if args.gif_episodes:
        expected_gifs.append(
            args.gif_dir / f"{args.gif_prefix}_{args.gif_episodes}_episodes.gif"
        )
    for path in [args.output, metadata_output, *expected_gifs]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")


def info_value(info_dict: dict, source: str, env_idx: int):
    value = info_dict[source]
    if isinstance(value, torch.Tensor):
        return value[env_idx, 0].detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        selected = value[env_idx, 0]
        return selected.copy() if isinstance(selected, np.ndarray) else selected
    return value[env_idx][0]


def append_frame(
    buffers: list[defaultdict[str, list]],
    info_dict: dict,
    actions: np.ndarray,
) -> None:
    for env_idx, buffer in enumerate(buffers):
        buffer["action"].append(actions[env_idx].copy())
        for source in SOURCE_COLUMNS:
            buffer[flattened_column(source)].append(
                info_value(info_dict, source, env_idx)
            )


def collect_episodes(
    world,
    writer: CubeHDF5Writer,
    episodes: int,
    episode_length: int,
    seed: int,
    require_diagonal_lift: bool,
    min_lifted_targets: int,
    min_completed_tasks: int,
    max_abs_block_quat_xy: float | None,
    gif_episode_count: int,
) -> tuple[list[dict], list[dict], list[np.ndarray], int]:
    action_steps = episode_length - 1
    summaries: list[dict] = []
    candidate_summaries: list[dict] = []
    episode_frames: list[np.ndarray] = []
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
                append_frame(buffers, world.infos, actions)
                _, world.rewards, world.terminateds, world.truncateds, world.infos = (
                    world.envs.step(actions)
                )
                if step < action_steps - 1 and (
                    world.terminateds.any() or world.truncateds.any()
                ):
                    raise RuntimeError(f"Episode ended early at action step {step + 1}.")

            if not np.all(world.truncateds):
                raise RuntimeError("Batch did not end at the reference time limit.")
            append_frame(
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
                if require_diagonal_lift and not summary[
                    "lifted_with_opposite_edge_grasp"
                ]:
                    continue
                if len(summary["targets_with_diagonal_lift"]) < min_lifted_targets:
                    continue
                if summary["completed_diagonal_tasks"] < min_completed_tasks:
                    continue
                if (
                    max_abs_block_quat_xy is not None
                    and summary["max_abs_block_quat_xy"]
                    > max_abs_block_quat_xy
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
                if len(episode_frames) < gif_episode_count:
                    episode_frames.append(
                        np.asarray(buffer["pixels"], dtype=np.uint8)
                    )
                accepted += 1
                progress.update(1)
            progress.set_postfix(simulated=simulated)
    return summaries, candidate_summaries, episode_frames, simulated


def save_gif(
    frames: np.ndarray,
    output: Path,
    duration_ms: int,
    label: str | None = None,
) -> None:
    images = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        if label is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 92, 20), fill="black")
            draw.text((5, 4), label, fill="white")
        images.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def save_gifs(
    episode_frames: list[np.ndarray],
    gif_dir: Path,
    duration_ms: int,
    prefix: str,
) -> list[Path]:
    outputs = []
    for episode_index, frames in enumerate(episode_frames):
        output = gif_dir / f"{prefix}_episode_{episode_index}.gif"
        save_gif(frames, output, duration_ms, label=f"episode {episode_index}")
        outputs.append(output)

    columns = min(5, len(episode_frames))
    rows = int(np.ceil(len(episode_frames) / columns))
    frame_height, frame_width = episode_frames[0].shape[1:3]
    combined_frames = []
    for step in range(max(map(len, episode_frames))):
        canvas = np.zeros(
            (rows * frame_height, columns * frame_width, 3), dtype=np.uint8
        )
        for episode_index, frames in enumerate(episode_frames):
            row, column = divmod(episode_index, columns)
            canvas[
                row * frame_height : (row + 1) * frame_height,
                column * frame_width : (column + 1) * frame_width,
            ] = frames[min(step, len(frames) - 1)]
        combined_frames.append(canvas)
    combined_output = gif_dir / f"{prefix}_{len(episode_frames)}_episodes.gif"
    save_gif(
        np.asarray(combined_frames),
        combined_output,
        duration_ms,
        label="repeated diagonal tasks",
    )
    outputs.append(combined_output)
    return outputs


def validate_output(path: Path, episodes: int, episode_length: int) -> dict:
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


def run(args: argparse.Namespace) -> Path:
    metadata_output = args.metadata_output or args.output.with_suffix(
        ".metadata.json"
    )
    validate_args(args, metadata_output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_length = reference_episode_length(args.reference)
    action_steps = episode_length - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )

    world = swm.World(
        "anygoal/CubeDiagonal-v0",
        num_envs=args.num_parallel,
        image_shape=(args.resolution, args.resolution),
        max_episode_steps=action_steps,
        env_type="single",
        ob_type="states",
        mode="data_collection",
        terminate_at_goal=False,
        visualize_info=False,
        height=args.resolution,
        width=args.resolution,
        cube_yaw_center=np.deg2rad(args.cube_yaw_center_deg),
        cube_yaw_half_width=np.deg2rad(args.cube_yaw_half_width_deg),
    )
    policy = CubeDiagonalPolicy(seed=args.seed)
    world.set_policy(policy)
    try:
        summaries, candidate_summaries, frames, simulated = collect_episodes(
            world,
            CubeHDF5Writer(temporary_output),
            episodes=args.episodes,
            episode_length=episode_length,
            seed=args.seed,
            require_diagonal_lift=args.require_diagonal_lift,
            min_lifted_targets=args.min_lifted_targets,
            min_completed_tasks=args.min_completed_tasks,
            max_abs_block_quat_xy=args.max_abs_block_quat_xy,
            gif_episode_count=args.gif_episodes,
        )
    finally:
        world.close()

    os.replace(temporary_output, args.output)
    gif_outputs = (
        save_gifs(frames, args.gif_dir, args.gif_duration_ms, args.gif_prefix)
        if frames
        else []
    )
    metadata = {
        "dataset": str(args.output.resolve()),
        "reference_dataset": str(args.reference.resolve()),
        "environment_id": "anygoal/CubeDiagonal-v0",
        "episodes": args.episodes,
        "simulated_candidate_episodes": simulated,
        "evaluated_candidate_episodes": len(candidate_summaries),
        "required_diagonal_lift": args.require_diagonal_lift,
        "minimum_lifted_targets": args.min_lifted_targets,
        "minimum_completed_tasks": args.min_completed_tasks,
        "maximum_abs_block_quat_xy": args.max_abs_block_quat_xy,
        "seed": args.seed,
        "seed_scope": (
            "Python, NumPy global, Torch, policy, variation space, and per-episode env reset"
        ),
        "num_parallel": args.num_parallel,
        "episode_length": episode_length,
        "cube_side_length_m": CUBE_SIDE_LENGTH,
        "grasp_yaw_offset_rad": DIAGONAL_GRASP_OFFSET,
        "closed_grasp_target": DIAGONAL_CLOSED_GRASP,
        "expected_pad_gap_m": EXPECTED_PAD_GAP,
        "cube_yaw_center_deg": args.cube_yaw_center_deg,
        "cube_yaw_half_width_deg": args.cube_yaw_half_width_deg,
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
            sum(
                item["lifted_with_opposite_edge_grasp"]
                for item in candidate_summaries
            )
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
        "gif_outputs": [str(path.resolve()) for path in gif_outputs],
        "output_validation": validate_output(
            args.output, args.episodes, episode_length
        ),
    }
    temporary_metadata = metadata_output.with_name(
        f".{metadata_output.stem}.tmp-{os.getpid()}{metadata_output.suffix}"
    )
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_output)
    print(json.dumps(metadata, indent=2), flush=True)


    return args.output
