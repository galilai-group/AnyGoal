"""Collect and write the generated TwoRoom triangle dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from anygoal.envs.tworoom_triangle import (
    TRIANGLE_CONDITIONS,
    TRIANGLE_POINTS,
    TRIANGLE_ROUTES,
    TwoRoomTriangleEnv,
    TwoRoomTrianglePolicy,
)


CONDITION_COUNTS = {
    "clean_constant": 20,
    "noisy_constant": 20,
    "clean_varying": 10,
}
SMOKE_CONDITION_COUNTS = {condition: 1 for condition in CONDITION_COUNTS}
CONDITION_IDS = {
    condition: index for index, condition in enumerate(TRIANGLE_CONDITIONS)
}
MAX_EPISODE_STEPS = 120
LOGGER = logging.getLogger("tworoom_triangle")


@dataclass
class Episode:
    columns: dict[str, np.ndarray]
    frames: list[np.ndarray]
    route_id: int
    condition: str
    sampled_magnitude: float

    @property
    def length(self) -> int:
        return int(self.columns["action"].shape[0])


class TriangleHDF5Writer:
    """Append triangle episodes without holding all rendered frames in memory."""

    PER_STEP_SPECS = {
        "pixels": ((224, 224, 3), np.uint8),
        "action": ((2,), np.float32),
        "proprio": ((2,), np.float32),
        "next_proprio": ((2,), np.float32),
        "observation": ((10,), np.float32),
        "next_observation": ((10,), np.float32),
        "target": ((2,), np.float32),
        "waypoint": ((2,), np.float32),
        "action_magnitude": ((), np.float32),
        "route_id": ((), np.int16),
        "condition_id": ((), np.int8),
        "ep_idx": ((), np.int32),
        "waypoint_index": ((), np.int8),
        "step_idx": ((), np.int32),
        "terminated": ((), np.bool_),
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = h5py.File(path, "w")
        self.datasets = {}
        for key, (tail_shape, dtype) in self.PER_STEP_SPECS.items():
            kwargs = {"compression": "gzip", "compression_opts": 1} if key == "pixels" else {}
            self.datasets[key] = self.file.create_dataset(
                key,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                chunks=(1, *tail_shape) if key == "pixels" else True,
                dtype=dtype,
                **kwargs,
            )
        self.ep_len = self.file.create_dataset(
            "ep_len", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self.ep_offset = self.file.create_dataset(
            "ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self.num_steps = 0
        self.num_episodes = 0

    def write(self, episode: Episode) -> None:
        length = episode.length
        for key, dataset in self.datasets.items():
            values = (
                np.full(length, self.num_episodes, dtype=np.int32)
                if key == "ep_idx"
                else episode.columns[key]
            )
            dataset.resize(self.num_steps + length, axis=0)
            dataset[self.num_steps : self.num_steps + length] = values
        self.ep_len.resize(self.num_episodes + 1, axis=0)
        self.ep_offset.resize(self.num_episodes + 1, axis=0)
        self.ep_len[self.num_episodes] = length
        self.ep_offset[self.num_episodes] = self.num_steps
        self.num_steps += length
        self.num_episodes += 1

    def close(self, metadata: dict) -> None:
        for key, value in metadata.items():
            self.file.attrs[key] = value
        self.file.close()


def add_episode_index(path: Path) -> None:
    """Add the per-step episode index required by planning to an older file."""
    with h5py.File(path, "a") as dataset:
        dataset.attrs["environment"] = "anygoal/TwoRoom-Tri-v1"
        if "ep_idx" in dataset:
            return
        lengths = np.asarray(dataset["ep_len"][:], dtype=np.int64)
        episode_indices = np.repeat(
            np.arange(len(lengths), dtype=np.int32), lengths
        )
        if len(episode_indices) != len(dataset["action"]):
            raise ValueError("ep_len does not cover all action rows")
        dataset.create_dataset(
            "ep_idx",
            data=episode_indices,
            maxshape=(None,),
            chunks=True,
        )


def collect_episode(
    route_id: int,
    condition: str,
    rng: np.random.Generator,
    max_steps: int,
    capture_frames: bool,
) -> Episode:
    route = np.asarray(
        [TRIANGLE_POINTS[index] for index in TRIANGLE_ROUTES[route_id]],
        dtype=np.float32,
    )
    env = TwoRoomTriangleEnv(render_mode="rgb_array")
    policy = TwoRoomTrianglePolicy(route_id, condition, rng=rng)
    observation, _ = env.reset(
        seed=int(rng.integers(0, 2**31 - 1)),
        options={"variation": [], "state": route[0], "target_state": route[2]},
    )
    columns = {key: [] for key in TriangleHDF5Writer.PER_STEP_SPECS}
    frames = [env.render()] if capture_frames else []

    for step_index in range(max_steps):
        position = env.agent_position.detach().cpu().numpy().copy()
        current_frame = env.render()
        action, waypoint_index, magnitude = policy.action(position)
        waypoint = route[waypoint_index]
        next_observation, _, _, _, info = env.step(action)
        next_position = np.asarray(info["state"], dtype=np.float32)
        route_complete = (
            waypoint_index == 2
            and np.linalg.norm(next_position - route[2]) <= policy.waypoint_tolerance
        )
        columns["pixels"].append(current_frame)
        columns["action"].append(action)
        columns["proprio"].append(position)
        columns["next_proprio"].append(next_position)
        columns["observation"].append(np.asarray(observation, dtype=np.float32))
        columns["next_observation"].append(np.asarray(next_observation, dtype=np.float32))
        columns["target"].append(route[2])
        columns["waypoint"].append(waypoint)
        columns["action_magnitude"].append(np.float32(np.linalg.norm(action)))
        columns["route_id"].append(np.int16(route_id))
        columns["condition_id"].append(np.int8(CONDITION_IDS[condition]))
        columns["ep_idx"].append(np.int32(-1))
        columns["waypoint_index"].append(np.int8(waypoint_index))
        columns["step_idx"].append(np.int32(step_index))
        columns["terminated"].append(np.bool_(route_complete))
        observation = next_observation
        if capture_frames:
            frames.append(env.render())
        if route_complete:
            break

    env.close()
    arrays = {
        key: np.asarray(values, dtype=TriangleHDF5Writer.PER_STEP_SPECS[key][1])
        for key, values in columns.items()
    }
    return Episode(arrays, frames, route_id, condition, policy.sampled_magnitude)


def annotate_frame(frame: np.ndarray, episode: Episode, tile_size: int) -> np.ndarray:
    image = Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    label = {
        "clean_constant": "C",
        "noisy_constant": "N",
        "clean_varying": "V",
    }[episode.condition]
    draw.rectangle((0, 0, 54, 18), fill=(96, 96, 96))
    draw.text((3, 3), f"R{episode.route_id} {label}", fill=(255, 255, 255))
    return np.asarray(image)


def save_smoke_gif(episodes: list[Episode], output: Path, fps: int = 10) -> None:
    if len(episodes) != 24:
        raise ValueError(f"Expected 24 smoke episodes, got {len(episodes)}")
    tile_size = 160
    grid_height, grid_width = 4 * tile_size, 6 * tile_size
    square_size = grid_width
    y_offset = (square_size - grid_height) // 2
    max_frames = max(len(episode.frames) for episode in episodes)
    gif_frames = []
    for frame_index in range(max_frames):
        canvas = np.full((square_size, square_size, 3), 128, dtype=np.uint8)
        for episode_index, episode in enumerate(episodes):
            row, column = divmod(episode_index, 6)
            frame = episode.frames[min(frame_index, len(episode.frames) - 1)]
            tile = annotate_frame(frame, episode, tile_size)
            y = y_offset + row * tile_size
            x = column * tile_size
            canvas[y : y + tile_size, x : x + tile_size] = tile
        gif_frames.append(canvas)
    imageio.mimsave(output, gif_frames, duration=1000 / fps, loop=0)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-steps", type=int, default=26)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / ".stable_worldmodel"
        / "datasets"
        / "tworoom_route_test.h5",
    )
    parser.add_argument("--gif", type=Path, default=Path("tworoom_triangle.gif"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def run(args: argparse.Namespace) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    counts = SMOKE_CONDITION_COUNTS if args.smoke else CONDITION_COUNTS
    expected_episodes = len(TRIANGLE_ROUTES) * sum(counts.values())
    if args.min_steps <= 0 or args.max_steps < args.min_steps:
        raise ValueError("Require 0 < min-steps <= max-steps")
    if args.max_steps > MAX_EPISODE_STEPS:
        raise ValueError(
            f"max-steps cannot exceed the dataset limit of {MAX_EPISODE_STEPS}"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.smoke and args.gif.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.gif}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp-{os.getpid()}{args.output.suffix}"
    )
    rng = np.random.default_rng(args.seed)
    writer = TriangleHDF5Writer(temporary_output)
    accepted: list[Episode] = []
    attempts_by_route = {route_id: 0 for route_id in TRIANGLE_ROUTES}
    rejected_short = 0
    rejected_timeout = 0
    LOGGER.info(
        "START mode=%s expected_episodes=%d min_steps=%d max_steps=%d seed=%d output=%s",
        "smoke" if args.smoke else "full",
        expected_episodes,
        args.min_steps,
        args.max_steps,
        args.seed,
        args.output,
    )
    try:
        for route_id in TRIANGLE_ROUTES:
            route_start_accepted = len(accepted)
            LOGGER.info(
                "ROUTE %d/8 start points=%s",
                route_id,
                TRIANGLE_ROUTES[route_id],
            )
            for condition, count in counts.items():
                condition_accepted = 0
                condition_attempts = 0
                while condition_accepted < count:
                    condition_attempts += 1
                    attempts_by_route[route_id] += 1
                    if condition_attempts > args.max_attempts:
                        raise RuntimeError(
                            f"Route {route_id} condition {condition} could not produce "
                            f"{count} episodes with at least {args.min_steps} steps"
                        )
                    episode = collect_episode(
                        route_id,
                        condition,
                        rng,
                        args.max_steps,
                        capture_frames=args.smoke,
                    )
                    if episode.length < args.min_steps:
                        rejected_short += 1
                        continue
                    if not bool(episode.columns["terminated"][-1]):
                        rejected_timeout += 1
                        continue
                    writer.write(episode)
                    accepted.append(episode)
                    condition_accepted += 1
                    LOGGER.info(
                        "ACCEPT %03d/%03d route=%d condition=%s count=%d/%d "
                        "length=%d condition_attempts=%d rejected_short=%d "
                        "rejected_timeout=%d",
                        len(accepted),
                        expected_episodes,
                        route_id,
                        condition,
                        condition_accepted,
                        count,
                        episode.length,
                        condition_attempts,
                        rejected_short,
                        rejected_timeout,
                    )
            LOGGER.info(
                "ROUTE %d/8 complete accepted=%d attempts=%d",
                route_id,
                len(accepted) - route_start_accepted,
                attempts_by_route[route_id],
            )

        lengths = np.asarray([episode.length for episode in accepted])
        metadata = {
            "environment": "anygoal/TwoRoom-Tri-v1",
            "variant": "tworoom_triangle",
            "seed": args.seed,
            "action_noise": 0.25,
            "action_magnitude_min": 0.25,
            "action_magnitude_max": 1.0,
            "minimum_episode_steps": args.min_steps,
            "maximum_episode_steps": args.max_steps,
            "render_target": False,
            "points": json.dumps(TRIANGLE_POINTS),
            "routes": json.dumps(TRIANGLE_ROUTES),
            "condition_ids": json.dumps(CONDITION_IDS),
            "condition_counts_per_route": json.dumps(counts),
        }
        writer.close(metadata)
        os.replace(temporary_output, args.output)
    except Exception:
        writer.file.close()
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    if args.smoke:
        save_smoke_gif(accepted, args.gif)

    LOGGER.info("COMPLETE dataset=%s", args.output)
    if args.smoke:
        LOGGER.info("GIF %s", args.gif)
    LOGGER.info(
        "SUMMARY episodes=%d/%d steps=%d length_mean=%.3f min=%d median=%.3f "
        "max=%d rejected_short=%d rejected_timeout=%d",
        len(accepted),
        expected_episodes,
        int(lengths.sum()),
        float(lengths.mean()),
        int(lengths.min()),
        float(np.median(lengths)),
        int(lengths.max()),
        rejected_short,
        rejected_timeout,
    )
    LOGGER.info("ATTEMPTS_BY_ROUTE %s", attempts_by_route)
    return args.output
