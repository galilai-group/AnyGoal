"""Collect the generated TwoRoom route dataset."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from anygoal.datasets.tworoom import support as tworoom_support
from anygoal.envs.tworoom_route import env as tworoom_route_env
from anygoal.envs.tworoom_route import policy as tworoom_route_policy
from anygoal.paths import data_root


CONDITION_COUNTS = {
    "clean_constant": 20,
    "noisy_constant": 20,
    "clean_varying": 10,
}
SMOKE_CONDITION_COUNTS = {condition: 1 for condition in CONDITION_COUNTS}
CONDITION_IDS = {
    condition: index
    for index, condition in enumerate(tworoom_route_env.ROUTE_CONDITIONS)
}
MAX_EPISODE_STEPS = 120
LOGGER = logging.getLogger("tworoom_route")
DEFAULT_OUTPUT = data_root() / "tworoom_route_test.h5"


def _collect_episode(
    route_id: int,
    condition: str,
    rng: np.random.Generator,
    max_steps: int,
) -> tworoom_support.TwoRoomEpisode:
    route = np.asarray(
        [
            tworoom_route_env.ROUTE_POINTS[index]
            for index in tworoom_route_env.ROUTES[route_id]
        ],
        dtype=np.float32,
    )
    env = tworoom_route_env.TwoRoomRouteEnv(render_mode="rgb_array")
    policy = tworoom_route_policy.TwoRoomRoutePolicy(
        route_id, condition, rng=rng
    )
    observation, _ = env.reset(
        seed=int(rng.integers(0, 2**31 - 1)),
        options={"variation": [], "state": route[0], "target_state": route[2]},
    )
    columns = {key: [] for key in tworoom_support.TwoRoomHDF5Writer.PER_STEP_SPECS}

    for step_index in range(max_steps):
        position = env.agent_position.detach().cpu().numpy().copy()
        current_frame = env.render()
        action, waypoint_index, _magnitude = policy.action(position)
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
        columns["next_observation"].append(
            np.asarray(next_observation, dtype=np.float32)
        )
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
        if route_complete:
            break

    env.close()
    return tworoom_support.make_episode(
        columns,
        route_id,
        condition,
        policy.sampled_magnitude,
    )


def _validate_options(
    min_steps: int,
    max_steps: int,
    output: Path,
    overwrite: bool,
) -> None:
    if min_steps <= 0 or max_steps < min_steps:
        raise ValueError("Require 0 < min-steps <= max-steps")
    if max_steps > MAX_EPISODE_STEPS:
        raise ValueError(
            f"max-steps cannot exceed the dataset limit of {MAX_EPISODE_STEPS}"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")


def collect_dataset(
    *,
    smoke: bool = False,
    seed: int = 0,
    min_steps: int = 26,
    max_steps: int = MAX_EPISODE_STEPS,
    max_attempts: int = 1000,
    output: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    counts = SMOKE_CONDITION_COUNTS if smoke else CONDITION_COUNTS
    expected_episodes = len(tworoom_route_env.ROUTES) * sum(counts.values())
    _validate_options(min_steps, max_steps, output, overwrite)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )
    rng = np.random.default_rng(seed)
    writer = tworoom_support.TwoRoomHDF5Writer(temporary_output)
    lengths = []
    attempts_by_route = {route_id: 0 for route_id in tworoom_route_env.ROUTES}
    rejected_short = 0
    rejected_timeout = 0
    LOGGER.info(
        "START mode=%s expected_episodes=%d min_steps=%d max_steps=%d seed=%d output=%s",
        "smoke" if smoke else "full",
        expected_episodes,
        min_steps,
        max_steps,
        seed,
        output,
    )
    try:
        for route_id in tworoom_route_env.ROUTES:
            route_start_accepted = len(lengths)
            LOGGER.info(
                "ROUTE %d/8 start points=%s",
                route_id,
                tworoom_route_env.ROUTES[route_id],
            )
            for condition, count in counts.items():
                condition_accepted = 0
                condition_attempts = 0
                while condition_accepted < count:
                    condition_attempts += 1
                    attempts_by_route[route_id] += 1
                    if condition_attempts > max_attempts:
                        raise RuntimeError(
                            f"Route {route_id} condition {condition} could not produce "
                            f"{count} episodes with at least {min_steps} steps"
                        )
                    episode = _collect_episode(route_id, condition, rng, max_steps)
                    if episode.length < min_steps:
                        rejected_short += 1
                        continue
                    if not bool(episode.columns["terminated"][-1]):
                        rejected_timeout += 1
                        continue
                    writer.write(episode)
                    lengths.append(episode.length)
                    condition_accepted += 1
                    LOGGER.info(
                        "ACCEPT %03d/%03d route=%d condition=%s count=%d/%d "
                        "length=%d condition_attempts=%d rejected_short=%d "
                        "rejected_timeout=%d",
                        len(lengths),
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
                len(lengths) - route_start_accepted,
                attempts_by_route[route_id],
            )

        length_array = np.asarray(lengths)
        metadata = {
            "environment": "anygoal/TwoRoom-Route-v1",
            "variant": "tworoom_route",
            "seed": seed,
            "action_noise": 0.25,
            "action_magnitude_min": 0.25,
            "action_magnitude_max": 1.0,
            "minimum_episode_steps": min_steps,
            "maximum_episode_steps": max_steps,
            "render_target": False,
            "points": json.dumps(tworoom_route_env.ROUTE_POINTS),
            "routes": json.dumps(tworoom_route_env.ROUTES),
            "condition_ids": json.dumps(CONDITION_IDS),
            "condition_counts_per_route": json.dumps(counts),
        }
        writer.close(metadata)
        os.replace(temporary_output, output)
        tworoom_support.validate_dataset(output, expected_episodes)
    except Exception:
        writer.file.close()
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    LOGGER.info("COMPLETE dataset=%s", output)
    LOGGER.info(
        "SUMMARY episodes=%d/%d steps=%d length_mean=%.3f min=%d median=%.3f "
        "max=%d rejected_short=%d rejected_timeout=%d",
        len(lengths),
        expected_episodes,
        int(length_array.sum()),
        float(length_array.mean()),
        int(length_array.min()),
        float(np.median(length_array)),
        int(length_array.max()),
        rejected_short,
        rejected_timeout,
    )
    LOGGER.info("ATTEMPTS_BY_ROUTE %s", attempts_by_route)
    return output
