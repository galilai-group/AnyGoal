"""Collect balanced peg extraction, waypoint carry, and floor placement episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from anygoal.datasets.insert_peg.support import (
    ReferenceSupport,
    audit_support,
    load_reference,
    write_variant_dataset,
)
from anygoal.paths import data_root

os.environ.setdefault("MUJOCO_GL", "osmesa")


DEFAULT_REFERENCE = data_root() / "insert_peg_weak_expert_train.h5"
DEFAULT_OUTPUT = data_root() / "extract_peg_test.h5"
DEFAULT_SUPPORT_CACHE = (
    Path(
        os.environ.get("ANYGOAL_SUPPORT_CACHE_ROOT", "~/.cache/anygoal/support")
    ).expanduser()
    / "extract_peg_support_v1.npz"
)


def _sample_triangle(rng, vertices):
    weights = rng.dirichlet(np.ones(3))
    return weights @ np.asarray(vertices, dtype=np.float64)


def _load_reverse_insertion_anchors(path):
    anchors = {-1: [], 1: []}
    with h5py.File(path, "r") as dataset:
        offsets = dataset["ep_offset"][:].astype(np.int64)
        lengths = dataset["ep_len"][:].astype(np.int64)
        for offset, length in zip(offsets, lengths):
            reward = dataset["reward"][offset : offset + length]
            successes = np.flatnonzero(reward >= 1.0 - 1e-7)
            if not len(successes):
                continue
            end = offset + int(successes[0])
            final = dataset["observation"][end].astype(np.float64)
            side = -1 if final[40] < 0 else 1
            if not (0.10 <= abs(final[40]) <= 0.40 and 0.10 <= final[41] <= 0.30):
                continue
            start = max(offset, end - 250)
            states = dataset["observation"][start : end + 1].astype(np.float64)
            target_angle = 2.0 * np.arctan2(states[:, 43], states[:, 42])
            relative = states[:, 33:35] - states[:, 40:42]
            local_z = (
                np.sin(target_angle) * relative[:, 0]
                + np.cos(target_angle) * relative[:, 1]
            )
            extraction_start = int(np.argmin(np.abs(local_z + 0.045)))
            insertion = states[extraction_start:]
            if len(insertion) < 20:
                continue
            distance = np.linalg.norm(states[:, 29:31] - final[33:35], axis=1)
            approach_candidates = np.flatnonzero((distance > 0.10) & (distance < 0.20))
            if not len(approach_candidates):
                continue
            approach = states[int(approach_candidates[-1])]
            reverse_q = np.unwrap(
                np.arctan2(insertion[:, :8:2], insertion[:, 1:8:2]), axis=0
            )[::-1].copy()
            anchors[side].append((final, approach[:24].copy(), reverse_q))
    if not anchors[-1] or not anchors[1]:
        raise RuntimeError("Reference lacks usable left/right insertion anchors")
    return anchors


def _arm_target_contact(physics):
    names = physics.named.model.geom_pos.axes.row.names
    arm = {
        "upper_arm",
        "middle_arm",
        "lower_arm",
        "hand",
        "palm1",
        "palm2",
        "thumb1",
        "thumb2",
        "thumbtip1",
        "thumbtip2",
        "finger1",
        "finger2",
        "fingertip1",
        "fingertip2",
    }
    target = {"slot_0", "slot_1", "slot_2"}
    for contact in physics.data.contact[: physics.data.ncon]:
        pair = {names[contact.geom1], names[contact.geom2]}
        if pair & arm and pair & target:
            return True
    return False


def _initialize_episode(env, rng, side, support, anchors):
    from anygoal.envs.extract_peg import policy as extract_peg_policy

    for _ in range(500):
        final, approach_arm, reverse_q = anchors[side][
            int(rng.integers(len(anchors[side])))
        ]
        state = final.copy()
        state[:24] = approach_arm
        state[24:29] = 0.0
        env.reset(seed=int(rng.integers(2**30)), options={"state": state})
        restored = env.get_state()
        actual_distance = np.linalg.norm(restored[29:31] - restored[33:35])
        if not (0.10 < actual_distance < 0.20):
            continue
        if (
            _arm_target_contact(env.physics)
            or extract_peg_policy.peg_lowest_z(env.physics) < -0.002
        ):
            continue
        if support.check("arm", restored[:16]) and support.check(
            "hand", restored[29:33]
        ):
            return restored, actual_distance, reverse_q
    raise RuntimeError("Could not initialize a supported extract-peg episode")


def _run_episode(env, policy, support):
    from anygoal.envs.extract_peg import policy as extract_peg_policy

    states, actions = [env.get_state()], []
    state_support, action_support, floor_clearance = [], [], []
    target_contacts = []
    for _ in range(extract_peg_policy.TOTAL_STEPS):
        action = policy.action()
        env.physics.set_control(action)
        env.physics.step(nstep=25)
        state = env.get_state()
        states.append(state)
        actions.append(action)
        state_support.append(support.state(state))
        action_support.append(support.check("action", action))
        floor_clearance.append(extract_peg_policy.peg_lowest_z(env.physics))
        target_contacts.append(_arm_target_contact(env.physics))
        if floor_clearance[-1] < -0.002:
            break
    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "state_support": np.asarray(state_support),
        "action_support": np.asarray(action_support),
        "floor_clearance": np.asarray(floor_clearance),
        "target_contacts": np.asarray(target_contacts),
    }


def _episode_metrics(rollout, policy, initial_distance):
    from anygoal.envs.extract_peg import policy as extract_peg_policy
    from anygoal.envs.insert_peg import env as insert_peg_env

    grip_end = extract_peg_policy.APPROACH_STEPS + extract_peg_policy.GRIP_STEPS
    carry_end = grip_end + extract_peg_policy.CARRY_STEPS
    place_end = carry_end + extract_peg_policy.PLACE_STEPS
    final = rollout["states"][-1]
    peg = final[33:35]
    angle = insert_peg_env.planar_angle(final[35:37])
    destination_side_ok = 0.10 <= -policy.side * peg[0] <= 0.40
    downward_error = abs(insert_peg_env.wrapped_angle_error(0.0, angle))
    floor_error = abs(float(rollout["floor_clearance"][-1]))
    success = bool(
        len(rollout["actions"]) == extract_peg_policy.TOTAL_STEPS
        and policy.reached_waypoint
        and policy.waypoint_error <= 0.10
        and policy.released
        and destination_side_ok
        and downward_error <= np.deg2rad(25.0)
        and floor_error <= 0.015
        and rollout["floor_clearance"].min() >= -0.002
    )
    return {
        "success": success,
        "steps": len(rollout["actions"]),
        "initial_hand_peg_distance": float(initial_distance),
        "waypoint_error": float(policy.waypoint_error),
        "reached_waypoint": bool(policy.reached_waypoint),
        "released": bool(policy.released),
        "final_peg_position": peg.tolist(),
        "final_downward_angle_error_degrees": float(np.rad2deg(downward_error)),
        "final_floor_error": floor_error,
        "minimum_floor_clearance": float(rollout["floor_clearance"].min()),
        "target_contact_frames": int(rollout["target_contacts"].sum()),
        "strict_state_support_fraction": float(rollout["state_support"].mean()),
        "strict_action_support_fraction": float(rollout["action_support"].mean()),
        "peg_position_after_grip": rollout["states"][grip_end, 33:35].tolist()
        if len(rollout["states"]) > grip_end
        else None,
        "peg_position_after_carry": rollout["states"][carry_end, 33:35].tolist()
        if len(rollout["states"]) > carry_end
        else None,
        "peg_position_after_place": rollout["states"][place_end, 33:35].tolist()
        if len(rollout["states"]) > place_end
        else None,
        "peg_angle_after_carry_degrees": float(
            np.rad2deg(
                insert_peg_env.planar_angle(
                    rollout["states"][carry_end, 35:37]
                )
            )
        )
        if len(rollout["states"]) > carry_end
        else None,
        "peg_angle_after_place_degrees": float(
            np.rad2deg(
                insert_peg_env.planar_angle(
                    rollout["states"][place_end, 35:37]
                )
            )
        )
        if len(rollout["states"]) > place_end
        else None,
    }


def _validate_options(episodes: int, output: Path, overwrite: bool) -> None:
    if episodes <= 0 or episodes % 2:
        raise ValueError("--episodes must be a positive even number")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")


def collect_dataset(
    *,
    reference: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_OUTPUT,
    episodes: int = 400,
    max_attempts: int | None = None,
    seed: int = 20260915,
    support_cache: Path = DEFAULT_SUPPORT_CACHE,
    skip_pixels: bool = False,
    overwrite: bool = False,
) -> Path:
    from anygoal.envs.extract_peg import env as extract_peg_env
    from anygoal.envs.extract_peg import policy as extract_peg_policy

    _validate_options(episodes, output, overwrite)
    _, reference_features = load_reference(reference)
    anchors = _load_reverse_insertion_anchors(reference)
    support = ReferenceSupport(reference_features, support_cache)
    env = extract_peg_env.ExtractPegEnv(terminate_at_goal=False)
    rng = np.random.default_rng(seed)
    accepted, reports = [], []
    attempts = 0
    attempt_limit = max_attempts or episodes * 100
    try:
        while len(accepted) < episodes and attempts < attempt_limit:
            attempts += 1
            side = -1 if len(accepted) % 2 == 0 else 1
            try:
                _initial, initial_distance, reverse_q = _initialize_episode(
                    env, rng, side, support, anchors
                )
            except RuntimeError as error:
                print(
                    "discarded",
                    json.dumps(
                        {"attempt": attempts, "side": side, "reason": str(error)}
                    ),
                    flush=True,
                )
                continue
            vertices = np.array([[0.3, 0.4], [0.3, 0.6], [0.55, 0.4]])
            waypoint = _sample_triangle(rng, vertices)
            waypoint[0] *= -side
            destination_x = -side * rng.uniform(0.1, 0.4)
            policy = extract_peg_policy.ExtractPegPolicy(
                env, side, waypoint, destination_x, rng, reverse_q
            )
            rollout = _run_episode(env, policy, support)
            report = _episode_metrics(rollout, policy, initial_distance)
            report.update(
                {
                    "attempt": attempts,
                    "side": "left" if side < 0 else "right",
                    "waypoint": waypoint.tolist(),
                    "destination_x": destination_x,
                }
            )
            print(
                "episode" if report["success"] else "discarded",
                json.dumps(report),
                flush=True,
            )
            if report["success"]:
                accepted.append(rollout)
                reports.append(report)

        if len(accepted) != episodes:
            raise RuntimeError(
                f"Collected {len(accepted)}/{episodes} episodes in {attempts} attempts"
            )
        distribution = audit_support(
            reference_features,
            np.concatenate([item["states"][:-1] for item in accepted]),
            np.concatenate([item["actions"] for item in accepted]),
        )
        output.with_suffix(".collection.json").write_text(
            json.dumps(
                {
                    "episodes": len(accepted),
                    "left_episodes": episodes // 2,
                    "right_episodes": episodes // 2,
                    "attempts": attempts,
                    "episode_metrics": reports,
                    "distribution_support_pre_save": distribution,
                },
                indent=2,
            )
            + "\n"
        )
        write_variant_dataset(
            output,
            accepted,
            {
                "task": "extract_peg",
                "variant": "extract peg, carry via waypoint, place long end down",
                "reference": str(reference.resolve()),
                "seed": seed,
                "max_episode_length": 1000,
            },
            include_pixels=not skip_pixels,
            env=env,
            overwrite=overwrite,
        )
    finally:
        env.close()
    print(f"completed {output}", flush=True)
    return output
