"""Collect balanced peg extraction, waypoint carry, and floor placement episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import h5py
import hdf5plugin
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets.manipulator_insert_peg_v2_collection import (
    ReferenceSupport,
    audit_support,
    load_reference,
)
from anygoal.envs.manipulator_extract_peg import ManipulatorExtractPegEnv
from anygoal.envs.manipulator_extract_peg_policy import (
    APPROACH_STEPS,
    CARRY_STEPS,
    GRIP_STEPS,
    PLACE_STEPS,
    TOTAL_STEPS,
    ExtractPegPolicy,
    initialize_arm_above_peg,
    peg_lowest_z,
    rotate_planar,
)
from anygoal.envs.manipulator_insert_peg import planar_angle, wrapped_angle_error


DEFAULT_REFERENCE = Path.home() / ".stable_worldmodel/datasets/manipulator_insert_peg_weak_expert_train.h5"
DEFAULT_OUTPUT = Path.home() / ".stable_worldmodel/datasets/extract_peg_test.h5"
STATE_FIELDS = {
    "proprio": slice(0, 33), "proprio_arm_pos": slice(0, 16),
    "proprio_arm_vel": slice(16, 24), "proprio_touch": slice(24, 29),
    "proprio_hand_pos": slice(29, 33), "privileged_object_pos": slice(33, 37),
    "privileged_object_vel": slice(37, 40), "privileged_target_pos": slice(40, 44),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--support-cache",
        type=Path,
        default=(
            Path(os.environ.get("ANYGOAL_SUPPORT_CACHE_ROOT", "~/.cache/anygoal/support"))
            .expanduser()
            / "manipulator_extract_peg_support_v1.npz"
        ),
    )
    parser.add_argument("--skip-pixels", action="store_true")
    parser.add_argument("--sample-gifs", type=Path, default=REPO_ROOT / "reports/manipulator_extract_peg_samples")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_triangle(rng, vertices):
    weights = rng.dirichlet(np.ones(3))
    return weights @ np.asarray(vertices, dtype=np.float64)


def load_reverse_insertion_anchors(path):
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
            local_z = np.sin(target_angle) * relative[:, 0] + np.cos(target_angle) * relative[:, 1]
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


def arm_target_contact(physics):
    names = physics.named.model.geom_pos.axes.row.names
    arm = {"upper_arm", "middle_arm", "lower_arm", "hand", "palm1", "palm2",
           "thumb1", "thumb2", "thumbtip1", "thumbtip2", "finger1", "finger2",
           "fingertip1", "fingertip2"}
    target = {"slot_0", "slot_1", "slot_2"}
    for contact in physics.data.contact[: physics.data.ncon]:
        pair = {names[contact.geom1], names[contact.geom2]}
        if pair & arm and pair & target:
            return True
    return False


def initialize_episode(env, rng, side, support, anchors):
    for _ in range(500):
        final, approach_arm, reverse_q = anchors[side][int(rng.integers(len(anchors[side])))]
        state = final.copy()
        state[:24] = approach_arm
        state[24:29] = 0.0
        env.reset(seed=int(rng.integers(2**30)), options={"state": state})
        restored = env.get_state()
        actual_distance = np.linalg.norm(restored[29:31] - restored[33:35])
        if not (0.10 < actual_distance < 0.20):
            continue
        if arm_target_contact(env.physics) or peg_lowest_z(env.physics) < -0.002:
            continue
        if support.check("arm", restored[:16]) and support.check("hand", restored[29:33]):
            return restored, actual_distance, reverse_q
    raise RuntimeError("Could not initialize a supported extract-peg episode")


def run_episode(env, policy, support):
    states, actions = [env.get_state()], []
    state_support, action_support, floor_clearance = [], [], []
    target_contacts = []
    for _ in range(TOTAL_STEPS):
        action = policy.action()
        env.physics.set_control(action)
        env.physics.step(nstep=25)
        state = env.get_state()
        states.append(state)
        actions.append(action)
        state_support.append(support.state(state))
        action_support.append(support.check("action", action))
        floor_clearance.append(peg_lowest_z(env.physics))
        target_contacts.append(arm_target_contact(env.physics))
        if floor_clearance[-1] < -0.002:
            break
    return {
        "states": np.asarray(states), "actions": np.asarray(actions),
        "state_support": np.asarray(state_support), "action_support": np.asarray(action_support),
        "floor_clearance": np.asarray(floor_clearance),
        "target_contacts": np.asarray(target_contacts),
    }


def metrics(rollout, policy, initial_distance):
    grip_end = APPROACH_STEPS + GRIP_STEPS
    carry_end = grip_end + CARRY_STEPS
    place_end = carry_end + PLACE_STEPS
    final = rollout["states"][-1]
    peg = final[33:35]
    angle = planar_angle(final[35:37])
    destination_side_ok = 0.10 <= -policy.side * peg[0] <= 0.40
    downward_error = abs(wrapped_angle_error(0.0, angle))
    floor_error = abs(float(rollout["floor_clearance"][-1]))
    success = bool(
        len(rollout["actions"]) == TOTAL_STEPS
        and policy.reached_waypoint
        and policy.waypoint_error <= 0.10
        and policy.released
        and destination_side_ok
        and downward_error <= np.deg2rad(25.0)
        and floor_error <= 0.015
        and rollout["floor_clearance"].min() >= -0.002
    )
    return {
        "success": success, "steps": len(rollout["actions"]),
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
        if len(rollout["states"]) > grip_end else None,
        "peg_position_after_carry": rollout["states"][carry_end, 33:35].tolist()
        if len(rollout["states"]) > carry_end else None,
        "peg_position_after_place": rollout["states"][place_end, 33:35].tolist()
        if len(rollout["states"]) > place_end else None,
        "peg_angle_after_carry_degrees": float(
            np.rad2deg(planar_angle(rollout["states"][carry_end, 35:37]))
        ) if len(rollout["states"]) > carry_end else None,
        "peg_angle_after_place_degrees": float(
            np.rad2deg(planar_angle(rollout["states"][place_end, 35:37]))
        ) if len(rollout["states"]) > place_end else None,
    }


def write_dataset(
    path, accepted, reference, seed, include_pixels, env, *, overwrite=False
):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    lengths = np.asarray([len(item["actions"]) for item in accepted], np.int64)
    observation = np.concatenate([item["states"][:-1] for item in accepted]).astype(np.float32)
    next_observation = np.concatenate([item["states"][1:] for item in accepted]).astype(np.float32)
    actions = np.concatenate([item["actions"] for item in accepted]).astype(np.float32)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = np.cumsum(lengths) - 1
    with h5py.File(temporary, "x") as dataset:
        dataset.attrs["task"] = "manipulator_extract_peg"
        dataset.attrs["variant"] = "extract peg, carry via waypoint, place long end down"
        dataset.attrs["reference"] = str(reference.resolve())
        dataset.attrs["seed"] = seed
        dataset.attrs["max_episode_length"] = 1000
        dataset.create_dataset("observation", data=observation)
        dataset.create_dataset("next_observation", data=next_observation)
        dataset.create_dataset("action", data=actions)
        dataset["privilege"] = dataset["observation"]
        dataset["next_privilege"] = dataset["next_observation"]
        for name, columns in STATE_FIELDS.items():
            dataset.create_dataset(name, data=observation[:, columns])
            dataset.create_dataset(f"next_{name}", data=next_observation[:, columns])
        dataset.create_dataset("ep_len", data=lengths)
        dataset.create_dataset("ep_offset", data=offsets)
        dataset.create_dataset("ep_idx", data=np.repeat(np.arange(len(lengths)), lengths))
        dataset.create_dataset("step_idx", data=np.concatenate([np.arange(length) for length in lengths]))
        starts, terminal = np.zeros(len(actions), bool), np.zeros(len(actions), bool)
        starts[offsets], terminal[ends] = True, True
        dataset.create_dataset("fragment_start", data=starts)
        dataset.create_dataset("fragment_end", data=terminal)
        dataset.create_dataset("done", data=terminal)
        dataset.create_dataset("success", data=np.repeat(True, len(actions)))
        dataset.create_dataset("discount", data=(~terminal).astype(np.float32))
        dataset.create_dataset("reward", data=np.zeros(len(actions), np.float32))
        next_action = np.zeros_like(actions)
        next_action[:-1] = actions[1:]
        next_action[ends] = 0.0
        dataset.create_dataset("next_action", data=next_action)
        if include_pixels:
            pixels = dataset.create_dataset("pixels", shape=(len(observation), 224, 224, 3), dtype=np.uint8, chunks=(1, 224, 224, 3), **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
            for index, state in enumerate(observation):
                env.set_state(state)
                pixels[index] = env.render()
    os.replace(temporary, path)


def render_samples(path, accepted, env):
    path.mkdir(parents=True, exist_ok=True)
    for index, episode in enumerate(accepted[:10]):
        frames = []
        for state in episode["states"][:-1:5]:
            env.set_state(state)
            frames.append(env.render(width=180, height=180))
        imageio.mimsave(path / f"episode_{index:03d}.gif", frames, duration=50, loop=0)


def main():
    args = parse_args()
    if args.episodes <= 0 or args.episodes % 2:
        raise ValueError("--episodes must be a positive even number")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    _, reference_features = load_reference(args.reference)
    anchors = load_reverse_insertion_anchors(args.reference)
    support = ReferenceSupport(reference_features, args.support_cache)
    env = ManipulatorExtractPegEnv(terminate_at_goal=False)
    rng = np.random.default_rng(args.seed)
    accepted, reports = [], []
    attempts = 0
    max_attempts = args.max_attempts or args.episodes * 100
    while len(accepted) < args.episodes and attempts < max_attempts:
        attempts += 1
        side = -1 if len(accepted) % 2 == 0 else 1
        try:
            initial, initial_distance, reverse_q = initialize_episode(
                env, rng, side, support, anchors
            )
        except RuntimeError as error:
            print("discarded", json.dumps({"attempt": attempts, "side": side, "reason": str(error)}), flush=True)
            continue
        vertices = np.array([[0.3, 0.4], [0.3, 0.6], [0.55, 0.4]])
        waypoint = sample_triangle(rng, vertices)
        waypoint[0] *= -side
        destination_x = -side * rng.uniform(0.1, 0.4)
        policy = ExtractPegPolicy(env, side, waypoint, destination_x, rng, reverse_q)
        rollout = run_episode(env, policy, support)
        report = metrics(rollout, policy, initial_distance)
        report.update({"attempt": attempts, "side": "left" if side < 0 else "right", "waypoint": waypoint.tolist(), "destination_x": destination_x})
        print("episode" if report["success"] else "discarded", json.dumps(report), flush=True)
        if report["success"]:
            accepted.append(rollout)
            reports.append(report)
    if len(accepted) != args.episodes:
        raise RuntimeError(f"Collected {len(accepted)}/{args.episodes} episodes in {attempts} attempts")
    distribution = audit_support(reference_features, np.concatenate([item["states"][:-1] for item in accepted]), np.concatenate([item["actions"] for item in accepted]))
    args.output.with_suffix(".collection.json").write_text(json.dumps({"episodes": len(accepted), "left_episodes": args.episodes // 2, "right_episodes": args.episodes // 2, "attempts": attempts, "episode_metrics": reports, "distribution_support_pre_save": distribution}, indent=2) + "\n")
    render_samples(args.sample_gifs, accepted, env)
    write_dataset(
        args.output,
        accepted,
        args.reference,
        args.seed,
        not args.skip_pixels,
        env,
        overwrite=args.overwrite,
    )
    env.close()
    print(f"completed {args.output}", flush=True)


if __name__ == "__main__":
    main()
