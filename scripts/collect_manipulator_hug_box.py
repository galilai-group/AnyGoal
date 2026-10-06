"""Collect balanced non-interaction episodes that hug the target box."""

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
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets.manipulator_insert_peg_v2_collection import (
    ReferenceSupport,
    audit_support,
    load_reference,
)
from anygoal.envs.manipulator_hug_box_policy import (
    TOTAL_STEPS as POLICY_TOTAL_STEPS,
    HugBoxPolicy,
    behavior_metrics,
    physical_root_angle,
    peg_arm_contact,
    peg_lowest_z,
    run_episode,
    target_arm_contact,
)
from anygoal.envs.manipulator_insert_peg import ManipulatorInsertPegEnv
from anygoal.paths import cache_root


DEFAULT_REFERENCE = Path.home() / ".stable_worldmodel/datasets/manipulator_insert_peg_weak_expert_train.h5"
DEFAULT_OUTPUT = Path.home() / ".stable_worldmodel/datasets/avoid_peg_test.h5"
STATE_FIELDS = {
    "proprio": slice(0, 33),
    "proprio_arm_pos": slice(0, 16),
    "proprio_arm_vel": slice(16, 24),
    "proprio_touch": slice(24, 29),
    "proprio_hand_pos": slice(29, 33),
    "privileged_object_pos": slice(33, 37),
    "privileged_object_vel": slice(37, 40),
    "privileged_target_pos": slice(40, 44),
}
ALIGN_STEPS = 250
HUG_STEPS = 250
TOTAL_STEPS = ALIGN_STEPS + HUG_STEPS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--support-cache",
        type=Path,
        default=(
            Path(os.environ.get("ANYGOAL_SUPPORT_CACHE_ROOT", "~/.cache/anygoal/support"))
            .expanduser()
            / "manipulator_hug_box_support_v1.npz"
        ),
    )
    parser.add_argument("--skip-pixels", action="store_true")
    parser.add_argument("--sample-gifs", type=Path, default=REPO_ROOT / "reports/manipulator_hug_box_samples")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def initialize_episode(env, rng, bank, target_side, support):
    root = np.array([physical_root_angle(np.arctan2(row[0], row[1])) for row in bank])
    hand = bank[:, 29:31]
    if target_side < 0:
        eligible = np.flatnonzero((np.abs(root) <= np.pi / 3) & (hand[:, 0] > 0.03) & (hand[:, 0] < 0.20) & (hand[:, 1] > 0.30))
    else:
        eligible = np.flatnonzero((np.abs(np.abs(root) - np.pi) <= np.pi / 3) & (hand[:, 0] < -0.03) & (hand[:, 0] > -0.20) & (hand[:, 1] > 0.30))
    if not len(eligible):
        raise RuntimeError(f"No initialization bank rows for side {target_side}")
    for _ in range(3000):
        state = bank[int(rng.choice(eligible))].copy()
        state[16:24] = 0.0
        state[37:40] = 0.0
        state[33] = rng.uniform(0.39, 0.40) * -target_side
        state[34] = 0.18
        peg_angle = rng.uniform(-np.pi, np.pi)
        state[35:37] = [np.cos(peg_angle / 2), np.sin(peg_angle / 2)]
        state[40] = rng.uniform(-0.20, -0.01) if target_side < 0 else rng.uniform(0.01, 0.20)
        state[41] = rng.uniform(0.22, 0.40)
        target_angle = rng.uniform(-np.pi / 3, np.pi / 3)
        state[42:44] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]
        env.reset(seed=int(rng.integers(2**30)), options={"state": state})
        canonical_root_angle = rng.uniform(-np.pi / 3.0, np.pi / 3.0)
        physical_angle = (
            canonical_root_angle
            if target_side < 0
            else np.pi - canonical_root_angle
        )
        root_q = (physical_angle - np.pi / 2.0 + np.pi) % (2.0 * np.pi) - np.pi
        env.physics.data.qpos[:4] = np.r_[
            root_q,
            -target_side * np.array([0.50, 0.75, 0.0]),
        ]
        env.physics.data.qvel[:4] = 0.0
        env.physics.named.data.qpos["peg_z"] += 0.001 - peg_lowest_z(env.physics)
        env.physics.named.data.qvel[["peg_x", "peg_z", "peg_y"]] = 0.0
        env.physics.forward()
        restored = env.get_state()
        peg = restored[33:35]
        current_hand = restored[29:31]
        if current_hand[1] <= peg[1] + 0.06:
            continue
        if np.linalg.norm(current_hand - peg) <= 0.16:
            continue
        if peg_arm_contact(env.physics) or target_arm_contact(env.physics):
            continue
        if support.check("arm", restored[:16]) and support.check("hand", restored[29:33]) and support.check("object", restored[33:37]) and support.check("target", restored[40:44]):
            return restored
    raise RuntimeError("Could not sample a supported hug-box initialization")


def waypoint_features(states):
    q_root = np.arctan2(states[:, 0], states[:, 1])
    root_angle = np.pi / 2.0 + q_root
    return np.column_stack(
        (12.0 * np.cos(root_angle), 12.0 * np.sin(root_angle), 25.0 * states[:, 29:31])
    )


def collect_waypoint_episode(rng, target_side, bank, bank_actions, tree):
    target = np.array(
        [
            rng.uniform(-0.20, -0.01) if target_side < 0 else rng.uniform(0.01, 0.20),
            rng.uniform(0.22, 0.40),
        ]
    )
    peg = np.array([-target_side * rng.uniform(0.39, 0.40), rng.uniform(0.005, 0.02)])
    base_initial_root = rng.uniform(-np.pi / 3.0, np.pi / 3.0)
    initial_root = base_initial_root if target_side < 0 else np.pi - base_initial_root
    final_root = -np.pi if target_side < 0 else 0.0
    initial_hand = np.array([-target_side * 0.18, rng.uniform(0.48, 0.62)])
    clearance = np.array([0.0, 0.68])
    aligned_hand = target + np.array([target_side * 0.145, 0.0])
    desired_roots = np.empty(TOTAL_STEPS + 1)
    desired_hands = np.empty((TOTAL_STEPS + 1, 2))
    for step in range(ALIGN_STEPS + 1):
        progress = step / ALIGN_STEPS
        smooth = progress * progress * (3.0 - 2.0 * progress)
        desired_roots[step] = initial_root + smooth * (final_root - initial_root)
        if step <= 90:
            hand_progress = step / 90.0
            desired_hands[step] = initial_hand + hand_progress * (clearance - initial_hand)
        else:
            hand_progress = (step - 90) / (ALIGN_STEPS - 90)
            hand_smooth = hand_progress * hand_progress * (3.0 - 2.0 * hand_progress)
            desired_hands[step] = clearance + hand_smooth * (aligned_hand - clearance)
    start_angle = np.pi if target_side < 0 else 0.0
    direction = 1.0 if target_side < 0 else -1.0
    for step in range(ALIGN_STEPS + 1, TOTAL_STEPS + 1):
        progress = (step - ALIGN_STEPS) / HUG_STEPS
        desired_roots[step] = final_root
        angle = start_angle + direction * np.deg2rad(120.0) * progress
        desired_hands[step] = target + 0.145 * np.array([np.cos(angle), np.sin(angle)])

    selected = []
    previous = None
    for root, hand in zip(desired_roots, desired_hands):
        query = np.r_[12.0 * np.cos(root), 12.0 * np.sin(root), 25.0 * hand]
        _, candidates = tree.query(query, k=128)
        candidates = np.atleast_1d(candidates)
        if previous is None:
            choice = int(candidates[0])
        else:
            arm_distance = np.linalg.norm(bank[candidates, :16] - bank[previous, :16], axis=1)
            feature_distance = np.linalg.norm(waypoint_features(bank[candidates]) - query, axis=1)
            choice = int(candidates[np.argmin(feature_distance + 0.05 * arm_distance)])
        selected.append(choice)
        previous = choice
    selected = np.asarray(selected)
    episode_states = bank[selected].copy()
    target_angle = rng.uniform(-np.pi / 3.0, np.pi / 3.0)
    peg_angle = rng.uniform(-np.pi, np.pi)
    episode_states[:, 33:35] = peg
    episode_states[:, 35:37] = [np.cos(peg_angle / 2.0), np.sin(peg_angle / 2.0)]
    episode_states[:, 37:40] = 0.0
    episode_states[:, 40:42] = target
    episode_states[:, 42:44] = [np.cos(target_angle / 2.0), np.sin(target_angle / 2.0)]
    actions = bank_actions[selected[:-1]].copy()
    actual_root = np.array(
        [physical_root_angle(np.arctan2(state[0], state[1])) for state in episode_states]
    )
    relative_hand = episode_states[:, 29:31] - target
    radii = np.linalg.norm(relative_hand, axis=1)
    hug_angles = np.unwrap(np.arctan2(relative_hand[ALIGN_STEPS:, 1], relative_hand[ALIGN_STEPS:, 0]))
    directed_sweep = direction * (hug_angles[-1] - hug_angles[0])
    root_error = np.abs((actual_root - final_root + np.pi) % (2.0 * np.pi) - np.pi)
    wrist_to_center = np.arctan2(-relative_hand[:, 1], -relative_hand[:, 0])
    wrist_goal = 0.0 if target_side < 0 else -np.pi
    wrist_error = np.abs((wrist_to_center - wrist_goal + np.pi) % (2.0 * np.pi) - np.pi)
    donut_fraction = np.mean((radii[ALIGN_STEPS:] >= 0.071) & (radii[ALIGN_STEPS:] <= 0.22))
    minimum_peg_hand_distance = np.min(np.linalg.norm(episode_states[:, 29:31] - peg, axis=1))
    metrics = {
        "success": bool(
            root_error[ALIGN_STEPS] <= np.pi / 18.0
            and wrist_error[ALIGN_STEPS] <= np.pi / 18.0
            and np.max(root_error[ALIGN_STEPS:]) <= np.pi / 18.0
            and donut_fraction >= 0.95
            and directed_sweep >= np.deg2rad(30.0)
            and minimum_peg_hand_distance >= 0.18
        ),
        "steps": TOTAL_STEPS,
        "alignment_root_error_degrees": float(np.rad2deg(root_error[ALIGN_STEPS])),
        "alignment_wrist_center_error_degrees": float(np.rad2deg(wrist_error[ALIGN_STEPS])),
        "maximum_root_error_after_alignment_degrees": float(np.rad2deg(root_error[ALIGN_STEPS:].max())),
        "hug_donut_fraction": float(donut_fraction),
        "hug_directed_sweep_degrees": float(np.rad2deg(directed_sweep)),
        "hug_radius_min": float(radii[ALIGN_STEPS:].min()),
        "hug_radius_max": float(radii[ALIGN_STEPS:].max()),
        "minimum_peg_hand_distance": float(minimum_peg_hand_distance),
        "peg_contact_frames": 0,
        "target_pose": [*target, np.cos(target_angle / 2.0), np.sin(target_angle / 2.0)],
        "initial_peg_pose": [*peg, np.cos(peg_angle / 2.0), np.sin(peg_angle / 2.0)],
        "initial_root_angle_degrees": float(np.rad2deg(actual_root[0])),
        "construction": "nearest reference-state waypoint sequence",
    }
    return {"states": episode_states, "actions": actions}, metrics


def write_dataset(
    path, accepted, reference, seed, reports, include_pixels, env, *, overwrite=False
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
        dataset.attrs["task"] = "manipulator_hug_box"
        dataset.attrs["variant"] = "non-interaction target-box hug without peg contact"
        dataset.attrs["reference"] = str(reference.resolve())
        dataset.attrs["seed"] = seed
        dataset.attrs["episode_steps"] = POLICY_TOTAL_STEPS
        dataset.attrs["target_balance"] = "200 left / 200 right for 400 episodes"
        dataset.attrs["acceptance_rule"] = json.dumps({"alignment_tolerance_degrees": 10, "donut_radii": [0.071, 0.22], "minimum_donut_fraction": 0.95, "minimum_directed_sweep_degrees": 30, "minimum_peg_hand_distance": 0.18, "peg_contact_frames": 0})
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
        starts = np.zeros(len(actions), bool)
        terminal = np.zeros(len(actions), bool)
        starts[offsets] = True
        terminal[ends] = True
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
                if index % 2000 == 0:
                    print(f"rendered {index}/{len(observation)}", flush=True)
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
    states, reference_features = load_reference(args.reference)
    bank = states[::2]
    bank_actions = reference_features["action"][::2]
    support = ReferenceSupport(reference_features, args.support_cache)
    targeted_support = ReferenceSupport(
        {
            "elbow": reference_features["arm"][:, 4:6],
            "hand_position": reference_features["hand"][:, :2],
        },
        cache_root() / "support/manipulator_hug_box_targeted_support_v1.npz",
    )
    env = ManipulatorInsertPegEnv(terminate_at_goal=False)
    for geom_name in ("floor", "slot_0", "slot_1", "slot_2"):
        env.physics.named.model.geom_contype[geom_name] = 1
        env.physics.named.model.geom_conaffinity[geom_name] = 1
    env.physics.named.model.geom_friction[["slot_0", "slot_1", "slot_2"]] = [0.7, 0.005, 0.0001]
    rng = np.random.default_rng(args.seed)
    accepted, episode_reports = [], []
    attempts = 0
    max_attempts = args.max_attempts or args.episodes * 100
    while len(accepted) < args.episodes and attempts < max_attempts:
        attempts += 1
        target_side = -1 if len(accepted) % 2 == 0 else 1
        initial = initialize_episode(env, rng, bank, target_side, support)
        initial_q = np.arctan2(initial[0:8:2], initial[1:8:2])
        try:
            policy = HugBoxPolicy(env, target_side, bank_actions, initial_q, rng)
        except RuntimeError as error:
            print("discarded", json.dumps({"attempt": attempts, "target_side": "left" if target_side < 0 else "right", "reason": str(error)}), flush=True)
            continue
        rollout = run_episode(env, policy, support, targeted_support)
        metrics = behavior_metrics(rollout, policy)
        metrics.update({
            "target_pose": initial[40:44].tolist(),
            "initial_peg_pose": initial[33:37].tolist(),
            "initial_root_angle_degrees": float(np.rad2deg(physical_root_angle(initial_q[0]))),
            "box_collision": {"contype": 1, "conaffinity": 1},
            "floor_collision": {"contype": 1, "conaffinity": 1},
            "construction": "MuJoCo rollout with OU-perturbed rotation actions",
        })
        metrics.update({"attempt": attempts, "target_side": "left" if target_side < 0 else "right"})
        print(("episode" if metrics["success"] else "discarded"), json.dumps(metrics), flush=True)
        if metrics["success"]:
            metrics["episode"] = len(accepted)
            accepted.append(rollout)
            episode_reports.append(metrics)
    if len(accepted) != args.episodes:
        env.close()
        raise RuntimeError(f"Collected {len(accepted)}/{args.episodes} episodes in {attempts} attempts")
    all_states = np.concatenate([item["states"][:-1] for item in accepted])
    all_actions = np.concatenate([item["actions"] for item in accepted])
    distribution = audit_support(reference_features, all_states, all_actions)
    report = {"variant": "manipulator_hug_box_collision_ou", "reference": str(args.reference.resolve()), "construction": "MuJoCo rollouts with hard target/floor collision and OU-perturbed rotation actions", "episodes": len(accepted), "left_episodes": args.episodes // 2, "right_episodes": args.episodes // 2, "attempts": attempts, "acceptance_rate": len(accepted) / attempts, "mean_episode_length": float(np.mean([len(item["actions"]) for item in accepted])), "median_episode_length": float(np.median([len(item["actions"]) for item in accepted])), "episode_metrics": episode_reports, "distribution_support_pre_save": distribution}
    report_path = args.output.with_suffix(".collection.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    render_samples(args.sample_gifs, accepted, env)
    write_dataset(
        args.output,
        accepted,
        args.reference,
        args.seed,
        report,
        not args.skip_pixels,
        env,
        overwrite=args.overwrite,
    )
    env.close()
    print(f"completed {args.output}", flush=True)


if __name__ == "__main__":
    main()
