"""Collect balanced non-interaction episodes that hug the target box."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from anygoal.datasets.insert_peg.support import (
    ReferenceSupport,
    audit_support,
    load_reference,
    write_variant_dataset,
)
from anygoal.paths import cache_root, data_root

os.environ.setdefault("MUJOCO_GL", "osmesa")


DEFAULT_REFERENCE = data_root() / "insert_peg_weak_expert_train.h5"
DEFAULT_OUTPUT = data_root() / "avoid_peg_test.h5"
DEFAULT_SUPPORT_CACHE = (
    Path(
        os.environ.get("ANYGOAL_SUPPORT_CACHE_ROOT", "~/.cache/anygoal/support")
    ).expanduser()
    / "avoid_peg_support_v1.npz"
)


def _initialize_episode(env, rng, bank, target_side, support):
    from anygoal.envs.avoid_peg import policy as avoid_peg_policy

    root = np.array(
        [
            avoid_peg_policy.physical_root_angle(np.arctan2(row[0], row[1]))
            for row in bank
        ]
    )
    hand = bank[:, 29:31]
    if target_side < 0:
        eligible = np.flatnonzero(
            (np.abs(root) <= np.pi / 3)
            & (hand[:, 0] > 0.03)
            & (hand[:, 0] < 0.20)
            & (hand[:, 1] > 0.30)
        )
    else:
        eligible = np.flatnonzero(
            (np.abs(np.abs(root) - np.pi) <= np.pi / 3)
            & (hand[:, 0] < -0.03)
            & (hand[:, 0] > -0.20)
            & (hand[:, 1] > 0.30)
        )
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
        state[40] = (
            rng.uniform(-0.20, -0.01) if target_side < 0 else rng.uniform(0.01, 0.20)
        )
        state[41] = rng.uniform(0.22, 0.40)
        target_angle = rng.uniform(-np.pi / 3, np.pi / 3)
        state[42:44] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]
        env.reset(seed=int(rng.integers(2**30)), options={"state": state})
        canonical_root_angle = rng.uniform(-np.pi / 3.0, np.pi / 3.0)
        physical_angle = (
            canonical_root_angle if target_side < 0 else np.pi - canonical_root_angle
        )
        root_q = (physical_angle - np.pi / 2.0 + np.pi) % (2.0 * np.pi) - np.pi
        env.physics.data.qpos[:4] = np.r_[
            root_q,
            -target_side * np.array([0.50, 0.75, 0.0]),
        ]
        env.physics.data.qvel[:4] = 0.0
        env.physics.named.data.qpos["peg_z"] += 0.001 - avoid_peg_policy.peg_lowest_z(
            env.physics
        )
        env.physics.named.data.qvel[["peg_x", "peg_z", "peg_y"]] = 0.0
        env.physics.forward()
        restored = env.get_state()
        peg = restored[33:35]
        current_hand = restored[29:31]
        if current_hand[1] <= peg[1] + 0.06:
            continue
        if np.linalg.norm(current_hand - peg) <= 0.16:
            continue
        if avoid_peg_policy.peg_arm_contact(
            env.physics
        ) or avoid_peg_policy.target_arm_contact(env.physics):
            continue
        if (
            support.check("arm", restored[:16])
            and support.check("hand", restored[29:33])
            and support.check("object", restored[33:37])
            and support.check("target", restored[40:44])
        ):
            return restored
    raise RuntimeError("Could not sample a supported avoid-peg initialization")


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
    seed: int = 20260914,
    support_cache: Path = DEFAULT_SUPPORT_CACHE,
    skip_pixels: bool = False,
    overwrite: bool = False,
) -> Path:
    from anygoal.envs.avoid_peg import policy as avoid_peg_policy
    from anygoal.envs.insert_peg import env as insert_peg_env

    _validate_options(episodes, output, overwrite)
    states, reference_features = load_reference(reference)
    bank = states[::2]
    bank_actions = reference_features["action"][::2]
    support = ReferenceSupport(reference_features, support_cache)
    targeted_support = ReferenceSupport(
        {
            "elbow": reference_features["arm"][:, 4:6],
            "hand_position": reference_features["hand"][:, :2],
        },
        cache_root() / "support/avoid_peg_targeted_support_v1.npz",
    )
    env = insert_peg_env.InsertPegEnv(terminate_at_goal=False)
    for geom_name in ("floor", "slot_0", "slot_1", "slot_2"):
        env.physics.named.model.geom_contype[geom_name] = 1
        env.physics.named.model.geom_conaffinity[geom_name] = 1
    env.physics.named.model.geom_friction[["slot_0", "slot_1", "slot_2"]] = [
        0.7,
        0.005,
        0.0001,
    ]
    rng = np.random.default_rng(seed)
    accepted, episode_reports = [], []
    attempts = 0
    attempt_limit = max_attempts or episodes * 100
    try:
        while len(accepted) < episodes and attempts < attempt_limit:
            attempts += 1
            target_side = -1 if len(accepted) % 2 == 0 else 1
            initial = _initialize_episode(env, rng, bank, target_side, support)
            initial_q = np.arctan2(initial[0:8:2], initial[1:8:2])
            try:
                policy = avoid_peg_policy.HugBoxPolicy(
                    env, target_side, bank_actions, initial_q, rng
                )
            except RuntimeError as error:
                print(
                    "discarded",
                    json.dumps(
                        {
                            "attempt": attempts,
                            "target_side": "left" if target_side < 0 else "right",
                            "reason": str(error),
                        }
                    ),
                    flush=True,
                )
                continue
            rollout = avoid_peg_policy.run_episode(
                env, policy, support, targeted_support
            )
            metrics = avoid_peg_policy.behavior_metrics(rollout, policy)
            metrics.update(
                {
                    "target_pose": initial[40:44].tolist(),
                    "initial_peg_pose": initial[33:37].tolist(),
                    "initial_root_angle_degrees": float(
                        np.rad2deg(avoid_peg_policy.physical_root_angle(initial_q[0]))
                    ),
                    "box_collision": {"contype": 1, "conaffinity": 1},
                    "floor_collision": {"contype": 1, "conaffinity": 1},
                    "construction": "MuJoCo rollout with OU-perturbed rotation actions",
                    "attempt": attempts,
                    "target_side": "left" if target_side < 0 else "right",
                }
            )
            print(
                "episode" if metrics["success"] else "discarded",
                json.dumps(metrics),
                flush=True,
            )
            if metrics["success"]:
                metrics["episode"] = len(accepted)
                accepted.append(rollout)
                episode_reports.append(metrics)

        if len(accepted) != episodes:
            raise RuntimeError(
                f"Collected {len(accepted)}/{episodes} episodes in {attempts} attempts"
            )
        all_states = np.concatenate([item["states"][:-1] for item in accepted])
        all_actions = np.concatenate([item["actions"] for item in accepted])
        distribution = audit_support(reference_features, all_states, all_actions)
        lengths = [len(item["actions"]) for item in accepted]
        report = {
            "variant": "avoid_peg_collision_ou",
            "reference": str(reference.resolve()),
            "construction": "MuJoCo rollouts with hard target/floor collision and OU-perturbed rotation actions",
            "episodes": len(accepted),
            "left_episodes": episodes // 2,
            "right_episodes": episodes // 2,
            "attempts": attempts,
            "acceptance_rate": len(accepted) / attempts,
            "mean_episode_length": float(np.mean(lengths)),
            "median_episode_length": float(np.median(lengths)),
            "episode_metrics": episode_reports,
            "distribution_support_pre_save": distribution,
        }
        output.with_suffix(".collection.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        write_variant_dataset(
            output,
            accepted,
            {
                "task": "avoid_peg",
                "variant": "non-interaction target-box hug without peg contact",
                "reference": str(reference.resolve()),
                "seed": seed,
                "episode_steps": avoid_peg_policy.TOTAL_STEPS,
                "target_balance": "200 left / 200 right for 400 episodes",
                "acceptance_rule": json.dumps(
                    {
                        "alignment_tolerance_degrees": 10,
                        "donut_radii": [0.071, 0.22],
                        "minimum_donut_fraction": 0.95,
                        "minimum_directed_sweep_degrees": 30,
                        "minimum_peg_hand_distance": 0.18,
                        "peg_contact_frames": 0,
                    }
                ),
            },
            include_pixels=not skip_pixels,
            env=env,
            overwrite=overwrite,
            render_progress_interval=2000,
        )
    finally:
        env.close()
    print(f"completed {output}", flush=True)
    return output
