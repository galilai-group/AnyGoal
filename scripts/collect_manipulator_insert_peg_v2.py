"""Collect fixed-length expert episodes for manipulator insert-peg variant 2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets.manipulator_insert_peg_v2_collection import (
    ReferenceSupport,
    audit_support,
    enrich_dataset,
    load_reference,
    load_target_reset_pool,
    render_sample_gifs,
    save_dataset,
)
from anygoal.envs.manipulator_insert_peg import ManipulatorInsertPegEnv
from anygoal.envs.manipulator_insert_peg_v2_policy import (
    behavior_metrics,
    build_search_trees,
    initialize_episode,
    run_episode,
)


DEFAULT_REFERENCE = (
    Path.home()
    / ".stable_worldmodel/datasets/manipulator_insert_peg_weak_expert_train.h5"
)
DEFAULT_EXPERT_REFERENCE = (
    Path.home() / ".stable_worldmodel/rlu_manipulator_insert_peg_success_trimmed.h5"
)
ACCEPTANCE_RULE = {
    "episode_steps": 500,
    "minimum_contact_events": 10,
    "minimum_contact_frames": 30,
    "minimum_contact_frames_each_100_step_window": 1,
    "minimum_last_contact_step": 450,
    "minimum_late_100_step_object_path": 0.10,
    "minimum_completed_cycles": 2,
    "minimum_vertical_excursion": 0.055,
    "minimum_upward_reversals": 3,
    "minimum_object_path": 0.30,
    "minimum_same_ramp_fraction": 0.90,
    "minimum_ramp_occupancy_fraction": 0.75,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--expert-reference", type=Path, default=DEFAULT_EXPERT_REFERENCE
    )
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--support-cache", type=Path, required=True)
    parser.add_argument("--skip-enrich", action="store_true")
    return parser.parse_args()


def target_summary(values):
    labels = ("x", "z", "qw", "qy")
    return {
        label: {
            "min": float(values[:, index].min()),
            "q01": float(np.quantile(values[:, index], 0.01)),
            "median": float(np.median(values[:, index])),
            "q99": float(np.quantile(values[:, index], 0.99)),
            "max": float(values[:, index].max()),
        }
        for index, label in enumerate(labels)
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.steps != 500:
        raise ValueError("V2 expert requires exactly 500 steps")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")

    expert_states, expert_features = load_reference(
        args.expert_reference, successful_only=True
    )
    _, reference_features = load_reference(args.reference)
    target_pool = load_target_reset_pool(args.reference)
    selection = np.arange(0, len(expert_states), 4)
    bank = expert_states[selection]
    expert_actions = expert_features["action"][selection]
    action_candidates = reference_features["action"][::4]
    support = ReferenceSupport(reference_features, args.support_cache)
    pose_tree, action_tree = build_search_trees(bank, action_candidates)
    env = ManipulatorInsertPegEnv(terminate_at_goal=False)
    rng = np.random.default_rng(args.seed)

    episodes, episode_reports = [], []
    attempts = 0
    max_attempts = args.max_attempts or args.episodes * 100
    while len(episodes) < args.episodes and attempts < max_attempts:
        attempts += 1
        target_side = -1 if len(episodes) % 2 == 0 else 1
        policy = initialize_episode(
            env, rng, bank, expert_actions, action_candidates, target_pool,
            target_side, support, pose_tree, action_tree,
        )
        (
            states, actions, contacts, failure,
            supported_states, supported_actions,
        ) = run_episode(env, policy, support, args.steps)
        metrics = behavior_metrics(states, contacts, policy)
        metrics.update(
            {
                "attempt": attempts,
                "steps": len(actions),
                "target_side": "left" if target_side < 0 else "right",
                "target_pose": states[0, 40:44].tolist(),
                "validated_anchor_index": policy.anchor_index,
                "termination": failure,
                "strict_state_support_fraction": float(supported_states.mean())
                if len(supported_states)
                else 0.0,
                "strict_action_support_fraction": float(supported_actions.mean())
                if len(supported_actions)
                else 0.0,
            }
        )
        print(
            ("episode" if metrics["success"] else "discarded"),
            json.dumps(metrics),
            flush=True,
        )
        if metrics["success"]:
            metrics["episode"] = len(episodes)
            episodes.append((states, actions))
            episode_reports.append(metrics)

    if len(episodes) != args.episodes:
        env.close()
        raise RuntimeError(
            f"Collected {len(episodes)}/{args.episodes} episodes in {attempts} attempts"
        )

    all_states = np.concatenate([states[:-1] for states, _ in episodes])
    all_actions = np.concatenate([actions for _, actions in episodes])
    support_report = audit_support(reference_features, all_states, all_actions)
    failed = [name for name, result in support_report.items() if not result["final_verdict"]]
    if failed:
        env.close()
        raise RuntimeError(f"Pre-save distribution support audit failed: {failed}")

    sampled_targets = np.asarray([states[0, 40:44] for states, _ in episodes])
    report = {
        "variant": "variant_2_expert",
        "reference": str(args.reference.resolve()),
        "expert_reference": str(args.expert_reference.resolve()),
        "episodes": len(episodes),
        "attempts": attempts,
        "acceptance_rate": len(episodes) / attempts,
        "acceptance_rule": ACCEPTANCE_RULE,
        "target_sampling": {
            "method": (
                "uniform over source episode-reset target poses after excluding "
                "x in [-0.2, 0.2], with exactly balanced left/right accepted episodes"
            ),
            "eligible_source_episodes": len(target_pool),
            "eligible_left": int(np.sum(target_pool[:, 0] < -0.2)),
            "eligible_right": int(np.sum(target_pool[:, 0] > 0.2)),
            "source_pool": target_summary(target_pool),
            "collected": target_summary(sampled_targets),
        },
        "episode_metrics": episode_reports,
        "distribution_support": support_report,
    }
    report_path = args.figures / "variant_2_expert_collection_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    save_dataset(
        temporary_path, episodes, args.reference, args.expert_reference, ACCEPTANCE_RULE
    )
    render_sample_gifs(args.figures, episodes, ManipulatorInsertPegEnv)
    if not args.skip_enrich:
        enrich_dataset(temporary_path, ManipulatorInsertPegEnv)
    os.replace(temporary_path, args.output)
    env.close()
    print(f"completed dataset {args.output}", flush=True)
    print(f"collection report {report_path}", flush=True)


if __name__ == "__main__":
    main()
