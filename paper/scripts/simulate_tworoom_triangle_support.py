"""Dry-simulate candidate TwoRoom triangle datasets without rendering pixels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import distribution_support as support  # noqa: E402
from anygoal.envs.tworoom_triangle import (  # noqa: E402
    TRIANGLE_CONDITIONS,
    TRIANGLE_ROUTES,
    TwoRoomTriangleEnv,
    TwoRoomTrianglePolicy,
)


BASE_POINTS = {
    1: (25.0, 199.0),
    2: (95.0, 199.0),
    3: (25.0, 75.0),
    4: (25.0, 25.0),
    5: (95.0, 25.0),
    6: (129.0, 199.0),
    7: (199.0, 199.0),
    8: (199.0, 75.0),
    9: (129.0, 25.0),
    10: (199.0, 25.0),
}
CONDITION_COUNTS = {
    "clean_constant": 20,
    "noisy_constant": 20,
    "clean_varying": 10,
}


def scaled_points(scale: float) -> dict[int, tuple[float, float]]:
    return {
        index: tuple(np.rint(112.0 + scale * (np.asarray(point) - 112.0)))
        for index, point in BASE_POINTS.items()
    }


def simulate(scale: float, action_noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    points = scaled_points(scale)
    rng = np.random.default_rng(seed)
    positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rejected_short = 0
    rejected_timeout = 0

    for route_id, point_indices in TRIANGLE_ROUTES.items():
        route = np.asarray([points[index] for index in point_indices], dtype=np.float32)
        for condition, count in CONDITION_COUNTS.items():
            accepted = 0
            attempts = 0
            while accepted < count:
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError(
                        f"scale={scale} route={route_id} condition={condition} exhausted attempts"
                    )
                env = TwoRoomTriangleEnv(render_mode="rgb_array")
                policy = TwoRoomTrianglePolicy(
                    route_id,
                    condition,
                    rng=rng,
                    action_noise=action_noise,
                )
                policy.route = route
                env.reset(
                    seed=int(rng.integers(0, 2**31 - 1)),
                    options={"variation": [], "state": route[0], "target_state": route[2]},
                )
                episode_positions = []
                episode_actions = []
                route_complete = False
                for _ in range(120):
                    position = env.agent_position.detach().cpu().numpy().copy()
                    action, waypoint_index, _ = policy.action(position)
                    _, _, _, _, info = env.step(action)
                    next_position = np.asarray(info["state"], dtype=np.float32)
                    route_complete = bool(
                        waypoint_index == 2
                        and np.linalg.norm(next_position - route[2])
                        <= policy.waypoint_tolerance
                    )
                    episode_positions.append(position)
                    episode_actions.append(action)
                    if route_complete:
                        break
                env.close()
                if len(episode_actions) < 26:
                    rejected_short += 1
                    continue
                if not route_complete:
                    rejected_timeout += 1
                    continue
                positions.extend(episode_positions)
                actions.extend(episode_actions)
                accepted += 1

    print(
        f"scale={scale:.3f} points={points} rows={len(actions)} "
        f"rejected_short={rejected_short} rejected_timeout={rejected_timeout}"
    )
    return np.asarray(positions), np.asarray(actions)


def print_support(
    reference_position: np.ndarray,
    reference_action: np.ndarray,
    target_position: np.ndarray,
    target_action: np.ndarray,
) -> None:
    features = (
        ("agent pos x", reference_position[:, :1], target_position[:, :1]),
        ("agent pos y", reference_position[:, 1:], target_position[:, 1:]),
        ("action x", reference_action[:, :1], target_action[:, :1]),
        ("action y", reference_action[:, 1:], target_action[:, 1:]),
        ("agent pos (2D)", reference_position, target_position),
        ("action (2D)", reference_action, target_action),
    )
    for label, reference, target in features:
        result = support.evaluate_feature(
            support.Feature(label, reference), support.Feature(label, target)
        )
        print(
            f"  {label}: mahalanobis={result['mahalanobis']['target_fraction']:.6f} "
            f"kde={result['kde']['target_fraction']:.6f} "
            f"range={result['range']['target_fraction']:.6f} "
            f"verdict={result['final_verdict']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--action-noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with h5py.File(args.reference) as dataset:
        reference_position = np.asarray(dataset["proprio"][:], dtype=np.float64)
        reference_action = np.asarray(dataset["action"][:], dtype=np.float64)
    reference_position = reference_position[np.isfinite(reference_position).all(axis=1)]
    reference_action = reference_action[np.isfinite(reference_action).all(axis=1)]

    for scale in args.scales:
        target_position, target_action = simulate(scale, args.action_noise, args.seed)
        print_support(
            reference_position,
            reference_action,
            target_position,
            target_action,
        )


if __name__ == "__main__":
    main()
