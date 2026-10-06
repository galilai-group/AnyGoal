"""TwoRoom triangle variant and the route policy used for data collection."""

from __future__ import annotations

import numpy as np

from stable_worldmodel.envs.two_room.env import TwoRoomEnv


TRIANGLE_POINTS = {
    1: (36.0, 188.0),
    2: (97.0, 188.0),
    3: (36.0, 80.0),
    4: (36.0, 36.0),
    5: (97.0, 36.0),
    6: (127.0, 188.0),
    7: (188.0, 188.0),
    8: (188.0, 80.0),
    9: (127.0, 36.0),
    10: (188.0, 36.0),
}
TRIANGLE_ROUTES = {
    1: (2, 3, 4),
    2: (6, 8, 10),
    3: (8, 9, 10),
    4: (9, 8, 10),
    5: (3, 5, 4),
    6: (5, 3, 4),
    7: (2, 1, 3),
    8: (6, 7, 8),
}
TRIANGLE_CONDITIONS = (
    "clean_constant",
    "noisy_constant",
    "clean_varying",
)


class TwoRoomTriangleEnv(TwoRoomEnv):
    """TwoRoom dynamics with an invisible target and unchanged success rule."""

    def __init__(self, **kwargs) -> None:
        kwargs["render_target"] = False
        super().__init__(**kwargs)
        self.env_name = "TwoRoom-Tri"

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        variation_values = dict(options.get("variation_values", {}))
        variation_values["rendering.render_target"] = 0
        options["variation_values"] = variation_values
        return super().reset(seed=seed, options=options)


class TwoRoomTrianglePolicy:
    """Feedback controller following one start-waypoint-goal triangle route."""

    def __init__(
        self,
        route_id: int,
        condition: str,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        action_noise: float = 0.25,
        waypoint_tolerance: float = 5.0,
    ) -> None:
        if route_id not in TRIANGLE_ROUTES:
            raise ValueError(f"Unknown route_id: {route_id}")
        if condition not in TRIANGLE_CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both")
        self.route_id = int(route_id)
        self.condition = condition
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.action_noise = float(action_noise)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.route = np.asarray(
            [TRIANGLE_POINTS[index] for index in TRIANGLE_ROUTES[route_id]],
            dtype=np.float32,
        )
        self.sampled_magnitude = float(self.rng.uniform(0.25, 1.0))
        self.target_index = 1

    def action(self, position: np.ndarray) -> tuple[np.ndarray, int, float]:
        target = self.route[self.target_index]
        if (
            np.linalg.norm(target - position) <= self.waypoint_tolerance
            and self.target_index == 1
        ):
            self.target_index = 2
            target = self.route[self.target_index]

        magnitude = (
            float(self.rng.uniform(0.25, 1.0))
            if self.condition == "clean_varying"
            else self.sampled_magnitude
        )
        direction = target - position
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-8:
            return np.zeros(2, dtype=np.float32), self.target_index, magnitude
        action = direction / direction_norm * magnitude

        if self.condition == "noisy_constant":
            action = action + self.rng.normal(0.0, self.action_noise, size=2)
            noisy_norm = float(np.linalg.norm(action))
            action = (
                direction / direction_norm * magnitude
                if noisy_norm <= 1e-8
                else action / noisy_norm * magnitude
            )

        action_norm = float(np.linalg.norm(action))
        if action_norm > 0.0:
            clipped_norm = np.clip(action_norm, 0.25, 1.0)
            action = action / action_norm * clipped_norm
        return action.astype(np.float32), self.target_index, magnitude


def make_tworoom_triangle_env(**kwargs) -> TwoRoomTriangleEnv:
    return TwoRoomTriangleEnv(**kwargs)


__all__ = [
    "TRIANGLE_CONDITIONS",
    "TRIANGLE_POINTS",
    "TRIANGLE_ROUTES",
    "TwoRoomTriangleEnv",
    "TwoRoomTrianglePolicy",
    "make_tworoom_triangle_env",
]
