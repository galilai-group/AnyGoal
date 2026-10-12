"""Policy for the TwoRoom route environment."""

from __future__ import annotations

import numpy as np

from .env import ROUTE_CONDITIONS, ROUTE_POINTS, ROUTES


class TwoRoomRoutePolicy:
    """Feedback controller following one start-waypoint-goal route."""

    def __init__(
        self,
        route_id: int,
        condition: str,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        action_noise: float = 0.25,
        waypoint_tolerance: float = 5.0,
    ) -> None:
        if route_id not in ROUTES:
            raise ValueError(f"Unknown route_id: {route_id}")
        if condition not in ROUTE_CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both")
        self.route_id = int(route_id)
        self.condition = condition
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.action_noise = float(action_noise)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.route = np.asarray(
            [ROUTE_POINTS[index] for index in ROUTES[route_id]],
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


__all__ = ["TwoRoomRoutePolicy"]

