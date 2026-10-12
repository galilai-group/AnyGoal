"""TwoRoom route variant and the route policy used for data collection."""

from __future__ import annotations

from stable_worldmodel.envs.two_room.env import TwoRoomEnv


ROUTE_POINTS = {
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
ROUTES = {
    1: (2, 3, 4),
    2: (6, 8, 10),
    3: (8, 9, 10),
    4: (9, 8, 10),
    5: (3, 5, 4),
    6: (5, 3, 4),
    7: (2, 1, 3),
    8: (6, 7, 8),
}
ROUTE_CONDITIONS = (
    "clean_constant",
    "noisy_constant",
    "clean_varying",
)


class TwoRoomRouteEnv(TwoRoomEnv):
    """TwoRoom dynamics with an invisible target and unchanged success rule."""

    def __init__(self, **kwargs) -> None:
        kwargs["render_target"] = False
        super().__init__(**kwargs)
        self.env_name = "TwoRoom-Route"

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        variation_values = dict(options.get("variation_values", {}))
        variation_values["rendering.render_target"] = 0
        options["variation_values"] = variation_values
        return super().reset(seed=seed, options=options)


def make_tworoom_route_env(**kwargs) -> TwoRoomRouteEnv:
    return TwoRoomRouteEnv(**kwargs)


__all__ = [
    "ROUTE_CONDITIONS",
    "ROUTE_POINTS",
    "ROUTES",
    "TwoRoomRouteEnv",
    "make_tworoom_route_env",
]
