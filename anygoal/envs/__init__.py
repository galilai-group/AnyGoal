"""Benchmark environments and policies, imported lazily by public name."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AvoidTEpisodeRandomizationWrapper": "avoidt.env",
    "AvoidTPPolicy": "avoidt.policy",
    "EpisodeRandomizationWrapper": "avoidt.env",
    "SampledGoalAvoidTPolicy": "avoidt.policy",
    "make_avoidt_env": "avoidt.env",
    "CubeDiagonalEnv": "cube_diagonal.env",
    "CubeDiagonalPolicy": "cube_diagonal.policy",
    "DiagonalCubeMarkovOracle": "cube_diagonal.policy",
    "DiagonalCubePlanOracle": "cube_diagonal.policy",
    "make_cube_diagonal_env": "cube_diagonal.env",
    "CubeHopEnv": "cube_hop.env",
    "CubeHopPolicy": "cube_hop.policy",
    "make_cube_hop_env": "cube_hop.env",
    "InsertPegEnv": "insert_peg.env",
    "make_insert_peg_env": "insert_peg.env",
    "PushTAntigoal": "pusht_antigoal.env",
    "PushTAntigoalV3EpisodeRandomizationWrapper": "pusht_antigoal.env",
    "PushTAntigoalV3Policy": "pusht_antigoal.policy",
    "make_push_t_antigoal_v3_env": "pusht_antigoal.env",
    "StackT": "stackt.env",
    "StackTV2EpisodeRandomizationWrapper": "stackt.env",
    "StackTV2Policy": "stackt.policy",
    "make_stackt_v2_env": "stackt.env",
    "stacked_block_pose": "stackt.env",
    "ROUTE_CONDITIONS": "tworoom_route.env",
    "ROUTE_POINTS": "tworoom_route.env",
    "ROUTES": "tworoom_route.env",
    "TwoRoomRouteEnv": "tworoom_route.env",
    "TwoRoomRoutePolicy": "tworoom_route.policy",
    "make_tworoom_route_env": "tworoom_route.env",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
