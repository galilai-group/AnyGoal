"""Benchmark environments and policies, imported lazily by public name."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AvoidTEpisodeRandomizationWrapper": "avoid_t",
    "AvoidTPPolicy": "avoid_t",
    "EpisodeRandomizationWrapper": "avoid_t",
    "SampledGoalAvoidTPolicy": "avoid_t",
    "make_avoidt_env": "avoid_t",
    "CubeDiagonalEnv": "cube_diagonal",
    "CubeDiagonalPolicy": "cube_diagonal",
    "DiagonalCubeMarkovOracle": "cube_diagonal",
    "DiagonalCubePlanOracle": "cube_diagonal",
    "make_cube_diagonal_env": "cube_diagonal",
    "CubeHopEnv": "cube_hop",
    "CubeHopPolicy": "cube_hop",
    "make_cube_hop_env": "cube_hop",
    "ManipulatorInsertPegEnv": "manipulator_insert_peg",
    "make_manipulator_insert_peg_env": "manipulator_insert_peg",
    "PushTAntigoal": "push_t_antigoal",
    "PushTAntigoalV3EpisodeRandomizationWrapper": "push_t_antigoal",
    "PushTAntigoalV3Policy": "push_t_antigoal",
    "make_push_t_antigoal_v3_env": "push_t_antigoal",
    "StackT": "stack_t",
    "StackTV2EpisodeRandomizationWrapper": "stack_t",
    "StackTV2Policy": "stack_t",
    "make_stackt_v2_env": "stack_t",
    "stacked_block_pose": "stack_t",
    "TRIANGLE_CONDITIONS": "tworoom_triangle",
    "TRIANGLE_POINTS": "tworoom_triangle",
    "TRIANGLE_ROUTES": "tworoom_triangle",
    "TwoRoomTriangleEnv": "tworoom_triangle",
    "TwoRoomTrianglePolicy": "tworoom_triangle",
    "make_tworoom_triangle_env": "tworoom_triangle",
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
