"""Dataset adapters and collection helpers, imported lazily by public name."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CubeActionSupport": "cube_collection",
    "CubeHDF5Writer": "cube_collection",
    "collect_fixed_length_episodes": "cube_collection",
    "reference_episode_length": "cube_collection",
    "CubeEverythingStateAdapter": "cube_true_state",
    "CubeTrueStateAdapter": "cube_true_state",
    "ManipulatorInsertPegStateAdapter": "manipulator_true_state",
    "BlockAngleSupportFilter": "pusht_collection",
    "PushTJointSupportFilter": "pusht_collection",
    "PushTTrainSchemaWriter": "pusht_collection",
    "collect_accepted_episodes": "pusht_collection",
    "complete_on_block_goal": "pusht_collection",
    "PushTTrueStateAdapter": "pusht_true_state",
    "TwoRoomTrueStateAdapter": "tworoom_true_state",
    "TriangleHDF5Writer": "tworoom_triangle_collection",
    "add_tworoom_triangle_episode_index": "tworoom_triangle_collection",
    "collect_tworoom_triangle_episode": "tworoom_triangle_collection",
    "MinMaxTransform": "toy_predictor",
    "TrueStateAdapter": "toy_predictor",
    "ZScoreTransform": "toy_predictor",
    "adapted_minmax_statistics": "toy_predictor",
    "adapted_zscore_statistics": "toy_predictor",
    "validate_hdf5": "validation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    module = import_module(f"{__name__}.{module_name}")
    if name == "add_tworoom_triangle_episode_index":
        value = getattr(module, "add_episode_index")
    elif name == "collect_tworoom_triangle_episode":
        value = getattr(module, "collect_episode")
    else:
        value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
