"""Dataset adapters and validation utilities, imported lazily by public name."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CubeEverythingStateAdapter": "cube.true_state",
    "CubeTrueStateAdapter": "cube.true_state",
    "InsertPegStateAdapter": "insert_peg.true_state",
    "PushTTrueStateAdapter": "pusht.true_state",
    "TwoRoomTrueStateAdapter": "tworoom.true_state",
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
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
