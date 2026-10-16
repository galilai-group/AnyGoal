import pytest

from anygoal.wm.checkpoints import compatible_vit_state_dict
from anygoal.wm.lewm_config import resolve_model_config
from omegaconf import OmegaConf


def test_legacy_vit_renaming_preserves_values_and_current_names():
    old = "encoder.encoder.layer.0.attention.attention.query.weight"
    new = "encoder.layers.0.attention.q_proj.weight"
    tensor = object()
    assert compatible_vit_state_dict({old: tensor}, [new]) == {new: tensor}
    assert compatible_vit_state_dict({old: tensor}, [old]) == {old: tensor}
    assert compatible_vit_state_dict({new: tensor}, [new]) == {new: tensor}


def test_duplicate_compatibility_keys_are_rejected():
    old = "encoder.encoder.layer.0.attention.attention.query.weight"
    new = "encoder.layers.0.attention.q_proj.weight"
    with pytest.raises(KeyError, match="Duplicate"):
        compatible_vit_state_dict({old: object(), new: object()}, [new])


def test_shared_lewm_defaults_preserve_explicit_overrides():
    config = OmegaConf.create(
        {
            "model": {
                "action_dim": 2,
                "encoder_scale": "small",
                "predictor": {"depth": 4},
            },
            "image_size": 196,
        }
    )
    resolved = resolve_model_config(config.model)
    assert resolved.encoder_scale == "small"
    assert resolved.predictor.depth == 4
    assert resolved.predictor.heads == 16
    assert resolved.action_dim == 2
