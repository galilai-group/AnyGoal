"""Compatibility for ViT state-dict names across Transformers releases."""

from collections.abc import Mapping


def remap_legacy_vit_key(key: str) -> str:
    key = key.replace("encoder.encoder.layer.", "encoder.layers.")
    for old, new in (
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    ):
        key = key.replace(old, new)
    return key


def compatible_vit_state_dict(state: Mapping, target_keys) -> dict:
    """Rename legacy keys only when the target model uses their newer names."""
    target_keys = set(target_keys)
    remapped = {}
    for key, value in state.items():
        candidate = remap_legacy_vit_key(key)
        target = (
            key if key in target_keys or candidate not in target_keys else candidate
        )
        if target in remapped:
            raise KeyError(f"Duplicate remapped state key: {target}")
        remapped[target] = value
    return remapped
