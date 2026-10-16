"""Evaluate LeWM object or state-dict checkpoints with MPC."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import hydra
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf

from anygoal.sources import source_path
from anygoal.wm.lewm_config import resolve_model_config
from anygoal.wm.checkpoints import compatible_vit_state_dict, remap_legacy_vit_key  # noqa: F401

from .planning import evaluate


def _source_root(cfg: DictConfig) -> Path:
    source_root = Path(cfg.model_source_root).expanduser().resolve()
    if not (source_root / "jepa.py").is_file():
        source_root = source_path("lewm")
    if not (source_root / "jepa.py").is_file():
        raise FileNotFoundError(f"LeWM source not found: {source_root}")
    return source_root


def build_lewm(cfg: DictConfig):
    OmegaConf.update(
        cfg, "model", resolve_model_config(cfg.model), merge=False, force_add=True
    )
    source_root = _source_root(cfg)
    sys.path.insert(0, str(source_root))
    try:
        from jepa import JEPA
        from module import ARPredictor, Embedder, MLP
    finally:
        sys.path.pop(0)

    encoder = spt.backbone.utils.vit_hf(
        str(cfg.model.encoder_scale),
        patch_size=int(cfg.model.patch_size),
        image_size=int(cfg.eval.img_size),
        pretrained=False,
        use_mask_token=False,
    )
    embed_dim = int(cfg.model.embed_dim)
    predictor_cfg = cfg.model.predictor
    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=int(cfg.model.history_size),
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=int(predictor_cfg.depth),
            heads=int(predictor_cfg.heads),
            mlp_dim=int(predictor_cfg.mlp_dim),
            dim_head=int(predictor_cfg.dim_head),
            dropout=float(predictor_cfg.dropout),
            emb_dropout=float(predictor_cfg.emb_dropout),
        ),
        action_encoder=Embedder(
            input_dim=int(cfg.model.action_dim) * int(cfg.plan_config.action_block),
            emb_dim=embed_dim,
        ),
        projector=MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
    )


def load_lewm_cost_model(cfg: DictConfig, checkpoint_root: str | Path):
    explicit_checkpoint = cfg.get("model", {}).get("checkpoint")
    checkpoint_path = (
        Path(str(explicit_checkpoint)).expanduser()
        if explicit_checkpoint
        else Path(checkpoint_root) / f"{cfg.policy}_object.ckpt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LeWM checkpoint not found: {checkpoint_path}")
    source_root = _source_root(cfg)
    sys.path.insert(0, str(source_root))
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    finally:
        sys.path.pop(0)
    if isinstance(payload, torch.nn.Module):
        if not hasattr(payload, "get_cost"):
            raise TypeError(f"Checkpoint model has no get_cost: {checkpoint_path}")
        print(
            f"Loaded serialized model {checkpoint_path}: "
            f"{type(payload).__module__}.{type(payload).__name__}"
        )
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected a model or state dict in {checkpoint_path}, got {type(payload)}"
        )

    model = build_lewm(cfg)
    remapped_state = compatible_vit_state_dict(payload, model.state_dict())
    model.load_state_dict(remapped_state, strict=True)
    print(
        f"Loaded state dict {checkpoint_path}: "
        f"{sum(parameter.numel() for parameter in model.parameters())} parameters"
    )
    return model


@hydra.main(
    version_base=None,
    config_path="../../resources/plan/lewm",
    config_name="cube",
)
def run(cfg: DictConfig):
    return evaluate(cfg, load_lewm_cost_model)


if __name__ == "__main__":
    run()
