#!/usr/bin/env python
"""Save counterfactual sensitivities for a trained sensorimotor world model."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

import save_counterfactual_sensitivity as base


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNING_DIR = (
    REPO_ROOT / "worldmodels" / "sensorimotor-world-model" / "planning"
)
RESULTS_DIR = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "counterfactual_sensitivity"
    / "sensorimotor"
)


class SensorimotorPushtInferenceTransform:
    """Match sensorimotor training without duplicating JEPA pixel preprocessing."""

    def __init__(
        self,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        img_size: int,
    ) -> None:
        self.action_mean = action_mean.float()
        self.action_std = action_std.float()
        self.img_size = img_size

    def __call__(self, sample: dict) -> dict:
        sample["action"] = (
            sample["action"].float() - self.action_mean
        ) / self.action_std
        sample["proprio"] = sample["proprio"].float()
        sample["state"] = sample["state"].float()
        return sample


def parse_args() -> argparse.Namespace:
    parser = base.build_parser(description=__doc__)
    parser.set_defaults(
        output_dir=RESULTS_DIR,
        output_prefix="sensorimotor_counterfactual_sensitivity",
        history_size=1,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training result directory containing config.yaml and checkpoints/last.ckpt.",
    )
    args = parser.parse_args()

    args.run_dir = args.run_dir.expanduser().resolve()
    args.checkpoint_path = args.run_dir / "checkpoints" / "last.ckpt"
    args.output_dir = args.output_dir.expanduser().resolve() / args.run_dir.name
    if args.run_dir.name not in args.output_prefix:
        args.output_prefix = f"{args.output_prefix}_{args.run_dir.name}"
    args.model_family = "sensorimotor-world-model"
    args.model_run = args.run_dir.name

    output_prefix = base.output_prefix_with_job_id(
        args.output_prefix, args.job_id
    )
    run_output_dir = args.output_dir / output_prefix
    if run_output_dir.exists() and any(run_output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {run_output_dir}"
        )
    return args


def load_sensorimotor_model(
    args: argparse.Namespace, device: torch.device
) -> torch.nn.Module:
    if str(PLANNING_DIR) in sys.path:
        sys.path.remove(str(PLANNING_DIR))
    sys.path.insert(0, str(PLANNING_DIR))

    from jepa import JEPA
    from module import ARPredictor, Embedder, InverseModel, MLP
    from transformers import ViTConfig, ViTModel

    config_path = args.run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)

    trained_history_size = int(config.wm.get("history_size", 1))
    trained_frameskip = int(config.data.dataset.frameskip)
    trained_img_size = int(config.img_size)
    requested_settings = {
        "history_size": args.history_size,
        "frameskip": args.frameskip,
        "img_size": args.img_size,
    }
    trained_settings = {
        "history_size": trained_history_size,
        "frameskip": trained_frameskip,
        "img_size": trained_img_size,
    }
    if requested_settings != trained_settings:
        raise ValueError(
            "Inference settings must match the training config: "
            f"requested={requested_settings}, trained={trained_settings}"
        )

    encoder_sizes = {
        "tiny": (192, 12, 3),
        "small": (384, 12, 6),
        "base": (768, 12, 12),
        "large": (1024, 24, 16),
        "huge": (1280, 32, 16),
    }
    if config.encoder_scale not in encoder_sizes:
        raise ValueError(f"Unsupported encoder scale: {config.encoder_scale}")
    hidden_dim, num_hidden_layers, num_attention_heads = encoder_sizes[
        config.encoder_scale
    ]
    encoder_config = ViTConfig(
        hidden_size=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_dim * 4,
        image_size=trained_img_size,
        patch_size=int(config.patch_size),
    )
    encoder = ViTModel(
        encoder_config, add_pooling_layer=False, use_mask_token=False
    )
    embed_dim = int(config.wm.get("embed_dim", hidden_dim))
    effective_action_dim = trained_frameskip * int(config.wm.action_dim)
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=trained_history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **config.predictor,
        ),
        action_encoder=Embedder(
            input_dim=effective_action_dim, emb_dim=embed_dim
        ),
        projector=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        inverse_model=InverseModel(
            embed_dim=embed_dim,
            action_dim=effective_action_dim,
            hidden_dim=int(config.inverse.get("hidden_dim", 256)),
        ),
    )
    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = {}
    encoder_key_replacements = (
        ("encoder.layers.", "encoder.encoder.layer."),
        (".attention.q_proj.", ".attention.attention.query."),
        (".attention.k_proj.", ".attention.attention.key."),
        (".attention.v_proj.", ".attention.attention.value."),
        (".attention.o_proj.", ".attention.output.dense."),
        (".mlp.fc1.", ".intermediate.dense."),
        (".mlp.fc2.", ".output.dense."),
    )
    for checkpoint_key, value in checkpoint["state_dict"].items():
        if not checkpoint_key.startswith("model."):
            continue
        model_key = checkpoint_key[len("model.") :]
        for source, target in encoder_key_replacements:
            model_key = model_key.replace(source, target)
        state_dict[model_key] = value
    model.load_state_dict(state_dict)
    return model.to(device)


if __name__ == "__main__":
    try:
        base.main(
            parse_args_fn=parse_args,
            transform_class=SensorimotorPushtInferenceTransform,
            model_loader=load_sensorimotor_model,
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
