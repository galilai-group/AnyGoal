"""Run one real-data forward/backward step for a toy predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_toy_predictor import (  # noqa: E402
    _load_normalization_statistics,
    _make_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        choices=(
            "cube_essential",
            "cube_extra",
            "cube_everything",
            "cube_everything_extreme",
            "pusht",
            "tworoom_essential",
            "tworoom_extra",
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("adaln", "action_duplicate"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    config_directory = REPO_ROOT / "configs" / "toy_predictors"
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_directory),
    ):
        cfg = compose(
            config_name="train",
            overrides=[f"data={args.data}", f"model={args.model}"],
        )

    adapter = hydra.utils.instantiate(cfg.data.adapter)
    normalizers, _ = _load_normalization_statistics(
        cfg.data.normalization_path,
        adapter=adapter,
        dataset_name=str(cfg.data.dataset_name),
    )
    dataset = swm.data.load_dataset(
        cfg.data.dataset_name,
        cache_dir=cfg.data.cache_dir,
        num_steps=cfg.history_size + cfg.num_preds,
        frameskip=cfg.data.frameskip,
        keys_to_load=list(cfg.data.keys_to_load),
        keys_to_cache=[],
        transform=_make_transform(adapter, normalizers),
    )
    sample_indices = torch.linspace(
        0,
        len(dataset) - 1,
        args.batch_size,
        dtype=torch.int64,
    ).tolist()
    batch = torch.utils.data.default_collate(
        [dataset[index] for index in sample_indices]
    )
    states = torch.nan_to_num(batch["state"].float()).to(args.device)
    actions = torch.nan_to_num(batch["action"].float()).to(args.device)

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("name")
    model = hydra.utils.instantiate(model_config).to(args.device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    output = model.loss(states, actions)
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    optimizer.step()

    result = {
        "data": args.data,
        "model": args.model,
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "prediction_shape": list(output["prediction"].shape),
        "loss": float(output["loss"].detach().cpu()),
        "finite_loss": bool(torch.isfinite(output["loss"]).item()),
    }
    if not result["finite_loss"]:
        raise RuntimeError(f"Non-finite smoke loss: {result}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
