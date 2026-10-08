#!/usr/bin/env python
"""Save TwoRoom LeWM or oracle-state AdaLN counterfactual sensitivities."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for source_dir in (SCRIPT_DIR, REPO_ROOT):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import save_counterfactual_sensitivity as base  # noqa: E402
from save_adaln_counterfactual_sensitivity import (  # noqa: E402
    predictor_outputs as adaln_predictor_outputs,
)
from anygoal.wm.toy_predictors import AdaLNActionPredictor  # noqa: E402
from anygoal.paths import checkpoint_root  # noqa: E402


DATASET_PATH = Path.home() / ".stable_worldmodel" / "datasets" / "tworoom.h5"
LEWM_CHECKPOINT = (
    Path.home()
    / ".stable_worldmodel"
    / "anygoal"
    / "lewm_tworoom.ckpt"
)
ADALN_CHECKPOINT = (
    Path.home() / ".stable_worldmodel/anygoal/toy-adaln_tworoom.pt"
)
ACTION_SET_PATH = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "tworoom_counterfactual_sensitivity"
    / "tworoom_counterfactual_actions.json"
)
OUTPUT_DIR = REPO_ROOT / "outputs/analysis/tworoom_counterfactual_sensitivity"


class TwoRoomTransform:
    def __init__(
        self,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        state_minimum: torch.Tensor | None,
        state_maximum: torch.Tensor | None,
        img_size: int,
    ) -> None:
        self.action_mean = action_mean.float()
        self.action_std = action_std.float()
        self.state_minimum = state_minimum
        self.state_maximum = state_maximum
        self.img_size = img_size

    def __call__(self, sample: dict) -> dict:
        sample["agent_position"] = sample["proprio"].float()
        sample["target_position"] = sample["pos_target"].float()
        sample["action"] = (
            sample["action"].float() - self.action_mean
        ) / self.action_std

        if "observation" in sample:
            if self.state_minimum is None or self.state_maximum is None:
                raise ValueError("AdaLN inference requires saved state min/max")
            state = sample["observation"].float()[..., :2]
            sample["state"] = (state - self.state_minimum) / (
                self.state_maximum - self.state_minimum
            )
        if "pixels" in sample:
            pixels = sample["pixels"].float()
            if pixels.max() > 2.0:
                pixels = pixels / 255.0
            if pixels.shape[-2:] != (self.img_size, self.img_size):
                pixels = F.interpolate(
                    pixels,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
            sample["pixels"] = (
                pixels - base.IMAGENET_MEAN.to(dtype=pixels.dtype)
            ) / base.IMAGENET_STD.to(dtype=pixels.dtype)
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=("lewm", "adaln"))
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--action-set", type=Path, default=ACTION_SET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--adaln-layers", default="0,1,2,3,4,5,final")
    parser.add_argument(
        "--intervention-scope", choices=("current", "all"), default="current"
    )
    parser.add_argument("--flush-every", type=int, default=100)
    return parser.parse_args()


def load_action_set(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text())
    if not payload.get("all_points_supported"):
        raise ValueError(f"Action set has unsupported points: {path}")
    actions = [tuple(float(value) for value in action) for action in payload["actions"]]
    if len(actions) != 9 or (0.0, 0.0) not in actions:
        raise ValueError("TwoRoom action set must contain nine points including zero")
    return actions


def load_model_and_transform(args: argparse.Namespace, device: torch.device):
    if args.model_family == "lewm":
        checkpoint = (
            LEWM_CHECKPOINT if args.checkpoint_path is None else args.checkpoint_path
        ).expanduser().resolve()
        args.checkpoint_path = checkpoint
        action_mean, action_std, raw_action_dim = base.get_action_stats(
            args.dataset_path
        )
        transform = TwoRoomTransform(
            action_mean=action_mean,
            action_std=action_std,
            state_minimum=None,
            state_maximum=None,
            img_size=args.img_size,
        )
        model = base.load_serialized_model(args, device)
        return model, transform, action_mean, action_std, raw_action_dim, None

    checkpoint = (
        ADALN_CHECKPOINT if args.checkpoint_path is None else args.checkpoint_path
    ).expanduser().resolve()
    config_path = (
        args.config_path.expanduser().resolve()
        if args.config_path is not None
        else checkpoint.parent / "config.yaml"
    )
    config = OmegaConf.load(config_path)
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs.pop("name")
    model_kwargs.pop("_target_")
    model = AdaLNActionPredictor(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    action_mean = torch.tensor(config.normalization.action.mean).view(1, -1)
    action_std = torch.tensor(config.normalization.action.std).view(1, -1)
    transform = TwoRoomTransform(
        action_mean=action_mean,
        action_std=action_std,
        state_minimum=torch.tensor(config.normalization.state.min),
        state_maximum=torch.tensor(config.normalization.state.max),
        img_size=args.img_size,
    )
    args.checkpoint_path = checkpoint
    return model, transform, action_mean, action_std, model.raw_action_dim, config_path


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.action_set = args.action_set.expanduser().resolve()
    if not args.dataset_path.is_file() or not args.action_set.is_file():
        raise FileNotFoundError(
            f"Missing dataset or action set: {args.dataset_path}, {args.action_set}"
        )
    if args.num_preds != 1:
        raise ValueError("num-preds must equal 1")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    (
        model,
        transform,
        action_mean,
        action_std,
        raw_action_dim,
        config_path,
    ) = load_model_and_transform(args, device)
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint_path)
    model.to(device).eval().requires_grad_(False)

    is_lewm = args.model_family == "lewm"
    expected_action_dim = args.frameskip * raw_action_dim
    model_action_dim = (
        base.get_action_encoder_input_dim(model) if is_lewm else model.action_dim
    )
    if model_action_dim != expected_action_dim:
        raise ValueError(
            f"Model action dimension is {model_action_dim}; expected "
            f"frameskip({args.frameskip}) * raw action dim({raw_action_dim}) = "
            f"{expected_action_dim}"
        )
    keys = ["action", "proprio", "pos_target"]
    keys.append("pixels" if is_lewm else "observation")
    dataset = base.PushtHDF5Dataset(
        path=args.dataset_path,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
        keys_to_load=keys,
        keys_to_cache=[key for key in keys if key != "pixels"],
        transform=transform,
    )
    indexed_dataset = base.IndexedDataset(dataset, max_instances=args.max_instances)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    adaln_count = base.get_adaln_count(model) if is_lewm else len(model.blocks)
    selected_layers = base.parse_layer_spec(args.adaln_layers, adaln_count)
    raw_actions = load_action_set(args.action_set)
    counterfactual_specs = [
        (
            raw_action,
            base.build_action_vector(
                raw_action,
                action_mean=action_mean.to(device),
                action_std=action_std.to(device),
                frameskip=args.frameskip,
                raw_action_dim=raw_action_dim,
                device=device,
            ),
        )
        for raw_action in raw_actions
    ]

    prefix = base.output_prefix_with_job_id(
        f"tworoom_{args.model_family}_counterfactual_sensitivity", args.job_id
    )
    run_output_dir = args.output_dir / args.model_family / prefix
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_output_dir / f"{prefix}.metadata.json"
    atomic_files = [
        {
            "layer": layer,
            "counterfactual_action": list(raw_action),
            "path": str(
                base.make_atomic_csv_path(run_output_dir, prefix, layer, raw_action)
            ),
        }
        for layer in selected_layers
        for raw_action, _ in counterfactual_specs
    ]
    metadata = {
        "dataset_path": str(args.dataset_path),
        "checkpoint_path": str(args.checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "action_set_path": str(args.action_set),
        "model_family": args.model_family,
        "job_id": str(args.job_id),
        "output_prefix": prefix,
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "intervention_scope": args.intervention_scope,
        "counterfactual_actions": [list(action) for action in raw_actions],
        "adaln_count": adaln_count,
        "selected_layers": selected_layers,
        "raw_action_dim": raw_action_dim,
        "action_mean": action_mean.view(-1).tolist(),
        "action_std": action_std.view(-1).tolist(),
        "total_instances": len(indexed_dataset),
        "storage_format": "atomic_layer_action_csv",
        "atomic_files": atomic_files,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    columns = [
        "dataset_index", "episode", "start", "global_t_index", "frame_t",
        "ball_x", "ball_y", "block_x", "block_y", "distance",
        "counterfactual_effect",
    ]
    rows_written = 0
    time_index = args.history_size - 1
    with ExitStack() as stack:
        writers = {}
        csv_files = []
        for spec in atomic_files:
            csv_file = stack.enter_context(Path(spec["path"]).open("w", newline=""))
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            writers[(spec["layer"], tuple(spec["counterfactual_action"]))] = writer
            csv_files.append(csv_file)

        with torch.inference_mode():
            progress = base.get_progress(
                loader, total=math.ceil(len(indexed_dataset) / args.batch_size)
            )
            for batch in progress:
                batch = base.move_batch_to_device(batch, device)
                real_actions = torch.nan_to_num(
                    batch["action"][:, : args.history_size]
                )
                if is_lewm:
                    encoded = model.encode(
                        {"pixels": batch["pixels"], "action": batch["action"]}
                    )
                    states = encoded["emb"][:, : args.history_size]
                    real_outputs = base.predictor_outputs(
                        model,
                        states,
                        encoded["act_emb"][:, : args.history_size],
                        selected_layers,
                    )
                else:
                    states = torch.nan_to_num(
                        batch["state"][:, : args.history_size]
                    )
                    real_outputs = adaln_predictor_outputs(
                        model, states, real_actions, selected_layers
                    )

                agent = batch["agent_position"][:, time_index]
                target = batch["target_position"][:, time_index]
                distance = (agent - target).norm(p=2, dim=-1)
                dataset_indices = batch["dataset_index"].cpu().numpy()
                episodes = batch["episode"].cpu().numpy()
                starts = batch["start"].cpu().numpy()
                common_columns = (
                    dataset_indices,
                    episodes,
                    starts,
                    dataset.offsets[episodes] + starts + time_index * args.frameskip,
                    starts + time_index * args.frameskip,
                    agent[:, 0].cpu().numpy(),
                    agent[:, 1].cpu().numpy(),
                    target[:, 0].cpu().numpy(),
                    target[:, 1].cpu().numpy(),
                    distance.cpu().numpy(),
                )
                for raw_action, counterfactual_action in counterfactual_specs:
                    cf_actions = base.make_counterfactual_context(
                        real_actions,
                        counterfactual_action,
                        scope=args.intervention_scope,
                    )
                    if is_lewm:
                        cf_outputs = base.predictor_outputs(
                            model,
                            states,
                            model.action_encoder(cf_actions),
                            selected_layers,
                        )
                    else:
                        cf_outputs = adaln_predictor_outputs(
                            model, states, cf_actions, selected_layers
                        )
                    for layer in selected_layers:
                        effect = base.layer_counterfactual_effect(
                            real_outputs, cf_outputs, layer
                        ).cpu().numpy()
                        writers[(layer, tuple(raw_action))].writerows(
                            zip(*common_columns, effect)
                        )
                rows_written += distance.shape[0]
                if rows_written % (args.flush_every * args.batch_size) == 0:
                    for csv_file in csv_files:
                        csv_file.flush()

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved {len(atomic_files)} atomic CSV files under: {run_output_dir}")
    print(f"Rows written per CSV: {rows_written}")


if __name__ == "__main__":
    main()
