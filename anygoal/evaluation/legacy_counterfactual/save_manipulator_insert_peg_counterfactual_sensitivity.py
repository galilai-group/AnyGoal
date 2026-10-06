#!/usr/bin/env python
"""Save Insert Peg LeWM/AdaLN counterfactual sensitivities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for source_dir in (SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "scripts"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import save_counterfactual_sensitivity as base  # noqa: E402
from planning_lewm import load_lewm_cost_model  # noqa: E402
from save_adaln_counterfactual_sensitivity import (  # noqa: E402
    predictor_outputs as adaln_predictor_outputs,
)
from anygoal.datasets.manipulator_true_state import (  # noqa: E402
    ManipulatorInsertPegStateAdapter,
)
from anygoal.wm.toy_predictors import AdaLNActionPredictor  # noqa: E402
from anygoal.paths import checkpoint_root  # noqa: E402


DATASET_PATH = (
    Path.home() / ".stable_worldmodel/datasets/"
    "manipulator_insert_peg_weak_expert_train.h5"
)
LEWM_CHECKPOINT = (
    checkpoint_root()
    / "checkpoints/20260910_manipulator_insert_peg_lewm/weights_epoch_10.pt"
)
LEWM_CONFIG = REPO_ROOT / "configs/lewm/planning/manipulator_insert_peg.yaml"
ADALN_CHECKPOINT = (
    checkpoint_root()
    / "20260910_manipulator_insert_peg_toy_predictors"
    / "20260910_manipulator_insert_peg_essential_adaln_01"
    / "weights_epoch_100.pt"
)
ACTION_SET_PATH = (
    REPO_ROOT / "paper/results/manipulator_insert_peg_counterfactual_sensitivity/"
    "manipulator_insert_peg_counterfactual_actions.json"
)
OUTPUT_DIR = (
    REPO_ROOT / "paper/results/manipulator_insert_peg_counterfactual_sensitivity"
)


class InsertPegTransform:
    def __init__(self, action_mean, action_std, state_mean, state_std, img_size):
        self.action_mean = action_mean.float()
        self.action_std = action_std.float()
        self.state_mean = state_mean
        self.state_std = state_std
        self.img_size = img_size
        self.state_adapter = ManipulatorInsertPegStateAdapter(False)

    def __call__(self, sample: dict) -> dict:
        observation = sample["observation"].float()
        sample["raw_observation"] = observation
        sample["action"] = (
            sample["action"].float() - self.action_mean
        ) / self.action_std
        if self.state_mean is not None:
            state = self.state_adapter.encode_state(observation)
            sample["state"] = (state - self.state_mean) / self.state_std
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
                pixels - base.IMAGENET_MEAN.to(pixels.dtype)
            ) / base.IMAGENET_STD.to(pixels.dtype)
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
    parser.add_argument("--batch-size", type=int, default=256)
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


def load_actions(path: Path) -> list[tuple[float, ...]]:
    payload = json.loads(path.read_text())
    actions = [tuple(map(float, action)) for action in payload["actions"]]
    if not payload.get("all_actions_supported") or len(actions) != 9:
        raise ValueError(f"Expected nine supported actions in {path}")
    if not any(all(value == 0.0 for value in action) for action in actions):
        raise ValueError("Action set must include the zero action")
    return actions


def load_model(args, device):
    if args.model_family == "lewm":
        checkpoint = (args.checkpoint_path or LEWM_CHECKPOINT).resolve()
        config_path = (args.config_path or LEWM_CONFIG).resolve()
        config = OmegaConf.load(config_path)
        config.model.checkpoint = str(checkpoint)
        base.patch_transformers_vit_pickle_compat()
        model = load_lewm_cost_model(config, checkpoint.parent)
        base.patch_loaded_model_compat(model)
        action_mean, action_std, raw_action_dim = base.get_action_stats(
            args.dataset_path
        )
        transform = InsertPegTransform(
            action_mean, action_std, None, None, args.img_size
        )
    else:
        checkpoint = (args.checkpoint_path or ADALN_CHECKPOINT).resolve()
        config_path = (
            args.config_path.resolve()
            if args.config_path
            else checkpoint.parent / "config.yaml"
        )
        config = OmegaConf.load(config_path)
        kwargs = OmegaConf.to_container(config.model, resolve=True)
        kwargs.pop("name")
        kwargs.pop("_target_")
        model = AdaLNActionPredictor(**kwargs)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        action_mean = torch.tensor(config.normalization.action.mean).view(1, -1)
        action_std = torch.tensor(config.normalization.action.std).view(1, -1)
        state_mean = torch.tensor(config.normalization.state.mean)
        state_std = torch.tensor(config.normalization.state.std)
        transform = InsertPegTransform(
            action_mean, action_std, state_mean, state_std, args.img_size
        )
        raw_action_dim = model.raw_action_dim
    args.checkpoint_path = checkpoint
    return model.to(device).eval().requires_grad_(False), transform, (
        action_mean,
        action_std,
        raw_action_dim,
        config_path,
    )


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.action_set = args.action_set.expanduser().resolve()
    if args.num_preds != 1:
        raise ValueError("num-preds must equal 1")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model, transform, model_info = load_model(args, device)
    action_mean, action_std, raw_action_dim, config_path = model_info
    is_lewm = args.model_family == "lewm"
    model_action_dim = (
        base.get_action_encoder_input_dim(model) if is_lewm else model.action_dim
    )
    expected_action_dim = args.frameskip * raw_action_dim
    if model_action_dim != expected_action_dim:
        raise ValueError(f"Model action dim {model_action_dim} != {expected_action_dim}")

    keys = ["action", "observation", "pixels"] if is_lewm else [
        "action", "observation"
    ]
    dataset = base.PushtHDF5Dataset(
        path=args.dataset_path,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
        keys_to_load=keys,
        keys_to_cache=[key for key in keys if key != "pixels"],
        transform=transform,
    )
    indexed_dataset = base.IndexedDataset(dataset, args.max_instances)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    layer_count = base.get_adaln_count(model) if is_lewm else len(model.blocks)
    layers = base.parse_layer_spec(args.adaln_layers, layer_count)
    raw_actions = load_actions(args.action_set)
    counterfactuals = [
        (
            action,
            base.build_action_vector(
                action,
                action_mean=action_mean.to(device),
                action_std=action_std.to(device),
                frameskip=args.frameskip,
                raw_action_dim=raw_action_dim,
                device=device,
            ),
        )
        for action in raw_actions
    ]

    prefix = base.output_prefix_with_job_id(
        f"manipulator_insert_peg_{args.model_family}_counterfactual_sensitivity",
        args.job_id,
    )
    run_dir = args.output_dir / args.model_family / prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_files = [
        {
            "layer": layer,
            "counterfactual_action": list(action),
            "path": str(base.make_atomic_csv_path(run_dir, prefix, layer, action)),
        }
        for layer in layers
        for action, _ in counterfactuals
    ]
    metadata = {
        "dataset_path": str(args.dataset_path),
        "checkpoint_path": str(args.checkpoint_path),
        "config_path": str(config_path),
        "action_set_path": str(args.action_set),
        "model_family": args.model_family,
        "job_id": str(args.job_id),
        "output_prefix": prefix,
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "intervention_scope": args.intervention_scope,
        "counterfactual_actions": [list(action) for action in raw_actions],
        "adaln_count": layer_count,
        "selected_layers": layers,
        "raw_action_dim": raw_action_dim,
        "action_mean": action_mean.view(-1).tolist(),
        "action_std": action_std.view(-1).tolist(),
        "total_instances": len(indexed_dataset),
        "storage_format": "atomic_layer_action_csv",
        "atomic_files": atomic_files,
    }
    metadata_path = run_dir / f"{prefix}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    columns = [
        "dataset_index", "episode", "start", "global_t_index", "frame_t",
        "hand_x", "hand_z", "peg_x", "peg_z", "target_x", "target_z",
        "distance", "hand_peg_distance", "target_peg_distance",
        "hand_target_distance", "min_hand_distance", "counterfactual_effect",
    ]
    time_index = args.history_size - 1
    rows_written = 0
    with ExitStack() as stack:
        writers, csv_files = {}, []
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
                real_actions = torch.nan_to_num(batch["action"][:, : args.history_size])
                if is_lewm:
                    encoded = model.encode(
                        {"pixels": batch["pixels"], "action": batch["action"]}
                    )
                    states = encoded["emb"][:, : args.history_size]
                    real_outputs = base.predictor_outputs(
                        model, states, encoded["act_emb"][:, : args.history_size], layers
                    )
                else:
                    states = torch.nan_to_num(batch["state"][:, : args.history_size])
                    real_outputs = adaln_predictor_outputs(
                        model, states, real_actions, layers
                    )
                observation = batch["raw_observation"][:, time_index]
                hand, peg, target = observation[:, 29:31], observation[:, 33:35], observation[:, 40:42]
                hand_peg = (hand - peg).norm(dim=-1)
                target_peg = (target - peg).norm(dim=-1)
                hand_target = (hand - target).norm(dim=-1)
                min_hand = torch.minimum(hand_target, hand_peg)
                indices = batch["dataset_index"].cpu().numpy()
                episodes = batch["episode"].cpu().numpy()
                starts = batch["start"].cpu().numpy()
                common = (
                    indices, episodes, starts,
                    dataset.offsets[episodes] + starts + time_index * args.frameskip,
                    starts + time_index * args.frameskip,
                    hand[:, 0].cpu().numpy(), hand[:, 1].cpu().numpy(),
                    peg[:, 0].cpu().numpy(), peg[:, 1].cpu().numpy(),
                    target[:, 0].cpu().numpy(), target[:, 1].cpu().numpy(),
                    hand_peg.cpu().numpy(), hand_peg.cpu().numpy(),
                    target_peg.cpu().numpy(), hand_target.cpu().numpy(),
                    min_hand.cpu().numpy(),
                )
                for raw_action, counterfactual_action in counterfactuals:
                    cf_actions = base.make_counterfactual_context(
                        real_actions, counterfactual_action,
                        scope=args.intervention_scope,
                    )
                    if is_lewm:
                        cf_outputs = base.predictor_outputs(
                            model, states, model.action_encoder(cf_actions), layers
                        )
                    else:
                        cf_outputs = adaln_predictor_outputs(
                            model, states, cf_actions, layers
                        )
                    for layer in layers:
                        effect = base.layer_counterfactual_effect(
                            real_outputs, cf_outputs, layer
                        ).cpu().numpy()
                        writers[(layer, tuple(raw_action))].writerows(
                            zip(*common, effect)
                        )
                rows_written += len(indices)
                if rows_written % (args.flush_every * args.batch_size) == 0:
                    for csv_file in csv_files:
                        csv_file.flush()
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved {len(atomic_files)} CSV files; {rows_written} rows per CSV")


if __name__ == "__main__":
    main()
