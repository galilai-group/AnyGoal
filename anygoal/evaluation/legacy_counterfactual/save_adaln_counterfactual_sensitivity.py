#!/usr/bin/env python
"""Save PushT oracle-state AdaLN counterfactual-action sensitivities."""

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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for source_dir in (SCRIPT_DIR, REPO_ROOT):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import save_counterfactual_sensitivity as base  # noqa: E402
from anygoal.datasets import PushTTrueStateAdapter  # noqa: E402
from anygoal.wm.toy_predictors import AdaLNActionPredictor  # noqa: E402
from anygoal.paths import checkpoint_root  # noqa: E402


DEFAULT_CHECKPOINT = (
    checkpoint_root()
    / "toy_predictors_20260723_frameskip5_10d"
    / "20260723_pusht_adaln"
    / "weights_epoch_100.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "paper" / "results" / "counterfactual_sensitivity" / "adaln"
)


class ToyPredictorTransform:
    """Reproduce the adapter and z-scoring saved with toy-predictor training."""

    def __init__(self, config) -> None:
        self.adapter = PushTTrueStateAdapter(
            goal_position=tuple(config.data.adapter.goal_position),
            goal_angle=float(config.data.adapter.goal_angle),
        )
        self.state_mean = torch.tensor(config.normalization.state.mean).float()
        self.state_std = torch.tensor(config.normalization.state.std).float()
        self.action_mean = torch.tensor(config.normalization.action.mean).float()
        self.action_std = torch.tensor(config.normalization.action.std).float()

    def __call__(self, sample: dict) -> dict:
        raw_state = sample["state"].float()
        sample["ball_position"] = raw_state[..., :2]
        sample["block_position"] = raw_state[..., 2:4]
        sample = self.adapter(sample)
        sample["state"] = (sample["state"].float() - self.state_mean) / self.state_std
        sample["action"] = (sample["action"].float() - self.action_mean) / self.action_std
        return sample


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser(description=__doc__)
    parser.set_defaults(
        checkpoint_path=DEFAULT_CHECKPOINT,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_prefix="adaln_counterfactual_sensitivity",
        adaln_layers="0,1,2,3,4,5,final",
        batch_size=1024,
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Training config; defaults to config.yaml beside the checkpoint.",
    )
    return parser


def predictor_outputs(
    model: AdaLNActionPredictor,
    states: torch.Tensor,
    actions: torch.Tensor,
    selected_layers: list[int | str],
) -> dict[int | str, torch.Tensor]:
    captures: dict[int | str, torch.Tensor] = {}
    handles = []
    for layer in selected_layers:
        if not isinstance(layer, int):
            continue

        def hook(_module, _inputs, output, idx=layer):
            captures[idx] = output[:, -1].detach()

        handles.append(model.blocks[layer].register_forward_hook(hook))
    try:
        prediction = model(states, actions)
    finally:
        for handle in handles:
            handle.remove()
    if "final" in selected_layers:
        captures["final"] = prediction.detach()
    return captures


def main() -> None:
    args = build_parser().parse_args()
    if args.history_size < 1 or args.num_preds != 1:
        raise ValueError("history-size must be positive and num-preds must equal 1")
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    required_dataset = base.default_dataset_path("pusht_expert_train.h5").resolve()
    if args.dataset_path != required_dataset:
        raise ValueError(
            f"Expected PushT training dataset {required_dataset}, "
            f"got {args.dataset_path}"
        )
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint_path)

    config_path = (
        args.config_path.expanduser().resolve()
        if args.config_path is not None
        else args.checkpoint_path.parent / "config.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)
    if str(config.model.name) != "adaln":
        raise ValueError(f"Expected an AdaLN training config, got {config.model.name!r}")
    if (
        int(config.history_size) != args.history_size
        or int(config.data.frameskip) != args.frameskip
    ):
        raise ValueError("--history-size and --frameskip must match the training config")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    transform = ToyPredictorTransform(config)
    dataset = base.PushtHDF5Dataset(
        path=args.dataset_path,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
        keys_to_load=["state", "action"],
        keys_to_cache=["state", "action"],
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

    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs.pop("name")
    model_kwargs.pop("_target_")
    model = AdaLNActionPredictor(**model_kwargs)
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.to(device).eval().requires_grad_(False)
    selected_layers = base.parse_layer_spec(args.adaln_layers, len(model.blocks))

    action_mean = transform.action_mean.view(1, -1)
    action_std = transform.action_std.view(1, -1)
    counterfactual_specs = [
        (
            raw_action,
            base.build_action_vector(
                raw_action,
                action_mean=action_mean.to(device),
                action_std=action_std.to(device),
                frameskip=args.frameskip,
                raw_action_dim=model.raw_action_dim,
                device=device,
            ),
        )
        for raw_action in args.counterfactual_actions
    ]

    output_prefix = base.output_prefix_with_job_id(args.output_prefix, args.job_id)
    run_output_dir = args.output_dir.expanduser().resolve() / output_prefix
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_output_dir / f"{output_prefix}.metadata.json"
    atomic_files = [
        {
            "layer": layer,
            "counterfactual_action": list(raw_action),
            "path": str(
                base.make_atomic_csv_path(
                    run_output_dir, output_prefix, layer, raw_action
                )
            ),
        }
        for layer in selected_layers
        for raw_action, _ in counterfactual_specs
    ]
    base.write_metadata(
        metadata_path,
        args=args,
        output_prefix=output_prefix,
        adaln_count=len(model.blocks),
        selected_layers=selected_layers,
        action_mean=action_mean,
        action_std=action_std,
        raw_action_dim=model.raw_action_dim,
        action_encoder_input_dim=model.action_dim,
        total_instances=len(indexed_dataset),
        output_dir=run_output_dir,
        atomic_files=atomic_files,
    )
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        model_family="adaln_true_state",
        config_path=str(config_path),
        state_representation=list(PushTTrueStateAdapter.state_names),
        sensitivity=(
            "L2(real output - counterfactual output) in the model's normalized "
            "representation"
        ),
    )
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
                states = torch.nan_to_num(batch["state"][:, : args.history_size])
                real_actions = torch.nan_to_num(batch["action"][:, : args.history_size])
                real_outputs = predictor_outputs(model, states, real_actions, selected_layers)

                ball = batch["ball_position"][:, time_index]
                block = batch["block_position"][:, time_index]
                distance = (ball - block).norm(p=2, dim=-1)
                dataset_indices = batch["dataset_index"].cpu().numpy()
                episodes = batch["episode"].cpu().numpy()
                starts = batch["start"].cpu().numpy()
                common_columns = (
                    dataset_indices,
                    episodes,
                    starts,
                    dataset.offsets[episodes] + starts + time_index * args.frameskip,
                    starts + time_index * args.frameskip,
                    ball[:, 0].cpu().numpy(),
                    ball[:, 1].cpu().numpy(),
                    block[:, 0].cpu().numpy(),
                    block[:, 1].cpu().numpy(),
                    distance.cpu().numpy(),
                )

                for raw_action, counterfactual_action in counterfactual_specs:
                    counterfactual_actions = base.make_counterfactual_context(
                        real_actions,
                        counterfactual_action,
                        scope=args.intervention_scope,
                    )
                    counterfactual_outputs = predictor_outputs(
                        model, states, counterfactual_actions, selected_layers
                    )
                    for layer in selected_layers:
                        effect = base.layer_counterfactual_effect(
                            real_outputs, counterfactual_outputs, layer
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
