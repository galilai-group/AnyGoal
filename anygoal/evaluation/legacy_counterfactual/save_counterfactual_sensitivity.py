#!/usr/bin/env python
# ruff: noqa: E402
"""Save LeWM counterfactual-action sensitivities to atomic CSV files."""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import csv
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def add_extra_site_packages() -> None:
    """Optionally add dependency paths after the active torch is imported."""
    extra_paths = os.environ.get("LEWM_EXTRA_SITE_PACKAGES", "")
    for path in reversed([item for item in extra_paths.split(os.pathsep) if item]):
        if path not in sys.path:
            sys.path.insert(0, path)


add_extra_site_packages()

import h5py
import hdf5plugin  # noqa: F401
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
LEWM_MODULE_DIR = REPO_ROOT / "worldmodels" / "lewm"
for source_dir in (REPO_ROOT, LEWM_MODULE_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

RESULTS_DIR = REPO_ROOT / "outputs" / "analysis"


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PushtInferenceTransform:
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
        mean = IMAGENET_MEAN.to(dtype=pixels.dtype)
        std = IMAGENET_STD.to(dtype=pixels.dtype)
        sample["pixels"] = (pixels - mean) / std

        action = sample["action"].float()
        sample["action"] = (action - self.action_mean) / self.action_std
        sample["proprio"] = sample["proprio"].float()
        sample["state"] = sample["state"].float()
        return sample


class PushtHDF5Dataset(Dataset):
    def __init__(
        self,
        *,
        path: Path,
        frameskip: int,
        num_steps: int,
        keys_to_load: list[str],
        keys_to_cache: list[str],
        transform: PushtInferenceTransform,
    ) -> None:
        self.h5_path = Path(path)
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        self.h5_file: h5py.File | None = None
        self._keys = keys_to_load
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024) as h5_file:
            self.lengths = h5_file["ep_len"][:]
            self.offsets = h5_file["ep_offset"][:]
            for key in keys_to_cache:
                self._cache[key] = h5_file[key][:]

        self.clip_indices = [
            (episode, start)
            for episode, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)

    def _load_slice(self, episode: int, start: int, end: int) -> dict:
        self._open()
        assert self.h5_file is not None
        global_start = int(self.offsets[episode] + start)
        global_end = int(self.offsets[episode] + end)

        sample = {}
        for key in self._keys:
            source = self._cache if key in self._cache else self.h5_file
            data = source[key][global_start:global_end]
            if key != "action":
                data = data[:: self.frameskip]
            value = torch.from_numpy(np.asarray(data))
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                value = value.permute(0, 3, 1, 2)
            sample[key] = value
        return self.transform(sample)

    def __getitem__(self, idx: int) -> dict:
        episode, start = self.clip_indices[idx]
        sample = self._load_slice(episode, start, start + self.span)
        sample["action"] = sample["action"].reshape(self.num_steps, -1)
        return sample


class IndexedDataset(Dataset):
    def __init__(self, dataset: PushtHDF5Dataset, max_instances: int | None = None) -> None:
        self.dataset = dataset
        self.max_instances = max_instances

    def __len__(self) -> int:
        if self.max_instances is None:
            return len(self.dataset)
        return min(len(self.dataset), self.max_instances)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        episode, start = self.dataset.clip_indices[idx]
        item["dataset_index"] = torch.tensor(idx, dtype=torch.long)
        item["episode"] = torch.tensor(episode, dtype=torch.long)
        item["start"] = torch.tensor(start, dtype=torch.long)
        return item


def default_swm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser()


def default_dataset_path(filename: str) -> Path:
    swm_home = default_swm_home()
    path = swm_home / "datasets" / filename
    legacy_path = swm_home / filename
    return path if path.exists() or not legacy_path.exists() else legacy_path


def parse_tuple_list(value: str) -> list[tuple[float, ...]]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, tuple) and all(isinstance(x, (int, float)) for x in parsed):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("expected a list of tuples, e.g. '[(0, 0), (1, 0)]'")

    out = []
    for item in parsed:
        if not isinstance(item, tuple):
            raise argparse.ArgumentTypeError("each counterfactual action must be a tuple")
        if not item or not all(isinstance(x, (int, float)) for x in item):
            raise argparse.ArgumentTypeError("counterfactual action tuples must contain numbers")
        out.append(tuple(float(x) for x in item))
    return out


def parse_layer_spec(value: str, adaln_count: int) -> list[int | str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("at least one --adaln-layers entry is required")

    layers: list[int | str] = []
    valid = [str(i) for i in range(adaln_count)] + ["final"]
    for item in items:
        if item == "final":
            layers.append(item)
            continue
        try:
            layer_idx = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid layer {item!r}; valid options are {valid}") from exc
        if not 0 <= layer_idx < adaln_count:
            raise ValueError(f"invalid layer {layer_idx}; valid options are {valid}")
        layers.append(layer_idx)

    return layers


def get_adaln_count(model: torch.nn.Module) -> int:
    return sum(
        1
        for layer in model.predictor.transformer.layers
        if hasattr(layer, "adaLN_modulation")
    )


def get_action_encoder_input_dim(model: torch.nn.Module) -> int | None:
    action_encoder = getattr(model, "action_encoder", None)
    patch_embed = getattr(action_encoder, "patch_embed", None)
    return getattr(patch_embed, "in_channels", None)


def build_action_vector(
    raw_action: tuple[float, ...],
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    frameskip: int,
    raw_action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.tensor(raw_action, dtype=torch.float32, device=device)
    if values.numel() == raw_action_dim:
        values = values.view(1, raw_action_dim).repeat(frameskip, 1)
    elif values.numel() == frameskip * raw_action_dim:
        values = values.view(frameskip, raw_action_dim)
    else:
        raise ValueError(
            f"counterfactual action {raw_action} has length {values.numel()}, "
            f"but expected {raw_action_dim} or {frameskip * raw_action_dim}"
        )

    normalized = (values - action_mean) / action_std
    return normalized.reshape(-1).to(device)


def make_counterfactual_context(
    action_context: torch.Tensor,
    counterfactual_action: torch.Tensor,
    *,
    scope: str,
) -> torch.Tensor:
    cf_action = action_context.clone()
    if scope == "current":
        cf_action[:, -1, :] = counterfactual_action
    elif scope == "all":
        cf_action[:, :, :] = counterfactual_action
    else:
        raise ValueError(f"unknown intervention scope: {scope}")
    return cf_action


def predictor_outputs(
    model: torch.nn.Module,
    ctx_emb: torch.Tensor,
    ctx_act_emb: torch.Tensor,
    selected_layers: Iterable[int | str],
) -> dict[int | str, torch.Tensor]:
    selected = list(selected_layers)
    layer_ids = [item for item in selected if isinstance(item, int)]
    captures: dict[int | str, torch.Tensor] = {}
    handles = []

    for layer_idx in layer_ids:
        layer = model.predictor.transformer.layers[layer_idx]

        def hook(_module, _inputs, output, idx=layer_idx):
            captures[idx] = output[:, -1, :].detach()

        handles.append(layer.register_forward_hook(hook))

    try:
        pred = model.predict(ctx_emb, ctx_act_emb)
    finally:
        for handle in handles:
            handle.remove()

    if "final" in selected:
        captures["final"] = pred[:, -1, :].detach()
    return captures


def counterfactual_effect(
    real_outputs: dict[int | str, torch.Tensor],
    cf_outputs: dict[int | str, torch.Tensor],
    selected_layers: list[int | str],
) -> torch.Tensor:
    values = []
    for layer in selected_layers:
        diff = real_outputs[layer] - cf_outputs[layer]
        values.append(diff.norm(p=2, dim=-1))
    return torch.stack(values, dim=0).mean(dim=0)


def layer_counterfactual_effect(
    real_outputs: dict[int | str, torch.Tensor],
    cf_outputs: dict[int | str, torch.Tensor],
    layer: int | str,
) -> torch.Tensor:
    return (real_outputs[layer] - cf_outputs[layer]).norm(p=2, dim=-1)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def patch_transformers_vit_pickle_compat() -> None:
    """Restore old ViT class names used by serialized checkpoints."""
    try:
        from transformers.models.vit import modeling_vit
        from transformers.pytorch_utils import apply_chunking_to_forward
    except ImportError:
        return

    if hasattr(modeling_vit, "ViTMLP"):
        return

    class ViTMLP(torch.nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.chunk_size_feed_forward = config.chunk_size_feed_forward
            self.seq_len_dim = 1
            self.intermediate = modeling_vit.ViTIntermediate(config)
            self.output = modeling_vit.ViTOutput(config)
            self.layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
            if hasattr(self, "fc1") and hasattr(self, "fc2"):
                hidden_states = self.fc1(hidden_states)
                hidden_states = self.activation_fn(hidden_states)
                return self.fc2(hidden_states)

            layer_output = self.layernorm(hidden_states)
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                layer_output,
            )
            return self.output(layer_output, input_tensor)

        def feed_forward_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.intermediate(hidden_states)

    ViTMLP.__module__ = modeling_vit.__name__
    modeling_vit.ViTMLP = ViTMLP


def patch_loaded_model_compat(model: torch.nn.Module) -> None:
    """Patch serialized HF ViT objects across Transformers minor versions."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None:
            continue
        if not hasattr(config, "output_attentions"):
            config.output_attentions = getattr(config, "_output_attentions", False)
        if not hasattr(config, "output_hidden_states"):
            config.output_hidden_states = getattr(config, "_output_hidden_states", False)
        if not hasattr(config, "use_return_dict"):
            config.__dict__["use_return_dict"] = getattr(config, "return_dict", True)

    vit = getattr(model, "encoder", None)
    if vit is None or not hasattr(vit, "layers"):
        return
    if hasattr(vit, "encoder"):
        return

    def legacy_attention_forward(attention, hidden_states, output_attentions: bool):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_heads = attention.num_attention_heads
        head_dim = attention.head_dim

        def shape(states):
            return states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        query = shape(attention.q_proj(hidden_states))
        key = shape(attention.k_proj(hidden_states))
        value = shape(attention.v_proj(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) * attention.scaling
        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=attention.attention_dropout, training=attention.training)
        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = attention.o_proj(context)
        return output, probs if output_attentions else None

    def legacy_layer_forward(layer, hidden_states, output_attentions: bool):
        attention_input = layer.layernorm_before(hidden_states)
        attention_output, attention_probs = legacy_attention_forward(
            layer.attention,
            attention_input,
            output_attentions,
        )
        hidden_states = hidden_states + layer.dropout(attention_output)
        mlp_input = layer.layernorm_after(hidden_states)
        if hasattr(layer.mlp, "fc1"):
            mlp_output = layer.mlp(mlp_input)
        else:
            mlp_output = layer.mlp(mlp_input, hidden_states)
        hidden_states = hidden_states + layer.dropout(mlp_output)
        return hidden_states, attention_probs

    def legacy_vit_forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = (
            getattr(self.config, "use_return_dict", getattr(self.config, "return_dict", True))
            if return_dict is None
            else return_dict
        )
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attention_probs = legacy_layer_forward(layer, hidden_states, output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attention_probs,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        sequence_output = self.layernorm(hidden_states)

        if not return_dict:
            output = (sequence_output, None)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            return output

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    vit.pooler = None
    vit.forward = types.MethodType(legacy_vit_forward, vit)


def get_action_stats(dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    with h5py.File(dataset_path, "r") as h5_file:
        action = torch.from_numpy(np.asarray(h5_file["action"][:])).float()
    action = action[~torch.isnan(action).any(dim=1)]
    mean = action.mean(0, keepdim=True)
    std = action.std(0, keepdim=True).clamp_min(1e-6)
    return mean, std, action.shape[1]


def get_progress(iterable, total: int):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total)
    except ImportError:
        return iterable


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    output_prefix: str,
    adaln_count: int,
    selected_layers: list[int | str],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    raw_action_dim: int,
    action_encoder_input_dim: int | None,
    total_instances: int,
    output_dir: Path,
    atomic_files: list[dict] | None = None,
) -> None:
    metadata = {
        "dataset_path": str(args.dataset_path),
        "checkpoint_path": str(args.checkpoint_path),
        "job_id": str(args.job_id),
        "output_prefix": output_prefix,
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "y_choice": args.y_choice,
        "y_choice_note": "Ignored during inference CSV storage; aggregate atomic layer/action CSVs at plot time.",
        "counterfactual_actions": args.counterfactual_actions,
        "zero_action": args.zero_action,
        "intervention_scope": args.intervention_scope,
        "adaln_count": adaln_count,
        "adaln_layer_options": [str(i) for i in range(adaln_count)] + ["final"],
        "selected_layers": selected_layers,
        "raw_action_dim": raw_action_dim,
        "raw_action_projector": action_encoder_input_dim is not None,
        "action_encoder_input_dim": action_encoder_input_dim,
        "action_mean": action_mean.squeeze(0).tolist(),
        "action_std": action_std.squeeze(0).tolist(),
        "total_instances": total_instances,
        "output_dir": str(output_dir),
        "storage_format": "atomic_layer_action_csv",
        "csv_columns": [
            "dataset_index",
            "episode",
            "start",
            "global_t_index",
            "frame_t",
            "ball_x",
            "ball_y",
            "block_x",
            "block_y",
            "distance",
            "counterfactual_effect",
        ],
        "atomic_files": atomic_files or [],
    }
    for key in ("model_family", "model_run"):
        if hasattr(args, key):
            metadata[key] = str(getattr(args, key))
    path.write_text(json.dumps(metadata, indent=2))


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    swm_home = default_swm_home()
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Save LeWM predictor counterfactual-action sensitivities for every "
            "valid trajectory window."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset_path("pusht_expert_train.h5"),
        help="PushT training HDF5 path. Other datasets are rejected.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=swm_home / "anygoal" / "lewm_pusht.ckpt",
        help="Serialized LeWM object checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "counterfactual_sensitivity",
        help="Parent directory for atomic CSVs and metadata JSON.",
    )
    parser.add_argument("--output-prefix", default="counterfactual_sensitivity")
    parser.add_argument(
        "--job-id",
        default=os.environ.get("SLURM_JOB_ID", "local"),
        help="Identifier appended to every output filename. Defaults to SLURM_JOB_ID.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Debug only. Default processes every valid training instance without sampling.",
    )
    parser.add_argument(
        "--y-choice",
        choices=["zero", "sum"],
        default="zero",
        help=(
            "zero: ||f(z,a)-f(z,do(a)=zero_action)||_2. "
            "sum: sum over --counterfactual-actions."
        ),
    )
    parser.add_argument(
        "--zero-action",
        type=parse_tuple_list,
        default=parse_tuple_list("[(0, 0)]"),
        help="Choice zero action tuple. Use a length-2 tuple or length frameskip*2 tuple.",
    )
    parser.add_argument(
        "--counterfactual-actions",
        type=parse_tuple_list,
        default=parse_tuple_list("[(0, 0), (0.25, 0), (-0.25, 0), (0.25, -0.25), (0.25, 0.25), (-0.25, -0.25), (-0.25, 0.25), (0, -0.25), (0, 0.25)]"),
        help="A for y-choice=sum, e.g. '[(0, 0), (1, 0), (-1, 0)]'.",
    )
    parser.add_argument(
        "--adaln-layers",
        default="final",
        help=(
            "Comma-separated outputs to compare. Options are 0..5 for the current "
            "PushT checkpoint's six AdaLN/ConditionalBlock outputs, plus final "
            "for model.predict output after pred_proj. Multiple entries are averaged."
        ),
    )
    parser.add_argument(
        "--intervention-scope",
        choices=["current", "all"],
        default="current",
        help=(
            "current replaces only the last context action, corresponding to time t. "
            "all replaces every action in the t-2,t-1,t context."
        ),
    )
    parser.add_argument("--flush-every", type=int, default=100)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def output_prefix_with_job_id(output_prefix: str, job_id: str) -> str:
    job_id = str(job_id).strip() or "local"
    job_token = f"job{job_id}"
    if job_token in output_prefix or job_id in output_prefix:
        return output_prefix
    return f"{output_prefix}_{job_token}"


def format_float_token(value: float) -> str:
    text = f"{float(value):g}"
    text = text.replace("-", "m").replace("+", "")
    return text.replace(".", "p")


def layer_token(layer: int | str) -> str:
    return str(layer)


def action_token(raw_action: tuple[float, ...]) -> str:
    return "_".join(format_float_token(value) for value in raw_action)


def make_atomic_csv_path(output_dir: Path, output_prefix: str, layer: int | str, raw_action: tuple[float, ...]) -> Path:
    return output_dir / f"{output_prefix}_layer{layer_token(layer)}_action{action_token(raw_action)}.csv"


def load_serialized_model(
    args: argparse.Namespace, device: torch.device
) -> torch.nn.Module:
    patch_transformers_vit_pickle_compat()
    model = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    patch_loaded_model_compat(model)
    return model


def main(
    *,
    parse_args_fn=parse_args,
    transform_class=PushtInferenceTransform,
    model_loader=load_serialized_model,
) -> None:
    args = parse_args_fn()
    if len(args.zero_action) != 1:
        raise ValueError("--zero-action must contain exactly one tuple")
    if args.history_size < 1:
        raise ValueError("--history-size must be positive")
    if not args.dataset_path.exists():
        raise FileNotFoundError(args.dataset_path)
    args.dataset_path = args.dataset_path.expanduser().resolve()
    required_dataset_path = default_dataset_path(
        "pusht_expert_train.h5"
    ).resolve()
    if args.dataset_path != required_dataset_path:
        raise ValueError(
            "Counterfactual sensitivity must use pusht_expert_train.h5: "
            f"{required_dataset_path}; got {args.dataset_path}"
        )
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(args.checkpoint_path)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_prefix = output_prefix_with_job_id(args.output_prefix, args.job_id)
    run_output_dir = args.output_dir / output_prefix
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_output_dir / f"{output_prefix}.metadata.json"

    action_mean, action_std, raw_action_dim = get_action_stats(args.dataset_path)
    transform = transform_class(
        action_mean=action_mean,
        action_std=action_std,
        img_size=args.img_size,
    )
    dataset = PushtHDF5Dataset(
        path=args.dataset_path,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        transform=transform,
    )
    indexed_dataset = IndexedDataset(dataset, max_instances=args.max_instances)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = model_loader(args, device)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    adaln_count = get_adaln_count(model)
    action_encoder_input_dim = get_action_encoder_input_dim(model)
    expected_action_encoder_input_dim = args.frameskip * raw_action_dim
    if action_encoder_input_dim != expected_action_encoder_input_dim:
        raise ValueError(
            "Unexpected action encoder input dimension: "
            f"model has {action_encoder_input_dim}, expected "
            f"{expected_action_encoder_input_dim} = frameskip({args.frameskip}) "
            f"* raw_action_dim({raw_action_dim})."
        )
    selected_layers = parse_layer_spec(args.adaln_layers, adaln_count)
    action_mean_device = action_mean.to(device)
    action_std_device = action_std.to(device)

    raw_counterfactuals = args.counterfactual_actions
    counterfactual_specs = [
        (
            raw_action,
            build_action_vector(
                raw_action,
                action_mean=action_mean_device,
                action_std=action_std_device,
                frameskip=args.frameskip,
                raw_action_dim=raw_action_dim,
                device=device,
            ),
        )
        for raw_action in raw_counterfactuals
    ]

    total_instances = len(indexed_dataset)
    csv_columns = [
        "dataset_index",
        "episode",
        "start",
        "global_t_index",
        "frame_t",
        "ball_x",
        "ball_y",
        "block_x",
        "block_y",
        "distance",
        "counterfactual_effect",
    ]
    atomic_files = []
    for layer in selected_layers:
        for raw_action, _counterfactual_vector in counterfactual_specs:
            atomic_files.append(
                {
                    "layer": layer,
                    "counterfactual_action": list(raw_action),
                    "path": str(make_atomic_csv_path(run_output_dir, output_prefix, layer, raw_action)),
                }
            )
    rows_written = 0
    t_index = args.history_size - 1

    write_metadata(
        metadata_path,
        args=args,
        output_prefix=output_prefix,
        adaln_count=adaln_count,
        selected_layers=selected_layers,
        action_mean=action_mean,
        action_std=action_std,
        raw_action_dim=raw_action_dim,
        action_encoder_input_dim=action_encoder_input_dim,
        total_instances=total_instances,
        output_dir=run_output_dir,
        atomic_files=atomic_files,
    )

    with ExitStack() as stack:
        writers: dict[tuple[int | str, tuple[float, ...]], csv.writer] = {}
        csv_files = []
        for spec in atomic_files:
            csv_file = stack.enter_context(Path(spec["path"]).open("w", newline=""))
            writer = csv.writer(csv_file)
            writer.writerow(csv_columns)
            writers[(spec["layer"], tuple(spec["counterfactual_action"]))] = writer
            csv_files.append(csv_file)

        with torch.inference_mode():
            for batch in get_progress(loader, total=math.ceil(total_instances / args.batch_size)):
                batch = move_batch_to_device(batch, device)
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)
                encoded = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
                ctx_emb = encoded["emb"][:, : args.history_size]
                real_action_context = batch["action"][:, : args.history_size]
                real_ctx_act_emb = encoded["act_emb"][:, : args.history_size]
                real_outputs = predictor_outputs(
                    model,
                    ctx_emb,
                    real_ctx_act_emb,
                    selected_layers,
                )

                ball = batch["proprio"][:, t_index, :2]
                block = batch["state"][:, t_index, 2:4]
                distance = (ball - block).norm(p=2, dim=-1)

                dataset_index_np = batch["dataset_index"].detach().cpu().numpy()
                episode_np = batch["episode"].detach().cpu().numpy()
                start_np = batch["start"].detach().cpu().numpy()
                global_t_index_np = dataset.offsets[episode_np] + start_np + t_index * args.frameskip
                frame_t_np = start_np + t_index * args.frameskip
                ball_x_np = ball[:, 0].detach().cpu().numpy()
                ball_y_np = ball[:, 1].detach().cpu().numpy()
                block_x_np = block[:, 0].detach().cpu().numpy()
                block_y_np = block[:, 1].detach().cpu().numpy()
                distance_np = distance.detach().cpu().numpy()

                common_columns = (
                    dataset_index_np,
                    episode_np,
                    start_np,
                    global_t_index_np,
                    frame_t_np,
                    ball_x_np,
                    ball_y_np,
                    block_x_np,
                    block_y_np,
                    distance_np,
                )

                batch_size = distance.shape[0]
                for raw_action, counterfactual_action in counterfactual_specs:
                    cf_action_context = make_counterfactual_context(
                        real_action_context,
                        counterfactual_action,
                        scope=args.intervention_scope,
                    )
                    cf_ctx_act_emb = model.action_encoder(cf_action_context)
                    cf_outputs = predictor_outputs(
                        model,
                        ctx_emb,
                        cf_ctx_act_emb,
                        selected_layers,
                    )
                    raw_action_key = tuple(raw_action)
                    for layer in selected_layers:
                        effect_np = (
                            layer_counterfactual_effect(real_outputs, cf_outputs, layer)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        rows = zip(*common_columns, effect_np)
                        writers[(layer, raw_action_key)].writerows(rows)

                rows_written += batch_size
                if rows_written % (args.flush_every * args.batch_size) == 0:
                    for csv_file in csv_files:
                        csv_file.flush()

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved {len(atomic_files)} atomic CSV files under: {run_output_dir}")
    print(f"Rows written per CSV: {rows_written}")
    print(f"AdaLN blocks found: {adaln_count}; valid --adaln-layers options: 0..{adaln_count - 1}, final")


if __name__ == "__main__":
    main()
