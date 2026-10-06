#!/usr/bin/env python
# ruff: noqa: E402
"""Save aggregated LeWM counterfactual sensitivities for OGBench Cube."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def add_extra_site_packages() -> None:
    extra_paths = os.environ.get("LEWM_EXTRA_SITE_PACKAGES", "")
    for path in reversed([item for item in extra_paths.split(os.pathsep) if item]):
        if path not in sys.path:
            sys.path.insert(0, path)


add_extra_site_packages()

import h5py
import hdf5plugin  # noqa: F401
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_MODULE_DIR = REPO_ROOT / "worldmodels" / "lewm"
for source_dir in (REPO_ROOT, LEWM_MODULE_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import save_counterfactual_sensitivity as base
from anygoal.datasets.cube_collection import CubeActionSupport


RESULTS_DIR = REPO_ROOT / "paper" / "results" / "cube_counterfactual_sensitivity"
DEFAULT_DATASET = (
    Path.home() / ".stable_worldmodel" / "datasets" / "cube_single_expert.h5"
)
DEFAULT_CHECKPOINT = Path.home() / ".stable_worldmodel" / "cube" / "lewm_object.ckpt"
DEFAULT_ACTION_GRIDS = RESULTS_DIR / "action_grids.json"


class CubeInferenceTransform:
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

    def preprocess_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        if pixels.max() > 2.0:
            pixels = pixels / 255.0
        if pixels.shape[-2:] != (self.img_size, self.img_size):
            leading_shape = pixels.shape[:-3]
            pixels = pixels.reshape(-1, *pixels.shape[-3:])
            pixels = F.interpolate(
                pixels,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            pixels = pixels.reshape(
                *leading_shape, pixels.shape[-3], self.img_size, self.img_size
            )
        mean = base.IMAGENET_MEAN.to(device=pixels.device, dtype=pixels.dtype)
        std = base.IMAGENET_STD.to(device=pixels.device, dtype=pixels.dtype)
        return (pixels - mean) / std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action.float() - self.action_mean) / self.action_std


class CubeHDF5Dataset(Dataset):
    def __init__(
        self,
        *,
        path: Path,
        frameskip: int,
        history_size: int,
        num_preds: int,
        transform: CubeInferenceTransform,
    ) -> None:
        self.h5_path = Path(path)
        self.frameskip = frameskip
        self.history_size = history_size
        self.num_steps = history_size + num_preds
        self.span = self.num_steps * frameskip
        self.current_offset = (history_size - 1) * frameskip
        self.transform = transform
        self.h5_file: h5py.File | None = None

        with h5py.File(
            self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
        ) as h5_file:
            self.lengths = h5_file["ep_len"][:]
            self.offsets = h5_file["ep_offset"][:]
            self.action = h5_file["action"][:]
            self.effector_position = h5_file["proprio_effector_pos"][:]

        self.clip_indices = [
            (episode, start)
            for episode, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(int(length) - self.span + 1)
        ]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path,
                "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
            )

    def __getitem__(self, index: int) -> dict:
        self._open()
        assert self.h5_file is not None
        episode, start = self.clip_indices[index]
        global_start = int(self.offsets[episode] + start)
        global_end = global_start + self.span
        current_global = global_start + self.current_offset

        pixels = np.asarray(
            self.h5_file["pixels"][global_start:global_end:self.frameskip]
        )
        pixels_tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
        action_tensor = torch.from_numpy(
            np.asarray(self.action[global_start:global_end])
        )
        sample = {
            "pixels": pixels_tensor,
            "action": self.transform.normalize_action(action_tensor).reshape(
                self.num_steps, -1
            ),
        }
        sample["dataset_index"] = torch.tensor(index, dtype=torch.long)
        sample["episode"] = torch.tensor(episode, dtype=torch.long)
        sample["start"] = torch.tensor(start, dtype=torch.long)
        sample["global_t_index"] = torch.tensor(current_global, dtype=torch.long)
        sample["frame_t"] = torch.tensor(
            start + self.current_offset, dtype=torch.long
        )
        sample["action_z"] = torch.tensor(
            self.action[current_global, 2], dtype=torch.float32
        )
        sample["effector_z"] = torch.tensor(
            self.effector_position[current_global, 2], dtype=torch.float32
        )
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--action-grids", type=Path, default=DEFAULT_ACTION_GRIDS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-prefix", default="cube_counterfactual_sensitivity")
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--counterfactual-chunk-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--adaln-layers", default="0,1,2,3,4,5,final")
    parser.add_argument("--max-instances", type=int)
    return parser.parse_args()


def load_and_verify_grids(path: Path, dataset_path: Path) -> tuple[dict, dict[str, list[tuple[float, ...]]]]:
    grid_metadata = json.loads(path.read_text())
    if Path(grid_metadata["dataset_path"]).resolve() != dataset_path:
        raise ValueError("Action grids were calibrated on a different dataset")
    support = CubeActionSupport(dataset_path)
    if not np.isclose(
        grid_metadata["support"]["support_radius"], support.support_radius
    ):
        raise ValueError("Recalibrated Cube support radius differs from action-grid metadata")

    if "fixed_negative_z_8point_grids" in grid_metadata:
        grid_specs = [
            (f"fixed_negative_z_set{index}", grid, 8)
            for index, grid in enumerate(
                grid_metadata["fixed_negative_z_8point_grids"], start=1
            )
        ]
    else:
        grid_specs = [
            ("grid_3d_27", grid_metadata["grid_3d_27"], 27),
            ("grid_xy_9", grid_metadata["grid_xy_9"], 9),
        ]

    grids = {}
    for output_name, grid_spec, expected_count in grid_specs:
        actions = [
            tuple(float(value) for value in row)
            for row in grid_spec["actions"]
        ]
        if len(actions) != expected_count or any(len(action) != 5 for action in actions):
            raise ValueError(
                f"{output_name} must contain {expected_count} 5-D actions"
            )
        radii = support._neighbor_radius(np.asarray(actions, dtype=np.float64))
        if np.any(radii > support.support_radius):
            raise ValueError(f"{output_name} contains an unsupported action")
        grids[output_name] = actions
    return grid_metadata, grids


def create_result_dataset(
    output_file: h5py.File,
    name: str,
    length: int,
    dtype: str,
) -> h5py.Dataset:
    return output_file.create_dataset(
        name,
        shape=(length,),
        dtype=dtype,
        chunks=(min(length, 65536),),
        compression="gzip",
        compression_opts=1,
    )


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.counterfactual_chunk_size,
        args.num_workers + 1,
        args.history_size,
        args.num_preds,
        args.frameskip,
    ) < 1:
        raise ValueError("batch, worker, history, prediction, and frameskip settings are invalid")
    dataset_path = args.dataset_path.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    action_grids_path = args.action_grids.expanduser().resolve()
    for path in (dataset_path, checkpoint_path, action_grids_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    grid_metadata, grids = load_and_verify_grids(action_grids_path, dataset_path)
    action_mean, action_std, raw_action_dim = base.get_action_stats(dataset_path)
    if raw_action_dim != 5:
        raise ValueError(f"Cube action dimension must be 5, got {raw_action_dim}")
    transform = CubeInferenceTransform(
        action_mean=action_mean, action_std=action_std, img_size=args.img_size
    )
    dataset = CubeHDF5Dataset(
        path=dataset_path,
        frameskip=args.frameskip,
        history_size=args.history_size,
        num_preds=args.num_preds,
        transform=transform,
    )
    total_instances = len(dataset)
    if args.max_instances is not None:
        if args.max_instances < 1:
            raise ValueError("max-instances must be positive")
        total_instances = min(total_instances, args.max_instances)
        dataset.clip_indices = dataset.clip_indices[:total_instances]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model_args = argparse.Namespace(checkpoint_path=checkpoint_path)
    model = base.load_serialized_model(model_args, device).to(device)
    model.eval()
    model.requires_grad_(False)
    adaln_count = base.get_adaln_count(model)
    selected_layers = base.parse_layer_spec(args.adaln_layers, adaln_count)
    action_encoder_input_dim = base.get_action_encoder_input_dim(model)
    if action_encoder_input_dim != args.frameskip * raw_action_dim:
        raise ValueError(
            f"Model action input is {action_encoder_input_dim}; expected "
            f"{args.frameskip * raw_action_dim}"
        )

    action_membership: dict[tuple[float, ...], list[str]] = defaultdict(list)
    for grid_name, actions in grids.items():
        for action in actions:
            action_membership[action].append(grid_name)
    unique_actions = list(action_membership)
    action_mean_device = action_mean.to(device)
    action_std_device = action_std.to(device)
    counterfactual_vectors = {
        action: base.build_action_vector(
            action,
            action_mean=action_mean_device,
            action_std=action_std_device,
            frameskip=args.frameskip,
            raw_action_dim=raw_action_dim,
            device=device,
        )
        for action in unique_actions
    }

    output_prefix = base.output_prefix_with_job_id(args.output_prefix, args.job_id)
    run_output_dir = args.output_dir.expanduser().resolve() / output_prefix
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_output_dir / f"{output_prefix}.metadata.json"
    partial_h5_path = run_output_dir / f"{output_prefix}.partial.h5"
    results_h5_path = run_output_dir / f"{output_prefix}.h5"
    if partial_h5_path.exists() or results_h5_path.exists():
        raise FileExistsError(f"Refusing to overwrite output in {run_output_dir}")

    metadata = {
        "status": "running",
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "action_grids_path": str(action_grids_path),
        "action_grids": grid_metadata,
        "output_prefix": output_prefix,
        "job_id": str(args.job_id),
        "results_h5": str(results_h5_path),
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "current_frame_offset": (args.history_size - 1) * args.frameskip,
        "selected_layers": selected_layers,
        "intervention_scope": "current action block only",
        "counterfactual_action_block": "the selected normalized 5-D action is repeated for all five controls in the current block",
        "sensitivity": "sum over grid actions of L2(real predictor output - counterfactual predictor output)",
        "action_mean": action_mean.squeeze(0).tolist(),
        "action_std": action_std.squeeze(0).tolist(),
        "total_instances": total_instances,
        "dataset_stored_rows": int(dataset.lengths.sum()),
        "valid_windows_per_episode": sorted(
            set(int(length) - dataset.span + 1 for length in dataset.lengths if length >= dataset.span)
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    common_specs = {
        "dataset_index": "i8",
        "episode": "i4",
        "start": "i4",
        "global_t_index": "i8",
        "frame_t": "i4",
        "action_z": "f4",
        "effector_z": "f4",
    }
    with h5py.File(partial_h5_path, "w") as output_file:
        common_datasets = {
            name: create_result_dataset(output_file, f"common/{name}", total_instances, dtype)
            for name, dtype in common_specs.items()
        }
        sensitivity_datasets = {
            (grid_name, layer): create_result_dataset(
                output_file,
                f"sensitivity/{grid_name}/layer_{layer}",
                total_instances,
                "f4",
            )
            for grid_name in grids
            for layer in selected_layers
        }

        rows_written = 0
        progress = base.get_progress(loader, total=math.ceil(total_instances / args.batch_size))
        with torch.inference_mode():
            for batch in progress:
                batch = base.move_batch_to_device(batch, device)
                batch["pixels"] = transform.preprocess_pixels(batch["pixels"])
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)
                encoded = model.encode(
                    {"pixels": batch["pixels"], "action": batch["action"]}
                )
                ctx_emb = encoded["emb"][:, : args.history_size]
                real_action_context = batch["action"][:, : args.history_size]
                real_outputs = base.predictor_outputs(
                    model,
                    ctx_emb,
                    encoded["act_emb"][:, : args.history_size],
                    selected_layers,
                )
                batch_size = ctx_emb.shape[0]
                sums = {
                    (grid_name, layer): torch.zeros(
                        batch_size, dtype=torch.float32, device=device
                    )
                    for grid_name in grids
                    for layer in selected_layers
                }

                for chunk_start in range(0, len(unique_actions), args.counterfactual_chunk_size):
                    chunk_actions = unique_actions[
                        chunk_start : chunk_start + args.counterfactual_chunk_size
                    ]
                    contexts = torch.stack(
                        [
                            base.make_counterfactual_context(
                                real_action_context,
                                counterfactual_vectors[action],
                                scope="current",
                            )
                            for action in chunk_actions
                        ],
                        dim=0,
                    )
                    chunk_count = len(chunk_actions)
                    flat_contexts = contexts.reshape(
                        chunk_count * batch_size,
                        args.history_size,
                        -1,
                    )
                    repeated_embeddings = (
                        ctx_emb.unsqueeze(0)
                        .expand(chunk_count, -1, -1, -1)
                        .reshape(chunk_count * batch_size, args.history_size, -1)
                    )
                    counterfactual_outputs = base.predictor_outputs(
                        model,
                        repeated_embeddings,
                        model.action_encoder(flat_contexts),
                        selected_layers,
                    )
                    for layer in selected_layers:
                        shaped_output = counterfactual_outputs[layer].reshape(
                            chunk_count, batch_size, -1
                        )
                        effects = (
                            real_outputs[layer].unsqueeze(0) - shaped_output
                        ).norm(p=2, dim=-1)
                        for action_index, action in enumerate(chunk_actions):
                            for grid_name in action_membership[action]:
                                sums[(grid_name, layer)].add_(effects[action_index])

                row_end = rows_written + batch_size
                for name, output_dataset in common_datasets.items():
                    output_dataset[rows_written:row_end] = (
                        batch[name].detach().cpu().numpy()
                    )
                for key, output_dataset in sensitivity_datasets.items():
                    output_dataset[rows_written:row_end] = sums[key].cpu().numpy()
                rows_written = row_end

        if rows_written != total_instances:
            raise RuntimeError(f"Wrote {rows_written} rows; expected {total_instances}")
        output_file.attrs["metadata_path"] = str(metadata_path)
        output_file.attrs["rows"] = rows_written

    partial_h5_path.replace(results_h5_path)
    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {results_h5_path}")
    print(f"Saved {metadata_path}")
    print(f"Rows: {total_instances:,}; unique interventions: {len(unique_actions)}")


if __name__ == "__main__":
    main()
