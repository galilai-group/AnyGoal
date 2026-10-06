"""Train Robomimic DriftWorld on StableWM's OGBench Cube HDF5 dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_driftworld import (  # noqa: E402
    CheckpointingTorchProxy,
    MaxStepsReached,
    MaxStepState,
    TrainingTelemetry,
    WandbTelemetryProxy,
    configure_distributed,
    configure_upstream_paths,
    load_config,
    resolve_run_directory,
)
from anygoal.wm.driftworld import (  # noqa: E402
    create_robomimic_driftworld_denoiser,
    get_driftworld_cube_hdf5_loader,
    robomimic_driftworld_source_root,
)


def validate_config(cfg: DictConfig) -> Path:
    if int(cfg.data.frameskip) < 1:
        raise ValueError("data.frameskip must be positive.")
    expected_horizon = int(cfg.model.num_history_frames) + int(
        cfg.model.num_future_frames
    )
    if int(cfg.data.pred_horizon) != expected_horizon:
        raise ValueError(
            f"data.pred_horizon must equal history + future ({expected_horizon})."
        )
    if int(cfg.data.num_history) + 1 != int(cfg.model.num_history_frames):
        raise ValueError("data.num_history + 1 must equal model.num_history_frames.")
    if int(cfg.data.num_future) != int(cfg.model.num_future_frames):
        raise ValueError("data.num_future must equal model.num_future_frames.")
    if int(cfg.model.action_dim) != 5:
        raise ValueError("OGBench Cube uses 5-D relative actions.")
    if str(cfg.data.action_mode) != "relative":
        raise ValueError("OGBench Cube DriftWorld requires data.action_mode=relative.")
    if str(cfg.model.unet_type) != "multi":
        raise ValueError("The Cube adapter uses Robomimic DriftWorld's one-view U-Net.")
    if int(cfg.data.resize_scale) != 96:
        raise ValueError("Robomimic DriftWorld's U-Net requires 96x96 images.")
    if not bool(cfg.data.normalize_img):
        raise ValueError("Robomimic DriftWorld training requires images in [-1, 1].")
    permanent_every = int(cfg.train.get("permanent_ckpt_every", 20_000))
    if permanent_every < 1 or permanent_every % int(cfg.train.ckpt_every) != 0:
        raise ValueError(
            "train.permanent_ckpt_every must be a positive multiple of "
            "train.ckpt_every."
        )
    max_steps = cfg.train.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 0 or max_steps % int(cfg.train.ckpt_every) != 0:
            raise ValueError(
                "train.max_steps must be non-negative and divisible by "
                "train.ckpt_every so the terminal checkpoint is saved."
            )

    dataset_name = Path(str(cfg.data.dataset_name)).expanduser()
    if dataset_name.is_absolute():
        dataset_path = dataset_name if dataset_name.suffix else dataset_name.with_suffix(".h5")
    else:
        dataset_path = (
            Path(cfg.data.dataset_root).expanduser()
            / "datasets"
            / f"{dataset_name}.h5"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"StableWM Cube H5 file not found: {dataset_path}")
    return dataset_path


def load_upstream_train_module():
    """Load the vendored Robomimic loop with local loader/model shims."""
    root = robomimic_driftworld_source_root()
    module_name = "anygoal.wm._driftworld_robomimic_train"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    data_package = types.ModuleType("data")
    data_package.__path__ = [str(root / "data")]
    loader_module = types.ModuleType("data.create_robomimic_loader")
    loader_module.get_dataloader = get_driftworld_cube_hdf5_loader
    data_package.create_robomimic_loader = loader_module
    sys.modules["data"] = data_package
    sys.modules["data.create_robomimic_loader"] = loader_module

    model_module = types.ModuleType("utils_model")
    model_module.create_model = create_robomimic_driftworld_denoiser
    sys.modules["utils_model"] = model_module

    # The vendored file has two Python-3.12-style nested-quote f-strings. Only
    # their quote characters are adapted in memory for the repository's 3.10 env.
    source = (root / "train.py").read_text()
    source = source.replace(
        'nbatch["obs"]["agentview_image"]',
        "nbatch['obs']['agentview_image']",
    ).replace('nbatch["actions"]', "nbatch['actions']")
    module = types.ModuleType(module_name)
    module.__file__ = str(root / "train.py")
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def run_training(cfg: DictConfig) -> None:
    upstream_train = load_upstream_train_module()
    original_loader = upstream_train.get_dataloader
    original_create_model = upstream_train.create_model
    original_wandb = upstream_train.wandb
    original_torch = upstream_train.torch
    original_barrier = upstream_train.barrier
    telemetry = TrainingTelemetry(cfg, original_wandb, upstream_train.log)
    max_step_state = MaxStepState(cfg)

    def create_logged_model(inner_cfg, device):
        denoiser = create_robomimic_driftworld_denoiser(
            unet_type=str(inner_cfg.model.unet_type),
            temp_list=inner_cfg.model.temp_list,
            n_neg=int(inner_cfg.model.n_neg),
            num_future_frames=int(inner_cfg.model.num_future_frames),
            num_history_frames=int(inner_cfg.model.num_history_frames),
            decay=float(inner_cfg.train.decay),
            action_dim=int(inner_cfg.model.action_dim),
            action_frameskip=int(inner_cfg.data.frameskip),
            device=device,
        )
        return telemetry.wrap_model(denoiser)

    def create_logged_loader(inner_cfg, rank=0, world_size=1):
        loader = get_driftworld_cube_hdf5_loader(
            inner_cfg, rank=rank, world_size=world_size
        )
        telemetry.register_loader(loader)
        return loader

    upstream_train.get_dataloader = create_logged_loader
    upstream_train.create_model = create_logged_model
    upstream_train.wandb = WandbTelemetryProxy(original_wandb, telemetry)
    upstream_train.torch = CheckpointingTorchProxy(
        original_torch, cfg, upstream_train.log, max_step_state
    )

    def stop_after_checkpoint_barrier(world_size):
        original_barrier(world_size)
        reached = max_step_state.reached
        if world_size > 1 and dist.is_initialized():
            device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
            signal = torch.tensor([int(reached)], device=device)
            dist.broadcast(signal, src=0)
            reached = bool(signal.item())
        if reached:
            raise MaxStepsReached

    upstream_train.barrier = stop_after_checkpoint_barrier
    try:
        try:
            upstream_train.train(cfg)
        except MaxStepsReached:
            if int(os.environ.get("RANK", 0)) == 0:
                upstream_train.log.info(
                    "Reached train.max_steps=%d after saving checkpoint; stopping.",
                    max_step_state.target,
                )
                original_wandb.finish()
            if dist.is_initialized():
                dist.destroy_process_group()
    finally:
        upstream_train.get_dataloader = original_loader
        upstream_train.create_model = original_create_model
        upstream_train.wandb = original_wandb
        upstream_train.torch = original_torch
        upstream_train.barrier = original_barrier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "driftworld" / "train" / "cube_lewm.yaml",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = [override for override in args.overrides if override != "--"]
    cfg = load_config(args.config.resolve(), overrides)
    dataset_path = validate_config(cfg)
    _, rank, world_size = configure_distributed()
    run_directory = resolve_run_directory(cfg, rank, world_size)
    configure_upstream_paths(cfg, run_directory)
    if rank == 0:
        OmegaConf.save(cfg, run_directory / "config.yaml")
        logging.info("Dataset: %s", dataset_path)
        logging.info("Run directory: %s", run_directory)

    if args.preflight:
        loader = get_driftworld_cube_hdf5_loader(
            cfg, rank=rank, world_size=world_size
        )
        batch = next(iter(loader))
        if rank == 0:
            print(
                "Preflight OK: "
                f"image={tuple(batch['obs']['agentview_image'].shape)} "
                f"action={tuple(batch['actions'].shape)} run={run_directory}"
            )
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        return
    run_training(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    main()
