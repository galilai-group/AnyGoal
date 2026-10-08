"""Train vendored IRASim on Stable WorldModel Push-T HDF5 demonstrations."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.wm.irasim import (  # noqa: E402
    create_irasim_transformer,
    get_irasim_hdf5_datasets,
    irasim_source_root,
    load_irasim_upstream_train_module,
)


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    source_root = irasim_source_root()
    cfg = OmegaConf.merge(
        OmegaConf.load(source_root / "configs" / "base" / "data.yaml"),
        OmegaConf.load(source_root / "configs" / "base" / "diffusion.yaml"),
        OmegaConf.load(path),
    )
    OmegaConf.set_struct(cfg, False)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        value = OmegaConf.from_dotlist([f"value={raw_value}"])["value"]
        OmegaConf.update(cfg, key, value, merge=True)
    OmegaConf.resolve(cfg)
    return cfg


def dataset_path(cfg: DictConfig) -> Path:
    name = Path(str(cfg.h5_dataset_name)).expanduser()
    if name.is_absolute():
        return name if name.suffix else name.with_suffix(".h5")
    return Path(cfg.h5_dataset_root).expanduser() / "datasets" / f"{name}.h5"


def validate_config(cfg: DictConfig) -> Path:
    if str(cfg.dataset) != "languagetable":
        raise ValueError(
            "The vendored IRASim dispatches its 2-D action encoder through "
            "dataset=languagetable; h5_dataset_name selects the actual H5 data."
        )
    if bool(cfg.pre_encode):
        raise ValueError("IRASim H5 training requires pre_encode=false.")
    if int(cfg.extras) != 3:
        raise ValueError("This adapter supports upstream frame-level Frame-Ada.")
    if int(cfg.mask_frame_num) != int(cfg.get("history_frames", 3)):
        raise ValueError("mask_frame_num must equal history_frames.")
    if int(cfg.num_frames) - int(cfg.mask_frame_num) != int(
        cfg.get("future_frames", 1)
    ):
        raise ValueError(
            "num_frames must equal history_frames + future_frames."
        )
    if int(cfg.frameskip) < 1 or int(cfg.num_frames) < 2:
        raise ValueError("frameskip must be positive and num_frames at least 2.")
    if any(int(size) % 8 for size in cfg.video_size):
        raise ValueError("Every video_size dimension must be divisible by 8.")
    action_dim = int(cfg.get("action_dim", 2))
    if action_dim < 1 or len(cfg.action_scale) != action_dim:
        raise ValueError("action_scale must contain one positive value per action dim.")
    if any(float(value) <= 0 for value in cfg.action_scale):
        raise ValueError("action_scale values must be positive.")
    path = dataset_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"StableWM Push-T H5 file not found: {path}")
    with h5py.File(path, "r", swmr=True) as handle:
        if handle["action"].ndim != 2 or handle["action"].shape[1] != action_dim:
            raise ValueError(
                f"IRASim expects a {action_dim}-D action column for this config."
            )
    scheduler_path = Path(cfg.scheduler_path).expanduser()
    if not scheduler_path.is_dir():
        raise FileNotFoundError(f"IRASim scheduler directory not found: {scheduler_path}")
    vae_path = Path(cfg.vae_model_path).expanduser()
    if not vae_path.exists() and str(cfg.vae_model_path).startswith("/"):
        raise FileNotFoundError(f"IRASim VAE path not found: {vae_path}")
    return path


def set_single_process_ddp_defaults() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29573")


class WandbEtaProxy:
    """Add throughput and ETA without changing the upstream training loop."""

    def __init__(self, wandb_module, total_steps: int) -> None:
        self._wandb = wandb_module
        self._total_steps = int(total_steps)
        self._started_at = time.monotonic()
        self._last_timed_step = 0

    def __getattr__(self, name):
        return getattr(self._wandb, name)

    def log(self, data, step=None, *args, **kwargs):
        payload = dict(data)
        if step is not None and int(step) > self._last_timed_step:
            current_step = int(step)
            elapsed = max(time.monotonic() - self._started_at, 1e-6)
            steps_per_second = current_step / elapsed
            remaining = max(self._total_steps - current_step, 0)
            eta_seconds = remaining / max(steps_per_second, 1e-12)
            payload.update(
                {
                    "timing/elapsed_seconds": elapsed,
                    "timing/eta_seconds": eta_seconds,
                    "timing/steps_per_second": steps_per_second,
                    "timing/steps_remaining": remaining,
                }
            )
            print(
                f"step={current_step}/{self._total_steps} "
                f"steps_per_second={steps_per_second:.6f} "
                f"eta_hours={eta_seconds / 3600:.2f}",
                flush=True,
            )
            self._last_timed_step = current_step
        return self._wandb.log(payload, step=step, *args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "scripts/train/config/irasim/pusht.yaml",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = [override for override in args.overrides if override != "--"]
    cfg = load_config(args.config.resolve(), overrides)
    path = validate_config(cfg)
    if args.preflight:
        train_dataset, validation_dataset = get_irasim_hdf5_datasets(cfg)
        sample = train_dataset[0]
        print(
            f"Preflight OK: dataset={path} train={len(train_dataset)} "
            f"validation={len(validation_dataset)} "
            f"video={tuple(sample['video'].shape)} "
            f"action={tuple(sample['action'].shape)}"
        )
        return

    set_single_process_ddp_defaults()
    upstream_train = load_irasim_upstream_train_module()
    original_wandb = upstream_train.wandb
    original_torch_load = upstream_train.torch.load
    original_get_models = upstream_train.get_models
    action_dim = int(cfg.get("action_dim", 2))

    def load_trusted_checkpoint(*load_args, **load_kwargs):
        # IRASim checkpoints include optimizer state and an OmegaConf config.
        # PyTorch 2.6+ otherwise defaults to weights_only=True.
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    upstream_train.wandb = WandbEtaProxy(original_wandb, cfg.max_train_steps)
    upstream_train.torch.load = load_trusted_checkpoint
    if action_dim != 2:
        upstream_train.get_models = lambda model_args: create_irasim_transformer(
            model_args, action_dim=action_dim
        )
    try:
        upstream_train.main(cfg)
    finally:
        upstream_train.wandb = original_wandb
        upstream_train.torch.load = original_torch_load
        upstream_train.get_models = original_get_models


if __name__ == "__main__":
    main()
