"""Train the vendored Push-T-architecture DriftWorld on a StableWM HDF5 dataset."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import shutil
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.wm.driftworld import (  # noqa: E402
    create_driftworld_denoiser,
    driftworld_source_root,
    get_driftworld_hdf5_loader,
    get_driftworld_modules,
)


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        value = OmegaConf.from_dotlist([f"value={raw_value}"])["value"]
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def create_run_directory(root: str | Path, description: str) -> Path:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{datetime.now():%Y%m%d}_{description}"
    for suffix in range(10_000):
        candidate = base if suffix == 0 else Path(f"{base}_{suffix:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory under {root}.")


def configure_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("DriftWorld DDP requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def resolve_run_directory(cfg: DictConfig, rank: int, world_size: int) -> Path:
    configured = cfg.output.get("run_directory")
    path_string = None
    if rank == 0:
        if configured:
            run_directory = Path(configured).expanduser().resolve()
            run_directory.mkdir(parents=True, exist_ok=True)
        else:
            run_directory = create_run_directory(
                cfg.output.checkpoint_root, cfg.output.description
            )
        path_string = str(run_directory)
    if world_size > 1:
        values = [path_string]
        dist.broadcast_object_list(values, src=0)
        path_string = values[0]
    return Path(path_string)


def configure_upstream_paths(cfg: DictConfig, run_directory: Path) -> None:
    OmegaConf.update(cfg, "output.run_directory", str(run_directory), force_add=True)
    OmegaConf.update(cfg, "output_dir", str(run_directory), force_add=True)
    OmegaConf.update(
        cfg, "path_ckpt_latest", str(run_directory / "ckpt-latest.pth"), force_add=True
    )
    OmegaConf.update(
        cfg,
        "path_ckpt_2nd_latest",
        str(run_directory / "ckpt-2nd-latest.pth"),
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "wandb_info.saved_run_id",
        str(run_directory / "wandb_run_id.json"),
        force_add=True,
    )
    OmegaConf.resolve(cfg)


def validate_config(cfg: DictConfig) -> Path:
    if int(cfg.data.frameskip) < 1:
        raise ValueError("data.frameskip must be positive.")
    expected_horizon = int(cfg.model.num_history_frames) + int(
        cfg.model.num_future_frames
    )
    if int(cfg.data.pred_horizon) != expected_horizon:
        raise ValueError(
            "data.pred_horizon must equal model.num_history_frames + "
            f"model.num_future_frames ({expected_horizon})."
        )
    if str(cfg.data.action_mode) != "relative":
        raise ValueError(
            "StableWM 2-D H5 training uses relative actions; H5 training "
            "requires data.action_mode=relative."
        )
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
        raise FileNotFoundError(f"StableWM H5 file not found: {dataset_path}")
    return dataset_path


def load_upstream_train_module():
    source_root = driftworld_source_root()
    module_name = "anygoal.wm._driftworld_upstream_train"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    get_driftworld_modules()
    loader_module = types.ModuleType("data.pushT_dataloader")
    loader_module.get_pushT_loader = get_driftworld_hdf5_loader
    sys.modules["data.pushT_dataloader"] = loader_module
    spec = importlib.util.spec_from_file_location(module_name, source_root / "train.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DriftWorld training loop from {source_root}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(source_root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(source_root))
    return module


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingTelemetry:
    """Add W&B metrics around the unmodified upstream training loop."""

    def __init__(self, cfg: DictConfig, wandb_module, log: logging.Logger) -> None:
        self.cfg = cfg
        self.wandb = wandb_module
        self.log = log
        self.total_steps: int | None = None
        self.first_step: int | None = None
        self.started_at: float | None = None
        self.forward_calls = 0

    def register_loader(self, loader) -> None:
        max_steps = self.cfg.train.get("max_steps")
        self.total_steps = (
            int(max_steps) + 1
            if max_steps is not None
            else len(loader) * int(self.cfg.train.num_epochs)
        )

    def wrap_model(self, denoiser):
        original_forward = denoiser.forward
        visualization = self.cfg.get("visualization", {})
        visualization_enabled = bool(visualization.get("enabled", True))
        visualization_every = int(visualization.get("every_steps", 500))
        visualization_samples = int(visualization.get("num_samples", 1))
        visualization_seed = int(visualization.get("seed", self.cfg.train.seed))
        if visualization_every < 1 or visualization_samples < 1:
            raise ValueError(
                "visualization.every_steps and visualization.num_samples must be positive."
            )
        is_main = int(os.environ.get("RANK", 0)) == 0

        def images_and_actions(batch):
            if "image" in batch and "action" in batch:
                return batch["image"], batch["action"]
            return batch["obs"]["agentview_image"], batch["actions"]

        def forward_with_telemetry(inner, batch, device):
            if is_main and self.started_at is None:
                self.started_at = time.monotonic()
            loss, raw_metrics = original_forward(batch, device)
            metrics = dict(raw_metrics)
            for key, value in tuple(metrics.items()):
                if key.startswith("loss_") and key != "loss_backprop":
                    metrics[f"loss/components/{key.removeprefix('loss_')}"] = value
            metrics["loss/final"] = metrics["loss_backprop"]

            should_visualize = (
                is_main
                and visualization_enabled
                and self.forward_calls % visualization_every == 0
            )
            if should_visualize:
                history_length = int(inner.num_history_frames)
                future_length = int(inner.num_future_frames)
                images, batch_actions = images_and_actions(batch)
                sample_count = min(visualization_samples, int(images.shape[0]))
                history = images[:sample_count, :history_length]
                history = history.permute(0, 2, 1, 3, 4).to(device)
                actions = batch_actions[
                    :sample_count,
                    history_length - 1 : history_length - 1 + future_length,
                ].to(device)
                targets = images[
                    :sample_count, history_length : history_length + future_length
                ].to(device)
                cuda_devices = [device.index] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(visualization_seed + self.forward_calls)
                    predictions = inner.sample(history, actions)

                target_pixels = targets.clamp(-1, 1).add(1).div(2)
                prediction_pixels = predictions.clamp(-1, 1).add(1).div(2)
                pixel_mse = torch.mean((prediction_pixels - target_pixels) ** 2)
                pixel_mse_value = float(pixel_mse.item())
                metrics["visualization/pixel_mse"] = pixel_mse_value
                metrics["visualization/pixel_psnr_db"] = -10.0 * math.log10(
                    max(pixel_mse_value, 1e-12)
                )

                rows = []
                for sample_index in range(sample_count):
                    rows.append(torch.cat(list(target_pixels[sample_index]), dim=-1))
                    rows.append(torch.cat(list(prediction_pixels[sample_index]), dim=-1))
                panel = torch.cat(rows, dim=-2).permute(1, 2, 0).float().cpu().numpy()
                metrics["visualization/ground_truth_vs_prediction"] = self.wandb.Image(
                    panel,
                    caption=(
                        "Each sample uses two rows: ground-truth future (top), "
                        "EMA prediction (bottom); time runs left to right."
                    ),
                )

            self.forward_calls += 1
            return loss, metrics

        denoiser.forward = types.MethodType(forward_with_telemetry, denoiser)
        return denoiser

    def augment_log(self, data, step: int | None):
        if step is None or "loss_backprop" not in data:
            return data
        now = time.monotonic()
        if self.started_at is None:
            self.started_at = now
        if self.first_step is None:
            self.first_step = int(step)
        elapsed = max(now - self.started_at, 1e-6)
        completed = int(step) - int(self.first_step) + 1
        steps_per_second = completed / elapsed
        total_steps = self.total_steps or (int(step) + 1)
        remaining_steps = max(total_steps - int(step) - 1, 0)
        eta_seconds = remaining_steps / max(steps_per_second, 1e-12)
        augmented = dict(data)
        augmented.update(
            {
                "timing/elapsed_seconds": elapsed,
                "timing/eta_seconds": eta_seconds,
                "timing/steps_per_second": steps_per_second,
                "timing/steps_remaining": remaining_steps,
            }
        )
        log_every = int(self.cfg.get("eta", {}).get("log_every_steps", 100))
        if log_every < 1:
            raise ValueError("eta.log_every_steps must be positive.")
        if int(step) % log_every == 0 or remaining_steps == 0:
            self.log.info(
                "step %d/%d | ETA %s | %.3f steps/s",
                int(step) + 1,
                total_steps,
                _format_duration(eta_seconds),
                steps_per_second,
            )
        return augmented


class WandbTelemetryProxy:
    def __init__(self, wandb_module, telemetry: TrainingTelemetry) -> None:
        self._wandb = wandb_module
        self._telemetry = telemetry

    def __getattr__(self, name):
        return getattr(self._wandb, name)

    def log(self, data, step=None, *args, **kwargs):
        return self._wandb.log(
            self._telemetry.augment_log(data, step), step=step, *args, **kwargs
        )


class MaxStepsReached(Exception):
    """Signal synchronized termination after the requested checkpoint is saved."""


class MaxStepState:
    def __init__(self, cfg: DictConfig) -> None:
        value = cfg.train.get("max_steps")
        self.target = int(value) if value is not None else None
        self.reached = False


class CheckpointingTorchProxy:
    """Preserve immutable step checkpoints after upstream latest saves."""

    def __init__(
        self,
        torch_module,
        cfg: DictConfig,
        log: logging.Logger,
        max_step_state: MaxStepState | None = None,
    ) -> None:
        self._torch = torch_module
        self._latest = Path(cfg.path_ckpt_latest).resolve()
        self._run_directory = Path(cfg.output.run_directory).resolve()
        self._every = int(cfg.train.get("permanent_ckpt_every", 20_000))
        self._log = log
        self._max_step_state = max_step_state or MaxStepState(cfg)

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def save(self, obj, path, *args, **kwargs):
        result = self._torch.save(obj, path, *args, **kwargs)
        saved_path = Path(path).resolve() if isinstance(path, (str, os.PathLike)) else None
        step = obj.get("step") if isinstance(obj, dict) else None
        if (
            saved_path == self._latest
            and step is not None
            and self._max_step_state.target == int(step)
        ):
            self._max_step_state.reached = True
        if saved_path != self._latest or not step or int(step) % self._every:
            return result

        permanent_path = self._run_directory / f"ckpt-step{int(step)}.pth"
        if permanent_path.exists():
            self._log.info("Permanent checkpoint already exists: %s", permanent_path)
            return result
        temporary_path = permanent_path.with_name(
            f".{permanent_path.name}.tmp-{os.getpid()}"
        )
        try:
            with self._latest.open("rb") as source, temporary_path.open("xb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.link(temporary_path, permanent_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._log.info("Saved permanent checkpoint: %s", permanent_path)
        return result


def run_training(cfg: DictConfig) -> None:
    upstream_train = load_upstream_train_module()
    original_loader = upstream_train.get_pushT_loader
    original_create_model = upstream_train.create_model
    original_wandb = upstream_train.wandb
    original_torch = upstream_train.torch
    original_barrier = upstream_train.barrier
    telemetry = TrainingTelemetry(cfg, original_wandb, upstream_train.log)
    max_step_state = MaxStepState(cfg)

    def create_logged_model(inner_cfg, device):
        if int(inner_cfg.data.frameskip) == 1:
            denoiser = original_create_model(inner_cfg, device)
        else:
            denoiser = create_driftworld_denoiser(
                unet_name=str(inner_cfg.model.unet_name),
                temp_list=inner_cfg.model.temp_list,
                n_neg=int(inner_cfg.model.n_neg),
                num_future_frames=int(inner_cfg.model.num_future_frames),
                num_history_frames=int(inner_cfg.model.num_history_frames),
                decay=float(inner_cfg.train.decay),
                action_frameskip=int(inner_cfg.data.frameskip),
                device=device,
            )
        return telemetry.wrap_model(denoiser)

    def create_logged_loader(inner_cfg, rank=0, world_size=1):
        loader = get_driftworld_hdf5_loader(
            inner_cfg, rank=rank, world_size=world_size
        )
        telemetry.register_loader(loader)
        return loader

    upstream_train.get_pushT_loader = create_logged_loader
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
        upstream_train.get_pushT_loader = original_loader
        upstream_train.create_model = original_create_model
        upstream_train.wandb = original_wandb
        upstream_train.torch = original_torch
        upstream_train.barrier = original_barrier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "scripts/train/config/driftworld/pusht.yaml",
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
        loader = get_driftworld_hdf5_loader(cfg, rank=rank, world_size=world_size)
        batch = next(iter(loader))
        if rank == 0:
            print(
                f"Preflight OK: image={tuple(batch['image'].shape)} "
                f"action={tuple(batch['action'].shape)} run={run_directory}"
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
