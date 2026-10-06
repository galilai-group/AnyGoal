"""Run Stable Worldmodel PreJEPA with explicit loss and ETA telemetry."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import types
from pathlib import Path

from hydra import compose, initialize_config_dir
import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


UPSTREAM_ROOT = Path(
    os.environ.get(
        "STABLE_WORLDMODEL_SOURCE",
        "~/.cache/anygoal/sources/stable-worldmodel",
    )
).expanduser().resolve()
UPSTREAM_TRAINER = UPSTREAM_ROOT / "scripts" / "train" / "prejepa.py"


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            "The pinned PreJEPA source no longer matches the expected wrapper patch."
        )
    return source.replace(old, new)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


class TrainingETACallback(Callback):
    """Log throughput and an estimated completion time to W&B and stdout."""

    def __init__(self, log_every_steps: int = 100) -> None:
        self.log_every_steps = int(log_every_steps)
        self.started_at = 0.0
        self.initial_step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self.started_at = time.monotonic()
        self.initial_step = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if not trainer.is_global_zero:
            return
        current_step = int(trainer.global_step)
        completed = current_step - self.initial_step
        if completed <= 0:
            return
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        total_steps = int(trainer.estimated_stepping_batches)
        remaining_steps = max(total_steps - current_step, 0)
        steps_per_second = completed / elapsed
        eta_seconds = remaining_steps / max(steps_per_second, 1e-12)
        pl_module.log_dict(
            {
                "timing/elapsed_seconds": elapsed,
                "timing/eta_seconds": eta_seconds,
                "timing/steps_per_second": steps_per_second,
                "timing/steps_remaining": float(remaining_steps),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            rank_zero_only=True,
        )
        if current_step % self.log_every_steps == 0 or remaining_steps == 0:
            print(
                f"step={current_step}/{total_steps} "
                f"rate={steps_per_second:.3f} steps/s "
                f"eta={_format_duration(eta_seconds)}",
                flush=True,
            )


def _load_upstream_module():
    if not UPSTREAM_TRAINER.is_file():
        raise FileNotFoundError(f"PreJEPA trainer not found: {UPSTREAM_TRAINER}")
    source = UPSTREAM_TRAINER.read_text()
    source = _replace_once(
        source,
        "def run(cfg):\n    # --- Dataset ---",
        "def run(cfg):\n    pl.seed_everything(cfg.seed, workers=True)\n    # --- Dataset ---",
    )
    source = _replace_once(
        source,
        "    logger = None\n    if cfg.wandb.enabled:\n        logger = WandbLogger(",
        "    wandb_logger = None\n    if cfg.wandb.enabled:\n        wandb_logger = WandbLogger(",
    )
    source = _replace_once(
        source,
        "        logger.log_hyperparams(OmegaConf.to_container(cfg))",
        "        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))",
    )
    source = _replace_once(
        source,
        "        logger=logger,\n        enable_checkpointing=True,",
        "        logger=wandb_logger,\n        enable_checkpointing=True,",
    )
    source = _replace_once(
        source,
        "        data=spt.data.DataModule(train=train_loader, val=val_loader),\n        ckpt_path=ckpt_path if ckpt_path.exists() else None,",
        "        data=spt.data.DataModule(train=train_loader, val=val_loader),\n        seed=cfg.seed,\n        ckpt_path=ckpt_path if ckpt_path.exists() else None,",
    )
    module = types.ModuleType("stable_worldmodel_prejepa_training")
    module.__file__ = str(UPSTREAM_TRAINER)
    exec(compile(source, str(UPSTREAM_TRAINER), "exec"), module.__dict__)
    return module


def main() -> None:
    upstream = _load_upstream_module()
    original_forward = upstream.dinowm_forward
    original_trainer = upstream.pl.Trainer
    original_checkpoint_hook = upstream.SaveCkptCallback.on_train_epoch_end

    def forward_with_total_loss(module, batch, stage, cfg):
        output = original_forward(module, batch, stage, cfg)
        metric_stage = "train" if stage == "fit" else stage
        module.log(
            f"{metric_stage}/loss",
            output["loss"].detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=output["emb"].shape[0],
        )
        return output

    class TrainerWithETA(original_trainer):
        def __init__(self, *args, callbacks=None, **kwargs):
            callbacks = list(callbacks or [])
            callbacks.append(
                TrainingETACallback(
                    log_every_steps=int(
                        os.environ.get("PREJEPA_ETA_EVERY", "100")
                    )
                )
            )
            super().__init__(*args, callbacks=callbacks, **kwargs)

    def save_checkpoint_once(callback, trainer, module) -> None:
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % callback.epoch_interval == 0 or epoch == trainer.max_epochs:
            callback._save(module.model, epoch)

    upstream.dinowm_forward = forward_with_total_loss
    upstream.pl.Trainer = TrainerWithETA
    upstream.SaveCkptCallback.on_train_epoch_end = save_checkpoint_once
    try:
        revision = subprocess.run(
            ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"Stable Worldmodel source: {UPSTREAM_ROOT} ({revision})", flush=True)
        overrides = [argument for argument in sys.argv[1:] if argument != "--print-config"]
        with initialize_config_dir(
            version_base=None,
            config_dir=str(UPSTREAM_TRAINER.parent / "config"),
        ):
            cfg = compose(config_name="prejepa", overrides=overrides)
        if "--print-config" in sys.argv:
            print(OmegaConf.to_yaml(cfg, resolve=True))
            return
        upstream.run.__wrapped__(cfg)
    finally:
        upstream.dinowm_forward = original_forward
        upstream.pl.Trainer = original_trainer
        upstream.SaveCkptCallback.on_train_epoch_end = original_checkpoint_hook


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    torch.set_float32_matmul_precision("high")
    main()
