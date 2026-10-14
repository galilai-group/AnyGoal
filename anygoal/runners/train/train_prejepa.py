"""Train the packaged Stable Worldmodel PreJEPA model on AnyGoal data."""

from __future__ import annotations

import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor

from anygoal.paths import checkpoint_root, stablewm_root


logger = logging.getLogger(__name__)
_PRINT_CONFIG = False


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _validate_dataset(cfg) -> None:
    dataset_name = OmegaConf.select(cfg, "dataset_name")
    if dataset_name is None:
        raise ValueError("PreJEPA training requires dataset_name.")
    dataset_path = Path(str(dataset_name)).expanduser()
    if dataset_path.is_absolute() and not dataset_path.is_file():
        raise ValueError(f"PreJEPA HDF5 dataset not found: {dataset_path}")


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


class SavePreJEPAWeights(Callback):
    def __init__(self, run_name: str, config, epoch_interval: int = 5) -> None:
        self.run_name = run_name
        self.config = config
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.config,
                filename=f"weights_epoch_{epoch}.pt",
            )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source="image", target="image") -> None:
        super().__init__()
        self.processor = processor
        self.source = source
        self.target = target

    def __call__(self, sample):
        frames = self.nested_get(sample, self.source)
        pixels = self.processor(frames, return_tensors="pt")[
            "pixel_values_videos"
        ].squeeze(0)
        self.nested_set(sample, pixels, self.target)
        return sample


def _image_preprocessor(source: str, target: str, image_size: int):
    statistics = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**statistics, source=source, target=target),
        spt.data.transforms.Resize(image_size, source=source, target=target),
    )


def _strip_action_dims(tensor: torch.Tensor, action_range: tuple[int, int]):
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]], dim=-1
    )


def prejepa_forward(module, batch, stage: str, cfg):
    for key in module.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()
    batch = module.model.encode(
        batch,
        target="emb",
        is_video=cfg.backbone.get("is_video_encoder", False),
    )
    embedding = batch["emb"][:, : cfg.wm.history_size, ...]
    predicted = module.model.predict(embedding)
    target = batch["emb"][:, cfg.wm.num_preds :, ...].detach()

    pixels_dim = batch["pixels_emb"].size(-1)
    batch["pixels_loss"] = F.mse_loss(
        predicted[..., :pixels_dim], target[..., :pixels_dim]
    )
    start = pixels_dim
    action_range = (0, 0)
    for key in module.model.extra_encoders:
        dimension = batch[f"{key}_emb"].size(-1)
        stop = start + dimension
        if key == "action":
            action_range = (start, stop)
        else:
            batch[f"{key}_loss"] = F.mse_loss(
                predicted[..., start:stop], target[..., start:stop]
            )
        start = stop

    batch["actionless_emb"] = _strip_action_dims(batch["emb"], action_range)
    batch["actionless_prev_emb"] = _strip_action_dims(embedding, action_range)
    batch["actionless_pred_emb"] = _strip_action_dims(predicted, action_range)
    batch["actionless_target_emb"] = _strip_action_dims(target, action_range)
    batch["loss"] = F.mse_loss(
        batch["actionless_pred_emb"], batch["actionless_target_emb"]
    )
    if batch["loss"].isnan():
        raise ValueError("NaN loss encountered.")

    metric_stage = "train" if stage == "fit" else stage
    metrics = {
        f"{metric_stage}/{key}": value.detach()
        for key, value in batch.items()
        if key.endswith("_loss")
    }
    metrics[f"{metric_stage}/loss"] = batch["loss"].detach()
    module.log_dict(
        metrics,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        batch_size=batch["emb"].shape[0],
    )
    return batch


def _load_data(cfg):
    encoding_keys = list(cfg.wm.get("encoding", {}).keys())
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        keys_to_load=["pixels", *encoding_keys],
        keys_to_cache=encoding_keys,
    )
    normalizers = [
        column_normalizer(dataset, column, column) for column in encoding_keys
    ]
    if cfg.backbone.get("is_video_encoder", False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source="pixels", target="pixels"),
            spt.data.transforms.Resize(
                cfg.image_size, source="pixels", target="pixels"
            ),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            _image_preprocessor("pixels", "pixels", cfg.image_size), *normalizers
        )
    dataset.transform = transform
    return dataset, encoding_keys


def _build_model(cfg, dataset, encoding_keys):
    with open_dict(cfg):
        cfg.extra_dims = {}
        for key in encoding_keys:
            if key not in dataset.column_names:
                raise ValueError(f"Encoding key {key!r} not found in dataset columns.")
            dimension = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dimension * cfg.frameskip if key == "action" else dimension
            )

    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval().requires_grad_(False)
    is_cnn = hasattr(encoder.config, "hidden_sizes")
    embedding_dim = (
        encoder.config.hidden_sizes[-1] if is_cnn else encoder.config.hidden_size
    ) + sum(cfg.wm.get("encoding", {}).values())
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    if cfg.backbone.get("is_video_encoder", False):
        num_patches += num_patches * (cfg.n_steps // 4)
    with open_dict(cfg):
        cfg.model.predictor.dim = embedding_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            "_target_": "torch.nn.ModuleDict",
            "modules": {
                key: {
                    "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                    "in_chans": cfg.extra_dims[key],
                    "emb_dim": int(cfg.wm.encoding[key]),
                }
                for key in encoding_keys
            },
        }
    return hydra.utils.instantiate(cfg.model, encoder=encoder)


@hydra.main(
    version_base=None,
    config_path="../../resources/train/prejepa",
    config_name="prejepa",
)
def run(cfg) -> None:
    _validate_dataset(cfg)
    pl.seed_everything(cfg.seed, workers=True)
    if _PRINT_CONFIG:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    dataset, encoding_keys = _load_data(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, validation_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=generator
    )
    loader_options = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_set,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_set, **loader_options)

    model = _build_model(cfg, dataset, encoding_keys)
    module = spt.Module(
        model=model,
        forward=partial(prejepa_forward, cfg=cfg),
        optim={"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer)}},
    )
    base_checkpoint_root = (
        stablewm_root() / "checkpoints"
        if "STABLEWM_HOME" in os.environ
        else checkpoint_root()
    )
    run_directory = base_checkpoint_root / (cfg.get("subdir") or "")
    run_directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_directory / "config.yaml")

    wandb_logger = None
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(**cfg.wandb.config)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SavePreJEPAWeights(cfg.output_model_name, cfg.model),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
            TrainingETACallback(int(os.environ.get("PREJEPA_ETA_EVERY", "100"))),
        ],
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )
    checkpoint = run_directory / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=validation_loader),
        seed=cfg.seed,
        ckpt_path=checkpoint if checkpoint.exists() else None,
    )
    manager()


def main() -> None:
    global _PRINT_CONFIG
    _PRINT_CONFIG = "--print-config" in sys.argv
    if _PRINT_CONFIG:
        sys.argv.remove("--print-config")
    run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    torch.set_float32_matmul_precision("high")
    main()
