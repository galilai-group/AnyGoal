"""Train predictor-only DriftWorld on normalized true-state transitions."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from anygoal.datasets import MinMaxTransform, ZScoreTransform  # noqa: E402
from anygoal.wm.toy_predictors import (  # noqa: E402
    count_trainable_parameters,
)
from anygoal.resources import resource_path


class Compose:
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def _create_run_directory(root: str | Path, description: str) -> Path:
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


def _statistics_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else resource_path("train") / path


def _load_normalizers(cfg: DictConfig, adapter) -> tuple[dict, dict]:
    path = _statistics_path(cfg.data.normalization_path)
    if not path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {path}")
    payload = json.loads(path.read_text())
    configured_dataset = Path(str(cfg.data.dataset_name)).name
    if Path(str(payload.get("dataset", ""))).name != configured_dataset:
        raise ValueError(
            f"Statistics were fit on {payload.get('dataset')!r}, not "
            f"{str(cfg.data.dataset_name)!r}."
        )

    normalizers = {}
    for source, dimension in {
        "state": adapter.state_dim,
        "action": adapter.action_dim,
    }.items():
        values = payload[source]
        normalization_type = values.get("type", "zscore")
        if normalization_type not in {"zscore", "minmax"}:
            raise ValueError(
                f"Unsupported {source} normalization: {normalization_type!r}."
            )
        names = ("mean", "std") if normalization_type == "zscore" else ("min", "max")
        first = torch.tensor(values[names[0]], dtype=torch.float32)
        second = torch.tensor(values[names[1]], dtype=torch.float32)
        if first.shape != (dimension,) or second.shape != (dimension,):
            raise ValueError(
                f"{source} statistics must have dimension {dimension}, got "
                f"{tuple(first.shape)} and {tuple(second.shape)}."
            )
        transform_type = ZScoreTransform if normalization_type == "zscore" else MinMaxTransform
        normalizers[source] = transform_type(first, second, source=source)
    return normalizers, payload


def _make_dataset(cfg: DictConfig):
    adapter = hydra.utils.instantiate(cfg.data.adapter)
    if adapter.state_dim != cfg.data.state_dim or adapter.action_dim != cfg.data.action_dim:
        raise ValueError("Configured state/action dimensions do not match the adapter.")
    normalizers, statistics = _load_normalizers(cfg, adapter)
    dataset = swm.data.load_dataset(
        cfg.data.dataset_name,
        cache_dir=cfg.data.cache_dir,
        num_steps=cfg.history_size + cfg.num_preds,
        frameskip=cfg.data.frameskip,
        keys_to_load=list(cfg.data.keys_to_load),
        keys_to_cache=list(cfg.data.keys_to_cache),
        transform=Compose(adapter, normalizers["state"], normalizers["action"]),
    )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        lengths=[float(cfg.train_split), 1.0 - float(cfg.train_split)],
        generator=generator,
    )
    return train_set, validation_set, statistics


def _loader_kwargs(cfg: DictConfig) -> dict:
    kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    if int(kwargs["num_workers"]) == 0:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
    return kwargs


def _make_loader(dataset, cfg: DictConfig, epoch: int):
    generator = torch.Generator().manual_seed(int(cfg.seed) + 1 + int(epoch))
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **_loader_kwargs(cfg),
    )


def _save_checkpoint(
    run_directory: Path,
    completed_steps: int,
    epoch: int,
    model,
    optimizer,
    scheduler,
    wandb_id: str | None,
) -> tuple[Path, Path]:
    checkpoint = run_directory / f"checkpoint_step_{completed_steps:08d}.pth"
    weights = run_directory / f"weights_step_{completed_steps:08d}.pt"
    if checkpoint.exists() or weights.exists():
        raise FileExistsError(
            f"Refusing to overwrite checkpoint step {completed_steps} in {run_directory}."
        )
    payload = {
        "step": completed_steps,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "wandb_id": wandb_id,
    }
    torch.save(payload, checkpoint)
    torch.save(model.state_dict(), weights)
    return checkpoint, weights


def _load_checkpoint(path: str | Path, model, optimizer, scheduler) -> tuple[int, str | None]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["step"]), payload.get("wandb_id")


@hydra.main(
    version_base=None,
    config_path="../../resources/train/toy_driftworld",
    config_name="train",
)
def run(cfg: DictConfig) -> None:
    if int(cfg.num_preds) != 1:
        raise ValueError("Toy DriftWorld supports exactly one future state.")
    checkpoint_every = cfg.train.get("checkpoint_every_steps")
    if checkpoint_every is not None and int(checkpoint_every) <= 0:
        raise ValueError(
            "checkpoint_every_steps must be positive when intermediate "
            "checkpointing is enabled."
        )
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, validation_set, statistics = _make_dataset(cfg)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_name = model_config.pop("name")
    model = hydra.utils.instantiate(model_config).to(device)
    parameter_count = count_trainable_parameters(model)
    print(
        f"{cfg.data.name}/{model_name}: {parameter_count:,} trainable parameters; "
        f"{len(train_set):,} train and {len(validation_set):,} validation clips"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        betas=(0.9, float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / float(cfg.optimizer.lr),
        end_factor=1.0,
        total_iters=int(cfg.optimizer.warmup_steps),
    )

    resume = cfg.get("resume_checkpoint")
    if resume:
        resume_path = Path(resume).expanduser().resolve()
        run_directory = resume_path.parent
        completed_steps, wandb_id = _load_checkpoint(
            resume_path, model, optimizer, scheduler
        )
        if not (run_directory / "config.yaml").is_file():
            raise FileNotFoundError(f"Resume directory has no config.yaml: {run_directory}")
    else:
        run_directory = _create_run_directory(
            cfg.output.checkpoint_root, cfg.output.description
        )
        completed_steps = 0
        wandb_id = None
        resolved = OmegaConf.to_container(cfg, resolve=True)
        resolved["parameter_count"] = parameter_count
        resolved["normalization"] = statistics
        resolved["dataset_sizes"] = {
            "train": len(train_set),
            "validation": len(validation_set),
        }
        OmegaConf.save(OmegaConf.create(resolved), run_directory / "config.yaml")

    wandb_run = None
    if bool(cfg.wandb.enabled):
        import wandb

        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            id=wandb_id,
            resume="must" if wandb_id else None,
            dir=run_directory,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_id = wandb_run.id
        wandb_run.log({"parameter_count": parameter_count}, step=completed_steps)

    max_steps = cfg.train.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("train.max_steps must be positive when set.")
    max_epochs = int(cfg.train.num_epochs)
    checkpoint_every = (
        None if checkpoint_every is None else int(checkpoint_every)
    )
    log_every = int(cfg.train.log_every_steps)
    steps_per_epoch = len(_make_loader(train_set, cfg, 0))
    if steps_per_epoch == 0:
        raise ValueError("The training split is smaller than one complete batch.")
    start_epoch = completed_steps // steps_per_epoch
    skip_batches = completed_steps % steps_per_epoch
    last_saved_step = None
    last_epoch = max(start_epoch - 1, 0)

    model.train()
    for epoch in range(start_epoch, max_epochs):
        last_epoch = epoch
        loader = _make_loader(train_set, cfg, epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < skip_batches:
                continue
            states = torch.nan_to_num(batch["state"].float()).to(device)
            actions = torch.nan_to_num(batch["action"].float()).to(device)
            output = model.loss(states, actions)
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite drifting loss at step {completed_steps}.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optimizer.grad_clip))
            optimizer.step()
            scheduler.step()
            model.update_ema()
            completed_steps += 1

            metrics = {
                "loss/drifting": float(loss.detach().cpu()),
                "diagnostics/sample_mse": float(output["future_loss"].cpu()),
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
            metrics.update(
                {
                    f"loss/components/{key.removeprefix('loss_')}": value
                    for key, value in output["drift_metrics"].items()
                    if key.startswith("loss_")
                }
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=completed_steps)
            if completed_steps % log_every == 0:
                print(
                    f"step={completed_steps} epoch={epoch + 1}/{max_epochs} "
                    f"loss={metrics['loss/drifting']:.6g} lr={metrics['lr']:.6g}"
                )
            if (
                checkpoint_every is not None
                and completed_steps % checkpoint_every == 0
            ):
                checkpoint, weights = _save_checkpoint(
                    run_directory,
                    completed_steps,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    wandb_id,
                )
                last_saved_step = completed_steps
                print(f"Saved {checkpoint} and {weights}")
            if max_steps is not None and completed_steps >= max_steps:
                break
        if max_steps is not None and completed_steps >= max_steps:
            break

    if last_saved_step != completed_steps:
        checkpoint, weights = _save_checkpoint(
            run_directory,
            completed_steps,
            last_epoch,
            model,
            optimizer,
            scheduler,
            wandb_id,
        )
        print(f"Saved final {checkpoint} and {weights}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run()
