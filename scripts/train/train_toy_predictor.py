"""Train a true-state toy predictor with SWM and stable-pretraining."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets import (  # noqa: E402
    MinMaxTransform,
    ZScoreTransform,
)
from anygoal.wm.toy_predictors import (  # noqa: E402
    count_trainable_parameters,
)


class SavePredictorWeights(Callback):
    """Save immutable, epoch-labelled model weights in the experiment folder."""

    def __init__(self, run_directory: Path, every_n_epochs: int) -> None:
        self.run_directory = run_directory
        self.every_n_epochs = int(every_n_epochs)

    def on_train_epoch_end(self, trainer, module) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero or (
            epoch % self.every_n_epochs and epoch != trainer.max_epochs
        ):
            return
        checkpoint = self.run_directory / f"weights_epoch_{epoch:03d}.pt"
        if checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
        torch.save(module.model.state_dict(), checkpoint)


class MultiValidationDataModule(spt.data.DataModule):
    """SPT data module whose validation loaders have stable, named positions."""

    def __init__(self, train_loader, validation_loaders) -> None:
        super().__init__(train=train_loader, val=validation_loaders[0])
        self.validation_loaders = list(validation_loaders)

    def val_dataloader(self):
        return self.validation_loaders


class MultiValidationModule(spt.Module):
    """SPT module that assigns a metric namespace to every validation set."""

    def __init__(self, validation_names: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.validation_names = tuple(validation_names)

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict:
        batch["batch_idx"] = batch_idx
        validation_name = self.validation_names[dataloader_idx]
        return self.forward(batch, stage=f"validation/{validation_name}")


def _create_run_directory(root: str | Path, description: str) -> Path:
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    base = root / f"{date}_{description}"
    for suffix in range(10_000):
        candidate = base if suffix == 0 else Path(f"{base}_{suffix:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory under {root}.")


def _resolve_resume_checkpoint(checkpoint_path: str | Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError("resume.checkpoint_path must be an absolute path.")
    if checkpoint.is_dir():
        candidates = (
            checkpoint / "last.ckpt",
            checkpoint / "checkpoints" / "last.ckpt",
        )
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(
                "A checkpoint directory must contain exactly one last.ckpt at "
                "its root or under checkpoints/."
            )
        checkpoint = existing[0]
    if not checkpoint.is_file() or checkpoint.suffix != ".ckpt":
        raise FileNotFoundError(
            "resume.checkpoint_path must point to a full-state .ckpt file or "
            "a directory containing last.ckpt. weights_epoch_*.pt files are "
            "model weights only and cannot restore optimizer state."
        )
    return checkpoint


def _resolve_run_directory(
    checkpoint: Path | None,
    requested_directory: str | Path | None,
    checkpoint_root: str | Path,
    description: str,
) -> Path:
    if checkpoint is None:
        if requested_directory is not None:
            raise ValueError(
                "resume.run_directory requires resume.checkpoint_path."
            )
        return _create_run_directory(checkpoint_root, description)
    if requested_directory is None:
        raise ValueError(
            "resume.run_directory is required when resuming so epoch weights "
            "and W&B metadata continue in the intended experiment directory."
        )
    run_directory = Path(requested_directory).expanduser()
    if not run_directory.is_absolute():
        raise ValueError("resume.run_directory must be an absolute path.")
    if not run_directory.is_dir():
        raise FileNotFoundError(
            f"Resume run directory does not exist: {run_directory}"
        )
    return run_directory


def _wandb_logger_config(
    config: DictConfig,
    run_directory: Path,
    resuming: bool,
) -> dict:
    logger_config = OmegaConf.to_container(config, resolve=True)
    if not resuming:
        return logger_config
    sidecar = run_directory / "wandb_resume.json"
    if not sidecar.is_file():
        print(
            f"No {sidecar.name} in {run_directory}; W&B will start a new run."
        )
        return logger_config
    resume_info = json.loads(sidecar.read_text())
    run_id = resume_info.get("id")
    if not run_id:
        raise ValueError(f"W&B resume sidecar has no run id: {sidecar}")
    for key in ("project", "entity"):
        saved = resume_info.get(key)
        configured = logger_config.get(key)
        if saved and configured and saved != configured:
            raise ValueError(
                f"W&B {key} mismatch: sidecar has {saved!r}, config has "
                f"{configured!r}."
            )
    logger_config["id"] = run_id
    logger_config["resume"] = "must"
    return logger_config


def _loader_config(config: DictConfig) -> dict:
    loader = OmegaConf.to_container(config, resolve=True)
    if loader["num_workers"] == 0:
        loader.pop("persistent_workers", None)
        loader.pop("prefetch_factor", None)
    return loader


def _load_dataset(data_config: DictConfig, num_steps: int):
    dataset_name = Path(str(data_config.dataset_name)).expanduser()
    location = (
        {"path": dataset_name}
        if dataset_name.is_absolute() or dataset_name.suffix in {".h5", ".hdf5"}
        else {"name": str(data_config.dataset_name), "cache_dir": data_config.cache_dir}
    )
    return swm.data.HDF5Dataset(
        **location,
        num_steps=num_steps,
        frameskip=data_config.frameskip,
        keys_to_load=list(data_config.keys_to_load),
        keys_to_cache=list(data_config.keys_to_cache),
        transform=None,
    )


def _validate_adapter_dimensions(adapter, state_dim: int, action_dim: int) -> None:
    if adapter.state_dim != state_dim or adapter.action_dim != action_dim:
        raise ValueError(
            "Every validation adapter must match the training representation: "
            f"expected state/action dims ({state_dim}, {action_dim}), got "
            f"({adapter.state_dim}, {adapter.action_dim})."
        )


def _load_normalization_statistics(
    path: str | Path,
    *,
    adapter,
    dataset_name: str,
) -> tuple[dict, dict]:
    statistics_path = Path(path).expanduser()
    if not statistics_path.is_absolute():
        statistics_path = REPO_ROOT / statistics_path
    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"Normalization statistics not found: {statistics_path}"
        )
    payload = json.loads(statistics_path.read_text())
    statistics_dataset = payload.get("dataset")
    if Path(str(statistics_dataset or "")).name != Path(str(dataset_name)).name:
        raise ValueError(
            f"Normalization statistics were fit on {statistics_dataset!r}, "
            f"not {dataset_name!r}."
        )

    normalizers = {}
    statistics = {}
    expected_dimensions = {
        "state": adapter.state_dim,
        "action": adapter.action_dim,
    }
    for source, expected_dimension in expected_dimensions.items():
        source_statistics = payload.get(source, {})
        normalization_type = source_statistics.get("type", "zscore")
        if normalization_type == "zscore":
            first_name, second_name = "mean", "std"
        elif normalization_type == "minmax":
            first_name, second_name = "min", "max"
        else:
            raise ValueError(
                f"Unsupported {source} normalization type: {normalization_type!r}."
            )
        first = torch.as_tensor(
            source_statistics.get(first_name), dtype=torch.float32
        )
        second = torch.as_tensor(
            source_statistics.get(second_name), dtype=torch.float32
        )
        if first.shape != (expected_dimension,) or second.shape != (
            expected_dimension,
        ):
            raise ValueError(
                f"{source} statistics must have dimension {expected_dimension}; "
                f"got {first_name}/{second_name} shapes "
                f"{tuple(first.shape)}/{tuple(second.shape)}."
            )
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError(f"{source} normalization statistics must be finite.")
        if normalization_type == "zscore" and (second <= 0).any():
            raise ValueError(f"{source} normalization std must be positive.")
        if normalization_type == "minmax" and (second <= first).any():
            raise ValueError(
                f"{source} normalization max must exceed min in every dimension."
            )
        normalizers[source] = (normalization_type, first, second)
        statistics[source] = {
            "type": normalization_type,
            first_name: first.tolist(),
            second_name: second.tolist(),
        }
    return normalizers, statistics


def _make_transform(adapter, normalizers: dict):
    transforms = [adapter]
    for source in ("state", "action"):
        normalization_type, first, second = normalizers[source]
        transform_type = (
            ZScoreTransform
            if normalization_type == "zscore"
            else MinMaxTransform
        )
        transforms.append(transform_type(first, second, source=source))
    return spt.data.transforms.Compose(
        *transforms,
    )


def _indices_digest(indices) -> str:
    values = torch.as_tensor(list(indices), dtype=torch.int64)
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _select_validation_fraction(dataset, percentage: float, seed: int):
    percentage = float(percentage)
    if not 0.0 < percentage <= 100.0:
        raise ValueError("Validation percentage must be in (0, 100].")
    if percentage == 100.0:
        return dataset, range(len(dataset))
    generator = torch.Generator().manual_seed(int(seed))
    selected, _ = spt.data.random_split(
        dataset,
        lengths=[percentage / 100.0, 1.0 - percentage / 100.0],
        generator=generator,
    )
    return selected, selected.indices


def _split_primary_dataset(dataset, train_fraction: float, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    return spt.data.random_split(
        dataset,
        lengths=[train_fraction, 1.0 - train_fraction],
        generator=generator,
    )


def make_data(cfg: DictConfig):
    num_steps = cfg.history_size + cfg.num_preds
    adapter = hydra.utils.instantiate(cfg.data.adapter)
    _validate_adapter_dimensions(adapter, cfg.data.state_dim, cfg.data.action_dim)
    dataset = _load_dataset(cfg.data, num_steps)

    normalizers, statistics = _load_normalization_statistics(
        cfg.data.normalization_path,
        adapter=adapter,
        dataset_name=str(cfg.data.dataset_name),
    )
    dataset.transform = _make_transform(adapter, normalizers)

    train_set, validation_set = _split_primary_dataset(
        dataset, cfg.train_split, cfg.seed
    )
    data_metadata = {
        "primary_split": {
            "seed": int(cfg.seed),
            "dataset_size": len(dataset),
            "train_size": len(train_set),
            "validation_size": len(validation_set),
            "validation_indices_sha256": _indices_digest(validation_set.indices),
        },
        "extra_validation_sets": {},
    }

    loader = _loader_config(cfg.loader)
    validation_loader = _loader_config(cfg.validation_loader)
    shuffle_generator = torch.Generator().manual_seed(int(cfg.seed) + 1)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=shuffle_generator,
        **loader,
    )
    validation_loaders = [
        torch.utils.data.DataLoader(
            validation_set,
            shuffle=False,
            drop_last=False,
            **validation_loader,
        )
    ]
    validation_names = [str(cfg.data.name)]

    extra_validation_sets = cfg.get("extra_validation_sets")
    if extra_validation_sets is None:
        extra_validation_sets = {}
    for name in sorted(extra_validation_sets):
        if str(name) in validation_names:
            raise ValueError(f"Duplicate validation-set name: {name!r}.")
        validation_config = extra_validation_sets[name]
        validation_adapter = hydra.utils.instantiate(validation_config.adapter)
        _validate_adapter_dimensions(
            validation_adapter,
            cfg.data.state_dim,
            cfg.data.action_dim,
        )
        validation_dataset = _load_dataset(validation_config, num_steps)
        validation_dataset.transform = _make_transform(
            validation_adapter, normalizers
        )
        selected_dataset, selected_indices = _select_validation_fraction(
            validation_dataset,
            validation_config.percentage,
            validation_config.seed,
        )
        validation_loaders.append(
            torch.utils.data.DataLoader(
                selected_dataset,
                shuffle=False,
                drop_last=False,
                **validation_loader,
            )
        )
        validation_names.append(str(name))
        data_metadata["extra_validation_sets"][str(name)] = {
            "dataset_name": str(validation_config.dataset_name),
            "percentage": float(validation_config.percentage),
            "seed": int(validation_config.seed),
            "dataset_size": len(validation_dataset),
            "validation_size": len(selected_dataset),
            "validation_indices_sha256": _indices_digest(selected_indices),
        }

    return (
        MultiValidationDataModule(train_loader, validation_loaders),
        statistics,
        validation_names,
        data_metadata,
    )


def predictor_forward(self, batch: dict, stage: str) -> dict:
    states = torch.nan_to_num(batch["state"].float())
    actions = torch.nan_to_num(batch["action"].float())
    output = self.model.loss(states, actions)
    metric_stage = "train" if stage == "fit" else stage
    metrics = {}
    for name, value in output.items():
        if name == "loss":
            metrics[f"{metric_stage}/mse"] = value.detach()
        elif name.endswith("_loss"):
            metric_name = f"{name[:-5]}_mse"
            metrics[f"{metric_stage}/{metric_name}"] = value.detach()
    self.log_dict(
        metrics,
        on_step=stage == "fit",
        on_epoch=True,
        sync_dist=True,
        add_dataloader_idx=False,
        batch_size=states.shape[0],
    )
    return output


@hydra.main(
    version_base=None,
    config_path="config/toy_predictors",
    config_name="train",
)
def run(cfg: DictConfig) -> None:
    if cfg.num_preds != 1:
        raise ValueError("The current toy predictors support exactly one future frame.")
    pl.seed_everything(cfg.seed, workers=True)
    data_module, statistics, validation_names, data_metadata = make_data(cfg)
    primary_split = data_metadata["primary_split"]
    print(
        "Primary validation split: "
        f"{primary_split['validation_size']:,} clips, "
        f"sha256={primary_split['validation_indices_sha256']}"
    )
    for name, metadata in data_metadata["extra_validation_sets"].items():
        print(
            f"Extra validation {name}: {metadata['validation_size']:,}/"
            f"{metadata['dataset_size']:,} clips ({metadata['percentage']:g}%)"
        )
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("name")
    model = hydra.utils.instantiate(model_config)
    parameter_count = count_trainable_parameters(model)
    print(f"{cfg.model.name}: {parameter_count:,} trainable parameters")

    resume_checkpoint = _resolve_resume_checkpoint(cfg.resume.checkpoint_path)
    run_directory = _resolve_run_directory(
        resume_checkpoint,
        cfg.resume.run_directory,
        cfg.output.checkpoint_root,
        f"{cfg.data.name}_{cfg.model.name}",
    )
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    resolved_config["parameter_count"] = parameter_count
    resolved_config["normalization"] = statistics
    resolved_config["data_metadata"] = data_metadata
    config_path = run_directory / "config.yaml"
    if resume_checkpoint is None:
        OmegaConf.save(OmegaConf.create(resolved_config), config_path)
    elif not config_path.is_file():
        raise FileNotFoundError(
            f"Resume run directory has no original config.yaml: {run_directory}"
        )

    optimizer_config = {
        "modules": "model",
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "interval": "epoch",
    }
    if cfg.scheduler is not None:
        optimizer_config["scheduler"] = OmegaConf.to_container(
            cfg.scheduler, resolve=True
        )
    else:
        optimizer_config["scheduler"] = {
            "type": "ConstantLR",
            "factor": 1.0,
            "total_iters": 1,
        }
    optimizers = {"model_optimizer": optimizer_config}
    module = MultiValidationModule(
        validation_names=validation_names,
        model=model,
        forward=predictor_forward,
        optim=optimizers,
        hparams=resolved_config,
    )
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(
            **_wandb_logger_config(
                cfg.wandb.config,
                run_directory,
                resuming=resume_checkpoint is not None,
            )
        )
    callback = SavePredictorWeights(
        run_directory,
        cfg.output.save_every_n_epochs,
    )
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=[callback],
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=run_directory,
        num_sanity_val_steps=1,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
        weights_only=False,
    )
    manager()


if __name__ == "__main__":
    run()
