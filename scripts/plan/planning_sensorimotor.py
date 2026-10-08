"""Evaluate PushT-trained sensorimotor world models on ANYGOAL datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import anygoal  # noqa: F401
from anygoal.envs.cube_hop import (
    add_cube_evaluation_metrics,
    prepare_cube_evaluation,
)
from anygoal.envs.insert_peg import (
    add_insert_peg_evaluation_metrics,
)
from anygoal.evaluation.planning_artifacts import PlanningArtifactRecorder


def load_model_classes(source_root: Path):
    """Load the model definition from the vendored sensorimotor repository."""
    source_root = source_root.resolve()
    if not (source_root / "jepa.py").is_file():
        raise FileNotFoundError(f"Sensorimotor source not found: {source_root}")
    sys.path.insert(0, str(source_root))
    try:
        from jepa import JEPA
        from module import ARPredictor, Embedder, InverseModel, MLP
    finally:
        sys.path.pop(0)
    return JEPA, ARPredictor, Embedder, InverseModel, MLP


def build_model(train_cfg: DictConfig, source_root: Path):
    JEPA, ARPredictor, Embedder, InverseModel, MLP = load_model_classes(source_root)
    encoder = spt.backbone.utils.vit_hf(
        train_cfg.encoder_scale,
        patch_size=train_cfg.patch_size,
        image_size=train_cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = train_cfg.wm.get("embed_dim", hidden_dim)
    effective_action_dim = train_cfg.data.dataset.frameskip * train_cfg.wm.action_dim
    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=int(train_cfg.wm.get("history_size", 1)),
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **train_cfg.predictor,
        ),
        action_encoder=Embedder(
            input_dim=effective_action_dim,
            emb_dim=embed_dim,
        ),
        projector=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        inverse_model=InverseModel(
            embed_dim=embed_dim,
            action_dim=effective_action_dim,
            hidden_dim=train_cfg.inverse.get("hidden_dim", 256),
        ),
    )


def load_model(
    run_directory: Path,
    source_root: Path,
    device: str,
    train_config_path: Path | None = None,
    checkpoint_path: Path | None = None,
):
    train_config_path = train_config_path or run_directory / "config.yaml"
    checkpoint_path = checkpoint_path or run_directory / "checkpoints" / "last.ckpt"
    train_cfg = OmegaConf.load(train_config_path)
    model = build_model(train_cfg, source_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(size=image_size),
        ]
    )


def fit_processors(dataset, keys_to_cache: list[str]) -> dict:
    processors = {}
    for column in keys_to_cache:
        if column == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        processor.fit(values)
        processors[column] = processor
        if column != "action":
            processors[f"goal_{column}"] = processor
    return processors


def sample_start_indices(
    dataset,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
    start_selection: str = "random",
) -> tuple[list[int], list[int]]:
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode_ids = dataset.get_col_data(episode_column)
    step_indices = dataset.get_col_data("step_idx")
    episode_lengths = {
        int(episode): int(step_indices[episode_ids == episode].max()) + 1
        for episode in np.unique(episode_ids)
    }
    max_start_per_row = np.asarray(
        [episode_lengths[int(episode)] - goal_offset_steps - 1 for episode in episode_ids]
    )
    valid_indices = np.flatnonzero(step_indices <= max_start_per_row)
    if len(valid_indices) <= num_eval:
        raise ValueError(
            f"Only {len(valid_indices)} valid starts found; need more than {num_eval}."
        )

    if start_selection == "first_episodes":
        selected = np.asarray([
            np.flatnonzero((episode_ids == episode) & (step_indices == 0))[0]
            for episode in np.sort(np.unique(episode_ids))[:num_eval]
            if episode_lengths[int(episode)] > goal_offset_steps
        ], dtype=np.int64)
        if len(selected) < num_eval:
            raise ValueError("Not enough eligible episodes for first_episodes selection.")
    else:
        # Keep the original sensorimotor evaluation's sampling behavior exactly.
        generator = np.random.default_rng(seed)
        selected = generator.choice(len(valid_indices) - 1, size=num_eval, replace=False)
        selected = np.sort(valid_indices[selected])
    rows = dataset.get_row_data(selected)
    return rows[episode_column].tolist(), rows["step_idx"].tolist()


def resolve_run_directory(cfg: DictConfig, seed: int) -> Path:
    run_name = (
        str(cfg.model.run_name)
        if "run_name" in cfg.model
        else str(cfg.model.run_name_template).format(seed=seed)
    )
    return Path(cfg.model.checkpoint_root).expanduser() / run_name


def required_paths(cfg: DictConfig, seed: int) -> list[Path]:
    run_directory = resolve_run_directory(cfg, seed)
    train_config_path = Path(
        cfg.model.get("config", run_directory / "config.yaml")
    ).expanduser()
    checkpoint_path = Path(
        cfg.model.get("checkpoint", run_directory / "checkpoints" / "last.ckpt")
    ).expanduser()
    dataset_name = Path(str(cfg.eval.dataset_name)).expanduser()
    dataset_path = (
        dataset_name.with_suffix(".h5")
        if dataset_name.is_absolute()
        else Path(cfg.dataset.root).expanduser()
        / "datasets"
        / f"{dataset_name}.h5"
    )
    return [
        Path(cfg.model.source_root).expanduser() / "jepa.py",
        train_config_path,
        checkpoint_path,
        dataset_path,
    ]


def validate_config(cfg: DictConfig, seed: int) -> None:
    seeds = cfg.eval.seeds if "seeds" in cfg.eval else cfg.model.seeds
    if seed not in seeds:
        raise ValueError(f"seed must be one of {list(seeds)}, got {seed}")
    if cfg.plan_config.horizon * cfg.plan_config.action_block > cfg.eval.eval_budget:
        raise ValueError("Planning horizon exceeds the evaluation budget")
    missing = [path for path in required_paths(cfg, seed) if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Sensorimotor evaluation preflight failed:\n{formatted}")


def evaluate(cfg: DictConfig, seed: int) -> Path:
    validate_config(cfg, seed)
    policy_seed = int(cfg.seed.policy_base) + seed
    task_seed = (
        int(cfg.seed.task_base) + seed
        if "task_base" in cfg.seed
        else int(cfg.seed.task)
    )
    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed)

    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=list(cfg.dataset.keys_to_cache),
        cache_dir=Path(cfg.dataset.root).expanduser(),
    )
    episodes, start_steps = sample_start_indices(
        dataset,
        int(cfg.eval.num_eval),
        int(cfg.eval.goal_offset_steps),
        task_seed,
        str(cfg.eval.get("start_selection", "random")),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_directory = resolve_run_directory(cfg, seed)
    model = load_model(
        run_directory,
        Path(cfg.model.source_root).expanduser(),
        device,
        train_config_path=(
            Path(cfg.model.config).expanduser() if "config" in cfg.model else None
        ),
        checkpoint_path=(
            Path(cfg.model.checkpoint).expanduser()
            if "checkpoint" in cfg.model
            else None
        ),
    )
    world_config = OmegaConf.to_container(cfg.world, resolve=True)
    world_config["num_envs"] = int(cfg.eval.num_eval)
    world = swm.World(
        **world_config,
        image_shape=(int(cfg.eval.img_size), int(cfg.eval.img_size)),
    )
    solver = swm.solver.CEMSolver(
        model=model,
        seed=policy_seed,
        device=device,
        **OmegaConf.to_container(cfg.solver, resolve=True),
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True)),
        process=fit_processors(dataset, list(cfg.dataset.keys_to_cache)),
        transform={
            "pixels": image_transform(int(cfg.eval.img_size)),
            "goal": image_transform(int(cfg.eval.img_size)),
        },
    )
    world.set_policy(policy)

    output_directory = Path(cfg.output.directory).expanduser() / f"seed_{seed}"
    output_directory.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    callables = prepare_cube_evaluation(
        world,
        OmegaConf.to_container(cfg.eval.callables, resolve=True),
    )
    artifact_recorder = None
    if bool(cfg.output.get("save_artifacts", False)):
        artifact_recorder = PlanningArtifactRecorder(
            world, dataset, episodes, start_steps, cfg.eval.goal_offset_steps,
            output_directory / "artifacts", seed,
        )
        artifact_recorder.install()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=start_steps,
        goal_offset=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=episodes,
        callables=callables,
        video=output_directory if cfg.eval.save_video else None,
    )
    metrics = add_cube_evaluation_metrics(world, metrics)
    metrics = add_insert_peg_evaluation_metrics(world, metrics)
    if artifact_recorder is not None:
        artifact_recorder.save(metrics)
    result = {
        "name": f"seed_{seed}",
        "policy": "cem",
        "policy_seed": policy_seed,
        "task_seed": task_seed,
        "run_dir": str(run_directory),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "elapsed_seconds": time.time() - start_time,
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"metrics: {metrics}")
    print(f"wrote: {metrics_path}")
    world.close()
    return metrics_path


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, raw_value = override.split("=", 1)
        value = OmegaConf.from_dotlist([f"value={raw_value}"])["value"]
        OmegaConf.update(cfg, key, value, merge=True)
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    overrides = [override for override in args.overrides if override != "--"]
    cfg = load_config(args.config.resolve(), overrides)
    validate_config(cfg, args.seed)
    if args.preflight:
        print(f"Preflight OK for seed {args.seed}: {args.config}")
        return
    evaluate(cfg, args.seed)


if __name__ == "__main__":
    main()
