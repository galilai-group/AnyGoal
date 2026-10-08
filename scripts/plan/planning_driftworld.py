"""Evaluate a DriftWorld checkpoint with StableWM CEM MPC."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anygoal  # noqa: E402,F401
from anygoal.envs.cube_hop import (  # noqa: E402
    add_cube_evaluation_metrics,
    prepare_cube_evaluation,
)
from anygoal.envs.insert_peg import (  # noqa: E402
    add_insert_peg_evaluation_metrics,
)
from anygoal.evaluation.planning_artifacts import PlanningArtifactRecorder
from anygoal.wm.driftworld import (  # noqa: E402
    DriftWorldActionProcessor,
    DriftWorldCubeWorldModel,
    DriftWorldWorldModel,
    DriftWorldWorldModelPolicy,
    driftworld_image_transform,
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
    OmegaConf.resolve(cfg)
    return cfg


def dataset_path(cfg: DictConfig) -> Path:
    name = Path(str(cfg.eval.dataset_name)).expanduser()
    if name.is_absolute():
        return name if name.suffix else name.with_suffix(".h5")
    return Path(cfg.dataset.root).expanduser() / "datasets" / f"{name}.h5"


def validate_config(cfg: DictConfig) -> None:
    if int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) > int(
        cfg.eval.eval_budget
    ):
        raise ValueError("Planning horizon exceeds the evaluation budget.")
    if int(cfg.plan_config.action_block) != int(cfg.model.action_frameskip):
        raise ValueError("plan_config.action_block must match model.action_frameskip.")
    if int(cfg.plan_config.history_len) != int(cfg.model.num_history_frames):
        raise ValueError("plan_config.history_len must match model.num_history_frames.")
    missing = [
        path
        for path in (
            Path(cfg.model.checkpoint).expanduser(),
            dataset_path(cfg),
        )
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"DriftWorld evaluation preflight failed:\n{formatted}")


def load_model(
    cfg: DictConfig,
    device: torch.device,
    action_processor: DriftWorldActionProcessor | None = None,
):
    common = {
        "temp_list": tuple(cfg.model.temp_list),
        "n_neg": int(cfg.model.n_neg),
        "num_future_frames": int(cfg.model.num_future_frames),
        "num_history_frames": int(cfg.model.num_history_frames),
        "decay": float(cfg.model.decay),
        "action_frameskip": int(cfg.model.action_frameskip),
        "action_mode": str(cfg.model.action_mode),
        "rollout_batch_size": int(cfg.model.rollout_batch_size),
        "num_rollout_samples": int(cfg.model.num_rollout_samples),
        "rollout_seed": int(cfg.model.rollout_seed),
        "action_processor": action_processor,
    }
    if str(cfg.model.get("source", "pusht")) == "robomimic":
        model = DriftWorldCubeWorldModel(
            unet_type=str(cfg.model.unet_type),
            **common,
        )
    else:
        model = DriftWorldWorldModel(
            unet_name=str(cfg.model.unet_name),
            action_scale=float(cfg.model.action_scale),
            absolute_action_max=float(cfg.model.absolute_action_max),
            **common,
        )
    step = model.load_native_checkpoint(cfg.model.checkpoint)
    return model.to(device).eval().requires_grad_(False), step


def get_dataset(cfg: DictConfig):
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        cache_dir=Path(cfg.dataset.root).expanduser(),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )


def sample_start_indices(dataset, num_eval, goal_offset_steps, seed, start_selection="random"):
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode_ids = dataset.get_col_data(episode_column)
    step_indices = dataset.get_col_data("step_idx")
    episode_lengths = {
        int(episode): int(step_indices[episode_ids == episode].max()) + 1
        for episode in np.unique(episode_ids)
    }
    maximum_start = np.asarray(
        [episode_lengths[int(episode)] - goal_offset_steps - 1 for episode in episode_ids]
    )
    valid_indices = np.flatnonzero(step_indices <= maximum_start)
    if len(valid_indices) < num_eval:
        raise ValueError(f"Only {len(valid_indices)} valid starts; need {num_eval}.")
    if start_selection == "first_episodes":
        selected = np.asarray([
            np.flatnonzero((episode_ids == episode) & (step_indices == 0))[0]
            for episode in np.sort(np.unique(episode_ids))[:num_eval]
            if episode_lengths[int(episode)] > goal_offset_steps
        ], dtype=np.int64)
        if len(selected) < num_eval:
            raise ValueError("Not enough eligible episodes for first_episodes selection.")
    else:
        generator = np.random.default_rng(seed)
        sampled_positions = generator.choice(
            len(valid_indices) - 1, size=num_eval, replace=False
        )
        selected = np.sort(valid_indices[sampled_positions])
    rows = dataset.get_row_data(selected)
    return rows[episode_column].tolist(), rows["step_idx"].tolist()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}.")


def evaluate(cfg: DictConfig) -> Path:
    validate_config(cfg)
    policy_seed = int(cfg.seed.policy)
    task_seed = int(cfg.seed.task)
    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(cfg)
    action_processor = DriftWorldActionProcessor.fit(
        dataset.get_col_data("action")
    )
    model, checkpoint_step = load_model(cfg, device, action_processor)
    if dataset.get_dim("action") != model.action_dim:
        raise ValueError(
            f"Dataset action dim {dataset.get_dim('action')} does not match "
            f"DriftWorld action_dim={model.action_dim}."
        )
    episodes, start_steps = sample_start_indices(
        dataset,
        int(cfg.eval.num_eval),
        int(cfg.eval.goal_offset_steps),
        task_seed,
        str(cfg.eval.get("start_selection", "random")),
    )

    world_config = OmegaConf.to_container(cfg.world, resolve=True)
    world_config["num_envs"] = int(cfg.eval.num_eval)
    world = swm.World(
        **world_config,
        image_shape=(int(cfg.eval.render_size), int(cfg.eval.render_size)),
    )
    solver = swm.solver.CEMSolver(
        model=model,
        seed=policy_seed,
        device=device,
        **OmegaConf.to_container(cfg.solver, resolve=True),
    )
    image_transform = driftworld_image_transform(model.image_size)
    policy = DriftWorldWorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            **OmegaConf.to_container(cfg.plan_config, resolve=True)
        ),
        action_processor=action_processor,
        transform={"pixels": image_transform, "goal": image_transform},
    )
    world.set_policy(policy)

    output_directory = Path(cfg.output.directory).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {metrics_path}")
    start_time = time.time()
    callables = prepare_cube_evaluation(
        world,
        OmegaConf.to_container(cfg.eval.callables, resolve=True),
    )
    artifact_recorder = None
    if bool(cfg.output.get("save_artifacts", False)):
        artifact_recorder = PlanningArtifactRecorder(
            world, dataset, episodes, start_steps, cfg.eval.goal_offset_steps,
            output_directory / "artifacts", task_seed,
        )
        artifact_recorder.install()
    try:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=episodes,
            callables=callables,
            video=output_directory / "videos" if cfg.output.save_video else None,
        )
        metrics = add_cube_evaluation_metrics(world, metrics)
        metrics = add_insert_peg_evaluation_metrics(world, metrics)
        if artifact_recorder is not None:
            artifact_recorder.save(metrics)
    finally:
        world.close()
    result = {
        "checkpoint": str(Path(cfg.model.checkpoint).expanduser().resolve()),
        "checkpoint_step": checkpoint_step,
        "policy": "stable_worldmodel.CEMSolver",
        "policy_seed": policy_seed,
        "task_seed": task_seed,
        "action_scaler": {
            "mean": action_processor.mean.tolist(),
            "scale": action_processor.scale.tolist(),
        },
        "episodes": episodes,
        "start_steps": start_steps,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "elapsed_seconds": time.time() - start_time,
    }
    metrics_path.write_text(json.dumps(result, indent=2, default=json_default) + "\n")
    print(f"metrics: {metrics}")
    print(f"wrote: {metrics_path}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "scripts"
            / "plan"
            / "config"
            / "driftworld"
            / "pusht_lewm.yaml"
        ),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = [override for override in args.overrides if override != "--"]
    cfg = load_config(args.config.resolve(), overrides)
    validate_config(cfg)
    if args.preflight:
        print(f"Preflight OK: {args.config}")
        return
    evaluate(cfg)


if __name__ == "__main__":
    main()
