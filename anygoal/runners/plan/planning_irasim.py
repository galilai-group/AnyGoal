"""Evaluate an IRASim checkpoint with Stable WorldModel CEM MPC."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

import anygoal  # noqa: E402,F401
from anygoal.paths import dataset_file
from anygoal.resources import resource_path
from anygoal.envs.cube_hop.env import (  # noqa: E402
    add_cube_evaluation_metrics,
    prepare_cube_evaluation,
)
from anygoal.envs.insert_peg.env import (  # noqa: E402
    add_insert_peg_evaluation_metrics,
)
from anygoal._planning_artifacts import (  # noqa: E402
    PlanningArtifactRecorder,
)
from anygoal.wm.irasim import (  # noqa: E402
    IRASimActionProcessor,
    IRASimWorldModel,
    IRASimWorldModelPolicy,
    create_irasim_transformer,
    irasim_image_transform,
    irasim_source_root,
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
    if not os.environ.get("IRASIM_SCHEDULER_PATH"):
        scheduler_path = Path(str(cfg.model.scheduler_path)).expanduser()
        if not scheduler_path.is_dir():
            cfg.model.scheduler_path = str(
                irasim_source_root() / "pretrained_models" / "scheduler"
            )
    return cfg


def dataset_path(cfg: DictConfig) -> Path:
    return dataset_file(str(cfg.eval.dataset_name), root=cfg.dataset.root)


def validate_config(cfg: DictConfig) -> None:
    if int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) > int(
        cfg.eval.eval_budget
    ):
        raise ValueError("Planning horizon exceeds the evaluation budget.")
    if int(cfg.plan_config.action_block) != int(cfg.model.action_frameskip):
        raise ValueError("plan_config.action_block must match model.action_frameskip.")
    if int(cfg.plan_config.history_len) != int(cfg.model.num_history_frames):
        raise ValueError(
            "plan_config.history_len must match model.num_history_frames."
        )
    if float(cfg.model.guidance_scale) != 1.0:
        raise ValueError("The vendored IRASim inference path requires guidance_scale=1.")
    if int(cfg.solver.topk) > int(cfg.solver.num_samples):
        raise ValueError("solver.topk cannot exceed solver.num_samples.")
    missing = [
        path
        for path in (
            Path(cfg.model.checkpoint).expanduser(),
            dataset_path(cfg),
            Path(cfg.model.scheduler_path).expanduser(),
        )
        if not path.exists()
    ]
    vae_path = Path(cfg.model.vae_model_path).expanduser()
    if str(cfg.model.vae_model_path).startswith("/") and not vae_path.exists():
        missing.append(vae_path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"IRASim evaluation preflight failed:\n{formatted}")


def get_dataset(cfg: DictConfig):
    return swm.data.HDF5Dataset(
        path=dataset_path(cfg),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )


def get_action_processor(cfg: DictConfig, dataset) -> IRASimActionProcessor:
    stats_name = cfg.dataset.get("action_stats_name")
    if not stats_name:
        return IRASimActionProcessor.fit(dataset.get_col_data("action"))
    stats_dataset = swm.data.HDF5Dataset(
        path=dataset_file(str(stats_name), root=cfg.dataset.root),
        keys_to_cache=["action"],
    )
    return IRASimActionProcessor.fit(stats_dataset.get_col_data("action"))


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
        selected = np.sort(generator.choice(valid_indices, size=num_eval, replace=False))
    rows = dataset.get_row_data(selected)
    return rows[episode_column].tolist(), rows["step_idx"].tolist()


def load_model(
    cfg: DictConfig,
    device: torch.device,
    action_processor: IRASimActionProcessor,
):
    from diffusers.models import AutoencoderKL
    from diffusers.schedulers import PNDMScheduler

    checkpoint_path = Path(cfg.model.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "args" not in checkpoint:
        raise TypeError("Expected a native IRASim checkpoint containing saved args.")
    model_args = checkpoint["args"]
    action_dim = int(model_args.get("action_dim", 2))
    saved_frameskip = int(model_args.get("frameskip", 1))
    if saved_frameskip != int(cfg.model.action_frameskip):
        raise ValueError(
            f"Checkpoint frameskip={saved_frameskip}, but eval config requests "
            f"{cfg.model.action_frameskip}."
        )
    saved_history_frames = int(model_args.get("mask_frame_num", 1))
    if saved_history_frames != int(cfg.model.num_history_frames):
        raise ValueError(
            f"Checkpoint history={saved_history_frames}, but eval config requests "
            f"{cfg.model.num_history_frames}."
        )
    transformer = create_irasim_transformer(model_args, action_dim=action_dim)
    state_dict = checkpoint.get("ema", checkpoint.get("model"))
    if state_dict is None:
        raise KeyError("IRASim checkpoint contains neither ema nor model weights.")
    transformer.load_state_dict(state_dict, strict=True)
    vae_kwargs = {}
    subfolder = cfg.model.get("vae_subfolder", "vae")
    if subfolder:
        vae_kwargs["subfolder"] = str(subfolder)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae_model_path, **vae_kwargs)
    scheduler = PNDMScheduler.from_pretrained(
        cfg.model.scheduler_path,
        beta_start=float(cfg.model.beta_start),
        beta_end=float(cfg.model.beta_end),
        beta_schedule=str(cfg.model.beta_schedule),
        variance_type=str(cfg.model.variance_type),
    )
    model = IRASimWorldModel(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        action_dim=action_dim,
        action_frameskip=saved_frameskip,
        video_size=tuple(model_args.video_size),
        num_frames=int(model_args.num_frames),
        num_history_frames=saved_history_frames,
        inference_steps=int(cfg.model.inference_steps),
        guidance_scale=float(cfg.model.guidance_scale),
        rollout_batch_size=int(cfg.model.rollout_batch_size),
        num_rollout_samples=int(cfg.model.num_rollout_samples),
        rollout_seed=int(cfg.model.rollout_seed),
        action_scale=tuple(model_args.action_scale),
        action_processor=action_processor,
        cost_type=str(cfg.model.cost_type),
    )
    return model.to(device).eval().requires_grad_(False), checkpoint_path


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
    action_processor = get_action_processor(cfg, dataset)
    model, checkpoint_path = load_model(cfg, device, action_processor)
    if dataset.get_dim("action") != model.action_dim:
        raise ValueError(
            f"Dataset action dim {dataset.get_dim('action')} does not match "
            f"IRASim action_dim={model.action_dim}."
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
    image_transform = irasim_image_transform(model.video_size)
    policy = IRASimWorldModelPolicy(
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
        "checkpoint": str(checkpoint_path),
        "policy": "stable_worldmodel.CEMSolver",
        "cost_type": model.cost_type,
        "policy_seed": policy_seed,
        "task_seed": task_seed,
        "action_statistics": {
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
        default=resource_path("plan", "irasim", "pusht.yaml"),
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
