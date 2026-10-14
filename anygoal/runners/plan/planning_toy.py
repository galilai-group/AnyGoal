"""Plan in SWM environments with an encoder-free true-state predictor."""

from __future__ import annotations

import json
import os

os.environ["MUJOCO_GL"] = "egl"

import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from stable_worldmodel.plot import save_panel_videos
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

import anygoal  # noqa: E402,F401
import anygoal.datasets as anygoal_datasets  # noqa: E402
import anygoal.wm as anygoal_wm  # noqa: E402
import anygoal.wm.toy_predictors as anygoal_toy_predictors  # noqa: E402
from anygoal.paths import data_root  # noqa: E402

# Historical checkpoints store Hydra targets under the pre-release package name.
sys.modules.setdefault("unstable_worldmodel", anygoal)
sys.modules.setdefault("unstable_worldmodel.datasets", anygoal_datasets)
sys.modules.setdefault("unstable_worldmodel.wm", anygoal_wm)
sys.modules.setdefault("unstable_worldmodel.wm.toy_predictors", anygoal_toy_predictors)
from anygoal.envs.cube_hop.env import (  # noqa: E402
    add_cube_evaluation_metrics,
    prepare_cube_evaluation,
)
from anygoal.envs.insert_peg.env import (  # noqa: E402
    add_insert_peg_evaluation_metrics,
)
from anygoal.resources import resource_path  # noqa: E402
from anygoal.wm.toy_predictors import (  # noqa: E402
    FixedMinMaxScaler,
    FixedStandardScaler,
    ToyWorldModelPolicy,
)
from anygoal._planning_artifacts import PlanningArtifactRecorder  # noqa: E402


def get_episodes_length(dataset, episodes) -> np.ndarray:
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_indices = dataset.get_col_data(column)
    step_indices = dataset.get_col_data("step_idx")
    return np.asarray(
        [np.max(step_indices[episode_indices == episode]) + 1 for episode in episodes]
    )


def get_dataset(cfg: DictConfig):
    dataset_name = Path(str(cfg.eval.dataset_name)).expanduser()
    dataset_path = (
        dataset_name
        if dataset_name.is_absolute()
        else data_root() / dataset_name
    )
    if not dataset_path.suffix:
        dataset_path = dataset_path.with_suffix(".h5")
    return swm.data.HDF5Dataset(
        path=dataset_path,
        keys_to_cache=cfg.dataset.keys_to_cache,
    )


def load_predictor(cfg: DictConfig):
    checkpoint = Path(cfg.model.checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Toy predictor weights not found: {checkpoint}")
    config_path = (
        Path(cfg.model.config_path).expanduser()
        if cfg.model.config_path
        else checkpoint.parent / "config.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    training_config = OmegaConf.load(config_path)
    model_config = OmegaConf.to_container(training_config.model, resolve=True)
    model_config.pop("name")
    model = hydra.utils.instantiate(model_config)
    if cfg.model.get("rollout_seed") is not None and hasattr(model, "rollout_seed"):
        model.rollout_seed = int(cfg.model.rollout_seed)
    state_dict = torch.load(
        checkpoint,
        map_location=cfg.device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(cfg.device).eval().requires_grad_(False)

    adapter = hydra.utils.instantiate(training_config.data.adapter)
    normalization = training_config.get("normalization")
    if normalization is None:
        normalization_path = resource_path(
            "train", *Path(str(training_config.data.normalization_path)).parts
        )
        normalization = OmegaConf.create(json.loads(normalization_path.read_text()))
    if normalization.state.get("type", "zscore") == "minmax":
        state_processor = FixedMinMaxScaler(
            normalization.state.min,
            normalization.state.max,
        )
    else:
        state_processor = FixedStandardScaler(
            normalization.state.mean,
            normalization.state.std,
        )
    action_processor = FixedStandardScaler(
        normalization.action.mean,
        normalization.action.std,
    )
    return model, adapter, state_processor, action_processor, training_config


def select_evaluation_starts(dataset, cfg: DictConfig):
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode_ids = np.unique(dataset.get_col_data(episode_column))
    episode_lengths = get_episodes_length(dataset, episode_ids)
    max_start = episode_lengths - int(cfg.eval.goal_offset_steps) - 1
    max_start_by_episode = dict(zip(episode_ids, max_start))
    row_max_start = np.asarray(
        [
            max_start_by_episode[episode]
            for episode in dataset.get_col_data(episode_column)
        ]
    )
    valid_indices = np.flatnonzero(dataset.get_col_data("step_idx") <= row_max_start)
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Only {len(valid_indices)} valid starts; need {cfg.eval.num_eval}."
        )
    if str(cfg.eval.get("start_selection", "random")) == "first_episodes":
        selected = np.asarray([
            np.flatnonzero(
                (dataset.get_col_data(episode_column) == episode)
                & (dataset.get_col_data("step_idx") == 0)
            )[0]
            for episode in np.sort(episode_ids)[: int(cfg.eval.num_eval)]
            if max_start_by_episode[episode] >= 0
        ], dtype=np.int64)
        if len(selected) < int(cfg.eval.num_eval):
            raise ValueError("Not enough eligible episodes for first_episodes selection.")
    else:
        generator = np.random.default_rng(cfg.seed)
        selected = np.sort(
            generator.choice(valid_indices, size=cfg.eval.num_eval, replace=False)
        )
    rows = dataset.get_row_data(selected)
    return rows[episode_column].tolist(), rows["step_idx"].tolist()


def save_gif(path: Path, frames: list[np.ndarray], duration_ms: int = 100) -> None:
    if not frames:
        raise ValueError(f"Cannot save an empty GIF: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        if images:
            image = image.quantize(palette=images[0], dither=Image.Dither.NONE)
        else:
            image = image.convert("P", palette=Image.Palette.ADAPTIVE)
        images.append(image)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_states(environment, states: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
    frames = []
    for state in states:
        if "qpos" in state and "qvel" in state:
            environment.set_state(state["qpos"].copy(), state["qvel"].copy())
        elif "state" in state:
            environment._set_state(state["state"].copy())
        else:
            raise KeyError(f"Unsupported simulator replay state: {sorted(state)}")
        frames.append(np.asarray(environment.render()).copy())
    return frames


def get_replay_keys(dataset) -> tuple[str, ...]:
    columns = set(dataset.column_names)
    if {"qpos", "qvel"} <= columns:
        return ("qpos", "qvel")
    if "state" in columns:
        return ("state",)
    raise KeyError("Video replay requires either qpos/qvel or state columns.")


def select_video_successes(
    metrics: dict,
    requested_criterion: str | None,
) -> tuple[np.ndarray, str]:
    aliases = {
        "criterion_3": "cube_hop_criterion_3",
        "original_cube": "original_cube_criterion",
    }
    requested = requested_criterion or (
        "original_cube" if "original_cube_criterion" in metrics else "termination"
    )
    requested = aliases.get(requested, requested)
    if requested == "termination":
        if "original_cube_criterion" in metrics:
            raise ValueError(
                "Cube videos must use original_cube or criterion_3 success."
            )
        return np.asarray(metrics["episode_successes"], dtype=bool), requested

    if requested not in metrics:
        available = ["termination", *aliases, *metrics]
        raise ValueError(
            f"Unknown video success criterion {requested_criterion!r}; "
            f"choose one of {sorted(set(available))}."
        )
    return np.asarray(metrics[requested]["episode_successes"], dtype=bool), requested


def video_criterion_directory(criterion: str) -> str:
    return {
        "cube_hop_criterion_3": "criterion_3",
        "original_cube_criterion": "original_cube",
    }.get(criterion, criterion)


def save_state_rollout_gifs(
    world,
    rollout_states: list[list[dict[str, np.ndarray]]],
    ground_truth_states: list[list[dict[str, np.ndarray]]],
    successes: np.ndarray,
    episodes: list[int],
    start_steps: list[int],
    output_directory: Path,
    seed: int,
) -> None:
    saved = defaultdict(int)
    for index, (rollout, ground_truth) in enumerate(
        zip(rollout_states, ground_truth_states)
    ):
        environment = world.envs.envs[index].unwrapped
        outcome = "success" if successes[index] else "fail"
        filename = (
            f"{outcome}_seed_{seed}_chunk_{index:03d}_episode_{episodes[index]}_"
            f"start_{start_steps[index]}.gif"
        )
        outcome = "success" if successes[index] else "fail"
        save_gif(
            output_directory / outcome / filename,
            render_states(environment, rollout),
        )
        save_gif(
            output_directory / "gt" / filename,
            render_states(environment, ground_truth),
        )
        saved[outcome] += 1
        saved["gt"] += 1
    print(
        "Saved state-rendered GIFs: "
        f"{saved['success']} success, {saved['fail']} fail, {saved['gt']} gt "
        f"under {output_directory}"
    )


def evaluate_with_state_rollouts(
    world,
    dataset,
    episodes: list[int],
    start_steps: list[int],
    cfg: DictConfig,
    gif_directory: Path,
    panel_video_directory: Path | None,
) -> dict:
    """Evaluate once while retaining raw states for renderer-based GIFs."""
    episode_indices = np.asarray(episodes)
    starts = np.asarray(start_steps)
    ends = starts + int(cfg.eval.goal_offset_steps) + 1
    replay_keys = get_replay_keys(dataset)
    ground_truth_chunks = dataset.load_chunk(episode_indices, starts, ends)
    ground_truth_states = [
        [
            {
                key: np.asarray(chunk[key][step]).copy()
                for key in replay_keys
            }
            for step in range(len(chunk[replay_keys[0]]))
        ]
        for chunk in ground_truth_chunks
    ]

    init_state, goal_state, dataset_videos = _extract_init_goal(
        dataset,
        episodes,
        start_steps,
        cfg.eval.goal_offset_steps,
    )
    world.reset(seed=init_state.get("seed"))
    callables = prepare_cube_evaluation(
        world,
        OmegaConf.to_container(cfg.eval.callables, resolve=True),
    )
    if callables:
        merged = {**init_state, **goal_state}
        for index in range(world.num_envs):
            environment_init = {key: value[index] for key, value in merged.items()}
            _apply_callables(
                world.envs.envs[index].unwrapped,
                callables,
                environment_init,
            )

    shape_prefix = world.infos["pixels"].shape[:2]
    for source in (init_state, goal_state):
        for key, value in source.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}

    successes = np.zeros(world.num_envs, dtype=bool)
    finished = np.zeros(world.num_envs, dtype=bool)
    rollout_states = [
        [
            {
                key: np.asarray(init_state[key][index]).copy()
                for key in replay_keys
            }
        ]
        for index in range(world.num_envs)
    ]
    panel_frames = defaultdict(list) if panel_video_directory else None

    def on_step(current_world) -> None:
        current_world.infos.update(deepcopy(goal_snapshot))
        for index in range(current_world.num_envs):
            if not finished[index]:
                replay_state = {}
                for key in replay_keys:
                    value = current_world.infos[key][index]
                    value = value[-1] if value.ndim > 1 else value
                    replay_state[key] = np.asarray(value).copy()
                rollout_states[index].append(replay_state)
            if panel_frames is not None:
                frame = current_world.infos["pixels"][index]
                frame = frame[-1] if frame.ndim > 3 else frame
                panel_frames[index].append(np.asarray(frame).copy())
        successes[:] |= current_world.terminateds
        finished[:] |= current_world.terminateds | current_world.truncateds

    world._run(
        max_steps=cfg.eval.eval_budget,
        mode="wait",
        on_step=on_step,
    )
    metrics = {
        "success_rate": float(successes.sum()) / world.num_envs * 100.0,
        "episode_successes": successes,
        "seeds": init_state.get("seed"),
    }
    metrics = add_cube_evaluation_metrics(world, metrics)
    metrics = add_insert_peg_evaluation_metrics(world, metrics)
    video_successes, success_criterion = select_video_successes(
        metrics,
        cfg.output.get("success_criterion"),
    )
    metrics["video_success_criterion"] = success_criterion
    metrics["video_episode_successes"] = video_successes.tolist()
    classified_gif_directory = gif_directory
    if "cube_hop_criterion_3" in metrics:
        classified_gif_directory /= video_criterion_directory(success_criterion)
    if panel_frames:
        save_panel_videos(
            panel_video_directory,
            {
                "agent": panel_frames,
                "dataset": dataset_videos,
                "goal": goal_state["goal"],
            },
        )
    save_state_rollout_gifs(
        world,
        rollout_states,
        ground_truth_states,
        video_successes,
        episodes,
        start_steps,
        classified_gif_directory,
        int(cfg.seed),
    )
    return metrics


@hydra.main(
    version_base=None,
    config_path="../../resources/plan/toy_predictors",
    config_name="pusht",
)
def run(cfg: DictConfig) -> None:
    if cfg.plan_config.horizon * cfg.plan_config.action_block > cfg.eval.eval_budget:
        raise ValueError(
            "Planning horizon times action_block must not exceed eval_budget."
        )

    model, adapter, state_processor, action_processor, training_config = load_predictor(
        cfg
    )
    if cfg.plan_config.action_block != model.action_frameskip:
        raise ValueError(
            "Planning action_block must match the predictor's training "
            f"frameskip ({model.action_frameskip})."
        )
    if cfg.plan_config.history_len != training_config.history_size:
        raise ValueError(
            "Planning history_len must match the predictor's training history_size."
        )

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg)
    episodes, start_steps = select_evaluation_starts(dataset, cfg)

    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = ToyWorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        adapter=adapter,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    world.set_policy(policy)

    output_directory = cfg.output.get("directory")
    results_directory = (
        Path(output_directory).expanduser()
        if output_directory
        else Path(cfg.model.checkpoint_path).expanduser().parent
        / "planning"
        / str(cfg.eval.dataset_name).removesuffix(".h5")
    )
    results_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    artifact_recorder = None
    if bool(cfg.output.get("save_artifacts", False)):
        artifact_recorder = PlanningArtifactRecorder(
            world, dataset, episodes, start_steps, cfg.eval.goal_offset_steps,
            results_directory / "artifacts", int(cfg.seed),
        )
        artifact_recorder.install()
    if cfg.output.save_video:
        dataset_name = Path(str(cfg.eval.dataset_name).removesuffix(".h5")).name
        predictor_name = str(training_config.model.name)
        gif_directory = (
            Path(cfg.output.gif_directory).expanduser() / predictor_name / dataset_name
        )
        metrics = evaluate_with_state_rollouts(
            world,
            dataset,
            episodes,
            start_steps,
            cfg,
            gif_directory,
            results_directory if cfg.output.video else None,
        )
    else:
        callables = prepare_cube_evaluation(
            world,
            OmegaConf.to_container(cfg.eval.callables, resolve=True),
        )
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=episodes,
            callables=callables,
            video=results_directory if cfg.output.video else None,
        )
        metrics = add_cube_evaluation_metrics(world, metrics)
        metrics = add_insert_peg_evaluation_metrics(world, metrics)
    if artifact_recorder is not None:
        artifact_recorder.save(metrics)
    elapsed = time.time() - start_time
    world.close()
    print(metrics)

    results_path = results_directory / cfg.output.filename
    with results_path.open("a") as output_file:
        output_file.write("\n==== CONFIG ====\n")
        output_file.write(OmegaConf.to_yaml(cfg))
        output_file.write("\n==== RESULTS ====\n")
        output_file.write(f"metrics: {metrics}\n")
        output_file.write(f"evaluation_time: {elapsed} seconds\n")


if __name__ == "__main__":
    if "--save-video" in sys.argv:
        sys.argv.remove("--save-video")
        sys.argv.append("output.save_video=true")
    run()
