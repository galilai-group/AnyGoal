"""Save five-step (25 environment step) open-loop rollout GIFs.

Pixel-generative models are saved as RGB. Models without an image decoder are
saved as normalized representation heatmaps; this avoids pretending that a
latent prediction is an RGB reconstruction. A matching RGB ground-truth GIF is
always written beside every model rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "scripts")]

from eval import build_adapter, resolve_path  # noqa: E402
from anygoal.evaluation.prediction import PredictionProtocol, _gather_batch  # noqa: E402
from anygoal.evaluation.prediction_models import (  # noqa: E402
    DriftWorldPredictionAdapter,
    IRASimPredictionAdapter,
    LatentARPredictionAdapter,
    PreJEPAPredictionAdapter,
    TrueStatePredictionAdapter,
    standardize_action_blocks,
    standardize_tensor,
)

ALIASES = {
    "toy-adaln": "adaln_predictor",
    "toy-action-duplicate": "action_duplicated_predictor",
    "toy-driftworld": "toy_driftworld",
}
ENVIRONMENT_ALIASES = {
    "cube_hop": "cubehop",
}


def uint8_rgb(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.min() < 0:
            array = (array + 1.0) / 2.0
        array = array * 255.0 if array.max() <= 1.0 else array
    return np.clip(array, 0, 255).astype(np.uint8)


def representation_frame(value: torch.Tensor, label: str) -> np.ndarray:
    flat = value.detach().float().cpu().reshape(-1).numpy()
    side = int(np.ceil(np.sqrt(len(flat))))
    padded = np.full(side * side, np.nan, dtype=np.float32)
    padded[: len(flat)] = flat
    finite = padded[np.isfinite(padded)]
    low, high = np.percentile(finite, (2, 98)) if len(finite) else (0.0, 1.0)
    normalized = np.nan_to_num((padded - low) / max(high - low, 1e-8))
    red = normalized
    blue = 1.0 - normalized
    green = 1.0 - np.abs(normalized - 0.5) * 2.0
    rgb = (np.stack((red, green, blue), axis=-1).reshape(side, side, 3) * 255).astype(np.uint8)
    image = Image.fromarray(rgb).resize((224, 224), Image.Resampling.NEAREST)
    ImageDraw.Draw(image).rectangle((0, 0, 224, 18), fill="black")
    ImageDraw.Draw(image).text((4, 3), label, fill="white")
    return np.asarray(image)


def save_gif(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(uint8_rgb(frame)) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=250, loop=0, disposal=2)


@torch.inference_mode()
def rollout(adapter, observations, actions, seed: int, steps: int) -> tuple[list[np.ndarray], str]:
    history_length = adapter.history_length
    if isinstance(adapter, LatentARPredictionAdapter):
        pixels = adapter.pixel_preprocessor(observations["pixels"].to(adapter.device))
        embeddings = adapter.model.encode({"pixels": pixels})["emb"]
        normalized_actions = standardize_action_blocks(actions.to(adapter.device), adapter.action_processor)
        history = embeddings[:, :history_length]
        frames = []
        for step in range(steps):
            action_embeddings = adapter.model.action_encoder(normalized_actions[:, step:step + history_length])
            prediction = adapter.model.predict(history, action_embeddings)[:, -1]
            frames.append(representation_frame(prediction[0], "latent prediction (no RGB decoder)"))
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return frames, "latent_heatmap"

    if isinstance(adapter, PreJEPAPredictionAdapter):
        from anygoal.evaluation.prediction_models import imagenet_preprocess
        pixels = imagenet_preprocess(observations["pixels"].to(adapter.device), adapter.image_size)
        normalized_actions = standardize_action_blocks(actions.to(adapter.device), adapter.action_processor)
        padded = torch.cat((normalized_actions, normalized_actions.new_zeros(1, 1, normalized_actions.shape[-1])), dim=1)
        info = {"pixels": pixels, "action": padded}
        for key in adapter.predicted_keys:
            info[key] = standardize_tensor(observations[key].to(adapter.device), adapter.processors[key])
        encoded = adapter.model.encode(info)
        history = encoded["emb"][:, :history_length]
        frames = []
        for step in range(steps):
            prediction = adapter.model.predict(history)[:, -1:]
            frames.append(representation_frame(prediction[0], "latent prediction (no RGB decoder)"))
            if step + 1 < steps:
                next_action = normalized_actions[:, history_length + step:history_length + step + 1]
                prediction = adapter.model.replace_action_in_embedding(prediction[:, None], next_action[:, None])[:, 0]
            history = torch.cat((history[:, 1:], prediction), dim=1)
        return frames, "latent_heatmap"

    if isinstance(adapter, TrueStatePredictionAdapter):
        states = adapter._encode_states(observations)
        normalized_actions = standardize_action_blocks(actions.to(adapter.device), adapter.action_processor)
        history = states[:, :history_length]
        generator = torch.Generator(device=adapter.device).manual_seed(seed)
        frames = []
        for step in range(steps):
            context = normalized_actions[:, step:step + history_length]
            prediction = (adapter.model.predict_next_state(history, context, generator=generator)
                          if hasattr(adapter.model, "predict_next_state") else adapter.model(history, context))
            if prediction.ndim == 3:
                prediction = prediction[..., 0]
            frames.append(representation_frame(prediction[0], "predicted normalized state"))
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return frames, "state_heatmap"

    pixels = adapter.model.preprocess_pixels(observations["pixels"].to(adapter.device))
    normalized_actions = standardize_action_blocks(actions.to(adapter.device), adapter.action_processor)
    frames = []
    if isinstance(adapter, DriftWorldPredictionAdapter):
        history = pixels[:, :history_length]
        for step in range(steps):
            candidate = normalized_actions[:, None, step + history_length - 1:step + history_length]
            info = {}
            if "proprio" in observations:
                info["proprio"] = observations["proprio"][:, step + history_length - 1].to(adapter.device)[:, None]
            prediction = adapter.model._rollout_once(history, adapter.model.prepare_model_actions(info, candidate), seed + step)[:, 0].clamp(-1, 1)
            frames.append(uint8_rgb(prediction[0]))
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
    elif isinstance(adapter, IRASimPredictionAdapter):
        encoded = adapter.model._encode(pixels[:, :history_length].flatten(0, 1), seed)
        history = encoded.reshape(1, history_length, *encoded.shape[1:])
        for step in range(steps):
            history_actions = adapter.model.prepare_model_actions(normalized_actions[:, step:step + history_length - 1])[:, None]
            future_actions = adapter.model.prepare_model_actions(normalized_actions[:, step + history_length - 1:step + history_length])[:, None]
            prediction = adapter.model._rollout_once(history, history_actions, future_actions, seed + step)[:, 0]
            frames.append(uint8_rgb(adapter.model._decode(prediction).clamp(-1, 1)[0]))
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
    return frames, "rgb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "paper/results/model_rollouts")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["LEWM_SOURCE_ROOT"] = str(REPO_ROOT / "worldmodels/lewm")
    os.environ["IRASIM_SCHEDULER_PATH"] = str(REPO_ROOT / "worldmodels/IRASim/pretrained_models/scheduler")
    cfg = OmegaConf.load(REPO_ROOT / "configs/prediction_eval.yaml")
    environment = ENVIRONMENT_ALIASES.get(args.environment, args.environment)
    model_name = ALIASES.get(args.model, args.model)
    dataset_cfg = cfg.datasets[environment]
    model_cfg = cfg.models[model_name][str(dataset_cfg.base_environment)]
    if "source_root" in cfg.models[model_name]:
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.source_root = cfg.models[model_name].source_root
    dataset = swm.data.HDF5Dataset(path=resolve_path(dataset_cfg.path), keys_to_cache=["action"])
    output = args.output_root / args.environment / args.model
    if (output / "manifest.json").is_file():
        print(f"already complete: {output}")
        return
    adapter, checkpoint, metadata = build_adapter(model_name, model_cfg, dataset, device=torch.device(args.device), checkpoint_override=None, training_config_override=None, planning_config_override=None, run_directory_override=None, history_length=3)
    protocol = PredictionProtocol(history_length=3, frame_skip=5, horizons=(25,), num_episodes=5, batch_size=1)
    manifest = []
    for episode_index in range(5):
        episode = dataset.load_episode(episode_index)
        available_steps = min(5, (int(dataset.lengths[episode_index]) - 16) // 5 + 1)
        if available_steps < 1:
            continue
        observations, actions = _gather_batch(episode, torch.tensor([0]), protocol, adapter.observation_keys, available_steps)
        if model_name == "sensorimotor":
            observations["pixels"][:, :3] = observations["pixels"][:, 2:3]
        frames, visualization = rollout(adapter, observations, actions, episode_index, available_steps)
        stem = f"episode_{episode_index:03d}_target_frame_016_steps_{len(frames) * 5:02d}"
        save_gif(output / f"{stem}_{visualization}.gif", frames)
        gt = [episode["pixels"][15 + 5 * step] for step in range(len(frames))]
        save_gif(output / f"{stem}_gt.gif", gt)
        manifest.append({"episode": episode_index, "first_target_frame_1_based": 16, "context_frames_1_based": [11] if model_name == "sensorimotor" else [1, 6, 11], "environment_steps": len(frames) * 5, "visualization": visualization})
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"model": args.model, "environment": args.environment, "config_environment": environment, "checkpoint": str(checkpoint), "model_metadata": metadata, "episodes": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
