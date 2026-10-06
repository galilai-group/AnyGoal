"""Visualize trained TwoRoom model embeddings over a physical position grid."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.envs.tworoom_triangle import TwoRoomTriangleEnv
from anygoal.paths import checkpoint_root, source_root


LEWM_CHECKPOINT = checkpoint_root() / "lewm/tworoom/lewm_object.ckpt"
SMWM_RUN_DIRECTORY = checkpoint_root() / "sensorimotor/tworoom_inverse_lambda_0p1_seed0"
SMWM_SOURCE_ROOT = source_root() / "sensorimotor-world-model/planning"
IRASIM_RUN_DIRECTORY = checkpoint_root() / "irasim/tworoom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-x", type=int, default=25)
    parser.add_argument("--num-y", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("lewm", "smwm", "irasim"),
        default=("lewm", "smwm", "irasim"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "tworoom_latent_grid",
    )
    return parser.parse_args()


def position_grid(env: TwoRoomTriangleEnv, num_x: int, num_y: int):
    radius = float(env.variation_space["agent"]["radius"].value.item())
    lower = float(env.BORDER_SIZE) + radius
    upper = float(env.IMG_SIZE - env.BORDER_SIZE) - radius
    xs = np.linspace(lower, upper, num_x, dtype=np.float32)
    ys = np.linspace(lower, upper, num_y, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    states = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)

    half_wall = env.wall_thickness // 2
    wall_min = env.WALL_CENTER - half_wall - radius
    wall_max = env.WALL_CENTER + half_wall + radius
    in_wall_x = (states[:, 0] >= wall_min) & (states[:, 0] <= wall_max)

    in_door = np.zeros(len(states), dtype=bool)
    for center, half_size in zip(env.door_positions, env.door_sizes):
        safe_half_size = float(half_size) - radius
        if safe_half_size >= 0:
            in_door |= np.abs(states[:, 1] - float(center)) <= safe_half_size
    valid_mask = ~in_wall_x | in_door
    return states, valid_mask


def generate_frames(num_x: int, num_y: int):
    env = TwoRoomTriangleEnv(render_mode="rgb_array")
    env.reset(
        seed=0,
        options={
            "variation": [],
            "state": np.asarray([36.0, 188.0], dtype=np.float32),
            "target_state": np.asarray([188.0, 36.0], dtype=np.float32),
        },
    )
    states, valid_mask = position_grid(env, num_x, num_y)
    frames = []
    for state in states[valid_mask]:
        env.agent_position = torch.as_tensor(state, dtype=torch.float32)
        frame = env.render()
        if frame.shape != (env.IMG_SIZE, env.IMG_SIZE, 3):
            raise ValueError(f"Unexpected rendered frame shape: {frame.shape}")
        frames.append(frame)
    env.close()
    return states, valid_mask, np.stack(frames).astype(np.uint8)


def imagenet_pixels(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((pixels - mean) / std).to(device)


@torch.inference_mode()
def extract_lewm(
    frames: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    if not LEWM_CHECKPOINT.is_file():
        raise FileNotFoundError(LEWM_CHECKPOINT)
    model = torch.load(LEWM_CHECKPOINT, map_location="cpu", weights_only=False)
    model = model.to(device).eval().requires_grad_(False)
    latents = []
    for start in range(0, len(frames), batch_size):
        pixels = imagenet_pixels(frames[start : start + batch_size], device)
        encoded = model.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]
        latents.append(encoded.float().cpu().numpy())
    del model
    return np.concatenate(latents)


@torch.inference_mode()
def extract_smwm(
    frames: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    from scripts.planning_sensorimotor import load_model

    checkpoint = SMWM_RUN_DIRECTORY / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = load_model(SMWM_RUN_DIRECTORY, SMWM_SOURCE_ROOT, str(device))
    latents = []
    for start in range(0, len(frames), batch_size):
        pixels = torch.from_numpy(frames[start : start + batch_size])
        pixels = pixels.permute(0, 3, 1, 2).to(device)
        encoded = model.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]
        latents.append(encoded.float().cpu().numpy())
    del model
    return np.concatenate(latents)


@torch.inference_mode()
def extract_irasim(
    frames: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    from diffusers.models import AutoencoderKL

    checkpoint = IRASIM_RUN_DIRECTORY / "checkpoints" / "0300000.pt"
    config_path = IRASIM_RUN_DIRECTORY / "config.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = OmegaConf.load(config_path)
    if (
        str(config.h5_dataset_name) != "tworoom"
        or str(config.dataset) != "languagetable"
        or int(config.mask_frame_num) != 3
    ):
        raise ValueError("Unexpected TwoRoom IRASim training configuration")
    vae = AutoencoderKL.from_pretrained(
        str(config.vae_model_path), subfolder="vae", local_files_only=True
    ).to(device).eval().requires_grad_(False)
    video_size = tuple(int(value) for value in config.video_size)
    latents = []
    irasim_batch_size = min(batch_size, 16)
    for start in range(0, len(frames), irasim_batch_size):
        pixels = torch.from_numpy(frames[start : start + irasim_batch_size])
        pixels = pixels.permute(0, 3, 1, 2).float().div(255.0).to(device)
        if tuple(pixels.shape[-2:]) != video_size:
            pixels = F.interpolate(
                pixels,
                size=video_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pixels = pixels.mul(2.0).sub(1.0)
        posterior = vae.encode(pixels).latent_dist
        deterministic_latent = posterior.mean * vae.config.scaling_factor
        latents.append(deterministic_latent.flatten(1).float().cpu().numpy())
    del vae
    return np.concatenate(latents)


def state_colors(states: np.ndarray) -> np.ndarray:
    normalized = (states - states.min(axis=0)) / np.ptp(states, axis=0)
    x, y = normalized.T
    colors = np.column_stack((x * (1.0 - y), y, (1.0 - x) * (1.0 - y)))
    return 0.12 + 0.78 * colors


def save_figure(
    model_name: str,
    valid_states: np.ndarray,
    projection: np.ndarray,
    output_path: Path,
) -> None:
    colors = state_colors(valid_states)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(valid_states[:, 0], valid_states[:, 1], c=colors, s=24)
    axes[0].set(
        title="TwoRoom Physical State Grid",
        xlabel="Agent x",
        ylabel="Agent y",
        aspect="equal",
    )
    axes[0].invert_yaxis()
    axes[1].scatter(projection[:, 0], projection[:, 1], c=colors, s=24)
    axes[1].set(
        title=f"2D t-SNE Projection ({model_name})",
        xlabel="Projected Dim 1",
        ylabel="Projected Dim 2",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_x < 2 or args.num_y < 2:
        raise ValueError("Grid dimensions must both be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(5)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}", flush=True)

    states, valid_mask, frames = generate_frames(args.num_x, args.num_y)
    np.save(args.output_dir / "tworoom_grid_states.npy", states)
    np.save(args.output_dir / "tworoom_grid_valid_mask.npy", valid_mask)
    np.save(args.output_dir / "tworoom_grid_frames.npy", frames)
    print(
        f"Generated {len(frames)} valid frames from {len(states)} grid points",
        flush=True,
    )

    extractors = {
        "lewm": extract_lewm,
        "smwm": extract_smwm,
        "irasim": extract_irasim,
    }
    display_names = {"lewm": "LeWM", "smwm": "SMWM", "irasim": "IRASim"}
    valid_states = states[valid_mask]
    for model_name in args.models:
        latents = extractors[model_name](frames, device, args.batch_size)
        if latents.ndim != 2 or len(latents) != len(valid_states):
            raise ValueError(f"Invalid {model_name} latent shape: {latents.shape}")
        projection = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(latents) // 10)),
            learning_rate="auto",
            init="pca",
            random_state=args.seed,
        ).fit_transform(latents)
        np.save(args.output_dir / f"tworoom_latents_{model_name}.npy", latents)
        np.save(
            args.output_dir / f"tworoom_projection_{model_name}.npy", projection
        )
        save_figure(
            display_names[model_name],
            valid_states,
            projection,
            args.output_dir / f"tworoom_latent_grid_{model_name}.png",
        )
        print(
            f"Saved {model_name}: latents {latents.shape}, projection {projection.shape}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
