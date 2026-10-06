"""Render paired GT/regressed AvoidT GIFs from cached frozen-model latents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from anygoal.paths import data_root
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.scripts.avoidt_latent_regressors import ProbeMLP
from anygoal.envs.avoid_t import FIXED_GOAL_T_POSE


DEFAULT_DATASET = data_root() / "avoidt_test.h5"
DEFAULT_PROBE_DIRECTORY = REPO_ROOT / "paper/results/avoidt_latent_regressors"
DEFAULT_OUTPUT_DIRECTORY = REPO_ROOT / "paper/results/avoidt_regressed_episode_gifs"
MODELS = ("lewm", "smwm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--probe-directory", type=Path, default=DEFAULT_PROBE_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-ms", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.num_episodes < 1 or args.duration_ms < 1:
        raise ValueError("num-episodes and duration-ms must be positive")
    required = []
    for model in MODELS:
        required.extend(
            (
                args.probe_directory / f"avoidt_latents_{model}.npy",
                args.probe_directory / f"probe_{model}_agent_xy.pt",
                args.probe_directory / f"probe_{model}_block_xy.pt",
                args.probe_directory / f"probe_{model}_block_rotation.pt",
            )
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing probe artifacts: {missing}")


def load_probe(path: Path, device: str) -> tuple[ProbeMLP, np.ndarray, np.ndarray]:
    # These are trusted, locally generated probe artifacts and contain NumPy
    # target-normalization arrays, which the restricted loader rejects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    probe = ProbeMLP(int(checkpoint["latent_dim"]))
    probe.load_state_dict(checkpoint["state_dict"])
    probe = probe.to(device).eval().requires_grad_(False)
    return (
        probe,
        np.asarray(checkpoint["target_mean"], dtype=np.float32),
        np.asarray(checkpoint["target_std"], dtype=np.float32),
    )


@torch.inference_mode()
def regress_coordinates(
    latent: np.ndarray,
    probe_path: Path,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    probe, mean, std = load_probe(probe_path, device)
    predictions = []
    for start in range(0, len(latent), batch_size):
        value = probe(
            torch.from_numpy(np.array(latent[start : start + batch_size], copy=True))
            .float()
            .to(device)
        )
        predictions.append(value.cpu().numpy() * std + mean)
    return np.concatenate(predictions)


def regress_rotation(latent: np.ndarray, probe_path: Path, device: str) -> np.ndarray:
    features = regress_coordinates(latent, probe_path, device)
    return np.mod(np.arctan2(features[:, 0], features[:, 1]), 2.0 * np.pi)


def make_renderer(initial_state: np.ndarray):
    from stable_worldmodel.envs.pusht.env import PushT

    environment = PushT(resolution=224, render_mode="rgb_array", with_target=True)
    goal_state = initial_state.copy()
    environment.reset(
        seed=0,
        options={
            "variation": [],
            "variation_values": {
                "goal.position": FIXED_GOAL_T_POSE[:2],
                "goal.angle": float(FIXED_GOAL_T_POSE[2]),
            },
            "state": initial_state,
            "goal_state": goal_state,
        },
    )
    return environment


def add_panel_headers(
    ground_truth: np.ndarray,
    regressed: np.ndarray,
    model: str,
    episode: int,
) -> Image.Image:
    gt_image = Image.fromarray(np.asarray(ground_truth, dtype=np.uint8))
    regressed_image = Image.fromarray(np.asarray(regressed, dtype=np.uint8))
    width, height = gt_image.size
    header_height = 28
    canvas = Image.new("RGB", (2 * width, height + header_height), "white")
    canvas.paste(gt_image, (0, header_height))
    canvas.paste(regressed_image, (width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"GT | episode {episode}", fill="black")
    draw.text((width + 8, 7), f"{model.upper()} regressed", fill="black")
    return canvas


def save_pair_gif(
    path: Path,
    ground_truth_frames: np.ndarray,
    regressed_frames: list[np.ndarray],
    model: str,
    episode: int,
    duration_ms: int,
) -> None:
    frames = [
        add_panel_headers(gt, prediction, model, episode)
        for gt, prediction in zip(ground_truth_frames, regressed_frames)
    ]
    palette_frames = []
    for frame in frames:
        if palette_frames:
            palette_frames.append(
                frame.quantize(palette=palette_frames[0], dither=Image.Dither.NONE)
            )
        else:
            palette_frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    validate(args)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as dataset:
        episode_count = len(dataset["ep_len"])
        if args.num_episodes > episode_count:
            raise ValueError(
                f"Requested {args.num_episodes} episodes from {episode_count}"
            )
        generator = np.random.default_rng(args.seed)
        selected_episodes = np.sort(
            generator.choice(episode_count, args.num_episodes, replace=False)
        )
        offsets = dataset["ep_offset"][:].astype(np.int64)
        lengths = dataset["ep_len"][:].astype(np.int64)

        selection_metadata = {
            "dataset": str(args.dataset.resolve()),
            "seed": args.seed,
            "episodes": selected_episodes.tolist(),
            "lengths": [int(lengths[index]) for index in selected_episodes],
            "panels": ["dataset ground truth", "probe-regressed rendering"],
            "regressed_fields": ["agent_x", "agent_y", "block_x", "block_y", "block_angle"],
            "ground_truth_fields": ["target_pose"],
            "target_pose": FIXED_GOAL_T_POSE.tolist(),
        }

        for model in MODELS:
            latent = np.load(
                args.probe_directory / f"avoidt_latents_{model}.npy",
                mmap_mode="r",
            )
            if len(latent) != dataset["state"].shape[0]:
                raise ValueError(
                    f"{model} latent rows ({len(latent)}) do not match dataset "
                    f"rows ({dataset['state'].shape[0]})"
                )
            agent_prediction = regress_coordinates(
                latent,
                args.probe_directory / f"probe_{model}_agent_xy.pt",
                args.device,
            )
            block_prediction = regress_coordinates(
                latent,
                args.probe_directory / f"probe_{model}_block_xy.pt",
                args.device,
            )
            rotation_prediction = regress_rotation(
                latent,
                args.probe_directory / f"probe_{model}_block_rotation.pt",
                args.device,
            )

            for episode in selected_episodes:
                start = int(offsets[episode])
                end = start + int(lengths[episode])
                true_states = dataset["state"][start:end].astype(np.float64)
                ground_truth_frames = dataset["pixels"][start:end]
                renderer = make_renderer(true_states[0])
                regressed_frames = []
                try:
                    for local_index, true_state in enumerate(true_states):
                        row = start + local_index
                        regressed_state = true_state.copy()
                        regressed_state[:2] = agent_prediction[row]
                        regressed_state[2:4] = block_prediction[row]
                        regressed_state[4] = rotation_prediction[row]
                        renderer._set_state(regressed_state)
                        regressed_frames.append(np.asarray(renderer.render()).copy())
                finally:
                    renderer.close()
                output_path = (
                    args.output_directory
                    / f"avoidt_{model}_episode_{int(episode):03d}_gt_vs_regressed.gif"
                )
                save_pair_gif(
                    output_path,
                    ground_truth_frames,
                    regressed_frames,
                    model,
                    int(episode),
                    args.duration_ms,
                )
                print(f"Saved {output_path} ({len(regressed_frames)} frames)", flush=True)

    (args.output_directory / "selected_episodes.json").write_text(
        json.dumps(selection_metadata, indent=2)
    )


if __name__ == "__main__":
    main()
