"""Render ten randomly sampled dataset episodes as a 2-by-5 GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image

from anygoal.registry import get_registry


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
NUM_EPISODES = 10
GRID_ROWS = 2
GRID_COLUMNS = 5


def task_slug(value: str) -> str:
    slug = value.strip().lower().replace("-", "_")
    if not slug or not all(character.isalnum() or character == "_" for character in slug):
        raise argparse.ArgumentTypeError(
            "task must contain only letters, digits, underscores, or hyphens"
        )
    return slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        type=task_slug,
        help="Dataset ID or alias, resolved through the AnyGoal catalog.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional dataset override for tasks that use another filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional GIF path. Defaults to paper/results/<task>_episodes_grid.gif.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-duration-ms", type=int, default=80)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset = get_registry().dataset(args.task)
    dataset_path = (
        DATASET_DIR / dataset.filename
        if args.dataset is None
        else args.dataset.expanduser()
    ).resolve()
    output_path = (
        RESULTS_DIR / f"{args.task}_episodes_grid.gif"
        if args.output is None
        else args.output.expanduser()
    ).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output must end in .gif")
    if args.frame_duration_ms <= 0:
        raise ValueError("frame-duration-ms must be positive")
    return dataset_path, output_path


def rgb_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"pixels must contain 3D frames, got {frame.shape}")
    if frame.shape[-1] not in (1, 3, 4) and frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unsupported pixel channel shape: {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        finite_max = float(np.nanmax(frame)) if frame.size else 0.0
        scale = 255.0 if finite_max <= 1.0 else 1.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0) * scale
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame[..., :3]


def main() -> None:
    args = parse_args()
    dataset_path, output_path = resolve_paths(args)
    rng = np.random.default_rng(args.seed)

    with h5py.File(dataset_path, "r") as dataset:
        required = {"pixels", "ep_len", "ep_offset"}
        missing = required.difference(dataset)
        if missing:
            raise KeyError(f"missing dataset columns: {sorted(missing)}")
        episode_lengths = dataset["ep_len"][:].astype(np.int64)
        episode_offsets = dataset["ep_offset"][:].astype(np.int64)
        valid_episodes = np.flatnonzero(episode_lengths > 0)
        if len(valid_episodes) < NUM_EPISODES:
            raise ValueError(
                f"dataset has only {len(valid_episodes)} non-empty episodes; "
                f"need {NUM_EPISODES}"
            )
        selected = rng.choice(valid_episodes, size=NUM_EPISODES, replace=False)
        selected_lengths = episode_lengths[selected]
        sampled_pixels = [
            dataset["pixels"][
                episode_offsets[episode_index] :
                episode_offsets[episode_index] + episode_lengths[episode_index]
            ]
            for episode_index in selected
        ]
        first_frame = rgb_frame(sampled_pixels[0][0])
        frame_height, frame_width = first_frame.shape[:2]
        longest_episode = int(selected_lengths.max())
        gif_frames: list[Image.Image] = []

        for frame_index in range(longest_episode):
            canvas = np.zeros(
                (GRID_ROWS * frame_height, GRID_COLUMNS * frame_width, 3),
                dtype=np.uint8,
            )
            for grid_index, episode_frames in enumerate(sampled_pixels):
                source_step = min(frame_index, len(episode_frames) - 1)
                frame = rgb_frame(episode_frames[source_step])
                if frame.shape[:2] != (frame_height, frame_width):
                    raise ValueError("all sampled frames must have the same resolution")
                row, column = divmod(grid_index, GRID_COLUMNS)
                y_start = row * frame_height
                x_start = column * frame_width
                canvas[
                    y_start : y_start + frame_height,
                    x_start : x_start + frame_width,
                ] = frame
            rgb_image = Image.fromarray(canvas)
            if gif_frames:
                gif_frames.append(
                    rgb_image.quantize(
                        palette=gif_frames[0], dither=Image.Dither.NONE
                    )
                )
            else:
                gif_frames.append(
                    rgb_image.convert("P", palette=Image.Palette.ADAPTIVE)
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=args.frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"Task: {args.task}")
    print(f"Dataset: {dataset_path}")
    print(f"Sampled episode indices: {selected.tolist()}")
    print(f"Sampled episode lengths: {selected_lengths.tolist()}")
    print(f"GIF frames: {longest_episode}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
