"""Render random episode grid GIFs from the datasets listed in datasets.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
DATASETS = {
    "pusht": DATASET_DIR / "pusht_expert_train.h5",
    "avoidt": DATASET_DIR / "avoidt_test.h5",
    "push_t_antigoal_v3": DATASET_DIR / "pusht_antigoal_test.h5",
    "stackt_v2": DATASET_DIR / "stackt_test.h5",
    "cube": DATASET_DIR / "cube_single_expert.h5",
    "cube_hop": DATASET_DIR / "cube_hop_test.h5",
    "cube_diagonal": DATASET_DIR / "cube_diagonal_test.h5",
    "manipulator_insert_peg": (
        DATASET_DIR / "manipulator_insert_peg_weak_expert_train.h5"
    ),
    "manipulator_hug_box": DATASET_DIR / "avoid_peg_test.h5",
    "manipulator_extract_peg": DATASET_DIR / "extract_peg_test.h5",
    "tworoom": DATASET_DIR / "tworoom.h5",
    "tworoom_route": DATASET_DIR / "tworoom_route_test.h5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="Optional dataset key to render. If omitted, renders all supported datasets.",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-duration-ms", type=int, default=80)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT,
        help="Directory in which to save the rendered GIFs.",
    )
    return parser.parse_args()


def resolve_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"expected a 3D image frame, got {frame.shape}")
    if frame.shape[-1] not in (1, 3, 4) and frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unsupported channel layout: {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        finite_max = float(np.nanmax(frame)) if frame.size else 0.0
        scale = 255.0 if finite_max <= 1.0 else 1.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0) * scale
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame[..., :3]


def load_episode_frames(dataset: h5py.File, episode_index: int) -> np.ndarray:
    offset = int(dataset["ep_offset"][episode_index])
    length = int(dataset["ep_len"][episode_index])
    if length <= 0:
        raise ValueError(f"episode {episode_index} is empty")
    return dataset["pixels"][offset : offset + length]


def save_grid_gif(
    output_path: Path,
    episode_frames: list[np.ndarray],
    grid_rows: int,
    grid_columns: int,
    frame_duration_ms: int,
) -> None:
    first_frame = resolve_frame(episode_frames[0][0])
    frame_height, frame_width = first_frame.shape[:2]
    max_length = max(len(frames) for frames in episode_frames)
    gif_frames: list[Image.Image] = []

    for step in range(max_length):
        canvas = np.zeros(
            (grid_rows * frame_height, grid_columns * frame_width, 3),
            dtype=np.uint8,
        )
        for grid_index, frames in enumerate(episode_frames):
            source_index = min(step, len(frames) - 1)
            frame = resolve_frame(frames[source_index])
            if frame.shape[:2] != (frame_height, frame_width):
                raise ValueError("all frames must have the same resolution")
            row, column = divmod(grid_index, grid_columns)
            y_start = row * frame_height
            x_start = column * frame_width
            canvas[
                y_start : y_start + frame_height,
                x_start : x_start + frame_width,
            ] = frame
        image = Image.fromarray(canvas)
        if gif_frames:
            gif_frames.append(
                image.quantize(palette=gif_frames[0], dither=Image.Dither.NONE)
            )
        else:
            gif_frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    if args.frame_duration_ms <= 0:
        raise ValueError("frame-duration-ms must be positive")
    if args.num_episodes <= 0:
        raise ValueError("num-episodes must be positive")
    if args.grid_rows <= 0 or args.grid_columns <= 0:
        raise ValueError("grid-rows and grid-columns must be positive")
    if args.grid_rows * args.grid_columns != args.num_episodes:
        raise ValueError("grid size must exactly match num-episodes")

    rng = np.random.default_rng(args.seed)
    selected_datasets = (
        {args.dataset: DATASETS[args.dataset]}
        if args.dataset is not None
        else DATASETS
    )
    for dataset_name, dataset_path in selected_datasets.items():
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        output_path = args.output_directory / (
            f"{dataset_name}_random_{args.num_episodes}_episodes_"
            f"{args.grid_rows}x{args.grid_columns}.gif"
        )
        with h5py.File(dataset_path, "r") as dataset:
            required = {"pixels", "ep_len", "ep_offset"}
            missing = required.difference(dataset)
            if missing:
                raise KeyError(f"missing dataset columns: {sorted(missing)}")
            episode_lengths = dataset["ep_len"][:].astype(np.int64)
            valid_episodes = np.flatnonzero(episode_lengths > 0)
            if len(valid_episodes) < args.num_episodes:
                raise ValueError(
                    f"dataset {dataset_name} has only {len(valid_episodes)} non-empty episodes"
                )
            selected = rng.choice(valid_episodes, size=args.num_episodes, replace=False)
            episode_frames = [load_episode_frames(dataset, int(index)) for index in selected]
        save_grid_gif(
            output_path,
            episode_frames,
            args.grid_rows,
            args.grid_columns,
            args.frame_duration_ms,
        )
        print(f"Dataset: {dataset_name}")
        print(f"Selected episode indices: {selected.tolist()}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
