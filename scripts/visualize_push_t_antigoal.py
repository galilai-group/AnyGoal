"""Save one PushT Antigoal episode per corner as GIFs and a 2x2 grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image


ANTIGOAL_POSITIONS = {
    "top-left": (106.0, 106.0),
    "bottom-left": (106.0, 406.0),
    "top-right": (406.0, 106.0),
    "bottom-right": (406.0, 406.0),
}
DEFAULT_DATASET = Path(
    "pusht_antigoal_test.h5"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "push_t_antigoal_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    return parser.parse_args()


def save_gif(path: Path, frames: np.ndarray, duration_ms: int) -> None:
    images = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        if images:
            image = image.quantize(
                palette=images[0], dither=Image.Dither.NONE
            )
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


def main() -> None:
    args = parse_args()
    if args.frame_duration_ms <= 0:
        raise ValueError("frame-duration-ms must be positive.")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corner_names = tuple(ANTIGOAL_POSITIONS)
    corner_positions = np.asarray(tuple(ANTIGOAL_POSITIONS.values()))

    selected: dict[str, np.ndarray] = {}
    selected_indices: dict[str, int] = {}
    with h5py.File(args.dataset, "r") as dataset:
        required = {"pixels", "state", "ep_len", "ep_offset"}
        missing = required.difference(dataset)
        if missing:
            raise KeyError(f"missing dataset columns: {sorted(missing)}")
        episode_lengths = dataset["ep_len"][:].astype(np.int64)
        episode_offsets = dataset["ep_offset"][:].astype(np.int64)
        for episode_index, (offset, length) in enumerate(
            zip(episode_offsets, episode_lengths)
        ):
            terminal_block_position = dataset["state"][
                offset + length - 1, 2:4
            ]
            distances = np.linalg.norm(
                corner_positions - terminal_block_position[None], axis=1
            )
            corner_name = corner_names[int(np.argmin(distances))]
            if corner_name not in selected:
                selected[corner_name] = dataset["pixels"][
                    offset : offset + length
                ]
                selected_indices[corner_name] = episode_index

    missing_corners = set(corner_names).difference(selected)
    if missing_corners:
        raise ValueError(
            f"dataset has no terminal episode for: {sorted(missing_corners)}"
        )

    for corner_name, frames in selected.items():
        save_gif(
            args.output_dir / f"{corner_name}.gif",
            frames,
            args.frame_duration_ms,
        )

    grid_order = ("top-left", "top-right", "bottom-left", "bottom-right")
    longest_episode = max(len(selected[name]) for name in grid_order)
    frame_height, frame_width = selected[grid_order[0]].shape[1:3]
    grid_frames = np.empty(
        (longest_episode, 2 * frame_height, 2 * frame_width, 3),
        dtype=np.uint8,
    )
    for frame_index in range(longest_episode):
        for grid_index, corner_name in enumerate(grid_order):
            frames = selected[corner_name]
            source_index = min(frame_index, len(frames) - 1)
            row, column = divmod(grid_index, 2)
            grid_frames[
                frame_index,
                row * frame_height : (row + 1) * frame_height,
                column * frame_width : (column + 1) * frame_width,
            ] = frames[source_index]
    save_gif(
        args.output_dir / "all_corners.gif",
        grid_frames,
        args.frame_duration_ms,
    )
    Image.fromarray(grid_frames[-1]).save(
        args.output_dir / "all_corners_final.png"
    )
    print(f"Dataset: {args.dataset}")
    print(f"Selected episodes: {selected_indices}")
    print(f"Saved visualizations: {args.output_dir}")


if __name__ == "__main__":
    main()
