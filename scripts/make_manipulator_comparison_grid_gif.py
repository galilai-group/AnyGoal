"""Render two random insert/hug/extract rows into one comparison GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw


DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
DATASETS = (
    ("Insert Peg", DATASET_DIR / "manipulator_insert_peg_weak_expert_train.h5"),
    ("Hug Box", DATASET_DIR / "avoid_peg_test.h5"),
    ("Extract Peg", DATASET_DIR / "extract_peg_test.h5"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--frame-duration-ms", type=int, default=60)
    return parser.parse_args()


def load_episode(dataset: h5py.File, index: int, stride: int) -> np.ndarray:
    offset = int(dataset["ep_offset"][index])
    length = int(dataset["ep_len"][index])
    return np.asarray(dataset["pixels"][offset : offset + length : stride], dtype=np.uint8)


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1 or args.frame_duration_ms < 1:
        raise ValueError("frame stride and duration must be positive")
    rng = np.random.default_rng(args.seed)
    rows: list[list[np.ndarray]] = [[], []]
    selections: dict[str, list[int]] = {}
    for label, path in DATASETS:
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as dataset:
            valid = np.flatnonzero(np.asarray(dataset["ep_len"][:]) > 0)
            selected = rng.choice(valid, size=2, replace=False)
            selections[label] = [int(index) for index in selected]
            for row, episode_index in enumerate(selected):
                rows[row].append(load_episode(dataset, int(episode_index), args.frame_stride))

    height, width = rows[0][0][0].shape[:2]
    title_height = 22
    max_length = max(len(episode) for row in rows for episode in row)
    palette: Image.Image | None = None
    frames: list[Image.Image] = []
    for step in range(max_length):
        canvas = np.zeros((2 * height + title_height, 3 * width, 3), dtype=np.uint8)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for column, (label, _) in enumerate(DATASETS):
            draw.text((column * width + 4, 4), label, fill=(255, 255, 255))
        for row_index, row in enumerate(rows):
            for column, episode in enumerate(row):
                source = episode[min(step, len(episode) - 1)]
                y_start = title_height + row_index * height
                x_start = column * width
                image.paste(Image.fromarray(source), (x_start, y_start))
        if palette is None:
            palette = image.convert("P", palette=Image.Palette.ADAPTIVE)
            frames.append(palette)
        else:
            frames.append(image.quantize(palette=palette, dither=Image.Dither.NONE))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    metadata = {
        "datasets": {label: str(path) for label, path in DATASETS},
        "selected_episode_indices": selections,
        "frame_stride": args.frame_stride,
        "frame_duration_ms": args.frame_duration_ms,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
