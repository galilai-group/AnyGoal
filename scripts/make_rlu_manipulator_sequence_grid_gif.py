"""Render a state-based 2x5 GIF from random RL Unplugged Manipulator sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET = (
    Path.home()
    / ".stable_worldmodel/datasets/rlu_manipulator_insert_peg_fragments.h5"
)
CANVAS_SIZE = (1600, 640)
GRID_ROWS, GRID_COLUMNS = 2, 5
PANEL_WIDTH, PANEL_HEIGHT = CANVAS_SIZE[0] // GRID_COLUMNS, CANVAS_SIZE[1] // GRID_ROWS
X_RANGE = (-0.8, 0.8)
Z_RANGE = (-0.1, 0.9)


def point_to_pixel(point: np.ndarray, left: int, top: int) -> tuple[int, int]:
    x = left + 36 + (float(point[0]) - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (PANEL_WIDTH - 54)
    z = top + 28 + (Z_RANGE[1] - float(point[1])) / (Z_RANGE[1] - Z_RANGE[0]) * (PANEL_HEIGHT - 50)
    return round(x), round(z)


def draw_path(draw: ImageDraw.ImageDraw, points: np.ndarray, color: str, left: int, top: int) -> None:
    if len(points) < 2:
        return
    draw.line([point_to_pixel(point, left, top) for point in points], fill=color, width=3)


def render_frame(sequences: list[dict[str, np.ndarray | int]], step: int) -> Image.Image:
    image = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, sequence in enumerate(sequences):
        row, column = divmod(index, GRID_COLUMNS)
        left, top = column * PANEL_WIDTH, row * PANEL_HEIGHT
        right, bottom = left + PANEL_WIDTH - 1, top + PANEL_HEIGHT - 1
        draw.rectangle((left, top, right, bottom), outline="#b0b0b0", width=1)
        draw.text((left + 7, top + 7), f"sequence {sequence['index']}", fill="black", font=font)
        draw.text((left + 7, top + 20), f"step {step + 1}/1000", fill="#555555", font=font)
        draw.rectangle((left + 36, top + 28, right - 18, bottom - 22), outline="#d0d0d0", width=1)
        states = sequence["states"][: step + 1]
        draw_path(draw, states[:, 33:35], "#1f77b4", left, top)
        draw_path(draw, states[:, 29:31], "#ff7f0e", left, top)
        target = point_to_pixel(sequence["target"], left, top)
        draw.regular_polygon((target, 8), n_sides=5, rotation=0, fill="#2ca02c")
        for point, color in ((states[-1, 33:35], "#1f77b4"), (states[-1, 29:31], "#ff7f0e")):
            x, z = point_to_pixel(point, left, top)
            draw.ellipse((x - 4, z - 4, x + 4, z + 4), fill=color)
        draw.text((left + 7, bottom - 15), "blue=object  orange=hand  green=target", fill="#444444", font=font)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("reports/rlu_manipulator_insert_peg_random_10_sequences_2x5.gif"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--duration-ms", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.dataset.expanduser(), "r") as handle:
        lengths = handle["ep_len"][:]
        offsets = handle["ep_offset"][:]
        eligible = np.flatnonzero(lengths >= 1000)
        if len(eligible) < 10:
            raise ValueError("Need at least 10 sequences with 1,000 transitions")
        selected = np.random.default_rng(args.seed).choice(eligible, size=10, replace=False)
        sequences = []
        for index in selected:
            start = int(offsets[index])
            states = handle["observation"][start : start + 1000]
            sequences.append({"index": int(index), "states": states, "target": states[0, 40:42]})
    frames = [render_frame(sequences, step) for step in range(0, 1000, 10)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=args.duration_ms, loop=0, optimize=True)
    metadata = {"dataset": str(args.dataset.expanduser().resolve()), "seed": args.seed, "selected_sequence_indices": selected.tolist(), "frames": len(frames), "sampled_steps": list(range(0, 1000, 10)), "duration_ms": args.duration_ms}
    metadata_path = args.output.with_suffix(".json")
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    with metadata_path.open("x") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
