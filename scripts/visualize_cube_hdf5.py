"""Render selected Cube HDF5 episodes as a labeled 2-by-5 GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  Register the Blosc filter.
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.input, "r") as dataset:
        episode_count = len(dataset["ep_len"])
        if episode_count < 10:
            raise ValueError("Input must contain at least 10 episodes")
        episode_indices = np.linspace(0, episode_count - 1, 10).round().astype(int)
        episode_frames = []
        for episode_index in episode_indices:
            offset = int(dataset["ep_offset"][episode_index])
            length = int(dataset["ep_len"][episode_index])
            episode_frames.append(dataset["pixels"][offset : offset + length])

    frame_height, frame_width = episode_frames[0].shape[1:3]
    images = []
    for step in range(max(map(len, episode_frames))):
        canvas = Image.new("RGB", (5 * frame_width, 2 * frame_height))
        draw = ImageDraw.Draw(canvas)
        for panel, (episode_index, frames) in enumerate(
            zip(episode_indices, episode_frames)
        ):
            row, column = divmod(panel, 5)
            x, y = column * frame_width, row * frame_height
            frame = Image.fromarray(frames[min(step, len(frames) - 1)]).convert(
                "RGB"
            )
            canvas.paste(frame, (x, y))
            draw.rectangle((x, y, x + 88, y + 20), fill="black")
            draw.text((x + 5, y + 4), f"episode {episode_index}", fill="white")
        images.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.output,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"Selected episodes: {episode_indices.tolist()}")
    print(f"Saved {len(images)} frames to {args.output}")


if __name__ == "__main__":
    main()
