"""Render the first ten episodes of every catalogued dataset as individual GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import imageio.v2 as imageio


DATASETS = {
    "pusht": "pusht_expert_train.h5",
    "avoidt": "avoidt_test.h5",
    "stackt": "stackt_test.h5",
    "pusht-antigoal": "pusht_antigoal_test.h5",
    "cube": "cube_single_expert.h5",
    "cubehop": "cube_hop_test.h5",
    "cubediagonal": "cube_diagonal_test.h5",
    "insertpeg": "manipulator_insert_peg_weak_expert_train.h5",
    "hug_box": "avoid_peg_test.h5",
    "extractpeg": "extract_peg_test.h5",
    "tworoom": "tworoom.h5",
    "tworoom_route": "tworoom_route_test.h5",
}


def render_dataset(
    dataset_id: str,
    dataset_path: Path,
    output_root: Path,
    fps: int,
    video_format: str,
) -> None:
    destination = output_root / dataset_id
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    with h5py.File(dataset_path, "r") as source:
        for key in ("pixels", "ep_offset", "ep_len"):
            if key not in source:
                raise KeyError(f"{dataset_path} is missing {key!r}")
        pixels, offsets, lengths = source["pixels"], source["ep_offset"], source["ep_len"]
        if len(offsets) < 10:
            raise ValueError(f"{dataset_path} has only {len(offsets)} episodes")
        for episode_index in range(10):
            offset, length = int(offsets[episode_index]), int(lengths[episode_index])
            output = destination / f"episode_{episode_index:03d}.{video_format}"
            writer_options = (
                {"mode": "I", "duration": 1 / fps, "loop": 0}
                if video_format == "gif"
                else {
                    "fps": fps,
                    "codec": "libx264",
                    "pixelformat": "yuv420p",
                    "quality": 8,
                    "ffmpeg_params": ["-movflags", "+faststart"],
                }
            )
            with imageio.get_writer(output, **writer_options) as writer:
                for frame_index in range(offset, offset + length):
                    writer.append_data(pixels[frame_index])
            manifest.append(
                {
                    "episode": episode_index,
                    "start_frame": offset,
                    "num_frames": length,
                    "video": str(output),
                }
            )
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset": str(dataset_path),
                "fps": fps,
                "format": video_format,
                "episodes": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(manifest)} GIFs to {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", choices=DATASETS)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/grid/klindt/home/nam/.stable_worldmodel/datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper/results/dataset_first_10_episodes"),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    args = parser.parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    dataset_ids = [args.dataset_id] if args.dataset_id else list(DATASETS)
    for dataset_id in dataset_ids:
        dataset_path = args.data_root / DATASETS[dataset_id]
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        render_dataset(
            dataset_id,
            dataset_path,
            args.output_root,
            args.fps,
            args.format,
        )


if __name__ == "__main__":
    main()
