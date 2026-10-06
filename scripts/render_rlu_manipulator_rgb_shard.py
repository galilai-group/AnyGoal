"""Render one GPU shard of all-frame RL Unplugged Manipulator RGB data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import hdf5plugin
import numpy as np
from dm_control import suite

from make_rlu_manipulator_rgb_grid_gif import (
    observable_state,
    restore_state,
    state_restoration_error,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--minimum-length", type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index is outside [0, num-shards)")
    output_path = args.output_directory / f"pixels_shard_{args.shard_index:02d}.h5"
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.monotonic()
    with h5py.File(args.source.expanduser(), "r", swmr=True) as source:
        all_lengths = source["ep_len"][:].astype(np.int64)
        all_offsets = source["ep_offset"][:].astype(np.int64)
        eligible = np.flatnonzero(all_lengths >= args.minimum_length)
        episode_indices = np.array_split(eligible, args.num_shards)[args.shard_index]
        lengths = all_lengths[episode_indices]
        total_frames = int(lengths.sum())
        env = suite.load(domain_name="manipulator", task_name="insert_peg")
        env.reset()
        maximum_error = 0.0
        args.output_directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "x", libver="latest") as output:
            output.attrs["complete"] = False
            output.attrs["source"] = str(args.source.expanduser().resolve())
            output.attrs["minimum_length"] = args.minimum_length
            output.attrs["shard_index"] = args.shard_index
            output.attrs["num_shards"] = args.num_shards
            output.create_dataset("episode_index", data=episode_indices.astype(np.int32))
            output.create_dataset("ep_len", data=lengths.astype(np.int32))
            pixels = output.create_dataset(
                "pixels",
                shape=(total_frames, 224, 224, 3),
                dtype=np.uint8,
                chunks=(100, 224, 224, 3),
                **hdf5plugin.Blosc(
                    cname="lz4",
                    clevel=5,
                    shuffle=hdf5plugin.Blosc.SHUFFLE,
                ),
            )
            destination = 0
            for local_episode, episode in enumerate(episode_indices):
                source_offset = int(all_offsets[episode])
                length = int(all_lengths[episode])
                states = source["observation"][source_offset : source_offset + length]
                for block_start in range(0, length, 100):
                    block_end = min(block_start + 100, length)
                    frame_block = np.empty((block_end - block_start, 224, 224, 3), dtype=np.uint8)
                    for frame_index, state in enumerate(states[block_start:block_end]):
                        restore_state(env.physics, state)
                        error = state_restoration_error(observable_state(env), state)
                        maximum_error = max(maximum_error, error)
                        if error > 2e-5:
                            raise ValueError(
                                f"State restoration error {error} in episode {episode}, frame {block_start + frame_index}"
                            )
                        frame_block[frame_index] = env.physics.render(
                            height=224, width=224, camera_id=0
                        )
                    pixels[
                        destination + block_start : destination + block_end
                    ] = frame_block
                destination += length
                if local_episode % 10 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"shard {args.shard_index}: episode {local_episode}/{len(episode_indices)}, "
                        f"frames {destination}/{total_frames}, elapsed {elapsed:.1f}s",
                        flush=True,
                    )
            env.close()
            output.attrs["maximum_reconstruction_error"] = maximum_error
            output.attrs["render_seconds"] = time.monotonic() - started
            output.attrs["complete"] = True
            output.flush()
    report = {
        "shard_index": args.shard_index,
        "episodes": int(len(episode_indices)),
        "frames": total_frames,
        "maximum_reconstruction_error": maximum_error,
        "seconds": time.monotonic() - started,
        "output": str(output_path.resolve()),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
