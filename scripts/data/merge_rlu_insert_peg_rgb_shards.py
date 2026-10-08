"""Merge filtered state rows and RGB render shards into one SWM HDF5."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


DERIVED_KEYS = {
    "ep_len", "ep_offset", "ep_idx", "step_idx", "fragment_start", "fragment_end"
}
ALIASES = {"privilege": "observation", "next_privilege": "next_observation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("shard_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--minimum-length", type=int, default=50)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output_path = args.output.expanduser()
    if output_path.exists():
        raise FileExistsError(output_path)
    shard_paths = [
        args.shard_directory / f"pixels_shard_{index:02d}.h5"
        for index in range(args.num_shards)
    ]
    started = time.monotonic()
    with h5py.File(args.source.expanduser(), "r", swmr=True) as source:
        source_lengths = source["ep_len"][:].astype(np.int64)
        source_offsets = source["ep_offset"][:].astype(np.int64)
        eligible = np.flatnonzero(source_lengths >= args.minimum_length)
        lengths = source_lengths[eligible]
        successful = np.array([
            source["reward"][source_offsets[episode] + source_lengths[episode] - 1]
            == np.float32(1.0)
            for episode in eligible
        ])
        offsets = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
        total = int(lengths.sum())
        shard_episode_indices = []
        shard_render_seconds = []
        maximum_error = 0.0
        for path in shard_paths:
            with h5py.File(path, "r") as shard:
                if not shard.attrs.get("complete", False):
                    raise ValueError(f"Incomplete render shard: {path}")
                shard_episode_indices.extend(shard["episode_index"][:].tolist())
                shard_render_seconds.append(float(shard.attrs["render_seconds"]))
                maximum_error = max(
                    maximum_error, float(shard.attrs["maximum_reconstruction_error"])
                )
        np.testing.assert_array_equal(shard_episode_indices, eligible)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "x", libver="latest") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            output.attrs["conversion_complete"] = False
            output.attrs["rgb_available"] = True
            output.attrs["pixels_rendering"] = (
                "dm_control insert_peg; restored every recorded observation; "
                "224x224 camera_id=0; no temporal subsampling"
            )
            output.attrs["minimum_episode_length"] = args.minimum_length
            output.attrs["source_episode_count"] = len(source_lengths)
            output.attrs["retained_episode_count"] = len(eligible)
            output.attrs["dropped_episode_count"] = len(source_lengths) - len(eligible)
            output.attrs["fragment_count"] = len(eligible)
            output.attrs["trimmed_transition_count"] = total
            output.attrs["successful_episode_count"] = int(successful.sum())
            output.attrs["unsuccessful_episode_count"] = int((~successful).sum())
            output.attrs["maximum_reconstruction_error"] = maximum_error
            output.create_dataset("ep_len", data=lengths.astype(np.int32))
            output.create_dataset("ep_offset", data=offsets)
            for key, dataset in source.items():
                if key in DERIVED_KEYS or key in ALIASES:
                    continue
                output.create_dataset(
                    key,
                    shape=(total, *dataset.shape[1:]),
                    dtype=dataset.dtype,
                    chunks=(min(4096, total), *dataset.shape[1:]),
                    compression="lzf",
                    shuffle=True,
                )
            output.create_dataset(
                "pixels",
                shape=(total, 224, 224, 3),
                dtype=np.uint8,
                chunks=(100, 224, 224, 3),
                **hdf5plugin.Blosc(
                    cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                ),
            )
            for key, target in ALIASES.items():
                output[key] = output[target]
            repeated_episode = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
            repeated_source_episode = np.repeat(eligible.astype(np.int32), lengths)
            step_idx = np.arange(total, dtype=np.int64) - offsets[repeated_episode]
            output.create_dataset("ep_idx", data=repeated_episode)
            output.create_dataset("source_ep_idx", data=repeated_source_episode)
            output.create_dataset("step_idx", data=step_idx.astype(np.int32))
            output.create_dataset("fragment_start", data=step_idx == 0)
            output.create_dataset("fragment_end", data=step_idx == np.repeat(lengths - 1, lengths))
            for target_episode, (source_episode, target_offset, length) in enumerate(
                zip(eligible, offsets, lengths)
            ):
                source_offset = int(source_offsets[source_episode])
                source_slice = slice(source_offset, source_offset + int(length))
                target_slice = slice(int(target_offset), int(target_offset + length))
                for key in source:
                    if key in DERIVED_KEYS or key in ALIASES:
                        continue
                    output[key][target_slice] = source[key][source_slice]
                if target_episode % 100 == 0:
                    print(f"copied state episode {target_episode}/{len(eligible)}", flush=True)
            pixel_offset = 0
            for path in shard_paths:
                with h5py.File(path, "r") as shard:
                    shard_pixels = shard["pixels"]
                    for start in range(0, len(shard_pixels), 100):
                        end = min(start + 100, len(shard_pixels))
                        output["pixels"][pixel_offset + start : pixel_offset + end] = (
                            shard_pixels[start:end]
                        )
                    pixel_offset += len(shard_pixels)
                    print(f"merged {path.name}: pixels={pixel_offset}/{total}", flush=True)
            if pixel_offset != total:
                raise ValueError("Pixel shards do not cover all output rows")
            output.attrs["merge_seconds"] = time.monotonic() - started
            output.attrs["conversion_complete"] = True
            output.flush()
    retained_lengths = lengths
    report = {
        "output": str(output_path.resolve()),
        "episodes_before_minimum_filter": int(len(source_lengths)),
        "episodes_after_minimum_filter": int(len(eligible)),
        "episodes_dropped": int(len(source_lengths) - len(eligible)),
        "successful_episodes": int(successful.sum()),
        "unsuccessful_episodes": int((~successful).sum()),
        "frames": total,
        "length_min": int(retained_lengths.min()),
        "length_max": int(retained_lengths.max()),
        "length_mean": float(retained_lengths.mean()),
        "length_median": float(np.median(retained_lengths)),
        "length_q25": float(np.quantile(retained_lengths, 0.25)),
        "length_q75": float(np.quantile(retained_lengths, 0.75)),
        "rgb_shape": [total, 224, 224, 3],
        "frame_stride": 1,
        "render_shard_seconds": shard_render_seconds,
        "maximum_reconstruction_error": maximum_error,
        "merge_seconds": time.monotonic() - started,
        "output_bytes": output_path.stat().st_size,
    }
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as destination:
            json.dump(report, destination, indent=2)
            destination.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
