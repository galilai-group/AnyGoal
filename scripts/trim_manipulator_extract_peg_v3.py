"""Create extract-peg v3 by removing the first 40% of every v2 episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


DERIVED_KEYS = {"ep_len", "ep_offset", "ep_idx", "step_idx", "fragment_start", "fragment_end"}
ALIASES = {"privilege": "observation", "next_privilege": "next_observation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--output-metadata", type=Path)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--trim-fraction", type=float, default=0.4)
    parser.add_argument("--sample-seed", type=int, default=20261005)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--gif-stride", type=int, default=4)
    return parser.parse_args()


def dataset_kwargs(source: h5py.Dataset, total: int) -> dict:
    kwargs = {}
    if source.chunks is not None:
        kwargs["chunks"] = (min(source.chunks[0], total), *source.chunks[1:])
    if source.compression == "unknown":
        kwargs.update(hdf5plugin.Blosc(
            cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE
        ))
    elif source.compression is not None:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = True
    if source.fletcher32:
        kwargs["fletcher32"] = True
    if source.scaleoffset is not None:
        kwargs["scaleoffset"] = source.scaleoffset
    return kwargs


def write_trimmed_dataset(input_path: Path, output_path: Path, trim_fraction: float) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    temporary_path = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    with h5py.File(input_path, "r") as source:
        original_lengths = source["ep_len"][:].astype(np.int64)
        original_offsets = source["ep_offset"][:].astype(np.int64)
        removed_lengths = np.floor(original_lengths * trim_fraction).astype(np.int64)
        lengths = original_lengths - removed_lengths
        offsets = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
        total_before = int(original_lengths.sum())
        total_after = int(lengths.sum())
        transition_keys = {
            key for key, value in source.items()
            if isinstance(value, h5py.Dataset) and value.ndim and len(value) == total_before
        }
        unexpected = set(source) - transition_keys - DERIVED_KEYS - set(ALIASES)
        if unexpected:
            raise ValueError(f"Unclassified HDF5 keys: {sorted(unexpected)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temporary_path, "x", libver="latest") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            output.attrs["conversion_complete"] = False
            output.attrs["derived_from"] = str(input_path.resolve())
            output.attrs["trimmed_episode_prefix_fraction"] = trim_fraction
            output.attrs["trim_rounding"] = "floor"
            output.attrs["original_transition_count"] = total_before
            output.attrs["trimmed_transition_count"] = total_after

            for key in sorted(transition_keys - DERIVED_KEYS - set(ALIASES)):
                source_dataset = source[key]
                output.create_dataset(
                    key,
                    shape=(total_after, *source_dataset.shape[1:]),
                    dtype=source_dataset.dtype,
                    **dataset_kwargs(source_dataset, total_after),
                )
                for attr_key, attr_value in source_dataset.attrs.items():
                    output[key].attrs[attr_key] = attr_value
            for alias, target in ALIASES.items():
                if alias in source:
                    output[alias] = output[target]

            output.create_dataset("ep_len", data=lengths.astype(source["ep_len"].dtype))
            output.create_dataset("ep_offset", data=offsets.astype(source["ep_offset"].dtype))
            episode_ids = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
            step_indices = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
            output.create_dataset("ep_idx", data=episode_ids.astype(source["ep_idx"].dtype))
            output.create_dataset("step_idx", data=step_indices.astype(source["step_idx"].dtype))
            output.create_dataset("fragment_start", data=step_indices == 0)
            output.create_dataset("fragment_end", data=step_indices == np.repeat(lengths - 1, lengths))

            copy_keys = sorted(transition_keys - DERIVED_KEYS - set(ALIASES))
            for episode, (old_offset, removed, old_length, new_offset, new_length) in enumerate(
                zip(original_offsets, removed_lengths, original_lengths, offsets, lengths)
            ):
                source_slice = slice(int(old_offset + removed), int(old_offset + old_length))
                target_slice = slice(int(new_offset), int(new_offset + new_length))
                for key in copy_keys:
                    output[key][target_slice] = source[key][source_slice]
                if episode % 50 == 0:
                    print(f"copied episode {episode}/{len(lengths)}", flush=True)
            output.attrs["conversion_complete"] = True
            output.flush()
        os.replace(temporary_path, output_path)

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "trim_fraction": trim_fraction,
        "trim_rounding": "floor",
        "episode_count": int(len(lengths)),
        "transitions_before": total_before,
        "transitions_after": total_after,
        "transitions_removed": int(removed_lengths.sum()),
        "original_lengths": original_lengths.tolist(),
        "removed_lengths": removed_lengths.tolist(),
        "lengths": lengths.tolist(),
    }


def verify_dataset(path: Path, report: dict) -> None:
    with h5py.File(path, "r") as dataset:
        lengths = dataset["ep_len"][:].astype(np.int64)
        offsets = dataset["ep_offset"][:].astype(np.int64)
        total = int(lengths.sum())
        if not bool(dataset.attrs["conversion_complete"]):
            raise ValueError("conversion_complete is false")
        if total != report["transitions_after"]:
            raise ValueError("Unexpected output transition count")
        if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])]):
            raise ValueError("ep_offset is inconsistent with ep_len")
        for key, value in dataset.items():
            if isinstance(value, h5py.Dataset) and key not in {"ep_len", "ep_offset"}:
                if len(value) != total:
                    raise ValueError(f"Leading dimension mismatch for {key}")
        expected_episode = np.repeat(np.arange(len(lengths)), lengths)
        expected_step = np.concatenate([np.arange(length) for length in lengths])
        if not np.array_equal(dataset["ep_idx"][:], expected_episode):
            raise ValueError("ep_idx was not reindexed")
        if not np.array_equal(dataset["step_idx"][:], expected_step):
            raise ValueError("step_idx was not reindexed")
        if dataset["privilege"].id != dataset["observation"].id:
            raise ValueError("privilege hard link was not preserved")
        if dataset["next_privilege"].id != dataset["next_observation"].id:
            raise ValueError("next_privilege hard link was not preserved")


def write_metadata(source_path: Path | None, output_path: Path | None, report: dict) -> None:
    if source_path is None or output_path is None:
        return
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    metadata = json.loads(source_path.read_text())
    metadata["derivation"] = {
        "source_hdf5": report["input"],
        "operation": "removed the first floor(0.4 * episode_length) transitions from every episode",
        "trim_fraction": report["trim_fraction"],
        "trim_rounding": report["trim_rounding"],
        "transitions_before": report["transitions_before"],
        "transitions_after": report["transitions_after"],
    }
    metadata["episode_lengths_after_trim"] = report["lengths"]
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")


def make_visualizations(dataset_path: Path, report_directory: Path, report: dict, seed: int, count: int, stride: int) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray(report["lengths"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.flatnonzero(lengths > 0), size=count, replace=False)
    with h5py.File(dataset_path, "r") as dataset:
        for episode in selected:
            offset = int(dataset["ep_offset"][episode])
            length = int(dataset["ep_len"][episode])
            frames = dataset["pixels"][offset : offset + length : stride]
            images = []
            for sampled_step, frame in enumerate(frames):
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 150, 17), fill=(0, 0, 0))
                draw.text((4, 3), f"episode {episode}  step {sampled_step * stride}", fill=(255, 255, 255))
                images.append(image)
            gif_path = report_directory / f"episode_{int(episode):03d}.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(30, max(1, len(np.unique(lengths))))
    ax.hist(lengths, bins=bins, edgecolor="black")
    ax.set(title="Extract-Peg v3 Episode Lengths", xlabel="Transitions per episode", ylabel="Episode count")
    stats = (
        f"n={len(lengths)}  min={lengths.min()}  max={lengths.max()}\n"
        f"mean={lengths.mean():.2f}  median={np.median(lengths):.2f}  std={lengths.std():.2f}"
    )
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    fig.savefig(report_directory / "episode_length_histogram.png", dpi=180)
    plt.close(fig)
    report["visualization_sample_seed"] = seed
    report["visualized_episode_indices"] = selected.tolist()
    report["gif_stride"] = stride
    report["length_statistics"] = {
        "min": int(lengths.min()), "max": int(lengths.max()),
        "mean": float(lengths.mean()), "median": float(np.median(lengths)),
        "std": float(lengths.std()), "q25": float(np.quantile(lengths, 0.25)),
        "q75": float(np.quantile(lengths, 0.75)),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.trim_fraction < 1.0:
        raise ValueError("--trim-fraction must be in [0, 1)")
    report = write_trimmed_dataset(args.input.expanduser(), args.output.expanduser(), args.trim_fraction)
    verify_dataset(args.output.expanduser(), report)
    write_metadata(args.source_metadata, args.output_metadata, report)
    make_visualizations(args.output.expanduser(), args.report_directory, report, args.sample_seed, args.sample_count, args.gif_stride)
    report_path = args.report_directory / "trim_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"original_lengths", "removed_lengths", "lengths"}}, indent=2))


if __name__ == "__main__":
    main()
