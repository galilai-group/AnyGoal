"""Inspect and validate an RL Unplugged SWM HDF5; optionally plot a fragment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def inspect(path, plot=None, fragment=None, check_loader=False):
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("conversion_complete", False):
            raise ValueError("Conversion is incomplete")
        lengths, offsets = handle["ep_len"][:], handle["ep_offset"][:]
        n = len(handle["action"])
        assert len(lengths) and np.all(lengths > 0)
        np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])
        assert int(lengths.sum()) == n
        shapes = {key: list(value.shape) for key, value in handle.items()}
        for key, value in handle.items():
            if key not in ("ep_len", "ep_offset"):
                assert len(value) == n, key
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        source_rows = handle["source_row_index"][:]
        if len(np.unique(source_rows)) != n:
            raise ValueError("source_row_index contains duplicates")
        source_count = int(handle.attrs.get("original_transition_count", n))
        if np.any(source_rows < 0) or np.any(source_rows >= source_count):
            raise ValueError("source_row_index is outside the original source range")
        ep_idx = np.repeat(np.arange(len(lengths)), lengths)
        step_idx = np.arange(n) - offsets[ep_idx]
        np.testing.assert_array_equal(handle["ep_idx"][:], ep_idx)
        np.testing.assert_array_equal(handle["step_idx"][:], step_idx)
        np.testing.assert_array_equal(handle["fragment_start"][:], step_idx == 0)
        np.testing.assert_array_equal(handle["fragment_end"][:], step_idx == lengths[ep_idx] - 1)
        for start in range(0, n, 16384):
            end = min(start + 16384, n)
            obs = handle["observation"][start:end]
            np.testing.assert_array_equal(handle["proprio"][start:end], obs[:, :33])
            np.testing.assert_array_equal(handle["privilege"][start:end], obs)
            np.testing.assert_array_equal(handle["next_proprio"][start:end], handle["next_observation"][start:end, :33])
            np.testing.assert_array_equal(handle["next_privilege"][start:end], handle["next_observation"][start:end])
            for key in ("action", "observation", "next_observation", "reward", "discount"):
                assert np.isfinite(handle[key][start:end]).all(), key
            link_end = min(end, n - 1)
            linked = ep_idx[start:link_end] == ep_idx[start + 1:link_end + 1]
            np.testing.assert_array_equal(handle["next_observation"][start:link_end][linked],
                                          handle["observation"][start + 1:link_end + 1][linked])
            np.testing.assert_array_equal(handle["next_action"][start:link_end][linked],
                                          handle["action"][start + 1:link_end + 1][linked])
            assert np.all(handle["discount"][start:link_end][linked] != 0)
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        report = {
            "path": str(path.resolve()), "sequence_kind": str(handle.attrs["sequence_kind"]),
            "original_episode_count": None, "fragment_count": len(lengths), "transitions": n,
            "length_min": int(lengths.min()), "length_max": int(lengths.max()),
            "length_mean": float(lengths.mean()),
            "length_histogram": dict(zip(map(str, unique_lengths), map(int, counts))),
            "shapes": shapes, "action_dimension": int(handle["action"].shape[1]),
            "source_sha256": str(handle.attrs["source_sha256"]),
            "all_source_values_verified": bool(handle.attrs["all_source_values_verified"]),
        }
        chosen = int(np.argmax(lengths)) if fragment is None else fragment
        if not 0 <= chosen < len(lengths):
            raise ValueError("Fragment index out of bounds")
        if plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            start, length = int(offsets[chosen]), int(lengths[chosen])
            states = np.concatenate((handle["observation"][start:start + length],
                                     handle["next_observation"][start + length - 1:start + length]))
            action = handle["action"][start:start + length]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            axes[0].plot(states[:, 33], states[:, 34], "o-", markersize=3, label="object")
            axes[0].plot(states[:, 29], states[:, 30], ".-", label="hand")
            axes[0].scatter(states[0, 40], states[0, 41], marker="*", s=120, label="target")
            axes[0].set(xlabel="x (m)", ylabel="z (m)", title="Recorded planar state path")
            axes[0].axis("equal")
            axes[0].legend()
            axes[1].plot(np.arange(length), action)
            axes[1].set(xlabel="Transition index", ylabel="Expert action", title="Original controls")
            fig.suptitle(f"Fragment {chosen}: {length} transitions (original boundaries unknown; no RGB)")
            plot.parent.mkdir(parents=True, exist_ok=True)
            if plot.exists():
                raise FileExistsError(plot)
            fig.savefig(plot, dpi=150)
            plt.close(fig)
            report["plot"] = str(plot.resolve())
            report["plotted_fragment"] = chosen

    if check_loader:
        from stable_worldmodel.data import HDF5Dataset, load_dataset

        # Full episodes and clips exercise the actual installed pipeline reader.
        dataset = HDF5Dataset(path=path, num_steps=2,
                              keys_to_load=["action", "observation", "proprio", "privilege", "ep_idx"])
        for ep in {0, chosen, len(lengths) - 1}:
            loaded = dataset.load_episode(ep)
            assert tuple(loaded["action"].shape) == (int(lengths[ep]), 5)
            assert tuple(loaded["observation"].shape) == (int(lengths[ep]), 44)
        assert len(dataset) == int(np.maximum(lengths - 1, 0).sum())
        if len(dataset):
            for idx in {0, len(dataset) - 1}:
                clip = dataset[idx]
                assert tuple(clip["action"].shape) == (2, 5)
                assert (clip["ep_idx"] == clip["ep_idx"][0]).all()
        named = load_dataset(path.name, cache_dir=str(path.parent.parent), format="hdf5",
                             keys_to_load=["action", "proprio", "privilege"])
        assert tuple(named[0]["privilege"].shape) == (1, 44)
        report["stable_worldmodel_loader_verified"] = True
        report["two_step_clip_count"] = len(dataset)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--fragment", type=int, help="Default: longest fragment")
    parser.add_argument("--check-loader", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    report = inspect(args.input.expanduser(), args.plot, args.fragment, args.check_loader)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
