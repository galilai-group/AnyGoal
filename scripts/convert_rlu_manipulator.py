"""Convert trusted RL Unplugged s/a/r/discount/s_/a_ NumPy transitions to SWM.

Shuffled transition exports have no original episode boundaries. With explicit
--allow-fragments, recover only unambiguous exact state AND action chains.
HDF5 layout follows stable_worldmodel/data/formats/hdf5.py and this repository's
TriangleHDF5Writer; larger chunks avoid millions of tiny chunks for state data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


# dm_control.suite.manipulator.Bring.get_observation order, not dict sort order.
FIELDS = {
    "arm_pos": 16, "arm_vel": 8, "touch": 5, "hand_pos": 4,
    "object_pos": 4, "object_vel": 3, "target_pos": 4,
}
PROPRIO_DIM = 33
RECORD_KEYS = {"s", "a", "r", "discount", "s_", "a_"}


def pack_records(rows):
    if not isinstance(rows, np.ndarray) or rows.ndim != 1 or not len(rows):
        raise ValueError("Expected a nonempty one-dimensional transition array")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != RECORD_KEYS:
            raise ValueError(f"Unsupported transition schema at source row {i}")
        for state in ("s", "s_"):
            if set(row[state]) != set(FIELDS):
                raise ValueError(f"Unexpected observation fields at row {i}/{state}")

    def stack(values, shape, name):
        dtype = np.asarray(values[0]).dtype
        if dtype.kind != "f":
            raise ValueError(f"Expected floating point {name}, got {dtype}")
        if any(np.asarray(v).shape != shape or np.asarray(v).dtype != dtype for v in values):
            raise ValueError(f"Inconsistent shape/dtype in {name}")
        result = np.stack(values)
        if not np.isfinite(result).all():
            raise ValueError(f"Nonfinite values in {name}")
        return result

    columns = {}
    for source, target in (("s", "observation"), ("s_", "next_observation")):
        parts = [stack([row[source][key] for row in rows], (dim,), f"{source}/{key}")
                 for key, dim in FIELDS.items()]
        if len({part.dtype for part in parts}) != 1:
            raise ValueError("Mixed observation dtypes would require implicit conversion")
        columns[target] = np.concatenate(parts, axis=1)
        print(f"Packed {target}: {columns[target].shape}", flush=True)
    for source, target, shape in (
        ("a", "action", (5,)), ("a_", "next_action", (5,)),
        ("r", "reward", ()), ("discount", "discount", ()),
    ):
        columns[target] = stack([row[source] for row in rows], shape, source)
    if columns["observation"].dtype != columns["next_observation"].dtype:
        raise ValueError("Current and next observation dtypes differ")
    if columns["action"].dtype != columns["next_action"].dtype:
        raise ValueError("Current and next action dtypes differ")
    if np.any((columns["discount"] < 0) | (columns["discount"] > 1)):
        raise ValueError("Discount outside [0, 1]")
    return columns


def recover_fragments(columns):
    """Cut ambiguous edges; never discard duplicate records or guess links."""
    observation, action = columns["observation"], columns["action"]
    next_observation, next_action = columns["next_observation"], columns["next_action"]
    n = len(action)

    def key(state, control):
        return hashlib.blake2b(state.tobytes() + control.tobytes(), digest_size=24).digest()

    starts = {}
    duplicate_starts = 0
    for i in range(n):
        fingerprint = key(observation[i], action[i])
        if fingerprint in starts:
            duplicate_starts += 1
            starts[fingerprint] = -1
        else:
            starts[fingerprint] = i
    successor = np.full(n, -1, dtype=np.int64)
    indegree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if columns["discount"][i] == 0:
            continue
        j = starts.get(key(next_observation[i], next_action[i]), -1)
        if j >= 0:
            # Hash lookup is only an index: verify actual values before linking.
            if not (np.array_equal(next_observation[i], observation[j])
                    and np.array_equal(next_action[i], action[j])):
                raise ValueError("State/action fingerprint collision")
            successor[i] = j
            indegree[j] += 1
    ambiguous = indegree > 1
    linked = np.flatnonzero(successor >= 0)
    successor[linked[ambiguous[successor[linked]]]] = -1
    indegree[:] = 0
    np.add.at(indegree, successor[successor >= 0], 1)
    order = np.empty(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    lengths = []
    ptr = 0
    for start in np.flatnonzero(indegree == 0):
        i, length = int(start), 0
        while i >= 0:
            if seen[i]:
                raise ValueError("Cycle or merged fragments detected")
            seen[i] = True
            order[ptr] = i
            ptr += 1
            length += 1
            i = int(successor[i])
        lengths.append(length)
    if not seen.all():
        raise ValueError("Closed transition cycles have no recoverable start; refusing to guess")
    lengths = np.asarray(lengths, dtype=np.int32)
    stats = {
        "source_rows": n, "exact_links": int((successor >= 0).sum()),
        "duplicate_start_records": duplicate_starts,
        "ambiguous_predecessor_states": int(ambiguous.sum()),
        "original_episode_count": None, "original_episode_boundaries_preserved": False,
        "fragment_count": len(lengths),
    }
    return order, lengths, stats


def write_hdf5(path, columns, order, lengths, metadata):
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
    n = len(order)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x", libver="latest") as handle:
        handle.attrs["conversion_complete"] = False
        for key, value in metadata.items():
            handle.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) or value is None else value
        handle.create_dataset("ep_len", data=lengths, maxshape=(None,))
        handle.create_dataset("ep_offset", data=offsets, maxshape=(None,))
        extra = {
            "proprio": columns["observation"][:, :PROPRIO_DIM],
            "next_proprio": columns["next_observation"][:, :PROPRIO_DIM],
        }
        field_start = 0
        for field, dim in FIELDS.items():
            prefix = "proprio" if field_start < PROPRIO_DIM else "privileged"
            extra[f"{prefix}_{field}"] = columns["observation"][:, field_start:field_start + dim]
            field_start += dim
        stored_columns = {**columns, **extra}
        for key, values in stored_columns.items():
            handle.create_dataset(key, shape=values.shape, dtype=values.dtype,
                                  maxshape=(None, *values.shape[1:]),
                                  chunks=(min(4096, n), *values.shape[1:]),
                                  compression="lzf", shuffle=True)
        handle["privilege"] = handle["observation"]
        handle["next_privilege"] = handle["next_observation"]
        handle.create_dataset("source_row_index", data=order)
        ep_idx = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        step_idx = np.arange(n, dtype=np.int64) - offsets[ep_idx]
        handle.create_dataset("ep_idx", data=ep_idx)
        handle.create_dataset("step_idx", data=step_idx.astype(np.int32))
        handle.create_dataset("fragment_start", data=step_idx == 0)
        handle.create_dataset("fragment_end", data=step_idx == lengths[ep_idx] - 1)
        for start in range(0, n, 16384):
            end = min(start + 16384, n)
            for key, values in stored_columns.items():
                handle[key][start:end] = values[order[start:end]]
        handle.flush()
        # Verify all values, including expert actions, after a complete disk round trip.
        for start in range(0, n, 16384):
            end = min(start + 16384, n)
            for key, values in stored_columns.items():
                np.testing.assert_array_equal(handle[key][start:end], values[order[start:end]])
        handle.attrs["all_source_values_verified"] = True
        handle.attrs["conversion_complete"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trusted local pickle-backed .npy export")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-fragments", action="store_true",
                        help="Accept labeled fragments instead of unrecoverable original episodes")
    args = parser.parse_args()
    if args.input.stem not in ("manipulator_insert_peg", "manipulator_insert_ball"):
        parser.error("Input name must identify manipulator_insert_peg or manipulator_insert_ball")
    if not args.allow_fragments:
        parser.error("This transition schema lacks episode metadata; use an episode-preserving source "
                     "or explicitly accept --allow-fragments")
    output = (args.output or Path.home() / ".stable_worldmodel/datasets" /
              f"rlu_{args.input.stem}_fragments.h5").expanduser()
    if output.exists():
        raise FileExistsError(output)
    before = args.input.stat()
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    rows = np.load(args.input, allow_pickle=True)
    print(f"Loaded {len(rows):,} source transitions", flush=True)
    columns = pack_records(rows)
    del rows
    order, lengths, stats = recover_fragments(columns)
    print(json.dumps(stats, indent=2), flush=True)
    after = args.input.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during conversion")
    metadata = {
        **stats, "source_path": str(args.input.resolve()), "source_sha256": digest.hexdigest(),
        "source_size_bytes": before.st_size, "task": args.input.stem,
        "sequence_kind": "exact_transition_fragments", "rgb_available": False,
        "observation_kind": "state_vector", "observation_fields": FIELDS,
        "observation_field_order": list(FIELDS), "proprio_fields": list(FIELDS)[:4],
        "privilege_fields": list(FIELDS),
        "privilege_semantics": "Full available DMC state observation; not raw MuJoCo qpos/qvel or restorable simulator state",
        "action_alignment": "observation[t], action[t] -> next_observation[t]; no rescaling, repeats, or padding",
        "boundary_semantics": "ep_len/ep_offset index fragments, not original episodes; fragment ends are not terminal flags",
        "link_rule": "Exact next observation and next expert action match; unique predecessor/successor; never link discount=0",
    }
    write_hdf5(output, columns, order, lengths, metadata)
    print(f"Wrote and verified {output}", flush=True)


if __name__ == "__main__":
    main()
