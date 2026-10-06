"""Report marginal OOD values and phases for selected manipulator concepts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.scripts import distribution_support as support
from scripts.analyze_extract_peg_ood import support_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--episode-length", type=int, required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phases = {}
    for value in args.phase:
        name, start, end = value.split(":")
        phases[name] = (int(start), int(end))
    reference = {item.label: item.values for item in support.load_features(args.reference, "manipulator_insert_peg")}
    target = {item.label: item.values for item in support.load_features(args.target, "manipulator_insert_peg")}
    step = np.arange(len(next(iter(target.values())))) % args.episode_length
    result = {}
    for label in args.feature:
        masks = support_masks(reference[label], target[label])
        values = target[label][:, 0] if target[label].shape[1] == 1 else np.arctan2(target[label][:, 1], target[label][:, 0])
        result[label] = {}
        for metric, mask in masks.items():
            entry = {"id_fraction": float(mask.mean())}
            if np.any(~mask):
                entry["ood_value_quantiles"] = np.quantile(values[~mask], [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist()
            entry["phase_id_fraction"] = {
                name: float(mask[(step >= start) & (step < end)].mean())
                for name, (start, end) in phases.items()
            }
            result[label][metric] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
