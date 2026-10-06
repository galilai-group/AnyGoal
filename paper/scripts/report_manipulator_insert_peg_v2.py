"""Combine V2 collection, marginal support, and joint support into one report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    collection = json.loads(args.collection.read_text())
    distribution = json.loads(args.distribution.read_text())
    pair = distribution["pairs"][0]
    marginal_rows = []
    for feature, result in pair["features"].items():
        marginal_rows.append(
            {
                "feature": feature,
                "dimensions": result["dimensions"],
                "mahalanobis": result["mahalanobis"]["target_fraction"],
                "kde": result["kde"]["target_fraction"],
                "range": result["range"]["target_fraction"],
                "passed": result["final_verdict"],
            }
        )
    with args.joint.open(newline="") as joint_file:
        joint_rows = list(csv.DictReader(joint_file))
    failed = [row["feature"] for row in marginal_rows if not row["passed"]]
    if collection["episodes"] != 400:
        raise RuntimeError(f"Expected 400 episodes, got {collection['episodes']}")
    if failed:
        raise RuntimeError(f"distribution_support failed: {failed}")

    lines = [
        "# Manipulator Insert Peg V2 — 400 episode report",
        "",
        f"- Accepted episodes: {collection['episodes']}",
        f"- Attempts: {collection['attempts']}",
        f"- Acceptance rate: {collection['acceptance_rate']:.4f}",
        f"- Eligible source slots: {collection['target_sampling']['eligible_source_episodes']} "
        f"(left {collection['target_sampling']['eligible_left']}, "
        f"right {collection['target_sampling']['eligible_right']})",
        "- Accepted slot sides: exactly 200 left / 200 right",
        "",
        "## distribution_support.py",
        "",
        "| Feature | Dim | Mahalanobis ID | KDE ID | Range ID | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in marginal_rows:
        range_value = "ignored" if row["range"] is None else f"{row['range']:.6f}"
        lines.append(
            f"| {row['feature']} | {row['dimensions']} | {row['mahalanobis']:.6f} "
            f"| {row['kde']:.6f} | {range_value} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## joint_distribution_support.py",
            "",
            "| Distribution | Dim | Joint KDE ID |",
            "|---|---:|---:|",
        ]
    )
    for row in joint_rows:
        lines.append(
            f"| {row['distribution']} | {row['dimensions']} | "
            f"{float(row['id_fraction']):.6f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.read_text(), flush=True)


if __name__ == "__main__":
    main()

