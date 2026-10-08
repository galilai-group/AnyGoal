"""Evaluate only the factors in reports/stat_check.tex and render the full table."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import distribution_support as support


DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "analysis" / "stat_fullresult"
DEFAULT_TABLE = REPO_ROOT / "reports" / "stat_fullresult.tex"
FULL_QUATERNION_RESULT = "cube_block_quaternion_full_4d_kde.json"


@dataclass(frozen=True)
class PairSpec:
    reference_name: str
    target_name: str
    reference_label: str
    target_label: str
    features: tuple[str, ...]


TWOROOM_FEATURES = ("agent pos (2D)", "action (2D)")
PUSHT_FEATURES = ("block angle", "agent pos (2D)", "block pos (2D)", "action (2D)")
INSERT_PEG_FEATURES = (
    "arm root angle (cos,sin)",
    "arm shoulder angle (cos,sin)",
    "arm elbow angle (cos,sin)",
    "arm wrist angle (cos,sin)",
    "finger angle (cos,sin)",
    "fingertip angle (cos,sin)",
    "thumb angle (cos,sin)",
    "thumbtip angle (cos,sin)",
    "hand position (x,z)",
    "hand orientation (qw,qy)",
    "object position (x,z)",
    "object orientation (qw,qy)",
    "target position (x,z)",
    "target orientation (qw,qy)",
    "action dimension 0",
    "action dimension 1",
    "action dimension 2",
    "action dimension 3",
    "action dimension 4",
)
CUBE_FEATURES = (
    "joint pos 0",
    "joint pos 1",
    "joint pos 2",
    "joint pos 3",
    "joint pos 4",
    "joint pos 5",
    "gripper opening",
    "gripper contact",
    "EE yaw",
    "delta yaw",
    "delta gripper",
    "EE pos (3D)",
    "block pos (3D)",
    "block quat (4D)",
    "delta pos (3D)",
)


PAIRS = {
    "tworoom_route": PairSpec(
        "tworoom.h5", "tworoom_route_test.h5", "TwoRoom", "TR-Route", TWOROOM_FEATURES
    ),
    "avoidt": PairSpec(
        "pusht_expert_train.h5", "avoidt_test.h5", "PushT", "Avoid-T", PUSHT_FEATURES
    ),
    "stackt": PairSpec(
        "pusht_expert_train.h5", "stackt_test.h5", "PushT", "Stack-T", PUSHT_FEATURES
    ),
    "pusht_antigoal": PairSpec(
        "pusht_expert_train.h5", "pusht_antigoal_test.h5", "PushT", "Push-T Antigoal", PUSHT_FEATURES
    ),
    "avoidpeg": PairSpec(
        "insert_peg_weak_expert_train.h5",
        "avoid_peg_test.h5",
        "Insert-Peg",
        "Avoid-Peg",
        INSERT_PEG_FEATURES,
    ),
    "extractpeg": PairSpec(
        "insert_peg_weak_expert_train.h5",
        "extract_peg_test.h5",
        "Insert-Peg",
        "Extract-Peg",
        INSERT_PEG_FEATURES,
    ),
    "cubehop": PairSpec(
        "cube_single_expert.h5", "cube_hop_test.h5", "Cube", "Cube-Hop", CUBE_FEATURES
    ),
    "cubediagonal": PairSpec(
        "cube_single_expert.h5",
        "cube_diagonal_test.h5",
        "Cube",
        "Cube-Diagonal",
        CUBE_FEATURES,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=PAIRS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    if (args.pair is None) == (not args.render):
        parser.error("specify exactly one of --pair or --render")
    return args


def dataset_path(filename: str) -> Path:
    path = DATASET_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluate_pair(name: str, results_dir: Path) -> None:
    spec = PAIRS[name]
    reference_path = dataset_path(spec.reference_name)
    target_path = dataset_path(spec.target_name)
    reference_schema = support.detect_schema(reference_path)
    target_schema = support.detect_schema(target_path)
    if reference_schema != target_schema:
        raise ValueError(f"schema mismatch: {reference_schema} != {target_schema}")
    reference = {item.label: item for item in support.load_features(reference_path, reference_schema)}
    target = {item.label: item for item in support.load_features(target_path, target_schema)}
    missing = set(spec.features).difference(reference).union(set(spec.features).difference(target))
    if missing:
        raise KeyError(f"missing requested features: {sorted(missing)}")

    results = {}
    for feature in spec.features:
        result = support.evaluate_feature(reference[feature], target[feature])
        results[feature] = result
        range_value = result["range"]["target_fraction"]
        range_text = "--" if range_value is None else f"{range_value:.6f}"
        print(
            f"{spec.reference_label} -> {spec.target_label} | {feature} | "
            f"mahalanobis={result['mahalanobis']['target_fraction']:.6f} "
            f"kde={result['kde']['target_fraction']:.6f} "
            f"range={range_text} verdict={result['final_verdict']}",
            flush=True,
        )
    payload = {
        "pair": name,
        "reference_label": spec.reference_label,
        "target_label": spec.target_label,
        "reference_path": str(reference_path),
        "target_path": str(target_path),
        "features": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def presentation(feature: str) -> tuple[str, str, str]:
    if feature.startswith("joint pos"):
        return "low", "proprio", feature
    if feature in {"gripper opening", "gripper contact", "EE yaw"}:
        return "low", "proprio", feature
    if feature in {"delta yaw", "delta gripper"}:
        return "low", "action", feature
    if feature == "EE pos (3D)":
        return "high", "proprio", feature
    if feature in {"block pos (3D)", "block quat (4D)"}:
        return "high", "privileged", feature
    if feature == "delta pos (3D)":
        return "high", "action", feature
    if feature == "block angle":
        return "low", "state", feature
    if feature == "agent pos (2D)":
        return "high", "proprio", feature
    if feature == "block pos (2D)":
        return "high", "state", feature
    if feature == "action (2D)":
        return "high", "action", feature
    if "angle (cos,sin)" in feature:
        return "low", "proprio", feature.replace(" (cos,sin)", "")
    if feature.startswith("hand "):
        return "low", "proprio", feature
    if feature.startswith(("object ", "target ")):
        return "low", "privileged", feature.replace("object ", "peg ")
    if feature.startswith("action dimension"):
        return "low", "action", feature.replace("action dimension", "motor command")
    raise KeyError(feature)


def format_fraction(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def span_cell(value: str, count: int) -> str:
    return value if count == 1 else rf"\multirow{{{count}}}{{*}}{{{value}}}"


def render_table(results_dir: Path, output: Path) -> None:
    pair_order = tuple(PAIRS)
    payloads = {}
    for name in pair_order:
        path = results_dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[name] = json.loads(path.read_text())

    quaternion_path = results_dir / FULL_QUATERNION_RESULT
    if not quaternion_path.is_file():
        raise FileNotFoundError(quaternion_path)
    quaternion_results = json.loads(quaternion_path.read_text())["targets"]
    for pair_name, target_name in (
        ("cubehop", "cube_hop"),
        ("cubediagonal", "cube_diagonal"),
    ):
        result = payloads[pair_name]["features"]["block quat (4D)"]
        full_kde = quaternion_results[target_name]["full_4d_kde_support_fraction"]
        result["kde"] = {
            "mode": "full-covariance 4D Gaussian KDE after 256-bin quantization",
            "target_fraction": full_kde,
        }
        result["criteria"]["kde"] = full_kde >= support.KDE_TARGET_FRACTION
        result["final_verdict"] = all(
            value for value in result["criteria"].values() if value is not None
        )

    lines = [
        "% Required packages:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{multirow}",
        "% \\usepackage[table]{xcolor}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{0.8}",
        r"\caption{Distribution-support verification between each reference dataset",
        r"\(P\) and query dataset \(Q\). Results are reported at both the individual",
        r"factor level (Low) and grouped factor level (High). A dash indicates that the",
        r"range criterion is not applied to circular variables.}",
        r"\label{tab:distribution_support_full}",
        "",
        r"\begin{tabular}{cccccllll}",
        r"\rowcolor[HTML]{45818E}",
        r"{\color[HTML]{FFFFFF} P} & {\color[HTML]{FFFFFF} Q} & {\color[HTML]{FFFFFF} level} & {\color[HTML]{FFFFFF} info} & {\color[HTML]{FFFFFF} factor of variants} & \multicolumn{1}{c}{\cellcolor[HTML]{45818E}{\color[HTML]{FFFFFF} mahalanobis 0.99}} & \multicolumn{1}{c}{\cellcolor[HTML]{45818E}{\color[HTML]{FFFFFF} kde}} & \multicolumn{1}{c}{\cellcolor[HTML]{45818E}{\color[HTML]{FFFFFF} range}} & \multicolumn{1}{c}{\cellcolor[HTML]{45818E}{\color[HTML]{FFFFFF} final verdict}} \\",
        r"\midrule",
    ]

    for pair_index, name in enumerate(pair_order):
        spec = PAIRS[name]
        payload = payloads[name]
        rows = []
        for feature in spec.features:
            level, info, label = presentation(feature)
            rows.append((level, info, label, payload["features"][feature]))
        group_counts: dict[tuple[str, str], int] = {}
        level_counts: dict[str, int] = {}
        for level, info, _, _ in rows:
            group_counts[(level, info)] = group_counts.get((level, info), 0) + 1
            level_counts[level] = level_counts.get(level, 0) + 1
        seen_groups: set[tuple[str, str]] = set()
        seen_levels: set[str] = set()
        for row_index, (level, info, label, result) in enumerate(rows):
            p_cell = span_cell(spec.reference_label, len(rows)) if row_index == 0 else ""
            q_cell = span_cell(spec.target_label, len(rows)) if row_index == 0 else ""
            if level not in seen_levels:
                level_cell = span_cell(level, level_counts[level])
                seen_levels.add(level)
            else:
                level_cell = ""
            group = (level, info)
            if group not in seen_groups:
                info_cell = span_cell(info, group_counts[group])
                seen_groups.add(group)
            else:
                info_cell = ""
            values = (
                format_fraction(result["mahalanobis"]["target_fraction"]),
                format_fraction(result["kde"]["target_fraction"]),
                format_fraction(result["range"]["target_fraction"]),
                "pass" if result["final_verdict"] else "fail",
            )
            lines.append(
                " & ".join((p_cell, q_cell, level_cell, info_cell, label, *values)) + r" \\"
            )
        if pair_index + 1 < len(pair_order):
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    if args.render:
        render_table(args.results_dir, args.output)
    else:
        evaluate_pair(args.pair, args.results_dir)


if __name__ == "__main__":
    main()
