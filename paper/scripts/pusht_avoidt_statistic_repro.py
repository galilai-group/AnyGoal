"""Reproduce joint PushT-support statistics for AvoidT.

The three joint vectors are:
  1. all_state:             [agent_x, agent_y, physical_t_x, physical_t_y]
  2. action_all_state:      [action_x, action_y, all_state]
  3. agent_action:          [agent_x, agent_y, action_x, action_y]

Every AvoidT row is scored. A fixed PushT row subsample is used to fit
the high-dimensional scipy Gaussian KDE and to obtain its empirical 1st
percentile density threshold. The saved subsample indices make the analysis
exactly reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIR_DEFINITIONS = {
    "all_state": {
        "label": "All state: agent + physical T",
        "columns": [
            "agent_x",
            "agent_y",
            "physical_t_x",
            "physical_t_y",
        ],
    },
    "action_all_state": {
        "label": "Action + all state",
        "columns": [
            "action_x",
            "action_y",
            "agent_x",
            "agent_y",
            "physical_t_x",
            "physical_t_y",
        ],
    },
    "agent_action": {
        "label": "Agent position + action",
        "columns": ["agent_x", "agent_y", "action_x", "action_y"],
    },
}

PAIR_COLORS = {
    "all_state": "#78b7c5",
    "action_all_state": "#e7a977",
    "agent_action": "#9b8ac4",
}
PLOT_PAIR_ORDER = ("all_state", "agent_action", "action_all_state")
ID_OOD_SCORE_BOUNDARY = 0.0
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATASET_DIR = Path.home() / ".stable_worldmodel" / "datasets"


def default_dataset_path(filename: str) -> Path:
    path = DATASET_DIR / filename
    legacy_path = DATASET_DIR.parent / filename
    return path if path.exists() or not legacy_path.exists() else legacy_path


def parse_args(
    *,
    query_filename: str,
    query_label: str,
    query_slug: str,
    output_subdir: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Reproduce joint PushT-support statistics for {query_label}."
    )
    parser.add_argument(
        f"--{query_slug}",
        dest="query",
        type=Path,
        default=default_dataset_path(query_filename),
    )
    parser.add_argument(
        "--pusht",
        type=Path,
        default=default_dataset_path("pusht_expert_train.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / output_subdir,
    )
    parser.add_argument(
        "--train-subsample-size",
        type=int,
        default=10_000,
        help="Fixed PushT row count used for KDE fitting and thresholding.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-quantile", type=float, default=0.01)
    parser.add_argument("--kde-batch-size", type=int, default=512)
    parser.add_argument("--score-bins", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--reuse-existing-results",
        action="store_true",
        help=(
            "Regenerate summaries and plots from existing per-pair score CSVs "
            "without refitting KDEs."
        ),
    )
    parser.set_defaults(query_label=query_label, query_slug=query_slug)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.reuse_existing_results:
        required_outputs = [
            args.output_dir / "summary.json",
            *[
                args.output_dir / f"{pair_name}_scores.csv"
                for pair_name in PAIR_DEFINITIONS
            ],
        ]
        for output_path in required_outputs:
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
    else:
        for dataset_path in (args.query, args.pusht):
            if not dataset_path.is_file():
                raise FileNotFoundError(dataset_path)
    if args.train_subsample_size < 2:
        raise ValueError("--train-subsample-size must be at least 2.")
    if not 0.0 < args.support_quantile < 1.0:
        raise ValueError("--support-quantile must be between 0 and 1.")
    if args.kde_batch_size < 1:
        raise ValueError("--kde-batch-size must be positive.")
    if args.score_bins < 1:
        raise ValueError("--score-bins must be positive.")
    if args.eps <= 0.0:
        raise ValueError("--eps must be positive.")


def load_dataset(path: Path, require_indices: bool) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as dataset:
        action = dataset["action"][:].astype(np.float64)
        state = dataset["state"][:, :4].astype(np.float64)
        ep_len = dataset["ep_len"][:].astype(np.int64)
        result = {"action": action, "state": state, "ep_len": ep_len}
        if require_indices:
            if "episode_idx" in dataset and "step_idx" in dataset:
                result["episode_idx"] = dataset["episode_idx"][:].astype(
                    np.int64
                )
                result["step_idx"] = dataset["step_idx"][:].astype(np.int64)
            else:
                result["episode_idx"] = np.repeat(
                    np.arange(len(ep_len), dtype=np.int64), ep_len
                )
                result["step_idx"] = np.concatenate(
                    [np.arange(length, dtype=np.int64) for length in ep_len]
                )

    if action.ndim != 2 or action.shape[1] != 2:
        raise ValueError(f"Expected action shape [N, 2] in {path}, got {action.shape}")
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"Expected first four state values in {path}")
    if len(action) != len(state) or int(ep_len.sum()) != len(action):
        raise ValueError(f"Inconsistent row or episode lengths in {path}")
    if require_indices:
        if len(result["episode_idx"]) != len(action) or len(
            result["step_idx"]
        ) != len(action):
            raise ValueError(f"Inconsistent episode/frame indices in {path}")
    return result


def make_pair_vectors(
    pair_name: str, action: np.ndarray, state: np.ndarray
) -> np.ndarray:
    agent = state[:, :2]
    physical_t = state[:, 2:4]
    if pair_name == "all_state":
        return np.column_stack((agent, physical_t))
    if pair_name == "action_all_state":
        return np.column_stack((action, agent, physical_t))
    if pair_name == "agent_action":
        return np.column_stack((agent, action))
    raise KeyError(pair_name)


def evaluate_kde_batched(
    kde: gaussian_kde, points: np.ndarray, batch_size: int
) -> np.ndarray:
    density = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        density[start:stop] = kde(points[start:stop].T)
    return density


def score_pair(
    train_all: np.ndarray,
    query: np.ndarray,
    train_indices: np.ndarray,
    support_quantile: float,
    batch_size: int,
    eps: float,
) -> dict[str, np.ndarray | float | list[float]]:
    # Full PushT statistics make standardization stable; only KDE fitting and
    # its empirical support threshold use the saved fixed row subsample.
    train_mean = train_all.mean(axis=0)
    train_std = train_all.std(axis=0) + 1e-8
    train_z = (train_all[train_indices] - train_mean) / train_std
    query_z = (query - train_mean) / train_std

    kde = gaussian_kde(train_z.T, bw_method="scott")
    train_density = evaluate_kde_batched(kde, train_z, batch_size)
    query_density = evaluate_kde_batched(kde, query_z, batch_size)
    threshold = float(np.quantile(train_density, support_quantile))
    raw_margin = np.log(query_density + eps) - np.log(threshold + eps)
    id_score = np.maximum(0.0, raw_margin)
    is_id = id_score > ID_OOD_SCORE_BOUNDARY

    return {
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "kde_scott_factor": float(kde.factor),
        "threshold": threshold,
        "train_density": train_density,
        "query_density": query_density,
        "raw_margin": raw_margin,
        "id_score": id_score,
        "is_id": is_id,
    }


def write_pair_csv(
    path: Path,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    result: dict,
) -> None:
    is_id = result["is_id"]
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "global_row_index",
                "video_index",
                "frame_index",
                "id_ood",
                "is_id",
                "is_ood",
                "id_score",
                "raw_log_density_margin",
                "query_kde_density",
                "pusht_1pct_density_threshold",
            ]
        )
        threshold = result["threshold"]
        for row_index in range(len(episode_idx)):
            row_is_id = bool(is_id[row_index])
            writer.writerow(
                [
                    row_index,
                    int(episode_idx[row_index]),
                    int(step_idx[row_index]),
                    "ID" if row_is_id else "OOD",
                    int(row_is_id),
                    int(not row_is_id),
                    f'{result["id_score"][row_index]:.12g}',
                    f'{result["raw_margin"][row_index]:.12g}',
                    f'{result["query_density"][row_index]:.12g}',
                    f"{threshold:.12g}",
                ]
            )


def build_score_bins(
    pair_name: str,
    scores: np.ndarray,
    bin_count: int,
    query_slug: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    positive = scores[scores > 0.0]
    if len(positive):
        maximum = float(positive.max())
        edges = np.linspace(0.0, maximum, bin_count + 1)
        if maximum == 0.0:
            edges = np.linspace(0.0, 1.0, bin_count + 1)
        counts, edges = np.histogram(positive, bins=edges)
    else:
        edges = np.linspace(0.0, 1.0, bin_count + 1)
        counts = np.zeros(bin_count, dtype=np.int64)

    all_total = int(len(scores))
    ood_count = int(np.sum(scores == 0.0))
    rows = [
        {
            "pair_type": pair_name,
            "bin_kind": "ood_score_zero",
            "bin_index": -1,
            "score_lower": 0.0,
            "score_upper": 0.0,
            "lower_is_inclusive": True,
            "upper_is_inclusive": True,
            "count": ood_count,
            f"percent_of_all_{query_slug}_rows": 100.0 * ood_count / all_total,
            "percent_of_id_rows": None,
        }
    ]
    id_total = int(len(positive))
    for bin_index, count in enumerate(counts):
        rows.append(
            {
                "pair_type": pair_name,
                "bin_kind": "positive_id_score",
                "bin_index": bin_index,
                "score_lower": float(edges[bin_index]),
                "score_upper": float(edges[bin_index + 1]),
                "lower_is_inclusive": bin_index > 0,
                "upper_is_inclusive": bin_index == len(counts) - 1,
                "count": int(count),
                f"percent_of_all_{query_slug}_rows": 100.0 * count / all_total,
                "percent_of_id_rows": (
                    100.0 * count / id_total if id_total else 0.0
                ),
            }
        )
    return positive, edges, rows


def write_bin_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summaries: dict[str, dict]) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "pair_type",
                "dimensions",
                "row_count",
                "id_count",
                "ood_count",
                "id_percent",
                "ood_percent",
                "score_all_mean",
                "score_all_std",
                "score_id_mean",
                "score_id_median",
                "score_max",
                "pusht_density_threshold",
                "id_ood_score_boundary",
            ]
        )
        for pair_name, summary in summaries.items():
            writer.writerow(
                [
                    pair_name,
                    summary["dimensions"],
                    summary["row_count"],
                    summary["id_count"],
                    summary["ood_count"],
                    summary["id_percent"],
                    summary["ood_percent"],
                    summary["score_all_mean"],
                    summary["score_all_std"],
                    summary["score_id_mean"],
                    summary["score_id_median"],
                    summary["score_all_max"],
                    summary["pusht_density_threshold"],
                    summary["id_ood_score_boundary"],
                ]
            )


def regenerate_from_existing_results(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / "summary.json"
    with metadata_path.open() as input_file:
        metadata = json.load(input_file)

    results: dict[str, dict[str, np.ndarray]] = {}
    score_bin_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_bin_rows: list[dict] = []
    for pair_name in PAIR_DEFINITIONS:
        score_path = args.output_dir / f"{pair_name}_scores.csv"
        scores = []
        is_id = []
        with score_path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                scores.append(float(row["id_score"]))
                is_id.append(row["is_id"] == "1")
        score_array = np.asarray(scores, dtype=np.float64)
        is_id_array = np.asarray(is_id, dtype=bool)
        if not np.array_equal(is_id_array, score_array > ID_OOD_SCORE_BOUNDARY):
            raise ValueError(f"Inconsistent saved ID labels in {score_path}")
        results[pair_name] = {"id_score": score_array, "is_id": is_id_array}
        positive, edges, bin_rows = build_score_bins(
            pair_name, score_array, args.score_bins, args.query_slug
        )
        score_bin_data[pair_name] = (positive, edges)
        all_bin_rows.extend(bin_rows)

        pair_summary = metadata["pairs"][pair_name]
        pair_summary["id_ood_score_boundary"] = ID_OOD_SCORE_BOUNDARY
        pair_summary["id_rule"] = "id_score > 0.0"
        pair_summary["ood_rule"] = "id_score == 0.0"

    metadata["pusht_density_threshold_definition"] = (
        "Raw KDE density tau at the configured quantile of densities evaluated "
        "on the fixed PushT KDE training subsample. It is not an ID-score "
        "boundary."
    )
    metadata["id_ood_score_boundary"] = ID_OOD_SCORE_BOUNDARY
    metadata["id_rule"] = "id_score > 0.0"
    metadata["ood_rule"] = "id_score == 0.0"
    metadata["plot_pair_order"] = list(PLOT_PAIR_ORDER)
    metadata["query_label"] = args.query_label
    metadata["query_slug"] = args.query_slug

    write_bin_csv(args.output_dir / "score_bin_statistics.csv", all_bin_rows)
    write_summary_csv(args.output_dir / "summary.csv", metadata["pairs"])
    with metadata_path.open("w") as output_file:
        json.dump(metadata, output_file, indent=2, allow_nan=False)
    plot_summary(
        results,
        score_bin_data,
        args.output_dir,
        args.dpi,
        args.query_label,
    )
    print(f"Regenerated summaries and plots in {args.output_dir}", flush=True)


def plot_summary(
    results: dict[str, dict],
    score_bin_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    dpi: int,
    query_label: str,
) -> None:
    pair_names = list(PLOT_PAIR_ORDER)
    labels = [PAIR_DEFINITIONS[name]["label"] for name in pair_names]
    id_percent = [100.0 * np.mean(results[name]["is_id"]) for name in pair_names]
    ood_percent = [100.0 - value for value in id_percent]

    figure = plt.figure(figsize=(14, 8), facecolor="white")
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 2.2), hspace=0.42)
    support_axis = figure.add_subplot(grid[0, :])
    y = np.arange(len(pair_names))
    support_axis.barh(y, id_percent, color="#a8d5ba", label="ID")
    support_axis.barh(
        y, ood_percent, left=id_percent, color="#b8b8b8", label="OOD"
    )
    for index, (id_value, ood_value) in enumerate(zip(id_percent, ood_percent)):
        support_axis.text(
            min(id_value / 2.0, 47.0),
            index,
            f"ID {id_value:.2f}%",
            ha="center",
            va="center",
            fontsize=10,
        )
        support_axis.text(
            id_value + ood_value / 2.0,
            index,
            f"OOD {ood_value:.2f}%",
            ha="center",
            va="center",
            fontsize=9,
        )
    support_axis.set_yticks(y, labels)
    support_axis.invert_yaxis()
    support_axis.set_xlim(0.0, 100.0)
    support_axis.set_xlabel(f"Fraction of all {query_label} rows (%)")
    support_axis.set_title("PushT KDE empirical support (1% density threshold)")
    support_axis.grid(axis="x", alpha=0.25)

    for column, pair_name in enumerate(pair_names):
        axis = figure.add_subplot(grid[1, column])
        positive_scores, edges = score_bin_data[pair_name]
        weights = np.full(
            len(positive_scores), 100.0 / len(results[pair_name]["id_score"])
        )
        axis.hist(
            positive_scores,
            bins=edges,
            weights=weights,
            color=PAIR_COLORS[pair_name],
            edgecolor="white",
            linewidth=0.5,
        )
        if len(positive_scores):
            mean = float(positive_scores.mean())
            median = float(np.median(positive_scores))
            axis.axvline(mean, color="#444444", linestyle="--", linewidth=1.2)
            axis.axvline(median, color="#444444", linestyle=":", linewidth=1.2)
            axis.text(
                0.97,
                0.95,
                f"ID mean {mean:.3f}\nID median {median:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
        pair_ood_percent = 100.0 * np.mean(~results[pair_name]["is_id"])
        axis.set_title(
            f'{PAIR_DEFINITIONS[pair_name]["label"]}\n'
            f"OOD (score = 0): {pair_ood_percent:.2f}%"
        )
        axis.set_xlabel("Positive ID score: log density above threshold")
        if column == 0:
            axis.set_ylabel(f"Fraction of all {query_label} rows (%)")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        f"{query_label} joint-pair ID/OOD scores under PushT train",
        fontsize=15,
        y=0.99,
    )
    figure.savefig(
        output_dir / "joint_pair_id_ood_score_distributions.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "joint_pair_id_ood_score_distributions.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main(
    *,
    query_filename: str = "avoidt_test.h5",
    query_label: str = "AvoidT",
    query_slug: str = "avoidt",
    output_subdir: str = "pusht_avoidt_statistic_repro",
) -> None:
    args = parse_args(
        query_filename=query_filename,
        query_label=query_label,
        query_slug=query_slug,
        output_subdir=output_subdir,
    )
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing_results:
        regenerate_from_existing_results(args)
        return

    print(f"Loading {args.query_label}: {args.query}", flush=True)
    query = load_dataset(args.query, require_indices=True)
    print(f"Loading PushT: {args.pusht}", flush=True)
    pusht = load_dataset(args.pusht, require_indices=False)
    train_row_count = len(pusht["action"])
    if args.train_subsample_size > train_row_count:
        raise ValueError(
            "--train-subsample-size exceeds the number of PushT rows: "
            f"{args.train_subsample_size} > {train_row_count}"
        )

    rng = np.random.default_rng(args.seed)
    train_indices = np.sort(
        rng.choice(train_row_count, args.train_subsample_size, replace=False)
    )
    np.save(args.output_dir / "pusht_kde_training_indices.npy", train_indices)

    results: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    all_bin_rows: list[dict] = []
    score_bin_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for pair_name, definition in PAIR_DEFINITIONS.items():
        print(
            f"Scoring {pair_name} ({len(definition['columns'])}D)...",
            flush=True,
        )
        train_vectors = make_pair_vectors(
            pair_name, pusht["action"], pusht["state"]
        )
        query_vectors = make_pair_vectors(
            pair_name, query["action"], query["state"]
        )
        result = score_pair(
            train_vectors,
            query_vectors,
            train_indices,
            args.support_quantile,
            args.kde_batch_size,
            args.eps,
        )
        results[pair_name] = result
        write_pair_csv(
            args.output_dir / f"{pair_name}_scores.csv",
            query["episode_idx"],
            query["step_idx"],
            result,
        )

        scores = result["id_score"]
        is_id = result["is_id"]
        positive, edges, bin_rows = build_score_bins(
            pair_name, scores, args.score_bins, args.query_slug
        )
        score_bin_data[pair_name] = (positive, edges)
        all_bin_rows.extend(bin_rows)
        summaries[pair_name] = {
            "label": definition["label"],
            "dimensions": len(definition["columns"]),
            "columns": definition["columns"],
            "row_count": int(len(scores)),
            "id_count": int(is_id.sum()),
            "ood_count": int((~is_id).sum()),
            "id_percent": float(100.0 * is_id.mean()),
            "ood_percent": float(100.0 * (~is_id).mean()),
            "score_all_min": float(scores.min()),
            "score_all_max": float(scores.max()),
            "score_all_mean": float(scores.mean()),
            "score_all_std": float(scores.std(ddof=0)),
            "score_id_mean": float(positive.mean()) if len(positive) else None,
            "score_id_median": (
                float(np.median(positive)) if len(positive) else None
            ),
            "pusht_density_threshold": result["threshold"],
            "id_ood_score_boundary": ID_OOD_SCORE_BOUNDARY,
            "id_rule": "id_score > 0.0",
            "ood_rule": "id_score == 0.0",
            "pusht_standardization_mean": result["train_mean"],
            "pusht_standardization_std": result["train_std"],
            "kde_scott_factor": result["kde_scott_factor"],
        }
        print(
            f"  ID {summaries[pair_name]['id_percent']:.4f}% | "
            f"OOD {summaries[pair_name]['ood_percent']:.4f}% | "
            f"threshold {result['threshold']:.6g}",
            flush=True,
        )

        del train_vectors, query_vectors

    write_bin_csv(args.output_dir / "score_bin_statistics.csv", all_bin_rows)
    write_summary_csv(args.output_dir / "summary.csv", summaries)

    metadata = {
        "definition": (
            "score=max(0, log(query_kde_density + eps) - "
            "log(PushT training density q-quantile + eps)); score > 0 is ID, "
            "score == 0 is OOD"
        ),
        "query_path": str(args.query.resolve()),
        "query_label": args.query_label,
        "query_slug": args.query_slug,
        "pusht_path": str(args.pusht.resolve()),
        "query_episode_count": int(len(query["ep_len"])),
        "query_row_count": int(len(query["action"])),
        "pusht_episode_count": int(len(pusht["ep_len"])),
        "pusht_row_count": train_row_count,
        "kde_training_subsample_size": args.train_subsample_size,
        "kde_training_subsample_indices_file": "pusht_kde_training_indices.npy",
        "random_seed": args.seed,
        "support_quantile": args.support_quantile,
        "pusht_density_threshold_definition": (
            "Raw KDE density tau at the configured quantile of densities "
            "evaluated on the fixed PushT KDE training subsample. It is not "
            "an ID-score boundary."
        ),
        "id_ood_score_boundary": ID_OOD_SCORE_BOUNDARY,
        "id_rule": "id_score > 0.0",
        "ood_rule": "id_score == 0.0",
        "plot_pair_order": list(PLOT_PAIR_ORDER),
        "bandwidth": "scipy.stats.gaussian_kde bw_method='scott'",
        "standardization": (
            "Per dimension, using all PushT rows: "
            "z=(x-PushT_mean)/(PushT_std+1e-8)"
        ),
        "eps": args.eps,
        "score_bin_count": args.score_bins,
        "pairs": summaries,
    }
    with (args.output_dir / "summary.json").open("w") as output_file:
        json.dump(metadata, output_file, indent=2, allow_nan=False)

    plot_summary(
        results,
        score_bin_data,
        args.output_dir,
        args.dpi,
        args.query_label,
    )
    print(f"Saved all outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
