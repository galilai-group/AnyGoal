"""Train PushT latent probes for block rotation and evaluate them on AvoidT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.scripts.avoidt_latent_regressors import (
    MODEL_LABELS,
    load_cached,
    predict,
    train_probe,
)


DEFAULT_RESULTS = REPO_ROOT / "paper/results/avoidt_latent_regressors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="anygoal-regressor")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-id")
    return parser.parse_args()


def angle_features(angle: np.ndarray) -> np.ndarray:
    return np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32)


def decode_angle(features: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(features[:, 0], features[:, 1]), 2.0 * np.pi)


def wrapped_error(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (prediction - truth + np.pi) % (2.0 * np.pi) - np.pi


def plot_angles(truth: np.ndarray, prediction: np.ndarray, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(truth, prediction, s=7, alpha=0.25, color="#2878B5", edgecolors="none")
    limits = (0.0, 2.0 * np.pi)
    axis.plot(limits, limits, "--", color="0.35", lw=1)
    axis.set(
        xlim=limits,
        ylim=limits,
        aspect="equal",
        xlabel="True block angle (rad)",
        ylabel="Regressed block angle (rad)",
        title=title,
    )
    ticks = np.linspace(0.0, 2.0 * np.pi, 5)
    labels = ("0", "π/2", "π", "3π/2", "2π")
    axis.set_xticks(ticks, labels)
    axis.set_yticks(ticks, labels)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def update_metrics(path: Path, new_rows: list[dict]) -> None:
    fieldnames = ("model", "probe_type", "target", "dataset", "mse", "rmse", "degradation_ratio")
    existing = []
    if path.is_file():
        with path.open(newline="") as stream:
            existing = [row for row in csv.DictReader(stream) if row["target"] != "block_rotation"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing + new_rows)


def main() -> None:
    import wandb

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        resume="must",
    )
    metric_rows = []
    for model_name in MODEL_LABELS:
        train_z, validation_z, avoidt_z, data = load_cached(model_name, args.output_dir)
        n_train, n_validation = len(train_z), len(validation_z)
        angles = np.mod(data["states"][:, 4], 2.0 * np.pi)
        train_angle = angles[:n_train]
        validation_angle = angles[n_train : n_train + n_validation]
        avoidt_angle = angles[n_train + n_validation :]
        probe, mean, std = train_probe(
            train_z,
            validation_z,
            angle_features(train_angle),
            angle_features(validation_angle),
            args,
            run,
            model_name,
            "block_rotation",
        )
        validation_prediction = decode_angle(predict(probe, validation_z, mean, std, args))
        avoidt_prediction = decode_angle(predict(probe, avoidt_z, mean, std, args))
        validation_mse = float(np.square(wrapped_error(validation_prediction, validation_angle)).mean())
        avoidt_mse = float(np.square(wrapped_error(avoidt_prediction, avoidt_angle)).mean())
        ratio = avoidt_mse / validation_mse
        for dataset_name, mse in (("PushT Val", validation_mse), ("AvoidT", avoidt_mse)):
            metric_rows.append(
                {
                    "model": MODEL_LABELS[model_name],
                    "probe_type": "MLP (sin/cos)",
                    "target": "block_rotation",
                    "dataset": dataset_name,
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "degradation_ratio": ratio,
                }
            )
        checkpoint_path = args.output_dir / f"probe_{model_name}_block_rotation.pt"
        torch.save(
            {
                "state_dict": probe.state_dict(),
                "latent_dim": train_z.shape[1],
                "target_mean": mean,
                "target_std": std,
                "target_encoding": "sin_cos",
            },
            checkpoint_path,
        )
        validation_plot = args.output_dir / f"pusht_val_probe_{model_name}_block_rotation.png"
        avoidt_plot = args.output_dir / f"avoidt_probe_{model_name}_block_rotation.png"
        plot_angles(validation_angle, validation_prediction, f"{MODEL_LABELS[model_name]} block rotation: PushT validation", validation_plot)
        plot_angles(avoidt_angle, avoidt_prediction, f"{MODEL_LABELS[model_name]} block rotation: AvoidT", avoidt_plot)
        table_indices = np.linspace(0, len(validation_angle) - 1, min(5000, len(validation_angle)), dtype=int)
        run.log(
            {
                f"validation/{model_name}_block_rotation": wandb.Image(str(validation_plot)),
                f"validation/{model_name}_block_rotation_table": wandb.Table(
                    columns=["validation_row", "true_angle_rad", "pred_angle_rad", "wrapped_error_rad"],
                    data=[
                        [int(index), float(validation_angle[index]), float(validation_prediction[index]), float(wrapped_error(validation_prediction[index], validation_angle[index]))]
                        for index in table_indices
                    ],
                ),
            }
        )
    update_metrics(args.output_dir / "probe_metrics.csv", metric_rows)
    (args.output_dir / "block_rotation_metrics.json").write_text(json.dumps(metric_rows, indent=2))
    run.log({"block_rotation_metrics": wandb.Table(columns=list(metric_rows[0]), data=[[row[key] for key in metric_rows[0]] for row in metric_rows])})
    run.finish()


if __name__ == "__main__":
    main()
