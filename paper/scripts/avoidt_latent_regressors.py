"""Train PushT-only latent probes and evaluate physical-state decoding on AvoidT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anygoal.paths import checkpoint_root, data_root, source_root


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = data_root()
LEWM_ROOT = source_root() / "lewm"
SMWM_ROOT = source_root() / "sensorimotor-world-model/planning"
DEFAULT_LEWM = checkpoint_root() / "lewm/pusht/lewm_object.ckpt"
DEFAULT_SMWM_RUN = checkpoint_root() / "sensorimotor/pusht_inverse_lambda_30_seed0"
DEFAULT_OUTPUT = REPO_ROOT / "paper/results/avoidt_latent_regressors"
MODEL_LABELS = {"lewm": "LeWM", "smwm": "SMWM"}
TARGETS = {"agent_xy": slice(0, 2), "block_xy": slice(2, 4)}


class ProbeMLP(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "analyze"))
    parser.add_argument("--model", choices=("lewm", "smwm"))
    parser.add_argument("--pusht", type=Path, default=DATA_ROOT / "pusht_expert_train.h5")
    parser.add_argument("--avoidt", type=Path, default=DATA_ROOT / "avoidt_test.h5")
    parser.add_argument("--lewm-checkpoint", type=Path, default=DEFAULT_LEWM)
    parser.add_argument("--smwm-run", type=Path, default=DEFAULT_SMWM_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pusht-frames", type=int, default=200_000)
    parser.add_argument("--max-avoidt-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--keyframes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="anygoal-regressor")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default="pusht-avoidt-latent-regressors")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.stage == "extract" and args.model is None:
        raise ValueError("--model is required for the extract stage")
    for path in (args.pusht, args.avoidt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_pusht_frames < 1 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("frame, batch, and epoch counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def sample_pusht_indices(path: Path, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split and subsample whole episodes to avoid leakage and fragmented HDF5 I/O."""
    with h5py.File(path, "r") as dataset:
        offsets = dataset["ep_offset"][:].astype(np.int64)
        lengths = dataset["ep_len"][:].astype(np.int64)
    episode_numbers = np.arange(len(lengths))
    generator = np.random.default_rng(seed)
    generator.shuffle(episode_numbers)
    validation_count = max(1, round(0.1 * len(episode_numbers)))
    validation_episodes = episode_numbers[:validation_count]

    def choose_whole_episodes(pool: np.ndarray, target_rows: int) -> np.ndarray:
        chosen = []
        count = 0
        for episode in pool:
            rows = np.arange(offsets[episode], offsets[episode] + lengths[episode])
            if chosen and count + len(rows) > target_rows:
                break
            chosen.append(rows)
            count += len(rows)
        return np.concatenate(chosen)

    train = np.sort(choose_whole_episodes(episode_numbers[validation_count:], round(0.9 * maximum)))
    validation = np.sort(choose_whole_episodes(validation_episodes, max(1, round(0.1 * maximum))))
    return train, validation


def load_frozen_model(model_name: str, args: argparse.Namespace):
    if model_name == "lewm":
        sys.path.insert(0, str(LEWM_ROOT))
        model = torch.load(args.lewm_checkpoint, map_location="cpu", weights_only=False)
        model = model.to(args.device).eval().requires_grad_(False)

        # Both datasets are already 224x224. Apply the exact ImageNet training
        # normalization as one batched GPU operation rather than frame by frame.
        mean = torch.tensor((0.485, 0.456, 0.406), device=args.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=args.device).view(1, 3, 1, 1)
        return model, (mean, std)

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.planning_sensorimotor import load_model

    model = load_model(args.smwm_run, SMWM_ROOT, args.device)
    return model, None


@torch.inference_mode()
def encode_rows(
    model_name: str,
    model,
    transform,
    dataset_path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    latents, states = [], []
    with h5py.File(dataset_path, "r") as dataset:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            # h5py requires monotonically increasing fancy indices.
            pixels = dataset["pixels"][batch_indices]
            states.append(dataset["state"][batch_indices].astype(np.float32))
            if model_name == "lewm":
                tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
                mean, std = transform
                tensor = (tensor - mean) / std
                encoded = model.encode({"pixels": tensor[:, None]})["emb"][:, 0]
            else:
                tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device)
                encoded = model.encode({"pixels": tensor})["emb"][:, 0]
            latents.append(encoded.float().cpu().numpy())
            print(f"{model_name} {dataset_path.stem}: {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    return np.concatenate(latents), np.concatenate(states)


def extract(args: argparse.Namespace) -> None:
    train_indices, validation_indices = sample_pusht_indices(
        args.pusht, args.max_pusht_frames, args.seed
    )
    with h5py.File(args.avoidt, "r") as dataset:
        avoidt_count = dataset["state"].shape[0]
    if args.max_avoidt_frames is not None:
        avoidt_count = min(avoidt_count, args.max_avoidt_frames)
    avoidt_indices = np.arange(avoidt_count)
    model, transform = load_frozen_model(args.model, args)
    split_indices = {
        "pusht_train": train_indices,
        "pusht_val": validation_indices,
        "avoidt": avoidt_indices,
    }
    split_paths = {"pusht_train": args.pusht, "pusht_val": args.pusht, "avoidt": args.avoidt}
    all_latents, all_states = [], []
    offsets = {}
    cursor = 0
    for split in ("pusht_train", "pusht_val", "avoidt"):
        latent, state = encode_rows(
            args.model, model, transform, split_paths[split], split_indices[split],
            args.batch_size, args.device,
        )
        all_latents.append(latent)
        all_states.append(state)
        offsets[split] = [cursor, cursor + len(latent)]
        cursor += len(latent)
    np.save(args.output_dir / f"pusht_latents_{args.model}.npy", np.concatenate(all_latents[:2]))
    np.save(args.output_dir / f"avoidt_latents_{args.model}.npy", all_latents[2])
    np.savez(
        args.output_dir / f"probe_data_{args.model}.npz",
        states=np.concatenate(all_states),
        pusht_train_indices=train_indices,
        pusht_val_indices=validation_indices,
        avoidt_indices=avoidt_indices,
    )
    (args.output_dir / f"splits_{args.model}.json").write_text(json.dumps(offsets, indent=2))


def load_cached(model_name: str, output_dir: Path):
    pusht = np.load(output_dir / f"pusht_latents_{model_name}.npy")
    avoidt = np.load(output_dir / f"avoidt_latents_{model_name}.npy")
    data = np.load(output_dir / f"probe_data_{model_name}.npz")
    n_train = len(data["pusht_train_indices"])
    return pusht[:n_train], pusht[n_train:], avoidt, data


def train_probe(
    train_latent: np.ndarray,
    validation_latent: np.ndarray,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    args: argparse.Namespace,
    run,
    model_name: str,
    target_name: str,
) -> tuple[ProbeMLP, np.ndarray, np.ndarray]:
    mean, std = train_target.mean(0), train_target.std(0)
    std = np.maximum(std, 1e-6)
    device = args.device
    probe = ProbeMLP(train_latent.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_latent).float(),
        torch.from_numpy((train_target - mean) / std).float(),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_x = torch.from_numpy(validation_latent).float().to(device)
    validation_y = torch.from_numpy((validation_target - mean) / std).float().to(device)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(args.epochs):
        probe.train()
        for latent, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(probe(latent.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.inference_mode():
            validation_loss = nn.functional.mse_loss(probe(validation_x), validation_y).item()
        run.log({f"training/{model_name}_{target_name}_validation_normalized_mse": validation_loss, f"training/{model_name}_{target_name}_epoch": epoch})
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    probe.load_state_dict(best_state)
    return probe, mean, std


@torch.inference_mode()
def predict(probe: nn.Module, latent: np.ndarray, mean: np.ndarray, std: np.ndarray, args) -> np.ndarray:
    outputs = []
    for start in range(0, len(latent), args.batch_size):
        value = probe(torch.from_numpy(latent[start : start + args.batch_size]).float().to(args.device))
        outputs.append(value.cpu().numpy() * std + mean)
    return np.concatenate(outputs)


def select_grid_keyframes(points: np.ndarray, count: int) -> np.ndarray:
    side = max(1, round(np.sqrt(count)))
    x_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), side)
    y_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), side)
    selected = []
    for y in y_grid:
        for x in x_grid:
            distances = np.square(points - (x, y)).sum(1)
            for index in np.argsort(distances):
                if int(index) not in selected:
                    selected.append(int(index))
                    break
    return np.asarray(selected[:count], dtype=np.int64)


def plot_scatter(truth, prediction, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot(np.stack((truth[:, 0], prediction[:, 0])), np.stack((truth[:, 1], prediction[:, 1])), color="0.75", lw=0.35, alpha=0.35)
    axis.scatter(truth[:, 0], truth[:, 1], s=9, facecolors="none", edgecolors="#2878B5", label="Ground truth")
    axis.scatter(prediction[:, 0], prediction[:, 1], s=10, marker="x", color="#D95319", label="Probe")
    axis.set(xlim=(0, 512), ylim=(512, 0), aspect="equal", xlabel="x", ylabel="y", title=title)
    axis.legend()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(truth, prediction, title: str, path: Path) -> None:
    error = np.linalg.norm(prediction - truth, axis=1)
    figure, axis = plt.subplots(figsize=(7, 6))
    artist = axis.hexbin(truth[:, 0], truth[:, 1], C=error, reduce_C_function=np.mean, gridsize=25, extent=(0, 512, 0, 512), cmap="magma")
    axis.set(xlim=(0, 512), ylim=(512, 0), aspect="equal", xlabel="x", ylabel="y", title=title)
    figure.colorbar(artist, ax=axis, label="Mean Euclidean error")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_keyframes(dataset_path: Path, indices, truth, prediction, title: str, path: Path) -> None:
    with h5py.File(dataset_path, "r") as dataset:
        images = dataset["pixels"][np.sort(indices)]
    # Restore requested spatial order after sorted HDF5 access.
    order = np.argsort(np.argsort(indices))
    images = images[order]
    side = int(np.ceil(np.sqrt(len(indices))))
    figure, axes = plt.subplots(side, side, figsize=(3 * side, 3 * side))
    for panel, image, gt, pred in zip(np.asarray(axes).flat, images, truth[indices], prediction[indices]):
        panel.imshow(image)
        # PushT renders 512 physical units into the stored image resolution.
        scale_x, scale_y = image.shape[1] / 512.0, image.shape[0] / 512.0
        gt_pixel, pred_pixel = gt * (scale_x, scale_y), pred * (scale_x, scale_y)
        panel.plot([gt_pixel[0], pred_pixel[0]], [gt_pixel[1], pred_pixel[1]], color="white", lw=1.2)
        panel.scatter(*gt_pixel, s=45, facecolors="none", edgecolors="#00FFFF", linewidths=1.5)
        panel.scatter(*pred_pixel, s=45, marker="x", color="#FF3030", linewidths=1.5)
        panel.axis("off")
    for panel in np.asarray(axes).flat[len(indices):]:
        panel.axis("off")
    figure.suptitle(title)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    import wandb

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        id=args.wandb_run_id,
        resume="must" if args.wandb_run_id else None,
        mode=args.wandb_mode,
        config={
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_pusht_frames": args.max_pusht_frames,
            "pusht_split": "episode-disjoint 90/10",
            "lewm_checkpoint": str(args.lewm_checkpoint),
            "smwm_checkpoint": str(args.smwm_run / "checkpoints/last.ckpt"),
        },
    )
    metric_rows = []
    keyframes = {}
    for model_name in MODEL_LABELS:
        train_z, validation_z, avoidt_z, data = load_cached(model_name, args.output_dir)
        n_train, n_validation = len(train_z), len(validation_z)
        states = data["states"]
        train_states = states[:n_train]
        validation_states = states[n_train : n_train + n_validation]
        avoidt_states = states[n_train + n_validation :]
        for target_name, target_slice in TARGETS.items():
            probe, mean, std = train_probe(
                train_z, validation_z, train_states[:, target_slice], validation_states[:, target_slice], args,
                run, model_name, target_name,
            )
            validation_prediction = predict(probe, validation_z, mean, std, args)
            avoidt_prediction = predict(probe, avoidt_z, mean, std, args)
            validation_truth, avoidt_truth = validation_states[:, target_slice], avoidt_states[:, target_slice]
            validation_mse = float(np.square(validation_prediction - validation_truth).mean())
            avoidt_mse = float(np.square(avoidt_prediction - avoidt_truth).mean())
            ratio = avoidt_mse / validation_mse
            for dataset_name, mse in (("PushT Val", validation_mse), ("AvoidT", avoidt_mse)):
                metric_rows.append({"model": MODEL_LABELS[model_name], "probe_type": "MLP", "target": target_name, "dataset": dataset_name, "mse": mse, "rmse": np.sqrt(mse), "degradation_ratio": ratio})
            torch.save({"state_dict": probe.state_dict(), "latent_dim": train_z.shape[1], "target_mean": mean, "target_std": std}, args.output_dir / f"probe_{model_name}_{target_name}.pt")
            validation_plot = args.output_dir / f"pusht_val_probe_{model_name}_{target_name}.png"
            plot_scatter(validation_truth, validation_prediction, f"{MODEL_LABELS[model_name]} {target_name} on PushT validation", validation_plot)
            plot_scatter(avoidt_truth, avoidt_prediction, f"{MODEL_LABELS[model_name]} {target_name} on AvoidT", args.output_dir / f"avoidt_probe_{model_name}_{target_name}.png")
            plot_heatmap(avoidt_truth, avoidt_prediction, f"{MODEL_LABELS[model_name]} {target_name} error on AvoidT", args.output_dir / f"avoidt_error_heatmap_{model_name}_{target_name.removesuffix('_xy')}.png")
            indices = keyframes.setdefault(target_name, select_grid_keyframes(avoidt_truth, args.keyframes))
            plot_keyframes(args.avoidt, indices, avoidt_truth, avoidt_prediction, f"{MODEL_LABELS[model_name]} {target_name}: circle=true, cross=probe", args.output_dir / f"avoidt_keyframes_{model_name}_{target_name.removesuffix('_xy')}_overlay.png")
            table_indices = np.linspace(0, len(validation_truth) - 1, min(5000, len(validation_truth)), dtype=int)
            table_rows = [
                [int(index), *validation_truth[index].tolist(), *validation_prediction[index].tolist(), float(np.linalg.norm(validation_prediction[index] - validation_truth[index]))]
                for index in table_indices
            ]
            run.log(
                {
                    f"validation/{model_name}_{target_name}_physical_coordinates": wandb.Image(str(validation_plot)),
                    f"validation/{model_name}_{target_name}_coordinate_table": wandb.Table(
                        columns=["validation_row", "true_x", "true_y", "pred_x", "pred_y", "euclidean_error"],
                        data=table_rows,
                    ),
                }
            )
    for target_name, indices in keyframes.items():
        np.save(args.output_dir / f"avoidt_keyframe_indices_{target_name.removesuffix('_xy')}_grid.npy", indices)
    with (args.output_dir / "probe_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "probe_type", "target", "dataset", "mse", "rmse", "degradation_ratio"))
        writer.writeheader()
        writer.writerows(metric_rows)
    metadata = {"seed": args.seed, "pusht": str(args.pusht), "avoidt": str(args.avoidt), "lewm_checkpoint": str(args.lewm_checkpoint), "smwm_checkpoint": str(args.smwm_run / "checkpoints/last.ckpt"), "max_pusht_frames": args.max_pusht_frames, "split": "episode-disjoint 90/10", "coordinate_mapping": "state[:,0:2]=agent_xy; state[:,2:4]=block_xy"}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    run.log({"probe_metrics": wandb.Table(columns=list(metric_rows[0]), data=[[row[key] for key in metric_rows[0]] for row in metric_rows])})
    run.finish()


def main() -> None:
    args = parse_args()
    validate(args)
    extract(args) if args.stage == "extract" else analyze(args)


if __name__ == "__main__":
    main()
