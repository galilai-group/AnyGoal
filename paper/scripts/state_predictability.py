"""Analyze one-step state predictability over ten full dataset episodes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import hydra
import matplotlib
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_DIRECTORY = Path.home() / ".stable_worldmodel" / "datasets"
RESULTS_DIRECTORY = REPO_ROOT / "paper" / "results" / "state_predictability"
STATE_VARIABLES_10 = (
    "goal_x",
    "goal_y",
    "goal_angle_sin",
    "goal_angle_cos",
    "block_x",
    "block_y",
    "block_angle_sin",
    "block_angle_cos",
    "agent_x",
    "agent_y",
)
SCATTER_COLOR = "#C9829B"
KDE_COLORMAP = LinearSegmentedColormap.from_list(
    "pastel_berry", ("#FFF9FB", "#EBC4D0", "#C9829B", "#8E536E")
)
BAR_COLOR = "#8DB9DC"
BAR_EDGE_COLOR = "#527FA6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_dataset(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    filename = value if value.endswith((".h5", ".hdf5")) else f"{value}.h5"
    path = DATASET_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_path = resolve_dataset(args.dataset)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config
        else checkpoint_path.parent / "config.yaml"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.num_episodes <= 0:
        raise ValueError("num-episodes must be positive")
    if args.batch_size <= 0 or args.dpi <= 0:
        raise ValueError("batch-size and dpi must be positive")
    return dataset_path, checkpoint_path, config_path


def load_model(checkpoint_path: Path, config_path: Path, device: str):
    config = OmegaConf.load(config_path)
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model_name = model_config.pop("name")
    model = hydra.utils.instantiate(model_config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval().requires_grad_(False)
    adapter = hydra.utils.instantiate(config.data.adapter)
    state_mean = np.asarray(config.normalization.state.mean, dtype=np.float32)
    state_std = np.asarray(config.normalization.state.std, dtype=np.float32)
    action_mean = np.asarray(config.normalization.action.mean, dtype=np.float32)
    action_std = np.asarray(config.normalization.action.std, dtype=np.float32)
    return (
        model,
        adapter,
        str(model_name),
        int(config.history_size),
        int(config.data.frameskip),
        state_mean,
        state_std,
        action_mean,
        action_std,
    )


def select_and_load_episodes(
    dataset_path: Path,
    num_episodes: int,
    seed: int,
    minimum_length: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    generator = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as dataset:
        required = {"state", "action", "ep_len", "ep_offset"}
        missing = required.difference(dataset)
        if missing:
            raise KeyError(f"Missing dataset columns: {sorted(missing)}")
        lengths = dataset["ep_len"][:].astype(np.int64)
        offsets = dataset["ep_offset"][:].astype(np.int64)
        valid = np.flatnonzero(lengths >= minimum_length)
        if len(valid) < num_episodes:
            raise ValueError(
                f"Only {len(valid)} episodes have at least {minimum_length} "
                "frames; "
                f"need {num_episodes}."
            )
        selected = generator.choice(valid, size=num_episodes, replace=False)
        states = []
        actions = []
        for episode in selected:
            start = offsets[episode]
            end = start + lengths[episode]
            states.append(dataset["state"][start:end].astype(np.float32))
            actions.append(dataset["action"][start:end].astype(np.float32))
    return selected, states, actions


def prepare_windows(
    selected_episodes: np.ndarray,
    raw_states: list[np.ndarray],
    raw_actions: list[np.ndarray],
    adapter,
    history_size: int,
    frameskip: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    state_windows = []
    action_windows = []
    true_values = []
    episode_values = []
    target_steps = []
    for episode, states, actions in zip(selected_episodes, raw_states, raw_actions):
        canonical_states = adapter.encode_state(torch.from_numpy(states)).numpy()
        canonical_actions = adapter.encode_action(torch.from_numpy(actions)).numpy()
        normalized_states = (canonical_states - state_mean) / state_std
        normalized_actions = (canonical_actions - action_mean) / action_std
        for target_step in range(history_size * frameskip, len(states), frameskip):
            history_start = target_step - history_size * frameskip
            state_windows.append(
                normalized_states[history_start:target_step:frameskip]
            )
            action_windows.append(
                np.stack(
                    [
                        normalized_actions[start : start + frameskip].reshape(-1)
                        for start in range(history_start, target_step, frameskip)
                    ]
                )
            )
            true_values.append(canonical_states[target_step])
            episode_values.append(episode)
            target_steps.append(target_step)
    if not state_windows:
        raise ValueError("Selected episodes contain no complete prediction windows.")
    return (
        torch.from_numpy(np.stack(state_windows)).float(),
        torch.from_numpy(np.stack(action_windows)).float(),
        np.stack(true_values),
        np.asarray(episode_values, dtype=np.int64),
        np.asarray(target_steps, dtype=np.int64),
    )


@torch.inference_mode()
def predict_in_batches(
    model,
    state_windows: torch.Tensor,
    action_windows: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    predictions = []
    for start in range(0, len(state_windows), batch_size):
        states = torch.nan_to_num(state_windows[start : start + batch_size]).to(device)
        actions = torch.nan_to_num(action_windows[start : start + batch_size]).to(
            device
        )
        prediction = model(states, actions)
        if prediction.ndim == 3:
            prediction = prediction[..., 0]
        predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def axis_limits(
    true_values: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    pooled = np.concatenate((true_values, predictions))
    pooled = pooled[np.isfinite(pooled)]
    if not len(pooled):
        return -1.0, 1.0
    lower, upper = np.quantile(pooled, (0.005, 0.995))
    if upper <= lower:
        padding = max(abs(float(lower)) * 0.05, 0.05)
    else:
        padding = 0.08 * (upper - lower)
    return float(lower - padding), float(upper + padding)


def format_comparison_axis(axis, limits: tuple[float, float]) -> None:
    span = limits[1] - limits[0]
    if span >= 100.0:
        decimals = 0
    elif span >= 10.0:
        decimals = 1
    elif span >= 1.0:
        decimals = 2
    else:
        decimals = min(7, max(2, int(np.ceil(-np.log10(span))) + 1))
    formatter = FormatStrFormatter(f"%.{decimals}f")
    for axis_dimension in (axis.xaxis, axis.yaxis):
        axis_dimension.set_major_locator(MaxNLocator(nbins=4))
        axis_dimension.set_major_formatter(formatter)
    axis.tick_params(labelsize=8)


def plot_scatter(
    true_values: np.ndarray,
    predictions: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
    state_variables: tuple[str, ...],
) -> None:
    rows, columns = ((3, 4) if len(state_variables) == 12 else (2, 5))
    figure, axes = plt.subplots(
        rows, columns, figsize=(15, 4 * rows), constrained_layout=True
    )
    for index, axis in enumerate(np.asarray(axes).flat):
        if index >= len(state_variables):
            axis.set_visible(False)
            continue
        limits = axis_limits(true_values[:, index], predictions[:, index])
        axis.scatter(
            true_values[:, index],
            predictions[:, index],
            s=10,
            alpha=0.4,
            color=SCATTER_COLOR,
            edgecolors="none",
        )
        axis.plot(limits, limits, color="#666666", linestyle="--", linewidth=1)
        axis.set(xlim=limits, ylim=limits, title=state_variables[index])
        format_comparison_axis(axis, limits)
        axis.grid(alpha=0.15)
    figure.suptitle(f"{title}: ground truth vs. prediction", fontsize=16)
    figure.supxlabel("Ground truth")
    figure.supylabel("Predicted")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_kde(
    true_values: np.ndarray,
    predictions: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
    state_variables: tuple[str, ...],
) -> None:
    rows, columns = ((3, 4) if len(state_variables) == 12 else (2, 5))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows))
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.92,
        hspace=0.35,
        wspace=0.28,
    )
    for index, axis in enumerate(np.asarray(axes).flat):
        if index >= len(state_variables):
            axis.set_visible(False)
            continue
        limits = axis_limits(true_values[:, index], predictions[:, index])
        histogram, x_edges, y_edges = np.histogram2d(
            true_values[:, index],
            predictions[:, index],
            bins=80,
            range=(limits, limits),
        )
        density = gaussian_filter(histogram, sigma=1.35)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        positive = density[density > 0]
        if len(positive) and density.max() > positive.min():
            levels = np.linspace(positive.min(), density.max(), 12)
            axis.contourf(
                x_centers,
                y_centers,
                density.T,
                levels=levels,
                cmap=KDE_COLORMAP,
            )
        axis.plot(limits, limits, color="#666666", linestyle="--", linewidth=1)
        axis.set(xlim=limits, ylim=limits, title=state_variables[index])
        format_comparison_axis(axis, limits)
        axis.grid(alpha=0.1)
    figure.suptitle(f"{title}: 2D KDE", fontsize=16)
    figure.supxlabel("Ground truth")
    figure.supylabel("Predicted")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_prediction_pairs(
    output_path: Path,
    episodes: np.ndarray,
    target_steps: np.ndarray,
    true_values: np.ndarray,
    predictions: np.ndarray,
    state_variables: tuple[str, ...],
) -> None:
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ("episode", "target_step", "state_index", "variable", "true", "predicted")
        )
        for row in range(len(true_values)):
            for state_index, variable in enumerate(state_variables):
                writer.writerow(
                    (
                        episodes[row],
                        target_steps[row],
                        state_index,
                        variable,
                        true_values[row, state_index],
                        predictions[row, state_index],
                    )
                )


def write_mse_summary(
    output_path: Path,
    true_values: np.ndarray,
    predictions: np.ndarray,
    state_variables: tuple[str, ...],
) -> np.ndarray:
    mse = np.mean(np.square(predictions - true_values), axis=0)
    total = float(mse.sum())
    ranks = np.empty(len(mse), dtype=np.int64)
    ranks[np.argsort(mse)[::-1]] = np.arange(1, len(mse) + 1)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "state_index",
                "variable",
                "mean_squared_error",
                "rmse",
                "mse_contribution_fraction",
                "rank",
                "num_predictions",
            )
        )
        for index, variable in enumerate(state_variables):
            writer.writerow(
                (
                    index,
                    variable,
                    mse[index],
                    np.sqrt(mse[index]),
                    mse[index] / total if total else 0.0,
                    ranks[index],
                    len(true_values),
                )
            )
    return mse


def plot_mse(
    mse: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
    state_variables: tuple[str, ...],
) -> None:
    order = np.argsort(mse)[::-1]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.bar(
        np.arange(len(mse)),
        mse[order],
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
    )
    axis.set_xticks(
        np.arange(len(mse)),
        np.asarray(state_variables)[order],
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Average squared error")
    axis.set_title(f"{title}: per-variable MSE")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_path, checkpoint_path, config_path = validate_args(args)
    (
        model,
        adapter,
        model_name,
        history_size,
        frameskip,
        state_mean,
        state_std,
        action_mean,
        action_std,
    ) = load_model(checkpoint_path, config_path, args.device)
    state_variables = STATE_VARIABLES_10
    if len(state_mean) == 12:
        state_variables += ("agent_velocity_x", "agent_velocity_y")
    if len(state_variables) != len(state_mean):
        raise ValueError(
            f"No state-variable labels for representation dimension {len(state_mean)}."
        )
    selected, raw_states, raw_actions = select_and_load_episodes(
        dataset_path,
        args.num_episodes,
        args.seed,
        history_size * frameskip + 1,
    )
    (
        state_windows,
        action_windows,
        true_values,
        episode_values,
        target_steps,
    ) = prepare_windows(
        selected,
        raw_states,
        raw_actions,
        adapter,
        history_size,
        frameskip,
        state_mean,
        state_std,
        action_mean,
        action_std,
    )
    normalized_predictions = predict_in_batches(
        model,
        state_windows,
        action_windows,
        args.batch_size,
        args.device,
    )
    predictions = normalized_predictions * state_std + state_mean

    dataset_name = dataset_path.stem
    output_directory = (
        args.output_root.expanduser().resolve()
        / model_name
        / dataset_name
        / f"seed_{args.seed}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    title = f"{model_name} on {dataset_name} (seed {args.seed})"
    write_prediction_pairs(
        output_directory / "prediction_pairs.csv",
        episode_values,
        target_steps,
        true_values,
        predictions,
        state_variables,
    )
    mse = write_mse_summary(
        output_directory / "variable_mse.csv",
        true_values,
        predictions,
        state_variables,
    )
    plot_scatter(
        true_values,
        predictions,
        title,
        output_directory / "scatter.png",
        args.dpi,
        state_variables,
    )
    plot_kde(
        true_values,
        predictions,
        title,
        output_directory / "kde.png",
        args.dpi,
        state_variables,
    )
    plot_mse(
        mse,
        title,
        output_directory / "mse_bar.png",
        args.dpi,
        state_variables,
    )
    metadata = {
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "model": model_name,
        "seed": args.seed,
        "selected_episodes": selected.tolist(),
        "episode_lengths": [len(states) for states in raw_states],
        "history_size": history_size,
        "frameskip": frameskip,
        "num_predictions": len(true_values),
        "prediction_mode": "one-step sliding window with ground-truth history",
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Selected episodes: {selected.tolist()}")
    print(f"Prediction pairs: {len(true_values)}")
    print(f"Largest MSE variable: {state_variables[int(np.argmax(mse))]}")
    print(f"Saved: {output_directory}")


if __name__ == "__main__":
    main()
