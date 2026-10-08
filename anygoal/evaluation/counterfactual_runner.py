"""Run RAS and CSU on the four base datasets and supported predictors.

This is a repository-facing CLI; the metric functions themselves live in
``anygoal.evaluation.counterfactual`` and are installable APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from .counterfactual import (
    counterfactual_sensitivity_uniformity,
    factual_output_variation,
    relative_action_sensitivity,
)
from .prediction_models import (
    LatentARPredictionAdapter,
    TrueStatePredictionAdapter,
    standardize_action_blocks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STATISTICS_DIRECTORY = Path(__file__).with_name("statistics")
MODEL_NAMES = ("lewm", "sensorimotor", "adaln_predictor")
ENVIRONMENTS = ("pusht", "cube", "tworoom", "insert_peg")
COUNTERFACTUAL_CHUNK_SIZE = 4
INSERT_PEG_GRIP_FRAMES = 10
INSERT_PEG_GRIP_DISTANCE = 0.08
INSERT_PEG_SYNC_TOLERANCE = 0.01


def _load_counterfactual_protocols(environment: str) -> tuple[dict, Path]:
    filenames = {
        "pusht": "pusht_actions.json",
        "cube": "cube_action_grids.json",
        "tworoom": "tworoom_actions.json",
        "insert_peg": "insert_peg_actions.json",
    }
    path = STATISTICS_DIRECTORY / filenames[environment]
    payload = json.loads(path.read_text())
    if environment == "cube":
        protocols = {
            "vertical_action": {
                "actions": payload["grid_3d_27"]["actions"],
                "bias_axes": ("action_z",),
                "effector_z_range": (0.04, 0.35),
            },
            "horizontal_action_by_height": {
                "actions": payload["grid_xy_9"]["actions"],
                "bias_axes": ("effector_z",),
            },
        }
    else:
        bias_axes = {
            "pusht": ("agent_object_distance",),
            "tworoom": ("corridor_center_distance",),
            "insert_peg": ("peg_target_distance_after_grip",),
        }[environment]
        protocols = {"default": {"actions": payload["actions"], "bias_axes": bias_axes}}
    for protocol in protocols.values():
        protocol["actions"] = np.asarray(protocol["actions"], dtype=np.float32)
    return protocols, path


def _sample_windows(
    lengths: np.ndarray,
    *,
    frame_skip: int,
    max_samples: int | None,
    windows_per_episode: int,
    seed: int,
) -> list[tuple[int, int]]:
    # Sample against the longest (three-frame) context, then shift shorter
    # model contexts forward below. This keeps the factual time t and CSU bins
    # identical across all models.
    span = 4 * frame_skip
    counts = np.maximum(np.asarray(lengths, dtype=np.int64) - span + 1, 0)
    total = int(counts.sum())
    if total < 2:
        raise ValueError("Dataset has fewer than two valid counterfactual windows.")
    eligible_episodes = np.flatnonzero(counts)
    if max_samples is None:
        return [
            (int(episode), start)
            for episode in eligible_episodes
            for start in range(int(counts[episode]))
        ]
    generator = np.random.default_rng(seed)
    generator.shuffle(eligible_episodes)
    target = min(max_samples, total)
    windows = []
    for episode in eligible_episodes:
        sample_count = min(int(counts[episode]), windows_per_episode, target - len(windows))
        starts = np.sort(
            generator.choice(int(counts[episode]), size=sample_count, replace=False)
        )
        windows.extend((int(episode), int(start)) for start in starts)
        if len(windows) == target:
            break
    if len(windows) != target:
        raise RuntimeError(f"Sampled {len(windows)} windows; expected {target}.")
    return windows


def _gather_episode_batch(
    episode: dict,
    starts: np.ndarray,
    *,
    observation_keys: tuple[str, ...],
    history_length: int,
    frame_skip: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, np.ndarray]:
    starts_tensor = torch.as_tensor(starts, dtype=torch.long)
    observation_offsets = torch.arange(history_length) * frame_skip
    action_offsets = (
        torch.arange(history_length)[:, None] * frame_skip
        + torch.arange(frame_skip)[None]
    )
    observation_indices = starts_tensor[:, None] + observation_offsets[None]
    action_indices = starts_tensor[:, None, None] + action_offsets[None]
    observations = {
        key: torch.as_tensor(episode[key])[observation_indices]
        for key in observation_keys
    }
    actions = torch.as_tensor(episode["action"])[action_indices].flatten(start_dim=2)
    current_indices = starts + (history_length - 1) * frame_skip
    return observations, actions, current_indices


def _bias_axes(
    environment: str,
    episode: dict,
    current_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    if environment == "pusht":
        state = np.asarray(episode["state"])[current_indices]
        return {"agent_object_distance": np.linalg.norm(state[:, :2] - state[:, 2:4], axis=1)}
    if environment == "cube":
        action = np.asarray(episode["action"])[current_indices]
        effector = np.asarray(episode["proprio_effector_pos"])[current_indices]
        return {"action_z": action[:, 2], "effector_z": effector[:, 2]}
    if environment == "tworoom":
        observation = np.asarray(episode["observation"])[current_indices]
        agent = observation[:, :2]
        return {
            "corridor_center_distance": np.linalg.norm(
                agent - np.asarray((112.0, 52.5)), axis=1
            )
        }
    if environment == "insert_peg":
        observation = np.asarray(episode["observation"])[current_indices]
        peg, target = observation[:, 33:35], observation[:, 40:42]
        return {
            "peg_target_distance_after_grip": np.linalg.norm(peg - target, axis=1),
        }
    raise ValueError(f"Unsupported environment: {environment}")


def _insert_peg_after_grip_mask(observation: np.ndarray) -> np.ndarray:
    """Mark frames preceded by ten close, synchronously moving hand/peg frames."""
    observation = np.asarray(observation)
    hand = observation[:, 29:31]
    peg = observation[:, 33:35]
    close = np.linalg.norm(hand - peg, axis=1) <= INSERT_PEG_GRIP_DISTANCE
    displacement_mismatch = np.linalg.norm(
        np.diff(hand, axis=0) - np.diff(peg, axis=0), axis=1
    )
    synchronized = displacement_mismatch <= INSERT_PEG_SYNC_TOLERANCE
    valid = np.zeros(len(observation), dtype=bool)
    for end in range(INSERT_PEG_GRIP_FRAMES - 1, len(observation)):
        start = end - INSERT_PEG_GRIP_FRAMES + 1
        valid[end] = close[start : end + 1].all() and synchronized[start:end].all()
    return valid


def _filter_insert_peg_windows(dataset, windows: list[tuple[int, int]]):
    """Keep common-time windows whose current frame satisfies the grip rule."""
    selected = []
    cursor = 0
    while cursor < len(windows):
        episode_index = windows[cursor][0]
        episode_end = cursor
        while episode_end < len(windows) and windows[episode_end][0] == episode_index:
            episode_end += 1
        episode = dataset.load_episode(episode_index)
        after_grip = _insert_peg_after_grip_mask(episode["observation"])
        selected.extend(
            (episode, start)
            for episode, start in windows[cursor:episode_end]
            if after_grip[start + 2 * 5]
        )
        cursor = episode_end
    if len(selected) < 2:
        raise ValueError("Insert Peg grip filter retained fewer than two windows.")
    return selected


def _capture_outputs(model, inputs: tuple, layers: tuple[int | str, ...]) -> dict:
    captures = {}
    handles = []
    modules = (
        model.predictor.transformer.layers
        if hasattr(model, "predictor")
        else model.blocks
    )
    for layer in layers:
        if not isinstance(layer, int):
            continue

        def hook(_module, _inputs, output, index=layer):
            captures[index] = output[:, -1].detach()

        handles.append(modules[layer].register_forward_hook(hook))
    try:
        prediction = model.predict(*inputs) if hasattr(model, "predict") else model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    if "final" in layers:
        captures["final"] = (
            prediction[:, -1].detach()
            if hasattr(model, "predict")
            else prediction.detach()
        )
    return captures


def _predict_latent(
    adapter: LatentARPredictionAdapter,
    observations: dict[str, torch.Tensor],
    raw_actions: torch.Tensor,
    protocols: dict,
    layers: tuple[int | str, ...],
) -> tuple[dict, dict[str, dict]]:
    device = adapter.device
    pixels = adapter.pixel_preprocessor(observations["pixels"].to(device))
    states = adapter.model.encode({"pixels": pixels})["emb"]
    actions = standardize_action_blocks(raw_actions.to(device), adapter.action_processor)
    factual = _capture_outputs(
        adapter.model, (states, adapter.model.action_encoder(actions)), layers
    )
    sensitivities = {}
    for protocol_name, protocol in protocols.items():
        actions_to_intervene = protocol["actions"]
        sums = {layer: torch.zeros(states.shape[0], device=device) for layer in layers}
        for chunk_start in range(0, len(actions_to_intervene), COUNTERFACTUAL_CHUNK_SIZE):
            action_chunk = actions_to_intervene[
                chunk_start : chunk_start + COUNTERFACTUAL_CHUNK_SIZE
            ]
            intervened_contexts = []
            for raw_action in action_chunk:
                block = torch.as_tensor(
                    raw_action, device=device, dtype=raw_actions.dtype
                )
                block = block.repeat(raw_actions.shape[-1] // block.numel())
                normalized = standardize_action_blocks(
                    block[None, None], adapter.action_processor
                )[0, 0]
                intervened = actions.clone()
                intervened[:, -1] = normalized
                intervened_contexts.append(intervened)
            chunk_size = len(action_chunk)
            flat_actions = torch.stack(intervened_contexts).flatten(0, 1)
            repeated_states = states.unsqueeze(0).expand(
                chunk_size, *states.shape
            ).flatten(0, 1)
            counterfactual = _capture_outputs(
                adapter.model,
                (repeated_states, adapter.model.action_encoder(flat_actions)),
                layers,
            )
            for layer in layers:
                output_shape = counterfactual[layer].shape[1:]
                chunk_outputs = counterfactual[layer].reshape(
                    chunk_size, states.shape[0], *output_shape
                )
                differences = factual[layer].unsqueeze(0) - chunk_outputs
                sums[layer] += differences.flatten(start_dim=2).norm(dim=2).sum(0)
        sensitivities[protocol_name] = {
            layer: values / len(actions_to_intervene) for layer, values in sums.items()
        }
    return factual, sensitivities


def _predict_true_state(
    adapter: TrueStatePredictionAdapter,
    observations: dict[str, torch.Tensor],
    raw_actions: torch.Tensor,
    protocols: dict,
    layers: tuple[int | str, ...],
) -> tuple[dict, dict[str, dict]]:
    actions = standardize_action_blocks(
        raw_actions.to(adapter.device), adapter.action_processor
    )
    states = adapter._encode_states(observations)
    factual = _capture_outputs(adapter.model, (states, actions), layers)
    sensitivities = {}
    for protocol_name, protocol in protocols.items():
        actions_to_intervene = protocol["actions"]
        sums = {
            layer: torch.zeros(states.shape[0], device=adapter.device)
            for layer in layers
        }
        for chunk_start in range(0, len(actions_to_intervene), COUNTERFACTUAL_CHUNK_SIZE):
            action_chunk = actions_to_intervene[
                chunk_start : chunk_start + COUNTERFACTUAL_CHUNK_SIZE
            ]
            intervened_contexts = []
            for raw_action in action_chunk:
                block = torch.as_tensor(
                    raw_action, device=adapter.device, dtype=raw_actions.dtype
                )
                block = block.repeat(raw_actions.shape[-1] // block.numel())
                normalized = standardize_action_blocks(
                    block[None, None], adapter.action_processor
                )[0, 0]
                intervened = actions.clone()
                intervened[:, -1] = normalized
                intervened_contexts.append(intervened)
            chunk_size = len(action_chunk)
            flat_actions = torch.stack(intervened_contexts).flatten(0, 1)
            repeated_states = states.unsqueeze(0).expand(
                chunk_size, *states.shape
            ).flatten(0, 1)
            counterfactual = _capture_outputs(
                adapter.model, (repeated_states, flat_actions), layers
            )
            for layer in layers:
                output_shape = counterfactual[layer].shape[1:]
                chunk_outputs = counterfactual[layer].reshape(
                    chunk_size, states.shape[0], *output_shape
                )
                differences = factual[layer].unsqueeze(0) - chunk_outputs
                sums[layer] += differences.flatten(start_dim=2).norm(dim=2).sum(0)
        sensitivities[protocol_name] = {
            layer: values / len(actions_to_intervene) for layer, values in sums.items()
        }
    return factual, sensitivities


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if minimum == maximum:
        scale = max(abs(minimum), 1.0) * np.finfo(np.float64).eps
        minimum, maximum = minimum - scale, maximum + scale
    return np.linspace(minimum, maximum, bins + 1)


def _write_metrics(
    output_directory: Path,
    *,
    metadata: dict,
    factual_outputs: dict[str, np.ndarray],
    sensitivities: dict[str, dict[str, np.ndarray]],
    biases: dict[str, np.ndarray],
    protocols: dict,
    bins: int,
    visualize: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    cache_path = output_directory / "counterfactual_cache.npz"
    cache = {
        f"factual__{layer}": values.astype(np.float32)
        for layer, values in factual_outputs.items()
    }
    cache.update(
        {
            f"sensitivity__{protocol}__{layer}": values.astype(np.float32)
            for protocol, layer_values in sensitivities.items()
            for layer, values in layer_values.items()
        }
    )
    cache.update(
        {f"bias__{name}": values.astype(np.float32) for name, values in biases.items()}
    )
    np.savez_compressed(cache_path, **cache)

    layer_names = [str(index) for index in range(6)] + ["final"]
    factual_variations = {
        layer: factual_output_variation(factual_outputs[layer]) for layer in layer_names
    }
    factual_variations["average"] = float(np.mean(list(factual_variations.values())))
    results = {}
    for protocol_name, protocol in protocols.items():
        protocol_sensitivities = sensitivities[protocol_name]
        protocol_sensitivities["average"] = np.mean(
            np.stack([protocol_sensitivities[layer] for layer in layer_names]), axis=0
        )
        selected_biases = {
            name: biases[name] for name in protocol["bias_axes"]
        }
        mask = np.ones(len(next(iter(selected_biases.values()))), dtype=bool)
        if "effector_z_range" in protocol:
            minimum, maximum = protocol["effector_z_range"]
            mask &= (biases["effector_z"] >= minimum) & (biases["effector_z"] <= maximum)
        representation_results = {}
        for layer, layer_sensitivities in protocol_sensitivities.items():
            masked_sensitivities = layer_sensitivities[mask]
            ras, _ = relative_action_sensitivity(
                masked_sensitivities, factual_variations[layer]
            )
            csu = {}
            for name, values in selected_biases.items():
                masked_values = values[mask]
                edges = _bin_edges(masked_values, bins)
                score, medians, populated = counterfactual_sensitivity_uniformity(
                    masked_sensitivities, masked_values, edges
                )
                csu[name] = {
                    "value": score,
                    "bin_edges": edges.tolist(),
                    "populated_bin_indices": populated.tolist(),
                    "bin_median_sensitivities": medians.tolist(),
                }
            representation_results[layer] = {
                "ras": ras,
                "factual_output_variation": factual_variations[layer],
                "median_counterfactual_sensitivity": float(
                    np.median(masked_sensitivities)
                ),
                "csu": csu,
            }
            if visualize:
                from .counterfactual_plotting import plot_counterfactual_biases

                plot_counterfactual_biases(
                    masked_sensitivities,
                    {name: values[mask] for name, values in selected_biases.items()},
                    {"ras": ras, **csu},
                    output_directory / "figures" / protocol_name / f"layer_{layer}",
                    model_name=str(metadata["model"]),
                    environment=f"{metadata['environment']} · {protocol_name} · layer {layer}",
                )
        results[protocol_name] = {
            "counterfactual_actions": protocol["actions"].tolist(),
            "representations": representation_results,
        }
    result = {
        **metadata,
        "metric_definitions": "reports/counterfactual_analysis.tex",
        "epsilon": float(np.finfo(np.float64).eps),
        "protocols": results,
        "cache": str(cache_path.resolve()),
    }
    (output_directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--environment", required=True, choices=ENVIRONMENTS)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "scripts/eval/config/prediction_eval.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/counterfactual_metrics_layerwise_20260918",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional debug/subsampling limit. Default evaluates every valid window.",
    )
    parser.add_argument(
        "--variation-samples",
        type=int,
        default=4096,
        help="Uniform factual-output sample used for the exact pairwise D_f median.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write bias-axis figures (default: true; disable with --no-visualize).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = [args.batch_size, args.bins, args.variation_samples]
    if args.max_samples is not None:
        positive_values.append(args.max_samples)
    if min(positive_values) < 1:
        raise ValueError("sample counts, batch-size, and bins must be positive")
    output_directory = args.output_root / args.model / args.environment
    if (output_directory / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_directory}")
    for source_directory in (REPO_ROOT, REPO_ROOT / "scripts/eval"):
        if str(source_directory) not in sys.path:
            sys.path.insert(0, str(source_directory))
    from scripts.eval.eval import build_adapter, resolve_path

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    dataset_cfg = cfg.datasets[args.environment]
    dataset_path = resolve_path(dataset_cfg.path)
    dataset = swm.data.HDF5Dataset(path=dataset_path, keys_to_cache=["action"])
    base_environment = str(dataset_cfg.base_environment)
    model_cfg = cfg.models[args.model][base_environment]
    if "source_root" in cfg.models[args.model]:
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.source_root = cfg.models[args.model].source_root
    history_length = 1 if args.model == "sensorimotor" else 3
    device = torch.device(args.device)
    adapter, checkpoint, model_metadata = build_adapter(
        args.model,
        model_cfg,
        dataset,
        device=device,
        checkpoint_override=None,
        training_config_override=None,
        planning_config_override=None,
        run_directory_override=None,
        history_length=history_length,
    )
    if not isinstance(adapter, (LatentARPredictionAdapter, TrueStatePredictionAdapter)):
        raise TypeError(f"Unsupported adapter: {type(adapter)}")
    protocols, action_statistics_path = _load_counterfactual_protocols(
        args.environment
    )
    windows = _sample_windows(
        dataset.lengths,
        frame_skip=5,
        max_samples=args.max_samples,
        windows_per_episode=args.batch_size,
        seed=args.seed,
    )
    unfiltered_window_count = len(windows)
    if args.environment == "insert_peg":
        windows = _filter_insert_peg_windows(dataset, windows)
    layers: tuple[int | str, ...] = (0, 1, 2, 3, 4, 5, "final")
    factual_batches = {str(layer): [] for layer in layers}
    sensitivity_batches = {
        protocol_name: {str(layer): [] for layer in layers}
        for protocol_name in protocols
    }
    bias_batches: dict[str, list[np.ndarray]] = {}
    variation_generator = np.random.default_rng(args.seed + 1)
    variation_positions = np.sort(
        variation_generator.choice(
            len(windows), size=min(args.variation_samples, len(windows)), replace=False
        )
    )
    cursor = 0
    with torch.inference_mode():
        while cursor < len(windows):
            episode_index = windows[cursor][0]
            episode_end = cursor
            while episode_end < len(windows) and windows[episode_end][0] == episode_index:
                episode_end += 1
            episode = dataset.load_episode(episode_index)
            reference_starts = np.asarray(
                [start for _, start in windows[cursor:episode_end]]
            )
            episode_starts = reference_starts + (3 - history_length) * 5
            for batch_start in range(0, len(episode_starts), args.batch_size):
                starts = episode_starts[batch_start : batch_start + args.batch_size]
                observations, raw_actions, current_indices = _gather_episode_batch(
                    episode,
                    starts,
                    observation_keys=adapter.observation_keys,
                    history_length=history_length,
                    frame_skip=5,
                )
                if isinstance(adapter, LatentARPredictionAdapter):
                    factual, batch_sensitivities = _predict_latent(
                        adapter, observations, raw_actions, protocols, layers
                    )
                else:
                    factual, batch_sensitivities = _predict_true_state(
                        adapter, observations, raw_actions, protocols, layers
                    )
                global_start = cursor + batch_start
                global_end = global_start + len(starts)
                selected = variation_positions[
                    (variation_positions >= global_start)
                    & (variation_positions < global_end)
                ] - global_start
                for layer in layers:
                    layer_name = str(layer)
                    if len(selected):
                        factual_batches[layer_name].append(
                            factual[layer][selected.tolist()]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                    for protocol_name in protocols:
                        sensitivity_batches[protocol_name][layer_name].append(
                            batch_sensitivities[protocol_name][layer]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                for name, values in _bias_axes(args.environment, episode, current_indices).items():
                    bias_batches.setdefault(name, []).append(values)
            cursor = episode_end
            print(f"processed {cursor}/{len(windows)} windows")
    factual_outputs = {
        layer: np.concatenate(values) for layer, values in factual_batches.items()
    }
    sensitivities = {
        protocol: {
            layer: np.concatenate(values) for layer, values in layer_values.items()
        }
        for protocol, layer_values in sensitivity_batches.items()
    }
    biases = {name: np.concatenate(values) for name, values in bias_batches.items()}
    _write_metrics(
        output_directory,
        metadata={
            "model": args.model,
            "environment": args.environment,
            "dataset_path": str(dataset_path.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "history_length": history_length,
            "frame_skip": 5,
            "num_samples": len(windows),
            "num_samples_before_selection": unfiltered_window_count,
            "sampling_seed": args.seed,
            "sampling_reference_history_length": 3,
            "sampling": (
                "all valid common-time windows"
                if args.max_samples is None
                else "uniform episodes, then uniform windows per episode"
            ),
            "frame_selection": (
                {
                    "name": "after_grip_close_synchronized",
                    "definition": (
                        "current frame ends 10 consecutive raw frames with "
                        "hand-peg distance <= 0.08 m and each intervening "
                        "||(delta hand) - (delta peg)|| <= 0.01 m"
                    ),
                    "consecutive_frames": INSERT_PEG_GRIP_FRAMES,
                    "hand_peg_distance_max_m": INSERT_PEG_GRIP_DISTANCE,
                    "displacement_mismatch_max_m_per_frame": (
                        INSERT_PEG_SYNC_TOLERANCE
                    ),
                    "bias_axis": "peg_target_distance_after_grip",
                }
                if args.environment == "insert_peg"
                else {"name": "all_valid_common_time_windows"}
            ),
            "counterfactual_action_statistics": str(action_statistics_path.resolve()),
            "counterfactual_action_aggregation": "arithmetic_mean_of_l2_effects",
            "intervention_scope": "current_action_block",
            "num_bins": args.bins,
            "binning": "equal_width_over_the_evaluated_bias_range",
            "selected_layers": [str(layer) for layer in layers],
            "average_definition": (
                "sample-wise arithmetic mean of S_i over layers 0..5 and final; "
                "D_f is the arithmetic mean of their layer-wise D_f values"
            ),
            "factual_variation_samples": len(variation_positions),
            "factual_variation_sampling": "uniform without replacement",
            **model_metadata,
        },
        factual_outputs=factual_outputs,
        sensitivities=sensitivities,
        biases=biases,
        protocols=protocols,
        bins=args.bins,
        visualize=args.visualize,
    )
    print(f"wrote {output_directory / 'metrics.json'}")


if __name__ == "__main__":
    main()
