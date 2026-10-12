"""Evaluate short- and long-horizon rollout error for every world model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from anygoal.evaluation.rollout import (  # noqa: E402
    RolloutProtocol,
    evaluate_rollout,
    select_evaluation_episodes,
)
from anygoal.wm.rollout import (  # noqa: E402
    build_adapter as shared_build_adapter,
    resolve_path as shared_resolve_path,
)
from anygoal.resources import resource_path


MODEL_ALIASES = {
    "lewm": "lewm",
    "sensorimotor": "sensorimotor",
    "sensorimotor_wm": "sensorimotor",
    "prejepa": "prejepa",
    "adaln": "adaln_predictor",
    "adaln_predictor": "adaln_predictor",
    "action_duplicate": "action_duplicated_predictor",
    "action_duplicated": "action_duplicated_predictor",
    "action_duplicated_predictor": "action_duplicated_predictor",
    "toy_driftworld": "toy_driftworld",
    "driftworld": "driftworld",
    "irasim": "irasim",
}
ENVIRONMENT_ALIASES = {
    "pusht": "pusht",
    "avoidt": "avoidt",
    "stackt": "stackt",
    "push_t_antigoal": "pusht_antigoal",
    "pusht_antigoal": "pusht_antigoal",
    "cube": "cube",
    "cube_hop": "cubehop",
    "cubehop": "cubehop",
    "cube_diagonal": "cube_diagonal",
    "tworoom": "tworoom",
    "tworoom_route": "tworoom_route",
    "tworoom_route_test": "tworoom_route",
    "insert_peg": "insert_peg",
    "avoid_peg": "avoid_peg",
    "extract_peg": "extract_peg",
}


build_adapter = shared_build_adapter
resolve_path = shared_resolve_path


def write_results(result: dict, output_directory: Path, overwrite: bool) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "metrics.json"
    csv_path = output_directory / "metrics.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite rollout results: "
            + ", ".join(str(path) for path in existing)
        )

    json_path.write_text(json.dumps(result, indent=2) + "\n")
    row = {
        "model_name": result["model_name"],
        "environment": result["environment"],
        "dataset_name": result["dataset_name"],
        "base_environment": result["base_environment"],
        "variant_identifier": result["variant_identifier"],
        "is_base": result["is_base"],
        "seed": result["seed"],
        "error_space": result["error_space"],
        "num_episodes": result["protocol"]["num_episodes"],
        "num_starting_points": result["sampling"]["num_starting_points"],
        "short_horizon_error": result["short_horizon_error"],
        "long_horizon_error_h15": result["long_horizon_error"]["15"],
        "long_horizon_error_h25": result["long_horizon_error"]["25"],
        "checkpoint": result["checkpoint"],
    }
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def parse_args(default_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=resource_path("eval", "rollout.yaml"),
    )
    parser.add_argument("--model", default=default_model, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--environment", required=True, choices=sorted(ENVIRONMENT_ALIASES))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--planning-config", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rollout-batch-size", type=int)
    parser.add_argument("--max-starts-per-episode", type=int)
    parser.add_argument("--episode-start-position", type=int, default=0)
    parser.add_argument("--episode-end-position", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        parser.error("--model is required")
    return args


def main(default_model: str | None = None) -> None:
    args = parse_args(default_model)
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    model_name = MODEL_ALIASES[args.model]
    environment = ENVIRONMENT_ALIASES[args.environment]
    dataset_cfg = cfg.datasets[environment]
    base_environment = str(dataset_cfg.base_environment)
    model_cfg = cfg.models[model_name][base_environment]
    if "source_root" in cfg.models[model_name]:
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.source_root = cfg.models[model_name].source_root

    protocol = RolloutProtocol(
        history_length=int(cfg.protocol.history_length),
        frame_skip=int(cfg.protocol.frame_skip),
        horizons=tuple(int(value) for value in cfg.protocol.horizons),
        num_episodes=int(args.num_episodes or cfg.protocol.num_episodes),
        batch_size=int(args.batch_size or cfg.protocol.batch_size),
    )
    dataset_path = resolve_path(dataset_cfg.path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_override = args.checkpoint.resolve() if args.checkpoint else None
    run_directory_override = (
        args.run_directory.resolve() if args.run_directory else None
    )
    required_paths = [dataset_path]
    if checkpoint_override is not None:
        required_paths.append(checkpoint_override)
    elif "checkpoint" in model_cfg:
        required_paths.append(resolve_path(model_cfg.checkpoint))
    elif run_directory_override is not None:
        required_paths.extend(
            [
                run_directory_override / "config.yaml",
                run_directory_override / "checkpoints" / "last.ckpt",
            ]
        )
    elif "run_directory" in model_cfg:
        run_directory = resolve_path(model_cfg.run_directory)
        required_paths.extend(
            [
                run_directory / "config.yaml",
                run_directory / "checkpoints" / "last.ckpt",
            ]
        )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Rollout evaluation preflight failed:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    dataset = swm.data.HDF5Dataset(
        path=dataset_path,
        keys_to_cache=["action"],
    )
    selected, eligible, rollout_steps, full_eligible_count = select_evaluation_episodes(
        dataset.lengths,
        protocol=protocol,
        seed=args.seed,
    )
    evaluation_mode = (
        "short_and_long"
        if rollout_steps == protocol.max_rollout_steps
        else "short_only"
    )
    print(
        f"preflight OK: model={model_name}, environment={environment}, "
        f"mode={evaluation_mode}, eligible_episodes={len(eligible)}, "
        f"full_rollout_eligible_episodes={full_eligible_count}, "
        f"sampled_episodes={len(selected)}"
    )
    if args.preflight:
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.rollout_batch_size is not None:
        if model_name != "irasim":
            raise ValueError("--rollout-batch-size is only supported for IRASim.")
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.rollout_batch_size = int(args.rollout_batch_size)
    output_directory = args.output_directory or (
        resolve_path(cfg.output_root) / model_name / environment / f"seed_{args.seed}"
    )
    adapter, checkpoint, model_metadata = build_adapter(
        model_name,
        model_cfg,
        dataset,
        device=device,
        checkpoint_override=checkpoint_override,
        training_config_override=(
            args.training_config.resolve() if args.training_config else None
        ),
        planning_config_override=(
            args.planning_config.resolve() if args.planning_config else None
        ),
        run_directory_override=run_directory_override,
        history_length=protocol.history_length,
    )
    metadata = {
        "environment": environment,
        "dataset_name": str(dataset_cfg.dataset_name),
        "dataset_path": str(dataset_path),
        "base_environment": base_environment,
        "variant_identifier": str(dataset_cfg.variant_identifier),
        "is_base": str(dataset_cfg.variant_identifier) == "base",
        "checkpoint": str(checkpoint.resolve()),
        **model_metadata,
    }
    result = evaluate_rollout(
        dataset,
        adapter,
        protocol=protocol,
        seed=args.seed,
        metadata=metadata,
        max_starts_per_episode=args.max_starts_per_episode,
        progress_path=output_directory / "progress.json",
        episode_start_position=args.episode_start_position,
        episode_end_position=args.episode_end_position,
    )
    write_results(result, output_directory, args.overwrite)


if __name__ == "__main__":
    main()
