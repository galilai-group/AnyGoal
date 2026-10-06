"""Evaluate short- and long-horizon prediction error for every world model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.evaluation.prediction import (  # noqa: E402
    PredictionProtocol,
    evaluate_prediction,
    select_evaluation_episodes,
)
from anygoal.evaluation.prediction_models import (  # noqa: E402
    DriftWorldPredictionAdapter,
    IRASimPredictionAdapter,
    LatentARPredictionAdapter,
    PreJEPAPredictionAdapter,
    TrueStatePredictionAdapter,
    imagenet_preprocess,
)


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
    "tworoom_triangle": "tworoom_route",
    "tworoom_triangles": "tworoom_route",
    "tworoom_route": "tworoom_route",
    "tworoom_route_test": "tworoom_route",
    "manipulator_insert_peg": "manipulator_insert_peg",
    "manipulator_hug_box": "manipulator_hug_box",
    "manipulator_extract_peg": "manipulator_extract_peg",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    path = path if path.is_absolute() else REPO_ROOT / path
    if path.exists():
        return path
    cache_root = Path.home() / ".cache" / "anygoal" / "checkpoints"
    try:
        relative = path.relative_to(cache_root)
    except ValueError:
        return path
    if relative.parts and relative.parts[0] == "sensorimotor":
        candidate = (
            Path("/grid/klindt/home/nam/sensorimotor-world-model/planning/experiments/train/results")
            / Path(*relative.parts[1:])
        )
    else:
        candidate = Path("/grid/klindt/home/nam/ckpt") / relative
    return candidate if candidate.exists() else path


def fit_processor(dataset, key: str) -> StandardScaler:
    values = np.asarray(dataset.get_col_data(key))
    values = values[~np.isnan(values).any(axis=1)]
    if not len(values):
        raise ValueError(f"No finite values available to fit processor for {key!r}.")
    return StandardScaler().fit(values)


def load_planning_config(path: str | Path) -> DictConfig:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Planning config not found: {path}")
    return OmegaConf.load(path)


def _object_checkpoint_location(checkpoint: Path) -> tuple[Path, str]:
    suffix = "_object.ckpt"
    if not checkpoint.name.endswith(suffix):
        raise ValueError(f"Expected an *_object.ckpt checkpoint, got {checkpoint}.")
    return checkpoint.parent, checkpoint.name[: -len(suffix)]


def build_adapter(
    model_name: str,
    model_cfg: DictConfig,
    dataset,
    *,
    device: torch.device,
    checkpoint_override: Path | None,
    training_config_override: Path | None,
    planning_config_override: Path | None,
    run_directory_override: Path | None,
    history_length: int,
):
    action_processor = fit_processor(dataset, "action")

    if model_name == "lewm":
        from planning import patch_loaded_model_compat, patch_transformers_vit_pickle_compat
        from planning_lewm import load_lewm_cost_model

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        if checkpoint.name.endswith("_object.ckpt"):
            checkpoint_root, policy = _object_checkpoint_location(checkpoint)
            cfg.policy = policy
        else:
            checkpoint_root = checkpoint.parent
            cfg.model.checkpoint = str(checkpoint)
            if cfg.model.get("config"):
                cfg.model.config = str(resolve_path(cfg.model.config))
        patch_transformers_vit_pickle_compat()
        model = load_lewm_cost_model(cfg, checkpoint_root)
        patch_loaded_model_compat(model)
        model = model.to(device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        image_size = int(cfg.eval.img_size)
        return (
            LatentARPredictionAdapter(
                model=model,
                model_name="lewm",
                action_processor=action_processor,
                device=device,
                pixel_preprocessor=lambda pixels: imagenet_preprocess(
                    pixels, image_size
                ),
                history_length=history_length,
            ),
            checkpoint,
            {},
        )

    if model_name == "sensorimotor":
        from planning_sensorimotor import load_model

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        if run_directory_override is not None:
            run_directory = run_directory_override
        elif "run_directory" in model_cfg:
            run_directory = resolve_path(model_cfg.run_directory)
        else:
            run_directory = resolve_path(Path(model_cfg.checkpoint).parent.parent)
        source_root = resolve_path(model_cfg.get("source_root", cfg.model.source_root))
        checkpoint = checkpoint_override or resolve_path(
            model_cfg.get("checkpoint", run_directory / "checkpoints" / "last.ckpt")
        )
        training_config = training_config_override or resolve_path(
            model_cfg.get("training_config", run_directory / "config.yaml")
        )
        model = load_model(
            run_directory,
            source_root,
            str(device),
            train_config_path=training_config,
            checkpoint_path=checkpoint,
        )
        return (
            LatentARPredictionAdapter(
                model=model,
                model_name="sensorimotor_wm",
                action_processor=action_processor,
                device=device,
                pixel_preprocessor=lambda pixels: pixels,
                history_length=history_length,
            ),
            checkpoint,
            {"run_directory": str(run_directory)},
        )

    if model_name == "prejepa":
        from planning import patch_loaded_model_compat, patch_transformers_vit_pickle_compat
        from planning_prejepa import load_prejepa_cost_model

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        if checkpoint.name.endswith("_object.ckpt"):
            checkpoint_root, policy = _object_checkpoint_location(checkpoint)
            cfg.policy = policy
        else:
            checkpoint_root = checkpoint.parent
            cfg.model.checkpoint = str(checkpoint)
            if cfg.model.get("config"):
                cfg.model.config = str(resolve_path(cfg.model.config))
        patch_transformers_vit_pickle_compat()
        model = load_prejepa_cost_model(cfg, checkpoint_root)
        patch_loaded_model_compat(model)
        model = model.to(device).eval().requires_grad_(False)
        processors = {
            key: fit_processor(dataset, key)
            for key in model.extra_encoders
            if key != "action"
        }
        return (
            PreJEPAPredictionAdapter(
                model=model,
                action_processor=action_processor,
                processors=processors,
                device=device,
                image_size=int(cfg.eval.img_size),
                history_length=history_length,
            ),
            checkpoint,
            {"variant": str(cfg.get("variant", cfg.policy))},
        )

    if model_name in {
        "adaln_predictor",
        "action_duplicated_predictor",
        "toy_driftworld",
    }:
        from planning_toy import load_predictor

        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        training_config = training_config_override or checkpoint.parent / "config.yaml"
        cfg = OmegaConf.create(
            {
                "device": str(device),
                "model": {
                    "checkpoint_path": str(checkpoint),
                    "config_path": str(training_config),
                },
            }
        )
        model, state_adapter, state_processor, toy_action_processor, train_cfg = (
            load_predictor(cfg)
        )
        trained_model_name = str(train_cfg.model.name)
        expected = {
            "adaln_predictor": "adaln",
            "action_duplicated_predictor": "action_duplicate",
            "toy_driftworld": "drift_state_predictor",
        }[model_name]
        if trained_model_name != expected:
            raise ValueError(
                f"Checkpoint model is {trained_model_name!r}, expected {expected!r}."
            )
        return (
            TrueStatePredictionAdapter(
                model=model,
                model_name=model_name,
                state_adapter=state_adapter,
                state_processor=state_processor,
                action_processor=toy_action_processor,
                device=device,
                history_length=history_length,
            ),
            checkpoint,
            {"state_representation": str(train_cfg.data.name)},
        )

    if model_name == "driftworld":
        from planning_driftworld import load_model
        from anygoal.wm.driftworld import DriftWorldActionProcessor

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        cfg.model.checkpoint = str(checkpoint)
        driftworld_action_processor = DriftWorldActionProcessor.fit(
            dataset.get_col_data("action")
        )
        model, checkpoint_step = load_model(
            cfg, device, driftworld_action_processor
        )
        if int(model.action_frameskip) != 5:
            raise ValueError(
                "Prediction evaluation requires a DriftWorld frameskip-5 checkpoint."
            )
        return (
            DriftWorldPredictionAdapter(
                model=model,
                action_processor=driftworld_action_processor,
                device=device,
                history_length=history_length,
            ),
            checkpoint,
            {"checkpoint_step": checkpoint_step},
        )

    if model_name == "irasim":
        from planning_irasim import load_model
        from anygoal.wm.irasim import IRASimActionProcessor

        cfg = load_planning_config(
            planning_config_override or model_cfg.planning_config
        )
        scheduler_override = os.environ.get("IRASIM_SCHEDULER_PATH")
        if scheduler_override:
            cfg.model.scheduler_path = scheduler_override
        checkpoint = checkpoint_override or resolve_path(model_cfg.checkpoint)
        cfg.model.checkpoint = str(checkpoint)
        if "vae_model_path" in model_cfg:
            cfg.model.vae_model_path = str(model_cfg.vae_model_path)
        if "rollout_batch_size" in model_cfg:
            cfg.model.rollout_batch_size = int(model_cfg.rollout_batch_size)
        stats_name = cfg.dataset.get("action_stats_name")
        stats_dataset = dataset
        if stats_name:
            stats_dataset = swm.data.HDF5Dataset(
                str(stats_name),
                cache_dir=Path(cfg.dataset.root).expanduser(),
                keys_to_cache=["action"],
            )
        irasim_action_processor = IRASimActionProcessor.fit(
            stats_dataset.get_col_data("action")
        )
        model, checkpoint = load_model(cfg, device, irasim_action_processor)
        if int(model.action_frameskip) != 5:
            raise ValueError(
                "Prediction evaluation requires an IRASim frameskip-5 checkpoint."
            )
        return (
            IRASimPredictionAdapter(
                model=model,
                action_processor=irasim_action_processor,
                device=device,
                history_length=history_length,
            ),
            checkpoint,
            {
                "inference_steps": int(model.inference_steps),
                "video_size": list(model.video_size),
            },
        )

    raise ValueError(f"Unsupported model: {model_name}")


def write_results(result: dict, output_directory: Path, overwrite: bool) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "metrics.json"
    csv_path = output_directory / "metrics.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite prediction results: "
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
        default=REPO_ROOT / "configs" / "prediction_eval.yaml",
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

    protocol = PredictionProtocol(
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
            "Prediction evaluation preflight failed:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    dataset = swm.data.HDF5Dataset(
        path=dataset_path,
        keys_to_cache=["action"],
    )
    selected, eligible, prediction_steps, full_eligible_count = select_evaluation_episodes(
        dataset.lengths,
        protocol=protocol,
        seed=args.seed,
    )
    evaluation_mode = (
        "short_and_long"
        if prediction_steps == protocol.max_prediction_steps
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
    result = evaluate_prediction(
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
