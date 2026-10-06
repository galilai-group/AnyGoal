"""Publish the curated AnyGoal datasets and checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi


REPO_ID = "HazelNam/anygoal"
DATA_ROOT = Path("/grid/klindt/home/nam/.stable_worldmodel/datasets")

DATASETS = {
    "avoidt": "avoidt_test.h5",
    "stackt": "stackt_test.h5",
    "pusht-antigoal": "pusht_antigoal_test.h5",
    "cubehop": "cube_hop_test.h5",
    "cubediagonal": "cube_diagonal_test.h5",
    "insertpeg": "manipulator_insert_peg_weak_expert_train.h5",
    "hug_box": "avoid_peg_test.h5",
    "extractpeg": "extract_peg_test.h5",
    "tworoom_route": "tworoom_route_test.h5",
}

CHECKPOINTS = {
    ("sensorimotor", "pusht"): "/grid/klindt/home/nam/sensorimotor-world-model/planning/experiments/train/results/pusht_inverse_lambda_30_seed0/checkpoints/last.ckpt",
    ("driftworld", "pusht"): "/grid/klindt/home/nam/ckpt/20260820_driftworld_pusht_lewm_fs5_lr5e-4_resume435500_to500k/ckpt-step500000.pth",
    ("irasim", "pusht"): "/grid/klindt/home/nam/ckpt/08/20/irasim_pusht_h5_frameskip5_lrlr5em5002/checkpoints/0300000.pt",
    ("toy-adaln", "pusht"): "/grid/klindt/home/nam/ckpt/toy_predictors_20260723_frameskip5_10d/20260723_pusht_adaln/weights_epoch_100.pt",
    ("toy-action-duplicate", "pusht"): "/grid/klindt/home/nam/ckpt/toy_predictors_20260723_frameskip5_10d/20260723_pusht_action_duplicate/weights_epoch_100.pt",
    ("toy-driftworld", "pusht"): "/grid/klindt/home/nam/ckpt/toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_pusht_drift_state_predictor_bs128_500k_lr1e-4/weights_step_00500000.pt",
    ("sensorimotor", "cube"): "/grid/klindt/home/nam/sensorimotor-world-model/planning/experiments/train/results/cube_inverse_lambda_1_seed3072/checkpoints/last.ckpt",
    ("driftworld", "cube"): "/grid/klindt/home/nam/ckpt/20260820_driftworld_cube_lewm_fs5_lr5e-5_rerun/ckpt-step100000.pth",
    ("irasim", "cube"): "/grid/klindt/home/nam/ckpt/08/24/irasim_cube_h5_frameskip5_lr5em5/checkpoints/0300000.pt",
    ("toy-adaln", "cube"): "/grid/klindt/home/nam/ckpt/toy_predictors_20260826_cube_everything_toy_predictors/20260826_cube_everything_adaln/weights_epoch_100.pt",
    ("toy-action-duplicate", "cube"): "/grid/klindt/home/nam/ckpt/toy_predictors_20260826_cube_everything_toy_predictors/20260826_cube_everything_action_duplicate/weights_epoch_100.pt",
    ("toy-driftworld", "cube"): "/grid/klindt/home/nam/ckpt/toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_cube_drift_state_predictor_bs128_500k_lr1e-4/weights_step_00500000.pt",
    ("sensorimotor", "tworoom"): "/grid/klindt/home/nam/sensorimotor-world-model/planning/experiments/train/results/tworoom_inverse_lambda_0p1_seed0/checkpoints/last.ckpt",
    ("driftworld", "tworoom"): "/grid/klindt/home/nam/ckpt/20260824_driftworld_tworoom_lewm_fs5_lr1e-4/ckpt-step500000.pth",
    ("irasim", "tworoom"): "/grid/klindt/home/nam/ckpt/08/26/irasim_tworoom_h5_frameskip5_lr5em5/checkpoints/0300000.pt",
    ("toy-adaln", "tworoom"): "/grid/klindt/home/nam/ckpt/toy_predictors_20260825_tworoom_toy_predictors/20260825_tworoom_essential_adaln/weights_epoch_100.pt",
    ("toy-action-duplicate", "tworoom"): "/grid/klindt/home/nam/ckpt/toy_predictors_20260825_tworoom_toy_predictors/20260825_tworoom_essential_action_duplicate/weights_epoch_100.pt",
    ("toy-driftworld", "tworoom"): "/grid/klindt/home/nam/ckpt/toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_tworoom_drift_state_predictor_bs128_500k_lr1e-4/weights_step_00256900.pt",
    ("lewm", "insertpeg"): "/grid/klindt/home/nam/ckpt/checkpoints/20260910_manipulator_insert_peg_lewm/weights_epoch_10.pt",
    ("prejepa", "insertpeg"): "/grid/klindt/home/nam/ckpt/20260911_manipulator_insert_peg_prejepa_rgb_action_fs5/checkpoints/20260911_manipulator_insert_peg_prejepa_rgb_action_fs5_3173369/weights_epoch_10.pt",
    ("sensorimotor", "insertpeg"): "/grid/klindt/home/nam/ckpt/sensorimotor_world_model/20260910_manipulator_insert_peg_weak_expert_inverse_lambda_1_seed0/runs/20260910/012156/bed8c9921824/checkpoints/last.ckpt",
    ("driftworld", "insertpeg"): "/grid/klindt/home/nam/ckpt/20260910_driftworld_manipulator_insert_peg_lewm_fs5_lr5e-5/ckpt-step100000.pth",
    ("toy-adaln", "insertpeg"): "/grid/klindt/home/nam/ckpt/20260910_manipulator_insert_peg_toy_predictors/20260910_manipulator_insert_peg_essential_adaln_01/weights_epoch_100.pt",
    ("toy-action-duplicate", "insertpeg"): "/grid/klindt/home/nam/ckpt/20260910_manipulator_insert_peg_toy_predictors/20260910_manipulator_insert_peg_essential_action_duplicate_01/weights_epoch_100.pt",
    ("toy-driftworld", "insertpeg"): "/grid/klindt/home/nam/ckpt/20260910_manipulator_insert_peg_toy_predictors/20260910_manipulator_insert_peg_essential_drift_lr1e-4_01/weights_step_00411600.pt",
}

METADATA_KEYS = {
    "learning_rate", "lr", "seed", "frame_skip", "frameskip", "batch_size",
    "epochs", "num_epochs", "max_epochs", "steps", "max_steps", "weight_decay",
    "optimizer", "scheduler", "inverse_lambda", "action_dim", "state_dim",
    "context_length", "prediction_length", "image_size", "num_frames",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_selected(value: Any, prefix: str = "") -> dict[str, Any]:
    selected: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in METADATA_KEYS and isinstance(child, (str, int, float, bool, type(None))):
                selected[child_prefix] = child
            selected.update(flatten_selected(child, child_prefix))
    return selected


def nearest_config(path: Path) -> Path | None:
    for parent in [path.parent, *path.parents[:6]]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def inferred_metadata(model: str, dataset: str, path: Path) -> dict[str, Any]:
    text = "/".join(path.parts[-6:])
    metadata: dict[str, Any] = {
        "format_version": 1,
        "model": model,
        "dataset": dataset,
        "source_filename": path.name,
        "training_run": path.parent.name if path.parent.name != "checkpoints" else path.parent.parent.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    patterns = {
        "learning_rate": r"(?:^|[_/])lr(?:lr)?(\d+(?:\.\d+)?e[-m]?\d+)",
        "seed": r"seed(\d+)",
        "frame_skip": r"(?:fs|frameskip)(\d+)",
        "batch_size": r"bs(\d+)",
        "training_step": r"(?:step[_-]?)(\d+)",
        "epoch": r"epoch[_-]?(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).replace("em", "e-")
            metadata[key] = float(raw) if key == "learning_rate" else int(raw)
    config = nearest_config(path)
    if config:
        try:
            metadata["training_config"] = flatten_selected(yaml.safe_load(config.read_text()) or {})
            metadata["source_config_filename"] = config.name
        except (OSError, yaml.YAMLError) as error:
            metadata["config_read_error"] = str(error)
    return metadata


def validate() -> None:
    missing = [str(DATA_ROOT / name) for name in DATASETS.values() if not (DATA_ROOT / name).is_file()]
    missing += [path for path in CHECKPOINTS.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing artifacts:\n" + "\n".join(missing))
    names = [f"{model}_{dataset}" for model, dataset in CHECKPOINTS]
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint destination names are not unique")


def upload_datasets(api: HfApi, dataset_ids: list[str]) -> None:
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    for dataset_id in dataset_ids:
        filename = DATASETS[dataset_id]
        path = DATA_ROOT / filename
        print(f"Uploading dataset {dataset_id}: {path}", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=REPO_ID, repo_type="dataset")


def upload_models(api: HfApi) -> None:
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="anygoal-hf-metadata-") as temporary:
        temporary_dir = Path(temporary)
        for (model, dataset), source in CHECKPOINTS.items():
            path = Path(source)
            extension = ".ckpt" if path.suffix == ".ckpt" else ".pt"
            stem = f"{model}_{dataset}"
            destination = stem + extension
            metadata = inferred_metadata(model, dataset, path)
            sidecar = temporary_dir / f"{stem}.json"
            sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            print(f"Uploading checkpoint {stem}: {path}", flush=True)
            api.upload_file(path_or_fileobj=path, path_in_repo=destination, repo_id=REPO_ID, repo_type="model")
            api.upload_file(path_or_fileobj=sidecar, path_in_repo=sidecar.name, repo_id=REPO_ID, repo_type="model")


def finalize_repositories(api: HfApi) -> None:
    dataset_readme = """---\npretty_name: AnyGoal datasets\n---\n\n# AnyGoal datasets\n\nPrivate research datasets used by the AnyGoal project. The published subset intentionally excludes the base PushT, Cube, and TwoRoom datasets. Dataset definitions and validation commands are documented in the AnyGoal source repository.\n"""
    dataset_manifest = {}
    for dataset_id, filename in DATASETS.items():
        path = DATA_ROOT / filename
        dataset_manifest[dataset_id] = {
            "filename": filename,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    api.upload_file(path_or_fileobj=dataset_readme.encode(), path_in_repo="README.md", repo_id=REPO_ID, repo_type="dataset")
    api.upload_file(path_or_fileobj=(json.dumps(dataset_manifest, indent=2) + "\n").encode(), path_in_repo="manifest.json", repo_id=REPO_ID, repo_type="dataset")

    model_readme = """---\nlibrary_name: pytorch\n---\n\n# AnyGoal checkpoints\n\nPrivate research checkpoints used by the AnyGoal project. Each checkpoint has a same-named JSON sidecar containing provenance and recoverable training metadata. Base PushT/Cube/TwoRoom LeWM and PreJEPA checkpoints are intentionally excluded.\n"""
    model_manifest = {}
    for model, dataset in CHECKPOINTS:
        source = Path(CHECKPOINTS[(model, dataset)])
        extension = ".ckpt" if source.suffix == ".ckpt" else ".pt"
        stem = f"{model}_{dataset}"
        model_manifest[stem] = {"checkpoint": stem + extension, "metadata": stem + ".json"}
    api.upload_file(path_or_fileobj=model_readme.encode(), path_in_repo="README.md", repo_id=REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=(json.dumps(model_manifest, indent=2) + "\n").encode(), path_in_repo="manifest.json", repo_id=REPO_ID, repo_type="model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("validate", "datasets", "models", "finalize", "all"))
    parser.add_argument("--dataset-ids", nargs="*", choices=tuple(DATASETS), default=None)
    args = parser.parse_args()
    validate()
    print(f"Validated {len(DATASETS)} datasets and {len(CHECKPOINTS)} checkpoints", flush=True)
    if args.target == "validate":
        return
    api = HfApi()
    if args.target in {"datasets", "all"}:
        upload_datasets(api, args.dataset_ids or list(DATASETS))
    if args.target in {"models", "all"}:
        upload_models(api)
    if args.target in {"finalize", "all"}:
        finalize_repositories(api)


if __name__ == "__main__":
    main()
