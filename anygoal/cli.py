"""Command-line entry point for datasets, models, evaluation, and visualization."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from itertools import product
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import yaml

from anygoal.config import load_config
from anygoal.datasets.validation import validate_hdf5
from anygoal.registry import DatasetSpec, ModelSpec, get_registry
from anygoal.version import __version__


BASE_DATASET = {
    "pusht": "pusht",
    "cube": "cube",
    "tworoom": "tworoom",
    "manipulator": "insertpeg",
}
TOY_DATA_CONFIG = {
    "pusht": "pusht",
    "cube": "cube_everything",
    "tworoom": "tworoom_essential",
    "manipulator": "manipulator_insert_peg_essential",
}
TOY_DRIFT_DATA_CONFIG = {
    "pusht": "pusht",
    "cube": "cube",
    "tworoom": "tworoom",
    "manipulator": "manipulator_insert_peg_essential",
}


def _repository_root() -> Path:
    configured = os.environ.get("ANYGOAL_REPOSITORY")
    candidates = [Path(configured)] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[1]])
    for candidate in candidates:
        if (candidate / "scripts").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "This operation needs the research source checkout. Set ANYGOAL_REPOSITORY "
        "to its root; registry, validation, and visualization remain wheel-only APIs."
    )


def _clean_remainder(arguments: Sequence[str]) -> list[str]:
    return [argument for argument in arguments if argument != "--"]


def _python_command(root: Path, target: str) -> list[str]:
    """Resolve a checkout-relative script or an installed Python module."""
    if target.endswith(".py"):
        return [sys.executable, str(root / target)]
    return [sys.executable, "-m", target]


def _execute(command: Sequence[str], *, dry_run: bool, cwd: Path | None = None) -> int:
    printable = shlex.join(str(part) for part in command)
    if dry_run:
        print(printable)
        return 0
    return subprocess.run([str(part) for part in command], cwd=cwd, check=False).returncode


def _default_output(spec: DatasetSpec, config: dict) -> Path:
    return Path(config["paths"]["data"]).expanduser() / spec.filename


def _collect_command(spec: DatasetSpec, preset: str) -> list[str]:
    if not spec.collector:
        raise ValueError(f"{spec.id} is imported from an upstream source, not generated.")
    command = _python_command(_repository_root(), spec.collector)
    smoke = {
        "avoidt": ["--episodes-per-goal", "1", "--num-envs", "1"],
        "stackt": ["--episodes", "1", "--num-envs", "1"],
        "pusht-antigoal": ["--episodes-per-goal", "1", "--num-envs", "1"],
        "cubehop": ["--episodes", "1", "--num-envs", "1"],
        "cubediagonal": ["--episodes", "1", "--num-envs", "1", "--gif-episodes", "0"],
        # These collectors balance left/right targets and intentionally require
        # a positive even episode count.
        "hug_box": ["--episodes", "2"],
        "extractpeg": ["--episodes", "2"],
        "tworoom_route": ["--smoke"],
    }
    return command + (smoke.get(spec.id, []) if preset == "smoke" else [])


def _run_collect(args: argparse.Namespace) -> int:
    registry = get_registry()
    spec = registry.dataset(args.dataset_id)
    config = load_config(site=args.site, experiment=args.config)
    output = (args.output or _default_output(spec, config)).expanduser().resolve()
    if spec.source == "import":
        if args.source is None:
            raise ValueError(
                f"{spec.id} requires --source because its canonical data comes from upstream."
            )
        source = args.source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {output}")
        report = validate_hdf5(source, dataset_id=spec.id)
        if not report.valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise ValueError(f"Source validation failed: {messages}")
        if args.dry_run:
            print(f"copy {source} -> {output}")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        print(output)
        return 0

    seed = spec.default_seed if args.seed is None else args.seed
    if spec.id == "cubediagonal" and args.preset == "paper":
        return _collect_cube_diagonal_paper(
            spec,
            output=output,
            seed=seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            arguments=_clean_remainder(args.arguments),
        )

    command = _collect_command(spec, args.preset)
    command.extend(["--output", str(output), "--seed", str(seed)])
    if args.overwrite:
        command.append("--overwrite")
    command.extend(_clean_remainder(args.arguments))
    return _execute(command, dry_run=args.dry_run, cwd=_repository_root())


def _collect_cube_diagonal_paper(
    spec: DatasetSpec,
    *,
    output: Path,
    seed: int,
    overwrite: bool,
    dry_run: bool,
    arguments: Sequence[str],
) -> int:
    """Reproduce the canonical 50+350 episode collection and merge recipe."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    root = _repository_root()
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_context = (
        tempfile.TemporaryDirectory(prefix="anygoal-cubediagonal-", dir=output.parent)
        if not dry_run
        else None
    )
    temporary_root = (
        Path(temporary_context.name)
        if temporary_context is not None
        else Path("/tmp/anygoal-cubediagonal-paper")
    )
    common = [
        "--num-envs",
        "5",
        "--gif-episodes",
        "0",
        "--cube-yaw-center-deg",
        "85",
        "--cube-yaw-half-width-deg",
        "5",
        "--require-diagonal-lift",
        "--min-lifted-targets",
        "2",
        "--max-abs-block-quat-xy",
        "0.2",
    ]
    parts = ((50, seed), (350, seed + 65))
    part_paths: list[Path] = []
    try:
        for episodes, part_seed in parts:
            part_path = temporary_root / f"part-{episodes}.h5"
            part_paths.append(part_path)
            command = [
                *_python_command(root, spec.collector),
                "--episodes",
                str(episodes),
                "--seed",
                str(part_seed),
                "--output",
                str(part_path),
                *common,
                *arguments,
            ]
            return_code = _execute(command, dry_run=dry_run, cwd=root)
            if return_code:
                return return_code
        merge_command = [
            sys.executable,
            str(root / "scripts/merge_cube_datasets.py"),
            "--output",
            str(output),
        ]
        if overwrite:
            merge_command.append("--overwrite")
        merge_command.extend(str(path) for path in part_paths)
        return _execute(merge_command, dry_run=dry_run, cwd=root)
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


def _run_validate(args: argparse.Namespace) -> int:
    dataset_id = None
    if args.dataset_id:
        dataset_id = get_registry().dataset(args.dataset_id).id
    report = validate_hdf5(args.path, dataset_id=dataset_id)
    payload = asdict(report)
    payload["path"] = str(payload["path"])
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"valid={report.valid} path={report.path}")
        print(f"keys={','.join(report.keys)} rows={report.rows} episodes={report.episodes}")
        for issue in report.issues:
            print(f"{issue.level}: {issue.message}")
    return 0 if report.valid else 1


def _base_training_config(root: Path, model: ModelSpec, dataset: DatasetSpec) -> Path | None:
    if dataset.id not in BASE_DATASET.values() or not model.training_config_dir:
        return None
    return (
        root
        / "configs"
        / model.training_config_dir
        / f"{dataset.config_stem}{model.training_config_suffix}.yaml"
    )


def _run_train(args: argparse.Namespace) -> int:
    registry = get_registry()
    model = registry.model(args.model_id)
    dataset = registry.dataset(args.dataset_id)
    if dataset.family not in model.families:
        raise ValueError(f"{model.id} does not declare support for {dataset.family} datasets.")
    root = _repository_root()
    command = _python_command(root, model.trainer)
    config = args.model_config or _base_training_config(root, model, dataset)
    if config is not None:
        config = config.expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        if model.trainer_config_style == "argparse":
            command.extend(["--config", str(config)])
        else:
            command.extend(["--config-path", str(config.parent), "--config-name", config.stem])
    elif dataset.id not in BASE_DATASET.values():
        raise ValueError(
            f"{dataset.id} is an evaluation dataset. Training it requires an explicit "
            "--model-config so AnyGoal does not silently train on the "
            f"{BASE_DATASET[dataset.family]} base dataset instead."
        )
    elif model.id in {"lewm", "sensorimotor"}:
        raise ValueError(
            f"{model.id} training needs --model-config pointing to the exact resolved "
            "Hydra config. No replacement defaults were invented because that could "
            "change model performance."
        )
    elif model.id == "prejepa" and dataset.id != "insertpeg":
        raise ValueError(
            "Only the repository's verified insertpeg PreJEPA recipe is selectable "
            "without an explicit configuration."
        )

    if model.id == "prejepa":
        dataset_path = Path("~/.stable_worldmodel/datasets").expanduser() / dataset.filename
        command.extend(
            [
                f"dataset_name={dataset_path}",
                "~wm.encoding.proprio",
                "wm.history_size=3",
                "wm.num_preds=1",
                "frameskip=5",
            ]
        )
    elif model.id == "toy-adaln":
        command.extend([f"data={TOY_DATA_CONFIG[dataset.family]}", "model=adaln"])
    elif model.id == "toy-action-duplicate":
        command.extend(
            [f"data={TOY_DATA_CONFIG[dataset.family]}", "model=action_duplicate"]
        )
    elif model.id == "toy-driftworld":
        command.append(f"data={TOY_DRIFT_DATA_CONFIG[dataset.family]}")
    command.extend(_clean_remainder(args.arguments))
    return _execute(command, dry_run=args.dry_run, cwd=root)


def _planning_config(root: Path, model: ModelSpec, dataset: DatasetSpec) -> Path:
    stem = dataset.planning_stems.get(model.id, dataset.config_stem)
    return (
        root
        / "configs"
        / model.planning_config_dir
        / "planning"
        / f"{stem}{model.planning_config_suffix}.yaml"
    )


def _run_eval(args: argparse.Namespace) -> int:
    registry = get_registry()
    model = registry.model(args.model_id)
    dataset = registry.dataset(args.dataset_id)
    if dataset.family not in model.families:
        raise ValueError(f"{model.id} does not declare support for {dataset.family} datasets.")
    root = _repository_root()
    extra = _clean_remainder(args.arguments)
    if args.eval_kind == "prediction":
        command = [
            *_python_command(root, model.predictor),
            "--model",
            model.evaluation_id,
            "--environment",
            dataset.evaluation_id,
            "--seed",
            str(args.seed),
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", str(args.checkpoint.expanduser().resolve())])
    elif args.eval_kind == "planning":
        config = args.model_config or _planning_config(root, model, dataset)
        if not config.is_file():
            raise FileNotFoundError(
                f"No verified planning config for this combination: {config}. "
                "Pass --model-config when an integration provides one."
            )
        if model.planner_config_style == "hydra":
            command = [
                *_python_command(root, model.planner),
                f"--config-path={config.parent}",
                f"--config-name={config.stem}",
            ]
            if model.planner_seed_style == "override":
                extra.append(f"seed={args.seed}")
        else:
            command = [*_python_command(root, model.planner), "--config", str(config)]
            if model.planner_seed_style == "argparse":
                command.extend(["--seed", str(args.seed)])
            elif model.planner_seed_style == "triple":
                extra.extend(
                    [
                        f"seed.task={args.seed}",
                        f"seed.policy={args.seed}",
                        f"model.rollout_seed={args.seed}",
                    ]
                )
        if dataset.id in {"hug_box", "extractpeg"}:
            extra.append(f"eval.dataset_name={Path(dataset.filename).stem}")
        if args.checkpoint:
            extra.append(
                f"{model.planning_checkpoint_key}="
                f"{args.checkpoint.expanduser().resolve()}"
            )
    else:
        if model.id not in {"lewm", "sensorimotor", "toy-adaln"}:
            raise ValueError(
                "The preserved counterfactual protocol currently supports lewm, "
                "sensorimotor, and toy-adaln."
            )
        if dataset.id not in {"pusht", "cube", "tworoom", "insertpeg"}:
            raise ValueError("Counterfactual evaluation is defined only for base datasets.")
        counterfactual_model = {
            "lewm": "lewm",
            "sensorimotor": "sensorimotor",
            "toy-adaln": "adaln_predictor",
        }[model.id]
        command = [
            sys.executable,
            "-m",
            "anygoal.evaluation.counterfactual_runner",
            "--model",
            counterfactual_model,
            "--environment",
            dataset.evaluation_id,
            "--seed",
            str(args.seed),
        ]
    command.extend(extra)
    return _execute(command, dry_run=args.dry_run, cwd=root)


def _run_visualize(args: argparse.Namespace) -> int:
    spec = get_registry().visualizer(args.visualizer_id)
    if args.visualizer_id in {"dataset-gif", "metrics-plot"} or spec.runner:
        if args.input is None or args.output is None:
            raise ValueError(f"{args.visualizer_id} requires --input and --output.")
        if args.dry_run and not spec.runner:
            print(
                shlex.join(
                    [
                        "anygoal",
                        "visualize",
                        args.visualizer_id,
                        "--input",
                        str(args.input),
                        "--output",
                        str(args.output),
                    ]
                )
            )
            return 0
    if args.visualizer_id == "dataset-gif":
        from anygoal.visualization import render_dataset_gif

        output = render_dataset_gif(
            args.input,
            args.output,
            start=args.start,
            frames=args.frames,
            fps=args.fps,
            overwrite=args.overwrite,
        )
    elif args.visualizer_id == "metrics-plot":
        from anygoal.visualization import plot_metrics

        output = plot_metrics(
            args.input, args.output, title=args.title, overwrite=args.overwrite
        )
    elif spec.runner:
        command = [
            *_python_command(_repository_root(), spec.runner),
            spec.input_option,
            str(args.input),
            spec.output_option,
            str(args.output),
            *_clean_remainder(args.arguments),
        ]
        return _execute(command, dry_run=args.dry_run, cwd=_repository_root())
    else:
        raise ValueError(f"{spec.id} does not declare an executable runner.")
    print(output)
    return 0


def _sbatch_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    config = load_config(site=args.site, experiment=args.config)
    slurm = config["slurm"]
    try:
        preset = slurm["presets"][args.preset]
    except KeyError as error:
        choices = ", ".join(sorted(slurm["presets"]))
        raise ValueError(f"Unknown resource preset {args.preset!r}; choose {choices}.") from error
    root = _repository_root()
    log_dir = Path(slurm["log_dir"])
    if not log_dir.is_absolute():
        log_dir = root / log_dir
    command = ["sbatch", f"--job-name={args.job_name}"]
    options = {
        "partition": slurm.get("partition"),
        "account": slurm.get("account"),
        "constraint": slurm.get("constraint"),
        "qos": preset.get("qos"),
        "cpus-per-task": preset.get("cpus"),
        "mem": preset.get("memory"),
        "time": preset.get("time"),
    }
    for key, value in options.items():
        if value is not None:
            command.append(f"--{key}={value}")
    if preset.get("gpus"):
        gpu_type = preset.get("gpu_type")
        gres = f"gpu:{gpu_type}:{preset['gpus']}" if gpu_type else f"gpu:{preset['gpus']}"
        command.append(f"--gres={gres}")
    if slurm.get("exclude"):
        command.append("--exclude=" + ",".join(slurm["exclude"]))
    if args.array:
        command.append(f"--array={args.array}")
    exported = ["ALL", f"ANYGOAL_PYTHON_ENV={slurm.get('python_env') or ''}"]
    wandb = config.get("wandb", {})
    if wandb.get("entity"):
        exported.append(f"WANDB_ENTITY={wandb['entity']}")
    if wandb.get("project"):
        exported.append(f"WANDB_PROJECT={wandb['project']}")
    if wandb.get("mode"):
        exported.append(f"WANDB_MODE={wandb['mode']}")
    command.extend(
        [
            f"--output={log_dir}/%x-%A_%a.out",
            f"--error={log_dir}/%x-%A_%a.err",
            "--export=" + ",".join(exported),
            str(root / "slurm/run.sh"),
        ]
    )
    payload = _clean_remainder(args.command)
    if not payload:
        raise ValueError("Pass the command after `--`, for example `-- anygoal list models`.")
    command.extend(payload)
    return command, log_dir


def _run_submit(args: argparse.Namespace) -> int:
    command, log_dir = _sbatch_command(args)
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    return _execute(command, dry_run=args.dry_run, cwd=_repository_root())


def _run_list(args: argparse.Namespace) -> int:
    registry = get_registry()
    collection = getattr(registry, args.kind)
    for item_id in sorted(collection):
        item = collection[item_id]
        if args.kind == "datasets":
            print(f"{item.id:22} family={item.family:11} source={item.source:9} {item.filename}")
        elif args.kind == "models":
            print(f"{item.id:22} capabilities={','.join(sorted(item.capabilities))}")
        else:
            required = "; ".join(item.required_inputs)
            print(f"{item.id:22} requires: {required}; output: {item.output}")
    return 0


def _run_reproduce(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction manifest schema version.")
    for matrix in manifest.get("matrices", ()):
        evaluation = matrix["evaluation"]
        preset = matrix.get("resource_preset", "gpu-eval")
        for model_id, dataset_id, seed in product(
            matrix["models"], matrix["datasets"], matrix.get("seeds", [0])
        ):
            payload = [
                "anygoal",
                "eval",
                evaluation,
                "--model",
                str(model_id),
                "--dataset",
                str(dataset_id),
                "--seed",
                str(seed),
            ]
            if args.submit:
                submit_args = argparse.Namespace(
                    preset=preset,
                    job_name=f"{evaluation}-{model_id}-{dataset_id}-s{seed}",
                    array=None,
                    site=args.site,
                    config=None,
                    dry_run=args.dry_run,
                    command=payload,
                )
                return_code = _run_submit(submit_args)
            else:
                return_code = _execute(payload, dry_run=True)
            if return_code:
                return return_code
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anygoal", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered integrations.")
    list_parser.add_argument("kind", choices=("datasets", "models", "visualizers"))
    list_parser.set_defaults(handler=_run_list)

    config_parser = subparsers.add_parser("config", help="Inspect merged configuration.")
    config_parser.add_argument("action", choices=("show",))
    config_parser.add_argument("--site", type=Path)
    config_parser.add_argument("--config", type=Path, help="Experiment config.")
    config_parser.set_defaults(
        handler=lambda args: print(
            yaml.safe_dump(load_config(site=args.site, experiment=args.config), sort_keys=False)
        )
        or 0
    )

    data_parser = subparsers.add_parser("data", help="Collect, import, or validate data.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    collect = data_subparsers.add_parser("collect", help="Collect or import a registered dataset.")
    collect.add_argument("dataset_id")
    collect.add_argument("--preset", choices=("smoke", "paper"), default="paper")
    collect.add_argument("--source", type=Path, help="Required for imported base datasets.")
    collect.add_argument("--output", type=Path)
    collect.add_argument(
        "--seed",
        type=int,
        help="Override the registered publication seed.",
    )
    collect.add_argument("--site", type=Path)
    collect.add_argument("--config", type=Path, help="Experiment config.")
    collect.add_argument("--overwrite", action="store_true")
    collect.add_argument("--dry-run", action="store_true")
    collect.add_argument("arguments", nargs="*", help="Collector options after --.")
    collect.set_defaults(handler=_run_collect)
    validate = data_subparsers.add_parser("validate", help="Validate common HDF5 invariants.")
    validate.add_argument("path", type=Path)
    validate.add_argument("--dataset", dest="dataset_id")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=_run_validate)

    train = subparsers.add_parser("train", help="Run an unchanged registered trainer.")
    train.add_argument("model_id")
    train.add_argument("--dataset", dest="dataset_id", required=True)
    train.add_argument("--model-config", type=Path)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("arguments", nargs="*", help="Trainer overrides after --.")
    train.set_defaults(handler=_run_train)

    evaluation = subparsers.add_parser("eval", help="Run a common evaluation protocol.")
    evaluation_subparsers = evaluation.add_subparsers(dest="eval_kind", required=True)
    for kind in ("prediction", "planning", "counterfactual"):
        child = evaluation_subparsers.add_parser(kind)
        child.add_argument("--model", dest="model_id", required=True)
        child.add_argument("--dataset", dest="dataset_id", required=True)
        child.add_argument("--checkpoint", type=Path)
        child.add_argument("--model-config", type=Path)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--dry-run", action="store_true")
        child.add_argument("arguments", nargs="*", help="Evaluator options after --.")
        child.set_defaults(handler=_run_eval)

    visualizer_lines = []
    for visualizer in get_registry().visualizers.values():
        required = "; ".join(visualizer.required_inputs)
        optional = "; ".join(visualizer.optional_inputs) or "none"
        visualizer_lines.append(
            f"  {visualizer.id}: requires {required}; optional {optional}; "
            f"output {visualizer.output}"
        )
    visualize = subparsers.add_parser(
        "visualize",
        help="Visualize registered artifacts; use `list visualizers` for input contracts.",
        description="Visualize a registered dataset or result artifact.",
        epilog="Required inputs:\n" + "\n".join(visualizer_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    visualize.add_argument("visualizer_id", choices=tuple(get_registry().visualizers))
    visualize.add_argument("--input", type=Path)
    visualize.add_argument("--output", type=Path)
    visualize.add_argument("--start", type=int, default=0)
    visualize.add_argument("--frames", type=int, default=100)
    visualize.add_argument("--fps", type=float, default=10.0)
    visualize.add_argument("--title")
    visualize.add_argument("--overwrite", action="store_true")
    visualize.add_argument("--dry-run", action="store_true")
    visualize.add_argument("arguments", nargs="*")
    visualize.set_defaults(handler=_run_visualize)

    submit = subparsers.add_parser("submit", help="Submit any AnyGoal command with a site profile.")
    submit.add_argument("--preset", default="gpu-train")
    submit.add_argument("--job-name", default="anygoal")
    submit.add_argument("--array")
    submit.add_argument("--site", type=Path)
    submit.add_argument("--config", type=Path, help="Experiment config.")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    submit.set_defaults(handler=_run_submit)

    reproduce = subparsers.add_parser(
        "reproduce", help="Expand a versioned experiment manifest."
    )
    reproduce.add_argument("manifest", type=Path)
    reproduce.add_argument("--submit", action="store_true")
    reproduce.add_argument("--site", type=Path)
    reproduce.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print sbatch commands without submission (default: true).",
    )
    reproduce.set_defaults(handler=_run_reproduce)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BrokenPipeError:
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
