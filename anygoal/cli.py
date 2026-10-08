"""Command-line entry point for datasets, models, evaluation, and visualization."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from itertools import product
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import yaml

from anygoal.config import load_config
from anygoal.collection import collect_from_cli
from anygoal.datasets.validation import validate_hdf5
from anygoal.registry import DatasetSpec, ModelSpec, get_registry
from anygoal.version import __version__


def _repository_root() -> Path:
    configured = os.environ.get("ANYGOAL_REPOSITORY")
    candidates = [Path(configured)] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[1]])
    for candidate in candidates:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "anygoal").is_dir()
            and (candidate / "scripts").is_dir()
        ):
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


def _run_collect(args: argparse.Namespace) -> int:
    output = collect_from_cli(args.dataset_id, args.arguments)
    print(output)
    return 0


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


def _run_train(args: argparse.Namespace) -> int:
    from anygoal.training import prepare_training, train

    model_options = {}
    for argument in _clean_remainder(args.arguments):
        if "=" not in argument:
            raise ValueError(
                f"Invalid model-specific option {argument!r}; expected key=value."
            )
        key, raw_value = argument.split("=", 1)
        model_options[key.replace(".", "__")] = yaml.safe_load(raw_value)
    common = dict(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        **model_options,
    )
    if args.dry_run:
        plan = prepare_training(
            args.model_id, data=args.data, task=args.task, **common
        )
        print(plan.shell_command())
        return 0
    _, result = train(
        args.model_id,
        data=args.data,
        task=args.task,
        device=args.device,
        **common,
    )
    print(result.checkpoint)
    return 0


def _planning_config(root: Path, model: ModelSpec, dataset: DatasetSpec) -> Path:
    stem = dataset.planning_stems.get(model.id, dataset.config_stem)
    return (
        root
        / "scripts"
        / "plan"
        / "config"
        / model.planning_config_dir
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
        if dataset.id in {"avoid_peg", "extractpeg"}:
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

    data_parser = subparsers.add_parser("data", help="Validate data.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    validate = data_subparsers.add_parser("validate", help="Validate common HDF5 invariants.")
    validate.add_argument("path", type=Path)
    validate.add_argument("--dataset", dest="dataset_id")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=_run_validate)

    collect = subparsers.add_parser("collect", help="Collect a generated dataset.")
    collect.add_argument("dataset_id")
    collect.add_argument("arguments", nargs=argparse.REMAINDER)
    collect.set_defaults(handler=_run_collect)

    train = subparsers.add_parser("train", help="Train a model on StableWM HDF5 data.")
    train.add_argument("model_id")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--task", required=True)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--device")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("arguments", nargs="*", help="Model-specific key=value options.")
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
