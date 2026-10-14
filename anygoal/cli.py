"""Command-line entry point for data, model sources, training, and evaluation."""

from __future__ import annotations

import argparse
import json
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
from anygoal.paths import resolve_path
from anygoal.registry import DatasetSpec, ModelSpec, get_registry
from anygoal.resources import resource_path
from anygoal.version import __version__


def _clean_remainder(arguments: Sequence[str]) -> list[str]:
    return [argument for argument in arguments if argument != "--"]


def _python_command(target: str) -> list[str]:
    """Resolve an explicit script or an installed Python module."""
    if target.endswith(".py"):
        return [sys.executable, str(Path(target).expanduser().resolve())]
    return [sys.executable, "-m", target]


def _hydra_override(config: Path, key: str, value: str) -> str:
    """Use Hydra's append syntax only when a key is absent from the config."""
    content = yaml.safe_load(config.read_text()) or {}
    current = content
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return f"+{key}={value}"
        current = current[part]
    return f"{key}={value}"


def _execute(command: Sequence[str], *, dry_run: bool, cwd: Path | None = None) -> int:
    printable = shlex.join(str(part) for part in command)
    if dry_run:
        print(printable)
        return 0
    return subprocess.run(
        [str(part) for part in command], cwd=cwd, check=False
    ).returncode


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
        print(
            f"keys={','.join(report.keys)} rows={report.rows} episodes={report.episodes}"
        )
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
        plan = prepare_training(args.model_id, data=args.data, task=args.task, **common)
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


def _planning_config(model: ModelSpec, dataset: DatasetSpec) -> Path:
    stem = dataset.planning_stems.get(model.id, dataset.config_stem)
    return resource_path(
        "plan",
        model.planning_config_dir,
        f"{stem}{model.planning_config_suffix}.yaml",
    )


def _planning_training_config(
    model: ModelSpec, dataset: DatasetSpec
) -> Path | None:
    if model.planning_training_config_key is None:
        return None
    evaluation = yaml.safe_load(
        resource_path("eval", "rollout.yaml").read_text(encoding="utf-8")
    )
    value = evaluation["models"][model.evaluation_id][dataset.family].get(
        "training_config"
    )
    return resolve_path(value).resolve() if value else None


def _run_eval(args: argparse.Namespace) -> int:
    registry = get_registry()
    model = registry.model(args.model_id)
    dataset = registry.dataset(args.dataset_id)
    if dataset.family not in model.families:
        raise ValueError(
            f"{model.id} does not declare support for {dataset.family} datasets."
        )
    if args.download:
        from anygoal.artifacts import download_checkpoint, download_dataset

        download_dataset(dataset)
        if args.checkpoint is None:
            args.checkpoint = download_checkpoint(model, dataset)
    extra = _clean_remainder(args.arguments)
    if args.eval_kind == "rollout":
        command = [
            *_python_command(model.rollout_runner),
            "--model",
            model.evaluation_id,
            "--environment",
            dataset.evaluation_id,
            "--seed",
            str(args.seed),
        ]
        if args.checkpoint:
            command.extend(
                ["--checkpoint", str(args.checkpoint.expanduser().resolve())]
            )
    else:
        config = args.model_config or _planning_config(model, dataset)
        if not config.is_file():
            raise FileNotFoundError(
                f"No verified planning config for this combination: {config}. "
                "Pass --model-config when an integration provides one."
            )
        if model.planner_config_style == "hydra":
            command = [
                *_python_command(model.planner),
                f"--config-path={config.parent}",
                f"--config-name={config.stem}",
            ]
            if model.planner_seed_style == "override":
                extra.append(f"seed={args.seed}")
        else:
            command = [*_python_command(model.planner), "--config", str(config)]
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
                _hydra_override(
                    config,
                    model.planning_checkpoint_key,
                    str(args.checkpoint.expanduser().resolve()),
                )
            )
        training_config = _planning_training_config(model, dataset)
        if training_config is not None:
            extra.append(
                _hydra_override(
                    config,
                    model.planning_training_config_key,
                    str(training_config),
                )
            )
    command.extend(extra)
    return _execute(command, dry_run=args.dry_run)


def _run_visualize(args: argparse.Namespace) -> int:
    spec = get_registry().visualizer(args.visualizer_id)
    if spec.runner:
        if args.input is None or args.output is None:
            raise ValueError(f"{args.visualizer_id} requires --input and --output.")
        command = [
            *_python_command(spec.runner),
            spec.input_option,
            str(args.input),
            spec.output_option,
            str(args.output),
            *_clean_remainder(args.arguments),
        ]
        return _execute(command, dry_run=args.dry_run)
    raise ValueError(f"{spec.id} does not declare an executable runner.")


def _run_list(args: argparse.Namespace) -> int:
    registry = get_registry()
    collection = getattr(registry, args.kind)
    for item_id in sorted(collection):
        item = collection[item_id]
        if args.kind == "datasets":
            print(
                f"{item.id:22} family={item.family:11} source={item.source:9} {item.filename}"
            )
        elif args.kind == "models":
            print(f"{item.id:22} capabilities={','.join(sorted(item.capabilities))}")
        else:
            required = "; ".join(item.required_inputs)
            print(f"{item.id:22} requires: {required}; output: {item.output}")
    return 0


def _run_models(args: argparse.Namespace) -> int:
    from anygoal.sources import get_sources, install_source, source_status

    source_ids = sorted({spec.id for spec in get_sources().values()})
    selected = source_ids if getattr(args, "all", False) else [args.model_id]
    for model_id in selected:
        status = (
            install_source(model_id)
            if args.models_command == "install"
            else source_status(model_id)
        )
        revision = status.revision or "-"
        print(
            f"{status.source.id:14} {status.state:10} "
            f"revision={revision} path={status.path}"
        )
    return 0


def _run_config(args: argparse.Namespace) -> int:
    config = load_config(site=args.site, experiment=args.config)
    if args.action == "show":
        print(yaml.safe_dump(config, sort_keys=False))
        return 0
    if args.path_name is None:
        raise ValueError("config path requires a path name.")
    print(config["paths"][args.path_name])
    return 0


def _run_artifacts(args: argparse.Namespace) -> int:
    from anygoal.artifacts import (
        download_all,
        download_pair,
    )

    if args.all:
        if args.model_id or args.dataset_id:
            raise ValueError("--all cannot be combined with --model or --dataset.")
        if not args.datasets and not args.checkpoints:
            raise ValueError("--all requires --datasets and/or --checkpoints.")
        downloaded = download_all(
            datasets=args.datasets, checkpoints=args.checkpoints
        )
    else:
        if args.datasets or args.checkpoints:
            raise ValueError("--datasets and --checkpoints require --all.")
        if not args.model_id or not args.dataset_id:
            raise ValueError("Specify both --model and --dataset, or use --all.")
        downloaded = download_pair(args.model_id, args.dataset_id)
    for path in downloaded:
        print(path)
    return 0


def _run_reproduce(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction manifest schema version.")
    for matrix in manifest.get("matrices", ()):
        evaluation = matrix["evaluation"]
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
            return_code = _execute(payload, dry_run=True)
            if return_code:
                return return_code
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anygoal", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered integrations.")
    list_parser.add_argument("kind", choices=("datasets", "models", "visualizers"))
    list_parser.set_defaults(handler=_run_list)

    models_parser = subparsers.add_parser(
        "models", help="Install or inspect pinned original model sources."
    )
    models_subparsers = models_parser.add_subparsers(
        dest="models_command", required=True
    )
    models_list = models_subparsers.add_parser("list", help="List source status.")
    models_list.set_defaults(handler=_run_models, all=True)
    models_status = models_subparsers.add_parser(
        "status", help="Show one source status."
    )
    models_status.add_argument("model_id")
    models_status.set_defaults(handler=_run_models, all=False)
    models_install = models_subparsers.add_parser(
        "install", help="Download a pinned original source checkout."
    )
    install_target = models_install.add_mutually_exclusive_group(required=True)
    install_target.add_argument("model_id", nargs="?")
    install_target.add_argument("--all", action="store_true")
    models_install.set_defaults(handler=_run_models)

    artifacts_parser = subparsers.add_parser(
        "artifacts", help="Download published datasets and checkpoints."
    )
    artifacts_subparsers = artifacts_parser.add_subparsers(
        dest="artifacts_command", required=True
    )
    artifacts_download = artifacts_subparsers.add_parser(
        "download", help="Download one evaluation pair or all published artifacts."
    )
    artifacts_download.add_argument("--model", dest="model_id")
    artifacts_download.add_argument("--dataset", dest="dataset_id")
    artifacts_download.add_argument("--datasets", action="store_true")
    artifacts_download.add_argument("--checkpoints", action="store_true")
    artifacts_download.add_argument("--all", action="store_true")
    artifacts_download.set_defaults(handler=_run_artifacts)

    config_parser = subparsers.add_parser(
        "config", help="Inspect merged configuration."
    )
    config_parser.add_argument("action", choices=("show", "path"))
    config_parser.add_argument(
        "path_name",
        nargs="?",
        choices=("home", "data", "checkpoints", "outputs", "sources", "cache"),
    )
    config_parser.add_argument("--site", type=Path)
    config_parser.add_argument("--config", type=Path, help="Experiment config.")
    config_parser.set_defaults(handler=_run_config)

    data_parser = subparsers.add_parser("data", help="Validate data.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    validate = data_subparsers.add_parser(
        "validate", help="Validate common HDF5 invariants."
    )
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
    for kind in ("rollout", "planning"):
        child = evaluation_subparsers.add_parser(kind)
        child.add_argument("--model", dest="model_id", required=True)
        child.add_argument("--dataset", dest="dataset_id", required=True)
        child.add_argument("--checkpoint", type=Path)
        child.add_argument("--model-config", type=Path)
        child.add_argument(
            "--download",
            action="store_true",
            help="Download the missing dataset and checkpoint from Hugging Face.",
        )
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
    visualize.add_argument("--dry-run", action="store_true")
    visualize.add_argument("arguments", nargs="*")
    visualize.set_defaults(handler=_run_visualize)

    reproduce = subparsers.add_parser(
        "reproduce", help="Expand a versioned experiment manifest."
    )
    reproduce.add_argument("manifest", type=Path)
    reproduce.set_defaults(handler=_run_reproduce)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BrokenPipeError:
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
