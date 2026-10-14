"""Lightweight registries backed by the packaged AnyGoal catalog."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, resources
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    aliases: tuple[str, ...]
    family: str
    filename: str
    environment: str
    evaluation_id: str
    config_stem: str
    planning_stems: Mapping[str, str]
    schema: str
    source: str
    collector: str | None = None
    default_seed: int = 0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    aliases: tuple[str, ...]
    capabilities: frozenset[str]
    families: frozenset[str]
    evaluation_id: str
    trainer: str
    rollout_runner: str
    planner: str
    training_config_dir: str | None
    training_config_suffix: str
    trainer_config_style: str
    training_profiles: Mapping[str, str | None]
    planning_config_dir: str
    planning_config_suffix: str
    planner_config_style: str
    planner_seed_style: str
    planning_checkpoint_key: str
    planning_training_config_key: str | None


@dataclass(frozen=True)
class VisualizerSpec:
    id: str
    description: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    output: str
    runner: str | None = None
    input_option: str = "--input"
    output_option: str = "--output"


def _catalog() -> Mapping[str, Any]:
    resource = resources.files("anygoal.resources").joinpath("catalog.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported AnyGoal catalog schema version.")
    return catalog


def _load_plugins(group: str) -> Iterable[Any]:
    try:
        entry_points = metadata.entry_points(group=group)
    except TypeError:  # Python 3.10 compatibility.
        entry_points = metadata.entry_points().get(group, ())
    for entry_point in entry_points:
        yield entry_point.load()()


class Registry:
    """Resolve canonical IDs and external entry-point integrations."""

    def __init__(self) -> None:
        catalog = _catalog()
        self.datasets = {
            item_id: DatasetSpec(
                id=item_id,
                aliases=tuple(item.get("aliases", ())),
                family=item["family"],
                filename=item["filename"],
                environment=item["environment"],
                evaluation_id=item.get("evaluation_id", item_id),
                config_stem=item.get("config_stem", item_id),
                planning_stems=dict(item.get("planning_stems", {})),
                schema=item["schema"],
                source=item["source"],
                collector=item.get("collector"),
                default_seed=int(item.get("default_seed", 0)),
            )
            for item_id, item in catalog["datasets"].items()
        }
        self.models = {
            item_id: ModelSpec(
                id=item_id,
                aliases=tuple(item.get("aliases", ())),
                capabilities=frozenset(item["capabilities"]),
                families=frozenset(item["families"]),
                evaluation_id=item.get("evaluation_id", item_id),
                trainer=item["trainer"],
                rollout_runner=item["rollout_runner"],
                planner=item["planner"],
                training_config_dir=item.get("training_config_dir"),
                training_config_suffix=item.get("training_config_suffix", ""),
                trainer_config_style=item.get("trainer_config_style", "hydra"),
                training_profiles=dict(item.get("training_profiles", {})),
                planning_config_dir=item.get("planning_config_dir", item_id),
                planning_config_suffix=item.get("planning_config_suffix", ""),
                planner_config_style=item.get("planner_config_style", "hydra"),
                planner_seed_style=item.get("planner_seed_style", "override"),
                planning_checkpoint_key=item.get(
                    "planning_checkpoint_key", "model.checkpoint"
                ),
                planning_training_config_key=item.get(
                    "planning_training_config_key"
                ),
            )
            for item_id, item in catalog["models"].items()
        }
        self.visualizers = {
            item_id: VisualizerSpec(
                id=item_id,
                description=item["description"],
                required_inputs=tuple(item["required_inputs"]),
                optional_inputs=tuple(item.get("optional_inputs", ())),
                output=item["output"],
                runner=item.get("runner"),
                input_option=item.get("input_option", "--input"),
                output_option=item.get("output_option", "--output"),
            )
            for item_id, item in catalog["visualizers"].items()
        }
        self._extend("anygoal.datasets", self.datasets)
        self._extend("anygoal.models", self.models)
        self._extend("anygoal.visualizers", self.visualizers)
        self._validate_aliases()
        self._validate_specs()

    @staticmethod
    def _extend(group: str, destination: dict[str, Any]) -> None:
        for item in _load_plugins(group):
            if item.id in destination:
                raise ValueError(f"Duplicate registry ID {item.id!r} from {group}.")
            destination[item.id] = item

    def _validate_aliases(self) -> None:
        for collection_name in ("datasets", "models", "visualizers"):
            collection = getattr(self, collection_name)
            seen = set(collection)
            for item in collection.values():
                for alias in getattr(item, "aliases", ()):
                    if alias in seen:
                        raise ValueError(
                            f"Duplicate {collection_name} ID or alias: {alias!r}."
                        )
                    seen.add(alias)

    def _validate_specs(self) -> None:
        valid_capabilities = {"train", "rollout", "plan"}
        for model in self.models.values():
            unknown = model.capabilities - valid_capabilities
            if unknown:
                raise ValueError(
                    f"Model {model.id!r} has unknown capabilities: {sorted(unknown)}"
                )
            if model.trainer_config_style not in {"argparse", "hydra"}:
                raise ValueError(
                    f"Model {model.id!r} has invalid trainer_config_style."
                )
            if model.planner_config_style not in {"argparse", "hydra"}:
                raise ValueError(
                    f"Model {model.id!r} has invalid planner_config_style."
                )
            if model.planner_seed_style not in {"argparse", "override", "triple"}:
                raise ValueError(f"Model {model.id!r} has invalid planner_seed_style.")
            unknown_profiles = set(model.training_profiles) - set(self.datasets)
            if unknown_profiles:
                raise ValueError(
                    f"Model {model.id!r} has training profiles for unknown tasks: "
                    f"{sorted(unknown_profiles)}"
                )
            if model.training_profiles and "train" not in model.capabilities:
                raise ValueError(
                    f"Model {model.id!r} declares profiles without train capability."
                )
        for dataset in self.datasets.values():
            if dataset.source not in {"generated", "external"}:
                raise ValueError(f"Dataset {dataset.id!r} has invalid source.")
            if dataset.source == "generated" and not dataset.collector:
                raise ValueError(f"Generated dataset {dataset.id!r} needs a collector.")
        for visualizer in self.visualizers.values():
            if not visualizer.required_inputs or not visualizer.output:
                raise ValueError(
                    f"Visualizer {visualizer.id!r} needs input and output contracts."
                )

    @staticmethod
    def _resolve(collection: Mapping[str, Any], item_id: str) -> Any:
        if item_id in collection:
            return collection[item_id]
        for item in collection.values():
            if item_id in getattr(item, "aliases", ()):
                return item
        choices = ", ".join(sorted(collection))
        raise KeyError(f"Unknown ID {item_id!r}. Available IDs: {choices}")

    def dataset(self, dataset_id: str) -> DatasetSpec:
        return self._resolve(self.datasets, dataset_id)

    def model(self, model_id: str) -> ModelSpec:
        return self._resolve(self.models, model_id)

    def visualizer(self, visualizer_id: str) -> VisualizerSpec:
        return self._resolve(self.visualizers, visualizer_id)


registry = Registry()


def get_registry() -> Registry:
    return registry
