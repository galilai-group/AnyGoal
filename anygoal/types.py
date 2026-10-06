"""Stable public data contracts for AnyGoal integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    dataset_id: str | None
    valid: bool
    keys: tuple[str, ...] = ()
    rows: int | None = None
    episodes: int | None = None
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_id: str
    path: Path
    schema_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointArtifact:
    model_id: str
    trained_on_dataset_id: str
    entry_file: Path
    format_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimum interface required from an external dataset integration."""

    id: str
    family: str
    schema_version: str

    def validate(self, path: Path) -> ValidationReport: ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Capability-based interface required from an external model integration."""

    id: str
    capabilities: frozenset[str]

    def validate_dataset(self, dataset: DatasetAdapter, task: str) -> None: ...


@runtime_checkable
class PredictionAdapter(Protocol):
    """Model-neutral prediction interface used by the evaluator."""

    def rollout_errors(
        self, observations: Mapping[str, Any], action_blocks: Any, *, rollout_seed: int
    ) -> Any: ...


@runtime_checkable
class PlanningAdapter(Protocol):
    """Model-neutral candidate-cost interface used by planners."""

    def cost(self, actions: Any, observation: Mapping[str, Any], goal: Any) -> Any: ...
