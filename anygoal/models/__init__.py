"""Public model integration contracts and registry access."""

from anygoal.registry import ModelSpec, get_registry
from anygoal.types import CheckpointArtifact, ModelAdapter, PlanningAdapter

__all__ = [
    "CheckpointArtifact",
    "ModelAdapter",
    "ModelSpec",
    "PlanningAdapter",
    "get_registry",
]
