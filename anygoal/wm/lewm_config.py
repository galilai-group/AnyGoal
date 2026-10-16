"""Shared LeWM architecture defaults for planning and rollout loaders."""

from omegaconf import DictConfig, OmegaConf

from anygoal.resources import resource_path


def resolve_model_config(config: DictConfig) -> DictConfig:
    """Merge model defaults with task dimensions and explicit user overrides."""
    defaults = OmegaConf.load(resource_path("models", "lewm.yaml"))
    # Resolve task-local interpolations while the model node still has its parent.
    overrides = OmegaConf.to_container(config, resolve=True)
    resolved = OmegaConf.merge(defaults, overrides)
    OmegaConf.to_container(resolved, resolve=True, throw_on_missing=True)
    return resolved
