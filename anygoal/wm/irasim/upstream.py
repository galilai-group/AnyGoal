"""Lazy loading and adaptation of vendored IRASim components."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

from torch import nn

from .data import get_irasim_hdf5_datasets


def irasim_source_root() -> Path:
    default = Path(__file__).resolve().parents[3] / "worldmodels" / "IRASim"
    root = Path(os.environ.get("IRASIM_SOURCE_ROOT", default)).expanduser().resolve()
    if not (root / "models" / "irasim.py").is_file():
        raise FileNotFoundError(
            "IRASim source was not found. Set IRASIM_SOURCE_ROOT to the "
            f"directory containing models/irasim.py: {root}"
        )
    return root


def _module_is_below(module, root: Path) -> bool:
    path = getattr(module, "__file__", None)
    if path is not None:
        return Path(path).resolve().is_relative_to(root)
    package_paths = getattr(module, "__path__", ())
    return any(Path(path).resolve().is_relative_to(root) for path in package_paths)


def _source_package(name: str, directory: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        if not _module_is_below(existing, irasim_source_root()):
            raise ImportError(
                f"Top-level module {name!r} is already loaded from another package; "
                "start a clean process before loading IRASim."
            )
        return existing
    package = types.ModuleType(name)
    package.__file__ = str(directory / "__init__.py")
    package.__path__ = [str(directory)]
    package.__package__ = name
    package.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=True
    )
    sys.modules[name] = package
    return package


@lru_cache(maxsize=1)
def get_irasim_model_module():
    """Load only IRASim's model file, avoiding unused upstream baselines."""
    root = irasim_source_root()
    _source_package("models", root / "models")
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module("models.irasim")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting


def _replace_linear_input(layer: nn.Linear, input_dim: int) -> nn.Linear:
    replacement = nn.Linear(
        input_dim,
        layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    # Match IRASim.initialize_weights(), which has already run in __init__.
    nn.init.xavier_uniform_(replacement.weight)
    if replacement.bias is not None:
        nn.init.constant_(replacement.bias, 0)
    return replacement


def adapt_irasim_action_frameskip(
    transformer: nn.Module,
    *,
    action_dim: int,
    action_frameskip: int,
) -> nn.Module:
    """Widen only IRASim's first action projection for blocked actions.

    The ``action_frameskip == 1`` branch returns the original module without
    replacing or reinitializing any layer.
    """
    if action_dim < 1 or action_frameskip < 1:
        raise ValueError("action_dim and action_frameskip must be positive.")
    expected_input_dim = int(action_dim) * int(action_frameskip)
    original_state_dim = int(transformer.state_dim)
    if original_state_dim != int(action_dim):
        raise ValueError(
            f"IRASim was built for {original_state_dim}-D actions, but the "
            f"adapter was configured for {action_dim}-D actions."
        )
    if action_frameskip == 1:
        return transformer

    if hasattr(transformer, "embed_arm_state"):
        projection = transformer.embed_arm_state
        if not isinstance(projection, nn.Linear):
            raise TypeError("IRASim embed_arm_state is not a Linear layer.")
        transformer.embed_arm_state = _replace_linear_input(
            projection, expected_input_dim
        )
    elif hasattr(transformer.embed_state, "fc1"):
        projection = transformer.embed_state.fc1
        if not isinstance(projection, nn.Linear):
            raise TypeError("IRASim embed_state.fc1 is not a Linear layer.")
        transformer.embed_state.fc1 = _replace_linear_input(
            projection, expected_input_dim
        )
    else:
        raise TypeError("Could not locate IRASim's first action projection.")
    transformer.state_dim = expected_input_dim
    return transformer


def adapt_irasim_action_dim(transformer: nn.Module, *, action_dim: int) -> nn.Module:
    """Widen IRASim's Language-Table action input without editing vendored code."""
    if action_dim < 1:
        raise ValueError("action_dim must be positive.")
    original_state_dim = int(transformer.state_dim)
    if original_state_dim == int(action_dim):
        return transformer
    if not hasattr(transformer, "embed_arm_state"):
        raise ValueError(
            "Only the vendored Language-Table action encoder supports a "
            "non-native action dimension."
        )
    projection = transformer.embed_arm_state
    if not isinstance(projection, nn.Linear):
        raise TypeError("IRASim embed_arm_state is not a Linear layer.")
    transformer.embed_arm_state = _replace_linear_input(projection, int(action_dim))
    transformer.state_dim = int(action_dim)
    return transformer


def create_irasim_transformer(args, *, action_dim: int = 2):
    """Construct the upstream IRASim model and apply the frame-skip adapter."""
    module = get_irasim_model_module()
    model_name = str(args.model)
    try:
        builder = module.IRASim_models[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported upstream IRASim model: {model_name}") from error
    args.latent_size = [int(size) // 8 for size in args.video_size]
    transformer = builder(
        input_size=args.latent_size,
        num_frames=int(args.num_frames),
        learn_sigma=bool(args.learn_sigma),
        extras=int(args.extras),
        attention_mode=str(args.attention_mode),
        args=args,
    )
    transformer = adapt_irasim_action_dim(transformer, action_dim=action_dim)
    return adapt_irasim_action_frameskip(
        transformer,
        action_dim=action_dim,
        action_frameskip=int(args.get("frameskip", 1)),
    )


@lru_cache(maxsize=1)
def get_irasim_pipeline_class():
    """Load IRASim's original inference pipeline lazily."""
    root = irasim_source_root()
    _source_package("sample", root / "sample")
    sys.path.insert(0, str(root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module("sample.pipeline_trajectory2videogen")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(root))
    return module.Trajectory2VideoGenPipeline


def load_irasim_upstream_train_module():
    """Load the original training loop with H5/model factories bound at import."""
    module_name = "anygoal.wm._irasim_upstream_train"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    root = irasim_source_root()
    models_package = _source_package("models", root / "models")
    models_package.get_models = create_irasim_transformer

    existing_dataset = sys.modules.get("dataset")
    if existing_dataset is not None and not getattr(
        existing_dataset, "_anygoal_irasim_adapter", False
    ):
        raise ImportError(
            "A different top-level 'dataset' module is already loaded; start a "
            "clean process before loading IRASim training."
        )
    dataset_package = types.ModuleType("dataset")
    dataset_package.__file__ = str(root / "dataset" / "__init__.py")
    dataset_package.__path__ = [str(root / "dataset")]
    dataset_package.__package__ = "dataset"
    dataset_package.__spec__ = importlib.machinery.ModuleSpec(
        "dataset", loader=None, is_package=True
    )
    dataset_package.get_dataset = get_irasim_hdf5_datasets
    dataset_package._anygoal_irasim_adapter = True
    sys.modules["dataset"] = dataset_package

    spec = importlib.util.spec_from_file_location(module_name, root / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IRASim training loop from {root}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(root))
    return module
