"""Lazy loading and construction of vendored DriftWorld components."""

from __future__ import annotations

import copy
import importlib
import importlib.machinery
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


def driftworld_source_root() -> Path:
    default = Path(__file__).resolve().parents[3] / "worldmodels" / "driftworld"
    root = Path(os.environ.get("DRIFTWORLD_SOURCE_ROOT", default)).expanduser()
    root = root.resolve() / "pusht_driftworld"
    if not (root / "drifting_denoiser_multi.py").is_file():
        raise FileNotFoundError(
            "DriftWorld Push-T source was not found. Set DRIFTWORLD_SOURCE_ROOT "
            f"to the vendored driftworld directory: {root.parent}"
        )
    return root


def robomimic_driftworld_source_root() -> Path:
    default = Path(__file__).resolve().parents[3] / "worldmodels" / "driftworld"
    root = Path(os.environ.get("DRIFTWORLD_SOURCE_ROOT", default)).expanduser()
    root = root.resolve() / "robomimic_driftworld"
    if not (root / "drifting_denoiser_multi.py").is_file():
        raise FileNotFoundError(
            "DriftWorld Robomimic source was not found. Set DRIFTWORLD_SOURCE_ROOT "
            f"to the vendored driftworld directory: {root.parent}"
        )
    return root


def _module_is_below(module, root: Path) -> bool:
    path = getattr(module, "__file__", None)
    return path is not None and Path(path).resolve().is_relative_to(root)


@lru_cache(maxsize=1)
def get_driftworld_modules():
    """Load the flat-import upstream modules without editing their source tree."""
    root = driftworld_source_root()
    for name in ("data", "unet_multi", "drifting"):
        existing = sys.modules.get(name)
        if existing is not None and not _module_is_below(existing, root):
            raise ImportError(
                f"Top-level module {name!r} is already loaded from another package; "
                "start a clean process before loading DriftWorld."
            )

    if "data" not in sys.modules:
        # The denoiser imports Batch only as a type annotation. Loading the
        # upstream data package would eagerly import its unused Zarr reader,
        # so expose the minimal package contract needed by the model.
        data_package = types.ModuleType("data")
        data_package.__file__ = str(root / "data" / "__init__.py")
        data_package.__path__ = [str(root / "data")]
        data_package.__package__ = "data"
        data_package.__spec__ = importlib.machinery.ModuleSpec(
            "data", loader=None, is_package=True
        )
        data_package.Batch = dict
        sys.modules["data"] = data_package

    sys.path.insert(0, str(root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        denoiser_module = importlib.import_module("drifting_denoiser_multi")
        unet_configs = importlib.import_module("unet_multi.unet_configs")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(root))
    return denoiser_module, unet_configs


@lru_cache(maxsize=1)
def get_robomimic_driftworld_modules():
    """Load the one-view Robomimic model without editing vendored sources."""
    root = robomimic_driftworld_source_root()
    for name in ("unet", "drifting", "drifting_denoiser_multi"):
        existing = sys.modules.get(name)
        if existing is not None and not _module_is_below(existing, root):
            raise ImportError(
                f"Top-level module {name!r} is already loaded from another package; "
                "start a clean process before loading Robomimic DriftWorld."
            )

    sys.path.insert(0, str(root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        denoiser_module = importlib.import_module("drifting_denoiser_multi")
        unet_configs = importlib.import_module("unet.unet_configs")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(root))
    return denoiser_module, unet_configs


def create_driftworld_denoiser(
    *,
    unet_name: str,
    temp_list,
    n_neg: int,
    num_future_frames: int,
    num_history_frames: int,
    decay: float,
    action_frameskip: int = 1,
    device: torch.device | str | None = None,
):
    """Construct the upstream denoiser, widening only its action input for skip > 1.

    ``action_frameskip=1`` calls the original constructor with the original U-Net
    registry entry. For larger skips, the original ``UNet_PushT`` builder is used
    and only its first action projection is replaced to accept concatenated 2-D
    actions, matching StableWM/LeWM action blocking.
    """
    if action_frameskip < 1:
        raise ValueError("action_frameskip must be positive.")
    denoiser_module, unet_configs = get_driftworld_modules()
    if unet_name != "UNet_PushT":
        raise ValueError(
            f"Only the upstream Push-T U-Net is supported, got {unet_name!r}."
        )

    if action_frameskip == 1:
        denoiser = denoiser_module.Denoiser(
            unet_name=unet_name,
            temp_list=temp_list,
            n_neg=n_neg,
            num_future_frames=num_future_frames,
            num_history_frames=num_history_frames,
            decay=decay,
        )
    else:
        registry_key = f"_UNet_PushT_frameskip_{action_frameskip}"

        def build_frameskip_unet(num_history=4):
            model = unet_configs.UNet_PushT(num_history=num_history)
            original_projection = model.action_embed[0]
            model.action_in = 2 * action_frameskip
            model.action_embed[0] = nn.Linear(
                model.action_in,
                original_projection.out_features,
                bias=original_projection.bias is not None,
            )
            return model

        unet_configs.UNet_model_dict[registry_key] = build_frameskip_unet
        try:
            denoiser = denoiser_module.Denoiser(
                unet_name=registry_key,
                temp_list=temp_list,
                n_neg=n_neg,
                num_future_frames=num_future_frames,
                num_history_frames=num_history_frames,
                decay=decay,
            )
        finally:
            unet_configs.UNet_model_dict.pop(registry_key, None)

    return denoiser.to(device) if device is not None else denoiser


def create_robomimic_driftworld_denoiser(
    *,
    unet_type: str,
    temp_list,
    n_neg: int,
    num_future_frames: int,
    num_history_frames: int,
    decay: float,
    action_dim: int = 5,
    action_frameskip: int = 5,
    device: torch.device | str | None = None,
):
    """Construct the upstream one-view Robomimic U-Net for blocked Cube actions."""
    if unet_type != "multi":
        raise ValueError(
            "OGBench Cube currently supports Robomimic's one-view U-Net only."
        )
    if action_dim < 1 or action_frameskip < 1:
        raise ValueError("action_dim and action_frameskip must be positive.")
    denoiser_module, _ = get_robomimic_driftworld_modules()
    denoiser = denoiser_module.Denoiser(
        unet_type=unet_type,
        temp_list=temp_list,
        n_neg=n_neg,
        num_future_frames=num_future_frames,
        num_history_frames=num_history_frames,
        decay=decay,
    )
    model_action_dim = int(action_dim) * int(action_frameskip)
    original_projection = denoiser.inner_model.action_embed[0]
    denoiser.inner_model.action_in = model_action_dim
    denoiser.inner_model.action_embed[0] = nn.Linear(
        model_action_dim,
        original_projection.out_features,
        bias=original_projection.bias is not None,
    )
    denoiser.ema_model = copy.deepcopy(denoiser.inner_model)
    denoiser.ema_model.requires_grad_(False)
    return denoiser.to(device) if device is not None else denoiser
