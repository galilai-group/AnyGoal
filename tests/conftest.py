"""Test-only defaults for deterministic headless rendering."""

import os
from pathlib import Path

import pytest


os.environ.setdefault("MUJOCO_GL", "osmesa")


@pytest.fixture(scope="session")
def matrix_data_root() -> Path:
    root = (
        Path(
            os.environ.get(
                "ANYGOAL_MATRIX_DATA_ROOT",
                os.environ.get("ANYGOAL_DATA_ROOT", "~/.stable_worldmodel/datasets"),
            )
        )
        .expanduser()
        .resolve()
    )
    if not root.is_dir():
        pytest.fail(
            f"matrix_e2e data root does not exist: {root}. Set "
            "ANYGOAL_MATRIX_DATA_ROOT."
        )
    return root


@pytest.fixture(scope="session")
def matrix_device() -> str:
    import torch

    device = os.environ.get("ANYGOAL_MATRIX_DEVICE", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.fail(
            "matrix_e2e requires an allocated CUDA device by default; set "
            "ANYGOAL_MATRIX_DEVICE=cpu only for integrations that support it."
        )
    return device


@pytest.fixture(scope="session")
def matrix_eval_config() -> Path:
    from anygoal.resources import resource_path

    path = (
        Path(
            os.environ.get(
                "ANYGOAL_MATRIX_EVAL_CONFIG",
                resource_path("eval", "rollout.yaml"),
            )
        )
        .expanduser()
        .resolve()
    )
    if not path.is_file():
        pytest.fail(
            f"matrix_e2e evaluation config does not exist: {path}. Set "
            "ANYGOAL_MATRIX_EVAL_CONFIG."
        )
    return path
