"""Explicit rollout checkpoint selection must reach the actual loader."""

import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from anygoal.resources import resource_path
from anygoal.wm import rollout


@pytest.mark.parametrize(
    "name,task",
    [
        ("lewm", "cube"),
        ("lewm", "insert_peg"),
        ("prejepa", "insert_peg"),
        ("prejepa", "pusht"),
    ],
)
def test_object_checkpoint_override_reaches_loader(name, task, monkeypatch, tmp_path):
    selected = tmp_path / "custom_object.ckpt"
    captured = []

    class Model:
        extra_encoders = {"action": object()}

        def to(self, *args):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

    def load(cfg, root):
        captured.append(cfg.model.checkpoint)
        return Model()

    monkeypatch.setattr(rollout, "fit_processor", lambda *_: object())
    monkeypatch.setattr(rollout, "LatentARRolloutAdapter", lambda **kwargs: kwargs)
    monkeypatch.setattr(rollout, "PreJEPARolloutAdapter", lambda **kwargs: kwargs)
    monkeypatch.setitem(
        sys.modules,
        "anygoal.runners.plan.planning",
        SimpleNamespace(
            patch_loaded_model_compat=lambda _: None,
            patch_transformers_vit_pickle_compat=lambda: None,
        ),
    )
    loader_name = "planning_lewm" if name == "lewm" else "planning_prejepa"
    attribute = "load_lewm_cost_model" if name == "lewm" else "load_prejepa_cost_model"
    monkeypatch.setitem(
        sys.modules,
        f"anygoal.runners.plan.{loader_name}",
        SimpleNamespace(**{attribute: load}),
    )
    config = OmegaConf.load(resource_path("eval", "rollout.yaml")).models[name][task]
    _, checkpoint, _ = rollout.build_adapter(
        name,
        config,
        object(),
        device=torch.device("cpu"),
        checkpoint_override=selected,
        training_config_override=None,
        planning_config_override=None,
        run_directory_override=None,
        history_length=3,
    )
    assert checkpoint == selected
    assert captured == [str(selected)]
