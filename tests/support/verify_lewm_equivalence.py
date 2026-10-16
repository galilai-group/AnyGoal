"""Compare the shared-config loader with the original builder on real weights."""

import ast
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from anygoal.artifacts import default_checkpoint_path
from anygoal.cli import _planning_config
from anygoal.registry import get_registry
from anygoal.runners.plan import planning_lewm
from anygoal.runners.plan.planning import patch_loaded_model_compat
from anygoal.wm.checkpoints import compatible_vit_state_dict


def main():
    original = subprocess.check_output(
        ["git", "show", "HEAD:anygoal/runners/plan/planning_lewm.py"], text=True
    )
    function = next(
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_lewm"
    )
    namespace = vars(planning_lewm).copy()
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "<original-build-lewm>",
            "exec",
        ),
        namespace,
    )
    reference_config = OmegaConf.create(
        subprocess.check_output(
            ["git", "show", "HEAD:anygoal/resources/plan/lewm/cube.yaml"], text=True
        )
    )
    registry = get_registry()
    model_spec = registry.model("lewm")
    references = {}
    for task in registry.datasets.values():
        checkpoint = default_checkpoint_path(model_spec, task)
        if task.family not in references:
            cfg = OmegaConf.create(
                OmegaConf.to_container(reference_config, resolve=False)
            )
            cfg.model.action_dim = 5 if task.family in {"cube", "insert_peg"} else 2
            reference = namespace["build_lewm"](cfg)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            reference.load_state_dict(
                compatible_vit_state_dict(payload, reference.state_dict()), strict=True
            )
            patch_loaded_model_compat(reference)
            references[task.family] = reference.cuda().eval().requires_grad_(False)
        path = _planning_config(model_spec, task)
        with initialize_config_dir(config_dir=str(path.parent), version_base=None):
            cfg = compose(config_name=path.stem)
        OmegaConf.update(cfg, "model.checkpoint", str(checkpoint), force_add=True)
        model = planning_lewm.load_lewm_cost_model(cfg, checkpoint.parent)
        patch_loaded_model_compat(model)
        model = model.cuda().eval().requires_grad_(False)
        torch.manual_seed(0)
        info = {
            "pixels": torch.rand(1, 2, 3, 3, 224, 224, device="cuda"),
            "goal": torch.rand(1, 2, 3, 3, 224, 224, device="cuda"),
            "action": torch.rand(1, 2, 3, cfg.model.action_dim * 5, device="cuda"),
        }
        actions = torch.rand(1, 2, 8, cfg.model.action_dim * 5, device="cuda")
        with torch.no_grad():
            expected = references[task.family].get_cost(
                {key: value.clone() for key, value in info.items()}, actions.clone()
            )
            actual = model.get_cost(
                {key: value.clone() for key, value in info.items()}, actions.clone()
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(
            f"EXACT_COST_EQUIVALENCE task={task.id} checkpoint={checkpoint.name}",
            flush=True,
        )
    print("ALL_12_LEWM_TASKS_STRICT_LOAD_AND_EXACT_COST_MATCH", flush=True)


if __name__ == "__main__":
    main()
