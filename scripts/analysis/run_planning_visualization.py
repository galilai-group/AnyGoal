"""Run one model/dataset planning visualization matrix cell by task ID."""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path.home() / ".stable_worldmodel/anygoal"
OLD_CKPT = Path("/grid/klindt/home/nam/ckpt")
DATASETS = [
    "pusht",
    "avoidt",
    "stackt",
    "pusht_antigoal",
    "cube",
    "cube_hop",
    "cube_diagonal",
    "insert_peg",
    "avoid_peg",
    "extract_peg",
    "tworoom",
    "tworoom_route",
]
MODELS = [
    "lewm",
    "sensorimotor",
    "prejepa",
    "driftworld",
    "irasim",
    "toy-adaln",
    "toy-action-duplicate",
    "toy-driftworld",
]
FAMILY = {
    **{x: "pusht" for x in DATASETS[:4]},
    **{x: "cube" for x in DATASETS[4:7]},
    **{x: "insert_peg" for x in DATASETS[7:10]},
    **{x: "tworoom" for x in DATASETS[10:]},
}
STEM = {
    "pusht_antigoal": "push_t_antigoal",
    "avoid_peg": "insert_peg",
    "extract_peg": "insert_peg",
}
DATASET_NAME = {
    "pusht": "pusht_expert_train",
    "avoidt": "avoidt_test",
    "stackt": "stackt_test",
    "pusht_antigoal": "pusht_antigoal_test",
    "cube": "cube_single_expert",
    "cube_hop": "cube_hop_test",
    "cube_diagonal": "cube_diagonal_test",
    "insert_peg": "insert_peg_weak_expert_train",
    "avoid_peg": "avoid_peg_test",
    "extract_peg": "extract_peg_test",
    "tworoom": "tworoom",
    "tworoom_route": "tworoom_route_test",
}
BASE = {
    "driftworld": {
        family: f"driftworld_{family if family != 'insert_peg' else 'insertpeg'}.pt"
        for family in ("pusht", "cube", "tworoom", "insert_peg")
    },
    "irasim": {
        "pusht": "irasim_pusht.pt",
        "cube": "irasim_cube.pt",
        "tworoom": "irasim_tworoom.pt",
        "insert_peg": "irasim_insertpeg.pt",
    },
    "toy-adaln": {
        family: f"toy-adaln_{family if family != 'insert_peg' else 'insertpeg'}.pt"
        for family in ("pusht", "cube", "tworoom", "insert_peg")
    },
    "toy-action-duplicate": {
        family: f"toy-action-duplicate_{family if family != 'insert_peg' else 'insertpeg'}.pt"
        for family in ("pusht", "cube", "tworoom", "insert_peg")
    },
    "toy-driftworld": {
        family: f"toy-driftworld_{family if family != 'insert_peg' else 'insertpeg'}.pt"
        for family in ("pusht", "cube", "tworoom", "insert_peg")
    },
}
TOY_CONFIG = {
    "toy-adaln": {
        "pusht": "toy_predictors_20260723_frameskip5_10d/20260723_pusht_adaln",
        "cube": "toy_predictors_20260826_cube_everything_toy_predictors/20260826_cube_everything_adaln",
        "tworoom": "toy_predictors_20260825_tworoom_toy_predictors/20260825_tworoom_essential_adaln",
        "insert_peg": "20260910_insert_peg_toy_predictors/20260910_insert_peg_essential_adaln_01",
    },
    "toy-action-duplicate": {
        "pusht": "toy_predictors_20260723_frameskip5_10d/20260723_pusht_action_duplicate",
        "cube": "toy_predictors_20260826_cube_everything_toy_predictors/20260826_cube_everything_action_duplicate",
        "tworoom": "toy_predictors_20260825_tworoom_toy_predictors/20260825_tworoom_essential_action_duplicate",
        "insert_peg": "20260910_insert_peg_toy_predictors/20260910_insert_peg_essential_action_duplicate_01",
    },
    "toy-driftworld": {
        "pusht": "toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_pusht_drift_state_predictor_bs128_500k_lr1e-4",
        "cube": "toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_cube_drift_state_predictor_bs128_500k_lr1e-4",
        "tworoom": "toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_tworoom_drift_state_predictor_bs128_500k_lr1e-4",
        "insert_peg": "20260910_insert_peg_toy_predictors/20260910_insert_peg_essential_drift_lr1e-4_01",
    },
}


def make_command(model, dataset):
    family, stem = FAMILY[dataset], STEM.get(dataset, dataset)
    out = ROOT / "outputs/analysis/planning_visualizations" / dataset / model
    common = [
        "eval.num_eval=5",
        "eval.goal_offset_steps=25",
        "eval.eval_budget=25",
        "eval.start_selection=first_episodes",
        f"eval.dataset_name={DATASET_NAME[dataset]}",
    ]
    hydra_common = [
        value if not value.startswith("eval.start_selection=") else "+" + value
        for value in common
    ]
    if model in {"lewm", "prejepa"}:
        cmd = [
            sys.executable,
            str(ROOT / f"scripts/plan/planning_{model}.py"),
            f"--config-name={stem}",
            *hydra_common,
            f"output.directory={out}",
            "+output.save_artifacts=true",
        ]
        if family == "insert_peg":
            checkpoint = CKPT / (
                "lewm_insertpeg.pt" if model == "lewm" else "prejepa_insertpeg.pt"
            )
            config = (
                OLD_CKPT
                / "20260911_insert_peg_prejepa_rgb_action_fs5/checkpoints/20260911_insert_peg_prejepa_rgb_action_fs5_3173369/config.json"
            )
            cmd += [f"model.checkpoint={checkpoint}"]
            if model == "prejepa":
                cmd += [f"model.config={config}"]
        return cmd
    if model == "sensorimotor":
        extra = [
            "model.checkpoint_root=/grid/klindt/home/nam/sensorimotor-world-model/planning/experiments/train/results"
        ]
        if family == "insert_peg":
            sensor_root = (
                OLD_CKPT
                / "sensorimotor_world_model/20260910_insert_peg_weak_expert_inverse_lambda_1_seed0"
            )
            extra = [
                f"model.config={sensor_root / 'config.yaml'}",
                f"model.checkpoint={CKPT / 'sensorimotor_insertpeg.ckpt'}",
            ]
        return [
            sys.executable,
            str(ROOT / "scripts/plan/planning_sensorimotor.py"),
            "--config",
            str(ROOT / f"scripts/plan/config/sensorimotor/{stem}.yaml"),
            "--seed",
            "0",
            "--",
            *common,
            *extra,
            f"output.directory={out}",
            "output.save_artifacts=true",
        ]
    if model in {"driftworld", "irasim"}:
        config_stem = (
            {"stackt": "stackt_v2", "pusht_antigoal": "push_t_antigoal_v3"}.get(
                dataset, stem
            )
            if model == "irasim"
            else stem + "_lewm"
        )
        extra = (
            [
                f"model.scheduler_path={ROOT / 'worldmodels/IRASim/pretrained_models/scheduler'}"
            ]
            if model == "irasim"
            else []
        )
        return [
            sys.executable,
            str(ROOT / f"scripts/plan/planning_{model}.py"),
            "--config",
            str(ROOT / f"scripts/plan/config/{model}/{config_stem}.yaml"),
            "--",
            *common,
            f"model.checkpoint={CKPT / BASE[model][family]}",
            *extra,
            f"output.directory={out}",
            "output.save_artifacts=true",
            "output.save_video=false",
        ]
    return [
        sys.executable,
        str(ROOT / "scripts/plan/planning_toy.py"),
        f"--config-name={stem}",
        *hydra_common,
        f"model.checkpoint_path={CKPT / BASE[model][family]}",
        f"model.config_path={OLD_CKPT / TOY_CONFIG[model][family] / 'config.yaml'}",
        f"output.directory={out}",
        "+output.save_artifacts=true",
        "output.save_video=true",
        f"output.gif_directory={out}",
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, required=True)
    a = p.parse_args()
    model = MODELS[a.task_id % 8]
    dataset = DATASETS[a.task_id // 8]
    cmd = make_command(model, dataset)
    output = (
        ROOT
        / "outputs/analysis/planning_visualizations"
        / dataset
        / model
        / "artifacts/actions.json"
    )
    if output.is_file():
        print(f"already complete: {output}")
        raise SystemExit(0)
    print("running:", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("LEWM_SOURCE_ROOT", str(ROOT / "worldmodels/lewm"))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
