"""Run one model/dataset planning visualization cell for a Slurm array."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = Path("/grid/klindt/home/nam/ckpt")
DATASETS = ["pusht", "avoidt", "stackt", "pusht_antigoal", "cube", "cube_hop", "cube_diagonal", "manipulator_insert_peg", "manipulator_hug_box", "manipulator_extract_peg", "tworoom", "tworoom_route"]
MODELS = ["lewm", "sensorimotor", "prejepa", "driftworld", "irasim", "toy-adaln", "toy-action-duplicate", "toy-driftworld"]
FAMILY = {**{x:"pusht" for x in DATASETS[:4]}, **{x:"cube" for x in DATASETS[4:7]}, **{x:"manipulator_insert_peg" for x in DATASETS[7:10]}, **{x:"tworoom" for x in DATASETS[10:]}}
STEM = {"pusht_antigoal":"push_t_antigoal", "tworoom_route":"tworoom_triangle", "manipulator_hug_box":"manipulator_insert_peg", "manipulator_extract_peg":"manipulator_insert_peg"}
DATASET_NAME = {"pusht":"pusht_expert_train", "avoidt":"avoidt_test", "stackt":"stackt_test", "pusht_antigoal":"pusht_antigoal_test", "cube":"cube_single_expert", "cube_hop":"cube_hop_test", "cube_diagonal":"cube_diagonal_test", "manipulator_insert_peg":"manipulator_insert_peg_weak_expert_train", "manipulator_hug_box":"avoid_peg_test", "manipulator_extract_peg":"extract_peg_test", "tworoom":"tworoom", "tworoom_route":"tworoom_route_test"}
BASE = {
 "driftworld": {"pusht":"20260820_driftworld_pusht_lewm_fs5_lr5e-4_resume435500_to500k/ckpt-step500000.pth", "cube":"20260820_driftworld_cube_lewm_fs5_lr5e-5_rerun/ckpt-step100000.pth", "tworoom":"20260824_driftworld_tworoom_lewm_fs5_lr1e-4/ckpt-step500000.pth", "manipulator_insert_peg":"20260910_driftworld_manipulator_insert_peg_lewm_fs5_lr5e-5/ckpt-step100000.pth"},
 "irasim": {"pusht":"08/20/irasim_pusht_h5_frameskip5_lrlr5em5002/checkpoints/0300000.pt", "cube":"08/24/irasim_cube_h5_frameskip5_lr5em5/checkpoints/0300000.pt", "tworoom":"08/26/irasim_tworoom_h5_frameskip5_lr5em5/checkpoints/0300000.pt", "manipulator_insert_peg":"09/11/irasim_manipulator_insert_peg_h5_frameskip5_lr5em5/checkpoints/0300000.pt"},
 "toy-adaln": {"pusht":"toy_predictors_20260723_frameskip5_10d/20260723_pusht_adaln/weights_epoch_100.pt", "cube":"toy_predictors_20260826_cube_everything_toy_predictors/20260826_cube_everything_adaln/weights_epoch_100.pt", "tworoom":"toy_predictors_20260825_tworoom_toy_predictors/20260825_tworoom_essential_adaln/weights_epoch_100.pt", "manipulator_insert_peg":"20260910_manipulator_insert_peg_toy_predictors/20260910_manipulator_insert_peg_essential_adaln_01/weights_epoch_100.pt"},
 "toy-action-duplicate": {"pusht":"toy_predictors_20260723_frameskip5_10d/20260723_pusht_action_duplicate/weights_epoch_100.pt", "cube":"toy_predictors_20260826_cube_everything_toy_predictors/20260826_cube_everything_action_duplicate/weights_epoch_100.pt", "tworoom":"toy_predictors_20260825_tworoom_toy_predictors/20260825_tworoom_essential_action_duplicate/weights_epoch_100.pt", "manipulator_insert_peg":"20260910_manipulator_insert_peg_toy_predictors/20260910_manipulator_insert_peg_essential_action_duplicate_01/weights_epoch_100.pt"},
 "toy-driftworld": {"pusht":"toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_pusht_drift_state_predictor_bs128_500k_lr1e-4/weights_step_00500000.pt", "cube":"toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_cube_drift_state_predictor_bs128_500k_lr1e-4/weights_step_00500000.pt", "tworoom":"toy_driftworld_bs128_500k_lr_sweep_20260902/20260902_tworoom_drift_state_predictor_bs128_500k_lr1e-4/weights_step_00256900.pt", "manipulator_insert_peg":"20260910_manipulator_insert_peg_toy_predictors/20260910_manipulator_insert_peg_essential_drift_lr1e-4_01/weights_step_00411600.pt"},
}

def make_command(model, dataset):
 family, stem = FAMILY[dataset], STEM.get(dataset, dataset)
 out = ROOT / "paper/results/planning_visualizations" / dataset / model
 common = ["eval.num_eval=5", "eval.goal_offset_steps=25", "eval.eval_budget=25", "eval.start_selection=first_episodes", f"eval.dataset_name={DATASET_NAME[dataset]}"]
 hydra_common = [value if not value.startswith("eval.start_selection=") else "+" + value for value in common]
 if model in {"lewm", "prejepa"}:
  cmd = [sys.executable, str(ROOT/f"scripts/planning_{model}.py"), f"--config-name={stem}", *hydra_common, f"output.directory={out}", "+output.save_artifacts=true"]
  if family == "manipulator_insert_peg":
   directory = CKPT/("checkpoints/20260910_manipulator_insert_peg_lewm" if model=="lewm" else "20260911_manipulator_insert_peg_prejepa_rgb_action_fs5/checkpoints/20260911_manipulator_insert_peg_prejepa_rgb_action_fs5_3173369")
   cmd += [f"model.checkpoint={directory/'weights_epoch_10.pt'}", f"model.config={directory/'config.json'}"]
  return cmd
 if model == "sensorimotor":
  extra = ["model.checkpoint_root=/grid/klindt/home/nam/sensorimotor-world-model/planning/experiments/train/results"]
  if family == "manipulator_insert_peg":
   sensor_root = CKPT/"sensorimotor_world_model/20260910_manipulator_insert_peg_weak_expert_inverse_lambda_1_seed0"
   extra = [f"model.config={sensor_root/'config.yaml'}", f"model.checkpoint={sensor_root/'runs/20260910/012156/bed8c9921824/checkpoints/last.ckpt'}"]
  return [sys.executable, str(ROOT/"scripts/planning_sensorimotor.py"), "--config", str(ROOT/f"configs/sensorimotor/planning/{stem}.yaml"), "--seed", "0", "--", *common, *extra, f"output.directory={out}", "output.save_artifacts=true"]
 if model in {"driftworld", "irasim"}:
  config_stem = ({"stackt":"stackt_v2", "pusht_antigoal":"push_t_antigoal_v3"}.get(dataset, stem) if model=="irasim" else stem+"_lewm")
  extra = [f"model.scheduler_path={ROOT/'worldmodels/IRASim/pretrained_models/scheduler'}"] if model == "irasim" else []
  return [sys.executable, str(ROOT/f"scripts/planning_{model}.py"), "--config", str(ROOT/f"configs/{model}/planning/{config_stem}.yaml"), "--", *common, f"model.checkpoint={CKPT/BASE[model][family]}", *extra, f"output.directory={out}", "output.save_artifacts=true", "output.save_video=false"]
 return [sys.executable, str(ROOT/"scripts/planning_toy.py"), f"--config-name={stem}", *hydra_common, f"model.checkpoint_path={CKPT/BASE[model][family]}", f"output.directory={out}", "+output.save_artifacts=true", "output.save_video=true", f"output.gif_directory={out}"]

if __name__ == "__main__":
 p=argparse.ArgumentParser(); p.add_argument("--task-id",type=int,required=True); a=p.parse_args()
 model=MODELS[a.task_id%8]; dataset=DATASETS[a.task_id//8]; cmd=make_command(model,dataset)
 output = ROOT/"paper/results/planning_visualizations"/dataset/model/"artifacts/actions.json"
 if output.is_file(): print(f"already complete: {output}"); raise SystemExit(0)
 print("running:", " ".join(map(str,cmd)), flush=True)
 env=os.environ.copy(); env["PYTHONPATH"]=str(ROOT)+(":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
 env.setdefault("LEWM_SOURCE_ROOT", str(ROOT/"worldmodels/lewm"))
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
