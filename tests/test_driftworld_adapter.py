import logging
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import stable_worldmodel as swm
import torch
from torch import nn

from anygoal.runners.train.train_driftworld import (
    CheckpointingTorchProxy,
    TrainingTelemetry,
)
from anygoal.resources import resource_path
from anygoal.wm.driftworld import (
    DriftWorldActionProcessor,
    DriftWorldCubeHDF5Dataset,
    DriftWorldCubeWorldModel,
    DriftWorldHDF5Dataset,
    DriftWorldWorldModel,
    DriftWorldWorldModelPolicy,
    driftworld_image_transform,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeEMA(nn.Module):
    def forward(self, noise, history, actions):
        value = actions.mean(dim=-1)[:, None, :, None, None]
        return value.expand(-1, 3, -1, history.shape[-2], history.shape[-1])


class FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.ema_model = FakeEMA()


class FakeTrainingDenoiser(nn.Module):
    num_history_frames = 3
    num_future_frames = 1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, batch, device):
        return self.anchor, {
            "loss_0.02": 1.0,
            "loss_0.05": 2.0,
            "loss_0.2": 3.0,
            "loss_backprop": 2.0,
        }

    def sample(self, history, actions):
        batch_size, _, _, height, width = history.shape
        return torch.zeros(batch_size, 1, 3, height, width, device=history.device)


class FakeWandb:
    @staticmethod
    def Image(panel, caption):
        return {"panel": panel, "caption": caption}


def make_model(action_frameskip=1, action_mode="relative"):
    return DriftWorldWorldModel(
        denoiser=FakeDenoiser(),
        action_frameskip=action_frameskip,
        action_mode=action_mode,
        num_history_frames=4,
        num_future_frames=4,
        rollout_batch_size=2,
    )


def write_h5(path):
    length = 20
    pixels = np.stack(
        [np.full((96, 96, 3), index, dtype=np.uint8) for index in range(length)]
    )
    actions = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_len", data=np.array([length], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))


def write_cube_h5(path):
    length = 20
    pixels = np.stack(
        [np.full((96, 96, 3), index, dtype=np.uint8) for index in range(length)]
    )
    actions = np.arange(length * 5, dtype=np.float32).reshape(length, 5)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_len", data=np.array([length], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))


def test_hdf5_frameskip_one_matches_original_contiguous_window(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    write_h5(datasets / "tiny.h5")
    dataset = DriftWorldHDF5Dataset(
        name="tiny",
        cache_dir=tmp_path,
        pred_horizon=8,
        frameskip=1,
        num_demos=1,
    )
    sample = dataset[0]
    assert sample["image"].shape == (8, 3, 96, 96)
    torch.testing.assert_close(
        sample["image"][:, 0, 0, 0], torch.arange(8, dtype=torch.float32) / 255
    )
    torch.testing.assert_close(
        sample["action"], torch.arange(16, dtype=torch.float32).reshape(8, 2)
    )


def test_hdf5_frameskip_groups_dense_actions_and_strides_images(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    write_h5(datasets / "tiny.h5")
    dataset = DriftWorldHDF5Dataset(
        name="tiny",
        cache_dir=tmp_path,
        pred_horizon=8,
        frameskip=2,
        num_demos=1,
    )
    sample = dataset[0]
    torch.testing.assert_close(
        sample["image"][:, 0, 0, 0],
        torch.arange(0, 16, 2, dtype=torch.float32) / 255,
    )
    torch.testing.assert_close(
        sample["action"], torch.arange(32, dtype=torch.float32).reshape(8, 4)
    )


def test_cube_hdf5_uses_robomimic_batch_and_blocked_five_dimensional_actions(
    tmp_path,
):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    write_cube_h5(datasets / "cube.h5")
    dataset = DriftWorldCubeHDF5Dataset(
        name="cube",
        cache_dir=tmp_path,
        pred_horizon=4,
        frameskip=2,
        num_demos=1,
    )
    sample = dataset[0]
    images = sample["obs"]["agentview_image"]
    assert images.shape == (4, 3, 96, 96)
    torch.testing.assert_close(
        images[:, 0, 0, 0], torch.arange(0, 8, 2, dtype=torch.float32) / 255 * 2 - 1
    )
    torch.testing.assert_close(
        sample["actions"], torch.arange(40, dtype=torch.float32).reshape(4, 10)
    )


def test_relative_action_cost_matches_stableworld_contract_and_is_deterministic():
    model = make_model(action_frameskip=2).eval()
    info = {
        "pixels": torch.zeros(1, 3, 4, 3, 96, 96, dtype=torch.uint8),
        "goal": torch.zeros(1, 3, 1, 3, 96, 96, dtype=torch.uint8),
    }
    candidates = torch.randn(1, 3, 2, 4)
    first = model.get_cost(info, candidates)
    second = model.get_cost(info, candidates)
    assert first.shape == (1, 3)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_action_processor_matches_standard_scaler_and_restores_raw_model_actions():
    raw_actions = np.asarray(
        [[-0.4, 0.1], [0.0, 0.3], [0.4, 0.5]], dtype=np.float32
    )
    processor = DriftWorldActionProcessor.fit(raw_actions)
    standardized = processor.transform(raw_actions)
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        processor.inverse_transform(standardized), raw_actions, atol=1e-6
    )

    model = DriftWorldWorldModel(
        denoiser=FakeDenoiser(),
        action_frameskip=2,
        num_history_frames=4,
        num_future_frames=4,
        rollout_batch_size=2,
        action_processor=processor,
    )
    blocked = torch.from_numpy(standardized[[0, 1]].reshape(1, 1, 1, 4))
    restored = model.prepare_model_actions({}, blocked)
    torch.testing.assert_close(
        restored,
        torch.from_numpy(raw_actions[[0, 1]].reshape(1, 1, 1, 4)),
    )


def test_cube_action_processor_restores_raw_blocked_actions():
    raw_actions = np.arange(20, dtype=np.float32).reshape(4, 5)
    processor = DriftWorldActionProcessor.fit(raw_actions)
    model = DriftWorldCubeWorldModel(
        denoiser=FakeDenoiser(),
        action_frameskip=2,
        num_history_frames=3,
        num_future_frames=1,
        action_processor=processor,
    )
    standardized = processor.transform(raw_actions[:2]).reshape(1, 1, 1, 10)
    restored = model.prepare_model_actions({}, torch.from_numpy(standardized))
    torch.testing.assert_close(
        restored, torch.from_numpy(raw_actions[:2].reshape(1, 1, 1, 10))
    )


def test_bundled_checkpoint_action_adapter_converts_relative_to_absolute():
    model = make_model(action_mode="absolute_target")
    info = {"proprio": torch.tensor([[[100.0, 200.0, 0.0, 0.0]]])}
    candidates = torch.tensor([[[[0.5, -0.5]]]])
    converted = model.prepare_model_actions(info, candidates)
    expected = torch.tensor([[[[150.0 / 511.0 * 2 - 1, 150.0 / 511.0 * 2 - 1]]]])
    torch.testing.assert_close(converted, expected)


def test_policy_accumulates_only_frameskip_aligned_history():
    model = make_model(action_frameskip=2).eval()
    solver = swm.solver.CEMSolver(model=model, num_samples=2, n_steps=1, topk=2)
    policy = DriftWorldWorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=1,
            receding_horizon=1,
            history_len=4,
            action_block=2,
        ),
        action_processor=model.action_processor,
        transform={
            "pixels": driftworld_image_transform(),
            "goal": driftworld_image_transform(),
        },
    )
    env = SimpleNamespace(
        num_envs=1,
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1, 2)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    policy.set_env(env)
    goal = np.ones((1, 1, 96, 96, 3), dtype=np.uint8)

    def prepare(value, action):
        return policy._prepare_info(
            {
                "pixels": np.full((1, 1, 96, 96, 3), value, dtype=np.uint8),
                "goal": goal,
                "action": np.asarray(action, dtype=np.float32).reshape(1, 1, 2),
                "proprio": np.zeros((1, 1, 4), dtype=np.float32),
            }
        )

    first = prepare(0, [np.nan, np.nan])
    second = prepare(1, [0.0, 0.0])
    third = prepare(2, [0.0, 0.0])
    assert first["pixels"].shape == (1, 4, 3, 96, 96)
    torch.testing.assert_close(second["pixels"][:, -1], first["pixels"][:, -1])
    torch.testing.assert_close(
        third["pixels"][:, -1], torch.full_like(third["pixels"][:, -1], 2)
    )


def test_training_telemetry_logs_components_eta_and_pixel_visualization():
    cfg = OmegaConf.create(
        {
            "train": {"seed": 1, "num_epochs": 2},
            "visualization": {
                "enabled": True,
                "every_steps": 1,
                "num_samples": 1,
                "seed": 1,
            },
            "eta": {"log_every_steps": 1},
        }
    )
    telemetry = TrainingTelemetry(cfg, FakeWandb(), logging.getLogger(__name__))
    telemetry.register_loader(range(3))
    model = telemetry.wrap_model(FakeTrainingDenoiser())
    batch = {
        "image": torch.zeros(1, 4, 3, 8, 8),
        "action": torch.zeros(1, 4, 2),
    }
    _, metrics = model(batch, torch.device("cpu"))
    augmented = telemetry.augment_log(metrics, step=0)

    assert metrics["loss/components/0.02"] == 1.0
    assert metrics["loss/components/0.05"] == 2.0
    assert metrics["loss/components/0.2"] == 3.0
    assert metrics["loss/final"] == 2.0
    assert metrics["visualization/pixel_mse"] == 0.0
    assert "visualization/ground_truth_vs_prediction" in metrics
    assert augmented["timing/steps_remaining"] == 5
    assert augmented["timing/eta_seconds"] >= 0


def test_permanent_checkpoint_is_independent_and_never_overwritten(tmp_path):
    cfg = OmegaConf.create(
        {
            "path_ckpt_latest": str(tmp_path / "ckpt-latest.pth"),
            "output": {"run_directory": str(tmp_path)},
            "train": {"permanent_ckpt_every": 20},
        }
    )
    proxy = CheckpointingTorchProxy(torch, cfg, logging.getLogger(__name__))
    proxy.save({"step": 20, "value": torch.tensor([1])}, cfg.path_ckpt_latest)
    permanent = tmp_path / "ckpt-step20.pth"
    assert permanent.is_file()
    assert permanent.stat().st_ino != (tmp_path / "ckpt-latest.pth").stat().st_ino
    proxy.save({"step": 20, "value": torch.tensor([2])}, cfg.path_ckpt_latest)
    assert torch.load(permanent, weights_only=True)["value"].item() == 1


def test_cube_train_and_planning_configs_are_lewm_aligned():
    train_cfg = OmegaConf.load(
        resource_path("train", "driftworld", "cube_lewm.yaml")
    )
    planning_cfg = OmegaConf.load(
        resource_path("plan", "driftworld", "cube_lewm.yaml")
    )
    assert train_cfg.model.source == planning_cfg.model.source == "robomimic"
    assert train_cfg.model.num_history_frames == planning_cfg.model.num_history_frames == 3
    assert train_cfg.model.num_future_frames == planning_cfg.model.num_future_frames == 1
    assert train_cfg.data.frameskip == planning_cfg.model.action_frameskip == 5
    assert train_cfg.model.action_dim == 5
    assert train_cfg.train.permanent_ckpt_every == 20_000


def test_tworoom_configs_reuse_pusht_model_and_are_lewm_aligned():
    train_cfg = OmegaConf.load(
        resource_path("train", "driftworld", "tworoom_lewm.yaml")
    )
    planning_cfg = OmegaConf.load(
        resource_path("plan", "driftworld", "tworoom_lewm.yaml")
    )
    assert train_cfg.model.unet_name == planning_cfg.model.unet_name == "UNet_PushT"
    assert train_cfg.model.num_history_frames == 3
    assert train_cfg.model.num_future_frames == 1
    assert train_cfg.data.frameskip == planning_cfg.model.action_frameskip == 5
    assert train_cfg.data.action_mode == planning_cfg.model.action_mode == "relative"
    assert planning_cfg.plan_config.history_len == 3
    assert planning_cfg.plan_config.action_block == 5
    assert planning_cfg.world.env_name == "swm/TwoRoom-v1"
    assert train_cfg.train.permanent_ckpt_every == 50_000


@pytest.mark.parametrize("task", ["pusht", "stackt", "avoidt", "push_t_antigoal"])
@pytest.mark.parametrize(
    ("mode", "history", "future", "frameskip", "planning_horizon"),
    [("lewm", 3, 1, 5, 5)],
)
def test_planning_configs_match_training_temporal_mode(
    task, mode, history, future, frameskip, planning_horizon
):
    config_path = resource_path("plan", "driftworld", f"{task}_{mode}.yaml")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    assert cfg.model.action_mode == "relative"
    assert cfg.model.num_history_frames == history
    assert cfg.model.num_future_frames == future
    assert cfg.model.action_frameskip == frameskip
    assert cfg.plan_config.history_len == history
    assert cfg.plan_config.action_block == frameskip
    assert cfg.plan_config.horizon == planning_horizon
