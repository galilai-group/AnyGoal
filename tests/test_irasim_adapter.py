from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torch import nn

from anygoal.wm.irasim import (
    IRASimActionProcessor,
    IRASimHDF5Dataset,
    IRASimWorldModel,
    IRASimWorldModelPolicy,
    adapt_irasim_action_frameskip,
    irasim_image_transform,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_h5(path):
    length = 40
    pixels = np.stack(
        [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(length)]
    )
    actions = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_len", data=np.array([length], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))


def test_hdf5_frameskip_one_uses_three_history_and_one_future_frame(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    write_h5(datasets / "tiny.h5")
    dataset = IRASimHDF5Dataset(
        name="tiny",
        cache_dir=tmp_path,
        num_frames=4,
        frameskip=1,
        video_size=(16, 16),
        action_scale=(20.0, 20.0),
    )
    sample = dataset[0]
    assert sample["video"].shape == (4, 3, 16, 16)
    torch.testing.assert_close(
        sample["video"][:, 0, 0, 0],
        torch.arange(4, dtype=torch.float32).div(255).mul(2).sub(1),
    )
    torch.testing.assert_close(
        sample["action"],
        torch.arange(6, dtype=torch.float32).reshape(3, 2) * 20,
    )


def test_hdf5_frameskip_strides_frames_and_groups_every_dense_action(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    write_h5(datasets / "tiny.h5")
    dataset = IRASimHDF5Dataset(
        name="tiny",
        cache_dir=tmp_path,
        num_frames=4,
        frameskip=2,
        video_size=(16, 16),
        action_scale=(1.0, 1.0),
    )
    sample = dataset[0]
    torch.testing.assert_close(
        sample["video"][:, 0, 0, 0],
        torch.arange(0, 8, 2, dtype=torch.float32).div(255).mul(2).sub(1),
    )
    torch.testing.assert_close(
        sample["action"], torch.arange(12, dtype=torch.float32).reshape(3, 4)
    )


class FakeFrameAda(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = 2
        self.embed_arm_state = nn.Linear(2, 16)


def test_frameskip_one_does_not_replace_original_action_projection():
    model = FakeFrameAda()
    projection = model.embed_arm_state
    returned = adapt_irasim_action_frameskip(
        model, action_dim=2, action_frameskip=1
    )
    assert returned is model
    assert returned.embed_arm_state is projection
    assert returned.state_dim == 2


def test_frameskip_widens_only_first_action_projection():
    model = FakeFrameAda()
    adapt_irasim_action_frameskip(model, action_dim=2, action_frameskip=5)
    assert model.embed_arm_state.in_features == 10
    assert model.embed_arm_state.out_features == 16
    torch.testing.assert_close(
        model.embed_arm_state.bias, torch.zeros_like(model.embed_arm_state.bias)
    )
    assert model.state_dim == 10


def test_action_processor_restores_raw_actions_before_upstream_scaling():
    raw = np.asarray([[-0.2, 0.1], [0.0, 0.4], [0.4, 0.7]], dtype=np.float32)
    processor = IRASimActionProcessor.fit(raw)
    standardized = processor.transform(raw)
    transformer = nn.Linear(1, 1)
    model = IRASimWorldModel(
        transformer=transformer,
        vae=FakeVAE(),
        scheduler=object(),
        action_frameskip=2,
        video_size=(16, 16),
        num_frames=4,
        inference_steps=1,
        action_scale=(20.0, 20.0),
        action_processor=processor,
        pipeline_factory=FakePipeline,
    )
    blocked = torch.from_numpy(standardized[:2].reshape(1, 1, 1, 4))
    restored = model.prepare_model_actions(blocked)
    torch.testing.assert_close(
        restored, torch.from_numpy(raw[:2].reshape(1, 1, 1, 4)) * 20
    )


class FakeDistribution:
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=1.0)

    def encode(self, images):
        latent = F.avg_pool2d(images, 2)
        latent = torch.cat([latent, latent[:, :1]], dim=1)
        return SimpleNamespace(latent_dist=FakeDistribution(latent))

    def decode(self, latents):
        images = F.interpolate(latents[:, :3], scale_factor=2, mode="nearest")
        return SimpleNamespace(sample=images)


class FakePipeline:
    def __init__(self, vae, scheduler, transformer):
        del vae, scheduler, transformer

    def __call__(self, actions, *, mask_x, latents, **kwargs):
        del kwargs
        output = latents.clone()
        future_length = latents.shape[1] - mask_x.shape[1]
        increments = actions[:, -future_length:].mean(dim=-1)[
            :, :, None, None, None
        ]
        output[:, mask_x.shape[1] :] = (
            mask_x[:, -1:] + increments.cumsum(dim=1)
        )
        return None, output


def test_cost_contract_is_deterministic_and_supports_blocked_actions():
    model = IRASimWorldModel(
        transformer=nn.Linear(1, 1),
        vae=FakeVAE(),
        scheduler=object(),
        action_frameskip=2,
        video_size=(16, 16),
        num_frames=3,
        inference_steps=1,
        rollout_batch_size=2,
        action_scale=(1.0, 1.0),
        pipeline_factory=FakePipeline,
    ).eval()
    info = {
        "pixels": torch.zeros(1, 3, 1, 3, 16, 16, dtype=torch.uint8),
        "goal": torch.zeros(1, 3, 1, 3, 16, 16, dtype=torch.uint8),
    }
    candidates = torch.zeros(1, 3, 3, 4)
    candidates[:, 1] = 1.0
    first = model.get_cost(info, candidates)
    second = model.get_cost(info, candidates)
    assert first.shape == (1, 3)
    assert torch.isfinite(first).all()
    assert first[0, 0] < first[0, 1]
    torch.testing.assert_close(first, second)


def test_three_frame_context_rolls_multiple_future_steps():
    model = IRASimWorldModel(
        transformer=nn.Linear(1, 1),
        vae=FakeVAE(),
        scheduler=object(),
        action_frameskip=2,
        video_size=(16, 16),
        num_frames=4,
        num_history_frames=3,
        inference_steps=1,
        rollout_batch_size=2,
        action_scale=(1.0, 1.0),
        pipeline_factory=FakePipeline,
    ).eval()
    info = {
        "pixels": torch.zeros(1, 2, 3, 3, 16, 16, dtype=torch.uint8),
        "goal": torch.zeros(1, 2, 1, 3, 16, 16, dtype=torch.uint8),
        "history_actions": torch.zeros(1, 2, 2, 4),
    }
    candidates = torch.zeros(1, 2, 3, 4)
    candidates[:, 1] = 1.0
    costs = model.get_cost(info, candidates)
    assert costs.shape == (1, 2)
    assert costs[0, 0] < costs[0, 1]


def test_policy_accumulates_frame_aligned_visual_and_action_context():
    model = IRASimWorldModel(
        transformer=nn.Linear(1, 1),
        vae=FakeVAE(),
        scheduler=object(),
        action_frameskip=2,
        video_size=(16, 16),
        num_frames=4,
        num_history_frames=3,
        inference_steps=1,
        action_scale=(1.0, 1.0),
        pipeline_factory=FakePipeline,
    ).eval()
    solver = swm.solver.CEMSolver(model=model, num_samples=2, n_steps=1, topk=2)
    policy = IRASimWorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=1,
            receding_horizon=1,
            history_len=3,
            action_block=2,
        ),
        action_processor=model.action_processor,
        transform={
            "pixels": irasim_image_transform((16, 16)),
            "goal": irasim_image_transform((16, 16)),
        },
    )
    env = SimpleNamespace(
        num_envs=1,
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1, 2)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    policy.set_env(env)
    goal = np.ones((1, 1, 16, 16, 3), dtype=np.uint8)

    def prepare(value, action):
        return policy._prepare_info(
            {
                "pixels": np.full((1, 1, 16, 16, 3), value, dtype=np.uint8),
                "goal": goal,
                "action": np.asarray(action, dtype=np.float32).reshape(1, 1, 2),
            }
        )

    first = prepare(0, [np.nan, np.nan])
    second = prepare(1, [1.0, 2.0])
    third = prepare(2, [3.0, 4.0])
    torch.testing.assert_close(second["pixels"], first["pixels"])
    torch.testing.assert_close(
        third["pixels"][:, -1], torch.full_like(third["pixels"][:, -1], 2)
    )
    torch.testing.assert_close(
        third["history_actions"],
        torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]]),
    )


def test_train_and_planning_configs_share_frameskip_and_upstream_defaults():
    train_cfg = OmegaConf.load(
        REPO_ROOT / "scripts" / "train" / "config" / "irasim" / "pusht.yaml"
    )
    planning_cfg = OmegaConf.load(
        REPO_ROOT / "scripts" / "plan" / "config" / "irasim" / "pusht.yaml"
    )
    assert train_cfg.frameskip == planning_cfg.model.action_frameskip == 5
    assert planning_cfg.plan_config.action_block == 5
    assert train_cfg.num_frames == 4
    assert train_cfg.history_frames == planning_cfg.model.num_history_frames == 3
    assert planning_cfg.plan_config.history_len == 3
    assert train_cfg.future_frames == 1
    assert train_cfg.extras == 3
    assert train_cfg.mask_frame_num == 3
    assert train_cfg.max_train_steps == 300_000
    assert train_cfg.learning_rate == 1e-4
    assert train_cfg.clip_max_norm == 0.1
    assert train_cfg.gradient_accumulation_steps == 1
