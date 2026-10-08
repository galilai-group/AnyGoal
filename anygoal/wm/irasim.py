"""Read-only adapters between vendored IRASim and Stable WorldModel."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from stable_worldmodel.policy import WorldModelPolicy
from torch import nn
from torchvision.transforms import v2 as transforms


def irasim_source_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "worldmodels" / "IRASim"
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


class IRASimActionProcessor:
    """Z-score CEM actions while IRASim receives its original scaled units."""

    def __init__(self, mean=None, scale=None) -> None:
        mean = np.zeros(2, dtype=np.float32) if mean is None else mean
        scale = np.ones(2, dtype=np.float32) if scale is None else scale
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.ndim != 1 or self.mean.shape != self.scale.shape:
            raise ValueError("Action mean and scale must be same-sized 1-D arrays.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Action statistics must be finite.")
        if np.any(self.scale <= 0):
            raise ValueError("Action scale must be positive.")

    @classmethod
    def fit(cls, values) -> "IRASimActionProcessor":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Action statistics require a 2-D array.")
        values = values[np.isfinite(values).all(axis=1)]
        if not len(values):
            raise ValueError("Cannot fit action statistics without finite rows.")
        return cls(values.mean(axis=0), values.std(axis=0))

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.scale

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
        return values * scale + mean


class IRASimHDF5Dataset(torch.utils.data.Dataset):
    """Expose StableWM H5 clips in the original IRASim batch format."""

    action_dim = 2

    def __init__(
        self,
        *,
        name: str,
        cache_dir: str | Path,
        num_frames: int,
        frameskip: int,
        video_size: tuple[int, int] | list[int],
        action_scale=(20.0, 20.0),
        action_dim: int = 2,
        episode_indices: np.ndarray | list[int] | None = None,
        max_samples: int | None = None,
    ) -> None:
        if num_frames < 2 or frameskip < 1:
            raise ValueError("num_frames must be at least 2 and frameskip positive.")
        dataset_path = Path(name).expanduser()
        location = (
            {"path": dataset_path}
            if dataset_path.is_absolute() or dataset_path.suffix in {".h5", ".hdf5"}
            else {"name": name, "cache_dir": Path(cache_dir).expanduser()}
        )
        self.dataset = swm.data.HDF5Dataset(
            **location,
            frameskip=int(frameskip),
            num_steps=int(num_frames),
            keys_to_load=["pixels", "action"],
            keys_to_cache=["action"],
        )
        self.action_dim = int(action_dim)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive.")
        if self.dataset.get_dim("action") != self.action_dim:
            raise ValueError(
                f"IRASim expected {self.action_dim}-D actions, got "
                f"{self.dataset.get_dim('action')}."
            )
        self.num_frames = int(num_frames)
        self.frameskip = int(frameskip)
        self.video_size = tuple(int(value) for value in video_size)
        if len(self.video_size) != 2 or any(value < 8 for value in self.video_size):
            raise ValueError("video_size must contain two dimensions of at least 8.")
        action_scale = np.asarray(action_scale, dtype=np.float32)
        if action_scale.shape != (self.action_dim,):
            raise ValueError(f"action_scale must have shape ({self.action_dim},).")
        self.action_scale = torch.from_numpy(
            np.tile(action_scale, self.frameskip)
        )

        allowed_episodes = (
            None
            if episode_indices is None
            else set(np.asarray(episode_indices, dtype=np.int64).tolist())
        )
        self.sample_indices = [
            index
            for index, (episode, _) in enumerate(self.dataset.clip_indices)
            if allowed_episodes is None or episode in allowed_episodes
        ]
        if max_samples is not None:
            if int(max_samples) < 1:
                raise ValueError("max_samples must be positive when set.")
            self.sample_indices = self.sample_indices[: int(max_samples)]
        selected_episodes = (
            range(len(self.dataset.lengths))
            if allowed_episodes is None
            else sorted(allowed_episodes)
        )
        self.ann_files = [f"h5_episode_{episode}" for episode in selected_episodes]
        self.samples = self.sample_indices
        self.training = False

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(
        self,
        index: int,
        cam_id: int | None = None,
        return_video: bool = False,
    ) -> dict[str, Any]:
        del cam_id, return_video
        source_index = self.sample_indices[index]
        sample = self.dataset[source_index]
        video = sample["pixels"]
        if video.dtype != torch.uint8:
            raise TypeError(f"IRASim expects uint8 H5 pixels, got {video.dtype}.")
        video = F.interpolate(
            video.float().div(255.0),
            size=self.video_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        video = video.sub(0.5).div(0.5)
        # HDF5Dataset groups dense actions into one flat block per strided
        # observation. The last observation has no outgoing transition.
        actions = sample["action"][:-1].float() * self.action_scale
        episode, start = self.dataset.clip_indices[source_index]
        return {
            "video": video,
            "action": actions,
            "video_name": {
                "episode_id": str(episode),
                "start_frame_id": str(start),
                "cam_id": "0",
            },
        }


def get_irasim_hdf5_datasets(args):
    """Drop-in replacement for the upstream IRASim ``get_dataset``."""
    if bool(args.pre_encode):
        raise ValueError(
            "StableWM H5 stores RGB frames, so IRASim H5 training requires "
            "pre_encode=false and uses the original online VAE path."
        )
    validation_fraction = float(args.get("validation_fraction", 0.05))
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    probe_name = Path(str(args.h5_dataset_name)).expanduser()
    probe_location = (
        {"path": probe_name}
        if probe_name.is_absolute() or probe_name.suffix in {".h5", ".hdf5"}
        else {
            "name": str(args.h5_dataset_name),
            "cache_dir": Path(args.h5_dataset_root).expanduser(),
        }
    )
    probe = swm.data.HDF5Dataset(
        **probe_location,
        keys_to_load=["action"],
        keys_to_cache=["action"],
    )
    episode_count = len(probe.lengths)
    generator = np.random.default_rng(int(args.global_seed))
    shuffled = generator.permutation(episode_count)
    validation_count = max(1, round(episode_count * validation_fraction))
    validation_episodes = np.sort(shuffled[:validation_count])
    training_episodes = np.sort(shuffled[validation_count:])
    if not len(training_episodes):
        raise ValueError("The H5 file needs at least two episodes for a split.")

    common = {
        "name": str(args.h5_dataset_name),
        "cache_dir": args.h5_dataset_root,
        "num_frames": int(args.num_frames),
        "frameskip": int(args.frameskip),
        "video_size": tuple(args.video_size),
        "action_scale": tuple(args.action_scale),
        "action_dim": int(args.get("action_dim", 2)),
    }
    max_samples = args.get("max_samples")
    max_validation_samples = args.get("max_validation_samples", max_samples)
    train = IRASimHDF5Dataset(
        **common,
        episode_indices=training_episodes,
        max_samples=None if max_samples is None else int(max_samples),
    )
    validation = IRASimHDF5Dataset(
        **common,
        episode_indices=validation_episodes,
        max_samples=(
            None if max_validation_samples is None else int(max_validation_samples)
        ),
    )
    return train, validation


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


class IRASimWorldModel(nn.Module):
    """IRASim video diffusion model implementing StableWM's cost contract."""

    action_dim = 2

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        scheduler,
        action_dim: int = 2,
        action_frameskip: int = 1,
        video_size=(224, 224),
        num_frames: int = 16,
        num_history_frames: int = 1,
        inference_steps: int = 50,
        guidance_scale: float = 1.0,
        rollout_batch_size: int = 1,
        num_rollout_samples: int = 1,
        rollout_seed: int = 0,
        action_scale=(20.0, 20.0),
        action_processor: IRASimActionProcessor | None = None,
        cost_type: str = "pixel_mse",
        pipeline_factory=None,
    ) -> None:
        super().__init__()
        if min(
            action_frameskip,
            num_history_frames,
            num_frames - num_history_frames,
            inference_steps,
            rollout_batch_size,
            num_rollout_samples,
        ) < 1:
            raise ValueError("IRASim frame, sampling, and rollout sizes must be positive.")
        if guidance_scale != 1.0:
            raise ValueError(
                "The vendored IRASim pipeline does not duplicate actions for CFG; "
                "guidance_scale must remain 1.0."
            )
        if cost_type not in {"pixel_mse", "latent_mse"}:
            raise ValueError("cost_type must be pixel_mse or latent_mse.")
        self.action_dim = int(action_dim)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive.")
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.action_frameskip = int(action_frameskip)
        self.model_action_dim = self.action_dim * self.action_frameskip
        self.video_size = tuple(int(value) for value in video_size)
        self.num_frames = int(num_frames)
        self.num_history_frames = int(num_history_frames)
        self.num_future_frames = self.num_frames - self.num_history_frames
        self.inference_steps = int(inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.rollout_batch_size = int(rollout_batch_size)
        self.num_rollout_samples = int(num_rollout_samples)
        self.rollout_seed = int(rollout_seed)
        self.cost_type = str(cost_type)
        self._action_processor = action_processor or IRASimActionProcessor()
        action_scale = np.asarray(action_scale, dtype=np.float32)
        if action_scale.shape != (self.action_dim,):
            raise ValueError(f"action_scale must have shape ({self.action_dim},).")
        self.register_buffer(
            "action_scale",
            torch.from_numpy(action_scale),
            persistent=True,
        )
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    @property
    def action_processor(self) -> IRASimActionProcessor:
        return self._action_processor

    def prepare_model_actions(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"Expected blocked action dim {self.model_action_dim}, got "
                f"{candidates.shape[-1]}."
            )
        blocked = candidates.reshape(
            *candidates.shape[:-1], self.action_frameskip, self.action_dim
        )
        raw = self.action_processor.inverse_transform_tensor(blocked)
        scaled = raw * self.action_scale.to(raw)
        return scaled.flatten(-2)

    def preprocess_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.shape[-1] == 3 and pixels.shape[-3] != 3:
            pixels = pixels.movedim(-1, -3)
        if pixels.shape[-3] != 3:
            raise ValueError(f"Expected RGB pixels, got {tuple(pixels.shape)}.")
        was_uint8 = pixels.dtype == torch.uint8
        pixels = pixels.float()
        if was_uint8 or pixels.detach().amax() > 1.0:
            pixels = pixels.div(255.0)
        leading = pixels.shape[:-3]
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
        if tuple(pixels.shape[-2:]) != self.video_size:
            pixels = F.interpolate(
                pixels,
                size=self.video_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pixels = pixels.mul(2.0).sub(1.0)
        return pixels.reshape(*leading, 3, *self.video_size)

    def _encode(self, images: torch.Tensor, seed: int) -> torch.Tensor:
        cuda_devices = (
            [images.device.index]
            if images.device.type == "cuda" and images.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            latent = self.vae.encode(images).latent_dist.sample()
        return latent * self.vae.config.scaling_factor

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = []
        scale = self.vae.config.scaling_factor
        for start in range(0, len(latents), self.rollout_batch_size):
            decoded.append(
                self.vae.decode(latents[start : start + self.rollout_batch_size] / scale).sample
            )
        return torch.cat(decoded)

    def _get_pipeline(self):
        if self._pipeline is None:
            factory = self.pipeline_factory or get_irasim_pipeline_class()
            self._pipeline = factory(
                vae=self.vae,
                scheduler=self.scheduler,
                transformer=self.transformer,
            )
        return self._pipeline

    def _rollout_once(
        self,
        context_latents: torch.Tensor,
        history_actions: torch.Tensor,
        actions: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        batch_size, num_candidates, horizon = actions.shape[:3]
        flat_actions = actions.reshape(-1, horizon, self.model_action_dim)
        flat_history_actions = (
            history_actions.reshape(
                -1, self.num_history_frames - 1, self.model_action_dim
            )
            if self.num_history_frames > 1
            else flat_actions.new_empty(len(flat_actions), 0, self.model_action_dim)
        )
        env_indices = torch.arange(batch_size, device=actions.device).repeat_interleave(
            num_candidates
        )
        current_context = context_latents[:, None].expand(
            batch_size, num_candidates, *context_latents.shape[1:]
        ).reshape(-1, *context_latents.shape[1:])
        current_history_actions = flat_history_actions
        pipeline = self._get_pipeline()
        generator = torch.Generator(device=actions.device)
        for segment_start in range(0, horizon, self.num_future_frames):
            segment_end = min(segment_start + self.num_future_frames, horizon)
            segment_length = segment_end - segment_start
            generator.manual_seed(seed + segment_start)
            shared_noise = torch.randn(
                batch_size,
                segment_length,
                *current_context.shape[2:],
                generator=generator,
                device=current_context.device,
                dtype=current_context.dtype,
            )
            segment_outputs = []
            for start in range(0, len(current_context), self.rollout_batch_size):
                end = min(start + self.rollout_batch_size, len(current_context))
                mask_x = current_context[start:end]
                initial_noise = torch.cat(
                    [mask_x, shared_noise[env_indices[start:end]]], dim=1
                )
                segment_actions = torch.cat(
                    [
                        current_history_actions[start:end],
                        flat_actions[start:end, segment_start:segment_end],
                    ],
                    dim=1,
                )
                _, latent_video = pipeline(
                    segment_actions,
                    mask_x=mask_x,
                    video_length=self.num_history_frames + segment_length,
                    height=self.video_size[0],
                    width=self.video_size[1],
                    num_inference_steps=self.inference_steps,
                    guidance_scale=self.guidance_scale,
                    device=actions.device,
                    latents=initial_noise,
                    output_type="latent",
                    return_dict=False,
                )
                segment_outputs.append(latent_video[:, -segment_length:])
            predicted = torch.cat(segment_outputs)
            current_context = torch.cat([current_context, predicted], dim=1)[
                :, -self.num_history_frames :
            ]
            if self.num_history_frames > 1:
                current_history_actions = torch.cat(
                    [
                        current_history_actions,
                        flat_actions[:, segment_start:segment_end],
                    ],
                    dim=1,
                )[:, -(self.num_history_frames - 1) :]
        terminal = current_context[:, -1]
        return terminal.reshape(
            batch_size, num_candidates, *terminal.shape[1:]
        )

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict[str, torch.Tensor],
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if "pixels" not in info_dict or "goal" not in info_dict:
            raise KeyError("IRASim planning requires pixels and goal.")
        if action_candidates.ndim != 4:
            raise ValueError("action_candidates must have shape (B,S,H,D).")
        batch_size, num_candidates = action_candidates.shape[:2]
        pixels = info_dict["pixels"]
        goals = info_dict["goal"]
        if pixels.ndim != 6 or goals.ndim != 6:
            raise ValueError("Expanded pixels/goal must have shape (B,S,T,C,H,W).")
        if pixels.shape[:2] != (batch_size, num_candidates):
            raise ValueError("Info and action candidate axes do not match.")

        if pixels.shape[2] < self.num_history_frames:
            raise ValueError(
                f"IRASim requires {self.num_history_frames} context frames, got "
                f"{pixels.shape[2]}."
            )
        current_pixels = self.preprocess_pixels(
            pixels[:, 0, -self.num_history_frames :]
        )
        goal_pixels = self.preprocess_pixels(goals[:, 0, -1])
        flat_context_latents = self._encode(
            current_pixels.flatten(0, 1), self.rollout_seed
        )
        context_latents = flat_context_latents.reshape(
            batch_size,
            self.num_history_frames,
            *flat_context_latents.shape[1:],
        )
        goal_latent = (
            self._encode(goal_pixels, self.rollout_seed + 1)
            if self.cost_type == "latent_mse"
            else None
        )
        model_actions = self.prepare_model_actions(action_candidates)
        if self.num_history_frames > 1:
            if "history_actions" not in info_dict:
                raise KeyError("IRASim planning requires frame-aligned history_actions.")
            history_actions = info_dict["history_actions"]
            expected_shape = (
                batch_size,
                num_candidates,
                self.num_history_frames - 1,
                self.model_action_dim,
            )
            if tuple(history_actions.shape) != expected_shape:
                raise ValueError(
                    f"Expected history_actions {expected_shape}, got "
                    f"{tuple(history_actions.shape)}."
                )
            history_actions = self.prepare_model_actions(history_actions)
        else:
            history_actions = model_actions.new_empty(
                batch_size, num_candidates, 0, self.model_action_dim
            )
        costs = []
        for rollout_index in range(self.num_rollout_samples):
            terminal = self._rollout_once(
                context_latents,
                history_actions,
                model_actions,
                self.rollout_seed + 10_000 * rollout_index,
            )
            if self.cost_type == "latent_mse":
                target = goal_latent[:, None]
            else:
                flat_terminal = terminal.flatten(0, 1)
                decoded = self._decode(flat_terminal).reshape(
                    batch_size, num_candidates, 3, *self.video_size
                )
                terminal = decoded.clamp(-1.0, 1.0)
                target = goal_pixels[:, None]
            costs.append((terminal - target).pow(2).mean(dim=(-3, -2, -1)))
        return torch.stack(costs).mean(dim=0)


class IRASimWorldModelPolicy(WorldModelPolicy):
    """Maintain frame-skip-aligned visual/action context for IRASim MPC."""

    def __init__(
        self,
        *,
        action_processor: IRASimActionProcessor | None = None,
        **kwargs,
    ) -> None:
        processor = action_processor or IRASimActionProcessor()
        process = dict(kwargs.pop("process", {}) or {})
        process["action"] = processor
        super().__init__(process=process, **kwargs)
        model = self.solver.model
        self.action_frameskip = int(model.action_frameskip)
        self.num_history_frames = int(model.num_history_frames)
        if self.cfg.action_block != int(model.action_frameskip):
            raise ValueError(
                "plan_config.action_block must match IRASim's training "
                f"frameskip ({model.action_frameskip})."
            )
        if self.cfg.history_len != self.num_history_frames:
            raise ValueError(
                "plan_config.history_len must match IRASim's training context "
                f"({self.num_history_frames})."
            )
        self._pixel_history: list[deque[torch.Tensor]] | None = None
        self._action_history: list[deque[torch.Tensor]] | None = None
        self._pending_actions: list[deque[torch.Tensor]] | None = None

    def set_env(self, env) -> None:
        super().set_env(env)
        self._pixel_history = [
            deque(maxlen=self.num_history_frames) for _ in range(env.num_envs)
        ]
        self._action_history = [
            deque(maxlen=self.num_history_frames - 1) for _ in range(env.num_envs)
        ]
        self._pending_actions = [
            deque(maxlen=self.action_frameskip) for _ in range(env.num_envs)
        ]

    @staticmethod
    def _latest(values: torch.Tensor) -> torch.Tensor:
        return values[:, -1] if values.ndim >= 3 else values

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        if (
            self._pixel_history is None
            or self._action_history is None
            or self._pending_actions is None
        ):
            raise RuntimeError("Set the policy environment before requesting actions.")
        prepared = super()._prepare_info(info_dict)
        if "pixels" not in prepared or "goal" not in prepared:
            raise KeyError("IRASim planning requires pixels and goal observations.")
        if self.num_history_frames > 1 and "action" not in prepared:
            raise KeyError("IRASim history alignment requires the last action.")

        current_pixels = self._latest(prepared["pixels"])
        current_actions = (
            torch.nan_to_num(self._latest(prepared["action"]).float(), nan=0.0)
            if self.num_history_frames > 1
            else None
        )
        needs_flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(len(current_pixels), dtype=bool))
        ).reshape(len(current_pixels), -1)[:, 0]

        pixel_batches = []
        action_batches = []
        for index, pixel in enumerate(current_pixels):
            if needs_flush[index]:
                self._pixel_history[index].clear()
                self._action_history[index].clear()
                self._pending_actions[index].clear()
            if not self._pixel_history[index]:
                self._pixel_history[index].append(pixel)
            elif self.num_history_frames > 1:
                pending = self._pending_actions[index]
                pending.append(current_actions[index])
                if len(pending) == self.action_frameskip:
                    self._pixel_history[index].append(pixel)
                    self._action_history[index].append(torch.cat(list(pending)))
                    pending.clear()

            pixels_for_env = list(self._pixel_history[index])
            pixels_for_env = [pixels_for_env[0]] * (
                self.num_history_frames - len(pixels_for_env)
            ) + pixels_for_env
            pixel_batches.append(torch.stack(pixels_for_env))

            if self.num_history_frames > 1:
                actions_for_env = list(self._action_history[index])
                zero_action = torch.zeros_like(current_actions[index]).repeat(
                    self.action_frameskip
                )
                actions_for_env = [zero_action] * (
                    self.num_history_frames - 1 - len(actions_for_env)
                ) + actions_for_env
                action_batches.append(torch.stack(actions_for_env))

        output = {
            "pixels": torch.stack(pixel_batches),
            "goal": prepared["goal"][:, -1:],
        }
        if self.num_history_frames > 1:
            output["history_actions"] = torch.stack(action_batches)
        if "_needs_flush" in info_dict:
            output["_needs_flush"] = torch.as_tensor(needs_flush)
        if "terminated" in prepared:
            output["terminated"] = prepared["terminated"]
        return output


def irasim_image_transform(video_size=(224, 224)):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(tuple(video_size), antialias=True),
        ]
    )


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


__all__ = [
    "IRASimActionProcessor",
    "IRASimHDF5Dataset",
    "IRASimWorldModel",
    "IRASimWorldModelPolicy",
    "adapt_irasim_action_frameskip",
    "create_irasim_transformer",
    "get_irasim_hdf5_datasets",
    "irasim_image_transform",
    "irasim_source_root",
    "load_irasim_upstream_train_module",
]
