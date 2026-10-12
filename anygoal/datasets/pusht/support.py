"""Reference-support and HDF5 helpers for generated PushT variants."""

from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import h5py
import hdf5plugin
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm


class EpisodeWriter(Protocol):
    """Minimal writer interface accepted by :func:`collect_accepted_episodes`."""

    def __enter__(self) -> "EpisodeWriter": ...

    def __exit__(self, *exc: object) -> None: ...

    def write_episode(self, episode: dict) -> None: ...


class BlockAngleSupportFilter:
    """Apply the predeclared PushT circular-angle support gates to episodes."""

    def __init__(
        self,
        reference_path: Path,
        bins: int = 256,
        minimum_fraction: float = 0.95,
    ) -> None:
        if bins < 4:
            raise ValueError("bins must be at least 4")
        if not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_fraction must be in (0, 1]")
        with h5py.File(reference_path, "r") as dataset:
            reference = self.wrap_angle(
                dataset["state"][:, 4].astype(np.float64)
            )

        self.bins = int(bins)
        self.minimum_fraction = float(minimum_fraction)
        self.edges = np.linspace(-np.pi, np.pi, self.bins + 1)
        width = float(self.edges[1] - self.edges[0])
        histogram, _ = np.histogram(reference, bins=self.edges)
        mean_angle = np.arctan2(
            np.sin(reference).mean(), np.cos(reference).mean()
        )
        centered = self.wrap_angle(reference - mean_angle)
        bandwidth = float(centered.std(ddof=0) * len(reference) ** (-1.0 / 5.0))
        sigma_bins = max(bandwidth / width, 0.5)
        self.density = gaussian_filter1d(
            histogram.astype(np.float64),
            sigma=sigma_bins,
            mode="wrap",
            truncate=4.0,
        )
        self.density /= self.density.sum() * width
        reference_density = self._lookup_density(reference)
        self.kde_threshold = float(np.quantile(reference_density, 0.01))

        reference_embedding = self._circular_embedding(reference)
        self.mean = reference_embedding.mean(axis=0)
        covariance = np.cov(reference_embedding, rowvar=False, bias=True)
        regularizer = max(float(np.trace(covariance) / 2.0) * 1e-8, 1e-12)
        self.inverse_covariance = np.linalg.inv(
            covariance + np.eye(2) * regularizer
        )
        reference_distance = self._mahalanobis_squared(reference_embedding)
        self.mahalanobis_threshold = float(
            np.quantile(reference_distance, 0.99)
        )

    @staticmethod
    def wrap_angle(values: np.ndarray) -> np.ndarray:
        return (values + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _circular_embedding(values: np.ndarray) -> np.ndarray:
        return np.column_stack((np.cos(values), np.sin(values)))

    def _lookup_density(self, values: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(
            self.edges, self.wrap_angle(values), side="right"
        ) - 1
        indices %= self.bins
        return self.density[indices]

    def _mahalanobis_squared(self, embedding: np.ndarray) -> np.ndarray:
        centered = embedding - self.mean
        return np.einsum(
            "ni,ij,nj->n",
            centered,
            self.inverse_covariance,
            centered,
            optimize=True,
        )

    def fractions(self, episode: dict) -> tuple[float, float]:
        state = np.asarray(episode["state"], dtype=np.float64)
        angles = self.wrap_angle(state[:, 4])
        kde_fraction = float(
            np.mean(self._lookup_density(angles) >= self.kde_threshold)
        )
        distance = self._mahalanobis_squared(
            self._circular_embedding(angles)
        )
        mahalanobis_fraction = float(
            np.mean(distance <= self.mahalanobis_threshold)
        )
        return kde_fraction, mahalanobis_fraction

    def accepts(self, episode: dict) -> bool:
        return all(
            fraction >= self.minimum_fraction
            for fraction in self.fractions(episode)
        )


class PushTJointSupportFilter:
    """Gate block-position and action episodes using the unified 2D metrics."""

    def __init__(
        self,
        reference_path: Path,
        bins: int = 256,
        minimum_fraction: float = 0.95,
        minimum_range_fraction: float = 0.99,
    ) -> None:
        if bins < 4:
            raise ValueError("bins must be at least 4")
        if not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_fraction must be in (0, 1]")
        if not 0.0 < minimum_range_fraction <= 1.0:
            raise ValueError("minimum_range_fraction must be in (0, 1]")
        with h5py.File(reference_path, "r") as dataset:
            reference_features = {
                "block_position": dataset["state"][:, 2:4].astype(np.float64),
                "action": dataset["action"][:].astype(np.float64),
            }
        self.bins = int(bins)
        self.minimum_fraction = float(minimum_fraction)
        self.minimum_range_fraction = float(minimum_range_fraction)
        self.models = {
            name: self._fit_feature(values)
            for name, values in reference_features.items()
        }

    def _fit_feature(self, reference: np.ndarray) -> dict:
        lower = reference.min(axis=0)
        upper = reference.max(axis=0)
        padding = 0.03 * (upper - lower)
        limits = np.column_stack((lower - padding, upper + padding))
        histogram, edges = np.histogramdd(
            reference, bins=self.bins, range=limits.tolist()
        )
        widths = np.array([edge[1] - edge[0] for edge in edges])
        bandwidth = reference.std(axis=0) * len(reference) ** (-1.0 / 6.0)
        sigma_bins = np.maximum(bandwidth / widths, 0.5)
        density = gaussian_filter(
            histogram.astype(np.float32),
            sigma=sigma_bins,
            mode="constant",
            truncate=4.0,
        )
        density /= density.sum(dtype=np.float64) * np.prod(widths)
        mean = reference.mean(axis=0)
        covariance = np.cov(reference, rowvar=False, bias=True)
        regularizer = max(float(np.trace(covariance) / 2.0) * 1e-8, 1e-12)
        inverse_covariance = np.linalg.inv(
            covariance + np.eye(2) * regularizer
        )
        model = {
            "edges": edges,
            "density": density,
            "mean": mean,
            "inverse_covariance": inverse_covariance,
            "lower": lower,
            "upper": upper,
        }
        reference_density = self._lookup_density(reference, model)
        reference_distance = self._mahalanobis_squared(reference, model)
        model["kde_threshold"] = float(
            np.quantile(reference_density, 0.01)
        )
        model["mahalanobis_threshold"] = float(
            np.quantile(reference_distance, 0.99)
        )
        return model

    @staticmethod
    def _lookup_density(values: np.ndarray, model: dict) -> np.ndarray:
        indices = [
            np.searchsorted(edge, values[:, axis], side="right") - 1
            for axis, edge in enumerate(model["edges"])
        ]
        inside = np.ones(len(values), dtype=bool)
        for axis, index in enumerate(indices):
            inside &= (index >= 0) & (index < model["density"].shape[axis])
        density = np.zeros(len(values), dtype=np.float64)
        density[inside] = model["density"][
            tuple(index[inside] for index in indices)
        ]
        return density

    @staticmethod
    def _mahalanobis_squared(values: np.ndarray, model: dict) -> np.ndarray:
        centered = values - model["mean"]
        return np.einsum(
            "ni,ij,nj->n",
            centered,
            model["inverse_covariance"],
            centered,
            optimize=True,
        )

    def fractions(self, episode: dict) -> dict[str, dict[str, float]]:
        features = {
            "block_position": np.asarray(episode["state"])[:, 2:4],
            "action": np.asarray(episode["action"]),
        }
        result = {}
        for name, values in features.items():
            values = values.astype(np.float64)
            model = self.models[name]
            result[name] = {
                "kde": float(
                    np.mean(
                        self._lookup_density(values, model)
                        >= model["kde_threshold"]
                    )
                ),
                "mahalanobis": float(
                    np.mean(
                        self._mahalanobis_squared(values, model)
                        <= model["mahalanobis_threshold"]
                    )
                ),
                "range": float(
                    np.mean(
                        np.all(
                            (values >= model["lower"])
                            & (values <= model["upper"]),
                            axis=1,
                        )
                    )
                ),
            }
        return result

    def accepts(self, episode: dict) -> bool:
        return all(
            metrics["kde"] >= self.minimum_fraction
            and metrics["mahalanobis"] >= self.minimum_fraction
            and metrics["range"] >= self.minimum_range_fraction
            for metrics in self.fractions(episode).values()
        )


def collect_accepted_episodes(
    world,
    writer: EpisodeWriter,
    episodes: int,
    min_frames: int,
    seed: int,
    columns: set[str] | None = None,
    accept_episode: Callable[[dict], bool] | None = None,
    complete_episode: Callable[[object, int], bool] | None = None,
) -> dict[str, int]:
    """Collect successful trajectories that meet a minimum episode length."""
    buffers = [defaultdict(list) for _ in range(world.num_envs)]

    def on_step(current_world) -> None:
        if complete_episode is not None:
            for env_idx in range(current_world.num_envs):
                if complete_episode(current_world, env_idx):
                    current_world.terminateds[env_idx] = True
        for column, data in current_world.infos.items():
            if columns is not None and column not in columns:
                continue
            if column.startswith("_") or not isinstance(
                data, (np.ndarray, torch.Tensor)
            ):
                continue
            if data.ndim > 1 and data.shape[1] == 1:
                data = data.squeeze(1)
            for env_idx in range(current_world.num_envs):
                value = data[env_idx]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                elif isinstance(value, np.ndarray):
                    value = value.copy()
                buffers[env_idx][column].append(value)

    accepted = 0
    attempted = 0
    rejected_short = 0
    rejected_failed = 0
    rejected_filtered = 0
    max_attempts = max(episodes * 100, world.num_envs)

    with writer as active_writer, tqdm(
        total=episodes, desc="Accepted episodes"
    ) as progress:
        rollout = world._run_iter(
            episodes=max_attempts,
            seed=seed,
            mode="auto",
            on_step=on_step,
        )
        for env_idx, _ in rollout:
            attempted += 1
            episode = {
                key: list(values) for key, values in buffers[env_idx].items()
            }
            buffers[env_idx].clear()
            if not episode:
                raise RuntimeError("The world produced no numeric info columns.")
            episode_length = len(next(iter(episode.values())))
            succeeded = bool(world.terminateds[env_idx])

            if not succeeded:
                rejected_failed += 1
                continue
            if episode_length < min_frames:
                rejected_short += 1
                continue
            if accept_episode is not None and not accept_episode(episode):
                rejected_filtered += 1
                continue

            if "action" in episode:
                episode["action"].append(episode["action"].pop(0))
            active_writer.write_episode(episode)
            accepted += 1
            progress.update(1)
            if accepted == episodes:
                rollout.close()
                break
        else:
            raise RuntimeError(
                f"Collected only {accepted}/{episodes} accepted episodes "
                f"after {max_attempts} attempts."
            )

    return {
        "attempted": attempted,
        "accepted": accepted,
        "rejected_short": rejected_short,
        "rejected_failed": rejected_failed,
        "rejected_filtered": rejected_filtered,
    }


def complete_on_block_goal(world, env_idx: int) -> bool:
    """End collection when the block reaches its goal; retain the actual pusher state."""
    env = world.envs.envs[env_idx].unwrapped
    state = env._get_obs()
    return bool(env.block_goal_reached(env.goal_state, state))


class PushTTrainSchemaWriter:
    """Write the compact schema used by the PushT training dataset."""

    def __init__(self, path: Path, resolution: int = 224) -> None:
        self.path = Path(path)
        self.resolution = resolution
        self.file: h5py.File | None = None
        self.opened_once = False

    def __enter__(self) -> "PushTTrainSchemaWriter":
        if not self.opened_once:
            if self.path.exists():
                raise FileExistsError(f"Refusing to overwrite {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = h5py.File(self.path, "w", libver="latest")
            self._create_schema()
            self.opened_once = True
        else:
            self.file = h5py.File(self.path, "a", libver="latest")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _create_schema(self) -> None:
        assert self.file is not None
        columns = {
            "action": ((2,), np.float32, (1000, 2), {}),
            "pixels": (
                (self.resolution, self.resolution, 3),
                np.uint8,
                (100, self.resolution, self.resolution, 3),
                hdf5plugin.Blosc(
                    cname="lz4",
                    clevel=5,
                    shuffle=hdf5plugin.Blosc.SHUFFLE,
                ),
            ),
            "proprio": ((4,), np.float32, (1000, 4), {}),
            "state": ((7,), np.float32, (1000, 7), {}),
        }
        for name, (shape, dtype, chunks, compression) in columns.items():
            self.file.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=chunks,
                dtype=dtype,
                **compression,
            )
        for name in ("episode_idx", "step_idx"):
            self.file.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(1000,),
                dtype=np.int64,
            )
        self.file.create_dataset(
            "ep_len",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.int32,
        )
        self.file.create_dataset(
            "ep_offset",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.int64,
        )

    def write_episode(self, episode: dict) -> None:
        assert self.file is not None
        arrays = {
            "action": np.asarray(episode["action"], dtype=np.float32),
            "pixels": np.asarray(episode["pixels"], dtype=np.uint8),
            "proprio": np.asarray(episode["proprio"], dtype=np.float32),
            "state": np.asarray(episode["state"], dtype=np.float32),
        }
        episode_length = len(arrays["action"])
        if any(len(values) != episode_length for values in arrays.values()):
            raise ValueError("Episode columns have inconsistent lengths.")

        episode_index = self.file["ep_len"].shape[0]
        episode_offset = self.file["action"].shape[0]
        arrays["episode_idx"] = np.full(
            episode_length, episode_index, dtype=np.int64
        )
        arrays["step_idx"] = np.arange(episode_length, dtype=np.int64)

        end = episode_offset + episode_length
        for name, values in arrays.items():
            dataset = self.file[name]
            dataset.resize(end, axis=0)
            dataset[episode_offset:end] = values

        self.file["ep_len"].resize(episode_index + 1, axis=0)
        self.file["ep_len"][episode_index] = episode_length
        self.file["ep_offset"].resize(episode_index + 1, axis=0)
        self.file["ep_offset"][episode_index] = episode_offset
