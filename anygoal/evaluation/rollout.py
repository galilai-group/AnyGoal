"""Shared episode sampling and aggregation for open-loop rollout evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutProtocol:
    history_length: int = 3
    frame_skip: int = 5
    horizons: tuple[int, ...] = (15, 25)
    num_episodes: int = 100
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.history_length < 1 or self.frame_skip < 1:
            raise ValueError("history_length and frame_skip must be positive.")
        if self.num_episodes < 1 or self.batch_size < 1:
            raise ValueError("num_episodes and batch_size must be positive.")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("At least one positive rollout horizon is required.")
        if any(horizon % self.frame_skip for horizon in self.horizons):
            raise ValueError("Every horizon must be divisible by frame_skip.")

    @property
    def max_rollout_steps(self) -> int:
        return max(self.horizons) // self.frame_skip

    @property
    def observation_offsets(self) -> torch.Tensor:
        count = self.history_length + self.max_rollout_steps
        return torch.arange(count, dtype=torch.long) * self.frame_skip

    @property
    def action_offsets(self) -> torch.Tensor:
        count = self.history_length - 1 + self.max_rollout_steps
        block_starts = torch.arange(count, dtype=torch.long) * self.frame_skip
        within_block = torch.arange(self.frame_skip, dtype=torch.long)
        return block_starts[:, None] + within_block[None]

    @property
    def required_episode_length(self) -> int:
        return int(self.observation_offsets[-1]) + 1

    def required_episode_length_for_steps(self, rollout_steps: int) -> int:
        if rollout_steps < 1 or rollout_steps > self.max_rollout_steps:
            raise ValueError("rollout_steps is outside the configured horizons.")
        return (self.history_length + rollout_steps - 1) * self.frame_skip + 1


class RolloutAdapter(Protocol):
    """Model-specific native-space rollout used by the shared evaluator."""

    model_name: str
    error_space: str
    observation_keys: tuple[str, ...]

    def rollout_errors(
        self,
        observations: dict[str, torch.Tensor],
        action_blocks: torch.Tensor,
        *,
        rollout_seed: int,
    ) -> torch.Tensor:
        """Return per-example errors shaped ``(batch, rollout_steps)``."""


def sample_episode_indices(
    lengths: np.ndarray,
    *,
    required_length: int,
    num_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample eligible internal episode indices reproducibly."""
    lengths = np.asarray(lengths, dtype=np.int64)
    eligible = np.flatnonzero(lengths >= int(required_length))
    if len(eligible) < num_episodes:
        raise ValueError(
            f"Only {len(eligible)} episodes have at least {required_length} steps; "
            f"need {num_episodes}."
        )
    generator = np.random.default_rng(int(seed))
    selected = np.sort(generator.choice(eligible, size=num_episodes, replace=False))
    return selected, eligible


def select_evaluation_episodes(
    lengths: np.ndarray,
    *,
    protocol: RolloutProtocol,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Select episodes, falling back to short-only evaluation when necessary."""
    lengths = np.asarray(lengths, dtype=np.int64)
    full_eligible = np.flatnonzero(lengths >= protocol.required_episode_length)
    if len(full_eligible) >= protocol.num_episodes:
        selected, eligible = sample_episode_indices(
            lengths,
            required_length=protocol.required_episode_length,
            num_episodes=protocol.num_episodes,
            seed=seed,
        )
        return selected, eligible, protocol.max_rollout_steps, len(full_eligible)

    short_required_length = protocol.required_episode_length_for_steps(1)
    selected, eligible = sample_episode_indices(
        lengths,
        required_length=short_required_length,
        num_episodes=protocol.num_episodes,
        seed=seed,
    )
    return selected, eligible, 1, len(full_eligible)


def _episode_identifier(episode: dict, fallback: int) -> int | str:
    for key in ("episode_idx", "ep_idx", "id"):
        if key not in episode:
            continue
        value = episode[key]
        if torch.is_tensor(value):
            value = value.flatten()[0].item()
        elif isinstance(value, np.ndarray):
            value = value.reshape(-1)[0].item()
        if isinstance(value, (np.integer, int)):
            return int(value)
        return str(value)
    return int(fallback)


def _gather_batch(
    episode: dict,
    starts: torch.Tensor,
    protocol: RolloutProtocol,
    observation_keys: tuple[str, ...],
    rollout_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    observation_offsets = torch.arange(
        protocol.history_length + rollout_steps, dtype=torch.long
    ) * protocol.frame_skip
    action_block_starts = torch.arange(
        protocol.history_length - 1 + rollout_steps, dtype=torch.long
    ) * protocol.frame_skip
    action_offsets = action_block_starts[:, None] + torch.arange(
        protocol.frame_skip, dtype=torch.long
    )[None]
    observation_indices = starts[:, None] + observation_offsets[None]
    action_indices = starts[:, None, None] + action_offsets[None]

    observations = {}
    for key in observation_keys:
        if key not in episode:
            raise KeyError(f"Dataset episode is missing required column {key!r}.")
        values = torch.as_tensor(episode[key])
        observations[key] = values[observation_indices]

    if "action" not in episode:
        raise KeyError("Dataset episode is missing required column 'action'.")
    raw_actions = torch.as_tensor(episode["action"])[action_indices]
    action_blocks = raw_actions.flatten(start_dim=2)
    return observations, action_blocks


def _sampling_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def evaluate_rollout(
    dataset,
    adapter: RolloutAdapter,
    *,
    protocol: RolloutProtocol,
    seed: int,
    metadata: dict,
    max_starts_per_episode: int | None = None,
    progress_path: Path | None = None,
    episode_start_position: int = 0,
    episode_end_position: int | None = None,
) -> dict:
    """Evaluate short and autoregressive errors with one shared protocol.

    The primary aggregate gives every sampled episode equal weight. The JSON
    also includes start-weighted aggregates and every per-episode average.
    """
    selected, eligible, step_count, full_eligible_count = select_evaluation_episodes(
        dataset.lengths,
        protocol=protocol,
        seed=seed,
    )
    episode_end_position = (
        len(selected) if episode_end_position is None else episode_end_position
    )
    if not 0 <= episode_start_position < episode_end_position <= len(selected):
        raise ValueError(
            "Episode position range must satisfy "
            f"0 <= start < end <= {len(selected)}."
        )
    selected = selected[episode_start_position:episode_end_position]
    long_horizon_available = step_count == protocol.max_rollout_steps
    required_episode_length = protocol.required_episode_length_for_steps(step_count)
    horizon_steps = {
        str(horizon): horizon // protocol.frame_skip for horizon in protocol.horizons
    }

    episode_results = []
    total_step_error = np.zeros(step_count, dtype=np.float64)
    total_starting_points = 0

    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("selected_episode_indices_sha256") != _sampling_digest(selected):
            raise ValueError(f"Progress sampling mismatch: {progress_path}")
        episode_results = progress.get("episodes", [])
        for episode_result in episode_results:
            count = int(episode_result["num_starting_points"])
            total_step_error += np.asarray(
                episode_result["step_errors"], dtype=np.float64
            ) * count
            total_starting_points += count
        print(f"resuming after {len(episode_results)} completed episodes")

    completed_count = len(episode_results)
    for selected_position, episode_index in enumerate(
        selected[completed_count:], start=completed_count
    ):
        episode = dataset.load_episode(int(episode_index))
        episode_length = int(dataset.lengths[episode_index])
        num_starts = episode_length - required_episode_length + 1
        starts = torch.arange(num_starts, dtype=torch.long)
        if max_starts_per_episode is not None:
            if max_starts_per_episode < 1:
                raise ValueError("max_starts_per_episode must be positive.")
            starts = starts[:max_starts_per_episode]
        num_used_starts = len(starts)
        episode_id = _episode_identifier(episode, int(episode_index))
        print(
            f"episode {selected_position + 1:03d}/{len(selected)}: "
            f"index={episode_index}, id={episode_id}, "
            f"starting_points={num_used_starts}"
        )

        episode_step_error = np.zeros(step_count, dtype=np.float64)
        for batch_start in range(0, num_used_starts, protocol.batch_size):
            batch_starts = starts[batch_start : batch_start + protocol.batch_size]
            observations, action_blocks = _gather_batch(
                episode,
                batch_starts,
                protocol,
                adapter.observation_keys,
                step_count,
            )
            with torch.inference_mode():
                errors = adapter.rollout_errors(
                    observations,
                    action_blocks,
                    rollout_seed=(
                        int(seed) * 1_000_003
                        + int(episode_index) * 10_007
                        + int(batch_start)
                    ),
                )
            errors = torch.as_tensor(errors).detach().float().cpu()
            expected_shape = (len(batch_starts), step_count)
            if tuple(errors.shape) != expected_shape:
                raise ValueError(
                    f"{adapter.model_name} returned errors shaped "
                    f"{tuple(errors.shape)}; expected {expected_shape}."
                )
            if not torch.isfinite(errors).all():
                raise FloatingPointError(
                    f"Non-finite rollout error in episode {episode_index}, "
                    f"batch starting at {batch_start}."
                )
            episode_step_error += errors.double().sum(dim=0).numpy()

        episode_step_mean = episode_step_error / num_used_starts
        total_step_error += episode_step_error
        total_starting_points += num_used_starts
        long_horizon = {
            horizon: ({
                "cumulative_error": float(episode_step_mean[:steps].sum()),
                "mean_step_error": float(episode_step_mean[:steps].mean()),
                "terminal_error": float(episode_step_mean[steps - 1]),
            } if long_horizon_available else None)
            for horizon, steps in horizon_steps.items()
        }
        episode_results.append(
            {
                "episode_index": int(episode_index),
                "episode_id": episode_id,
                "episode_length": episode_length,
                "num_starting_points": num_used_starts,
                "short_horizon_error": float(episode_step_mean[0]),
                "step_errors": episode_step_mean.tolist(),
                "long_horizon": long_horizon,
            }
        )
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary_path.write_text(json.dumps({
                "selected_episode_indices_sha256": _sampling_digest(selected),
                "selected_episode_indices": selected.tolist(),
                "episodes": episode_results,
            }, indent=2) + "\n")
            temporary_path.replace(progress_path)

    episode_step_means = np.asarray(
        [episode["step_errors"] for episode in episode_results], dtype=np.float64
    )
    macro_step_mean = episode_step_means.mean(axis=0)
    micro_step_mean = total_step_error / total_starting_points
    long_horizon_error = {
        horizon: (float(macro_step_mean[:steps].sum()) if long_horizon_available else None)
        for horizon, steps in horizon_steps.items()
    }
    long_horizon_metrics = {
        horizon: ({
            "rollout_steps": steps,
            "cumulative_error": long_horizon_error[horizon],
            "mean_step_error": float(macro_step_mean[:steps].mean()),
            "terminal_error": float(macro_step_mean[steps - 1]),
            "start_weighted_cumulative_error": float(
                micro_step_mean[:steps].sum()
            ),
        } if long_horizon_available else {
            "rollout_steps": steps,
            "available": False,
            "reason": (
                f"only {full_eligible_count} episodes meet the full-rollout "
                f"length requirement; need {protocol.num_episodes}"
            ),
        })
        for horizon, steps in horizon_steps.items()
    }

    result = {
        **metadata,
        "model_name": adapter.model_name,
        "seed": int(seed),
        "error_space": adapter.error_space,
        "short_horizon_error": float(macro_step_mean[0]),
        "long_horizon_error": long_horizon_error,
        "protocol": {
            "history_length": protocol.history_length,
            "frame_skip": protocol.frame_skip,
            "horizons": list(protocol.horizons),
            "num_episodes": protocol.num_episodes,
            "batch_size": protocol.batch_size,
            "aggregation": "mean_of_episode_means",
            "long_horizon_definition": "sum_of_autoregressive_step_mse",
            "evaluation_mode": (
                "short_and_long" if long_horizon_available else "short_only"
            ),
        },
        "sampling": {
            "seed": int(seed),
            "eligible_episode_count": int(len(eligible)),
            "full_rollout_eligible_episode_count": int(full_eligible_count),
            "selected_episode_indices": selected.tolist(),
            "selected_episode_indices_sha256": _sampling_digest(selected),
            "episode_start_position": episode_start_position,
            "episode_end_position": episode_end_position,
            "num_starting_points": int(total_starting_points),
            "max_starts_per_episode": max_starts_per_episode,
            "required_episode_length": required_episode_length,
        },
        "metrics": {
            "short_horizon": {
                "horizon_environment_steps": protocol.frame_skip,
                "rollout_steps": 1,
                "error": float(macro_step_mean[0]),
                "episode_std": float(
                    episode_step_means[:, 0].std(ddof=1)
                    if len(episode_step_means) > 1
                    else 0.0
                ),
                "start_weighted_error": float(micro_step_mean[0]),
            },
            "autoregressive_step_errors": macro_step_mean.tolist(),
            "long_horizon": long_horizon_metrics,
        },
        "episodes": episode_results,
    }
    print(
        f"used {total_starting_points} starting points from "
        f"{len(selected)} sampled episodes"
    )
    return result
