"""Internal recording of planning actions and rollout/ground-truth GIFs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _frame(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.min() >= 0.0 and array.max() <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _save_gif(path: Path, frames: list[np.ndarray], duration_ms: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(_frame(frame)) for frame in frames]
    if not images:
        return
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


class PlanningArtifactRecorder:
    def __init__(self, world, dataset, episodes, start_steps, goal_offset, output, seed):
        self.world = world
        self.dataset = dataset
        self.episodes = [int(value) for value in episodes]
        self.start_steps = [int(value) for value in start_steps]
        self.goal_offset = int(goal_offset)
        self.output = Path(output)
        self.seed = int(seed)
        self.actions = [[] for _ in self.episodes]
        self.frames = [[] for _ in self.episodes]
        self._original_step = world.envs.step

    def install(self) -> None:
        def recording_step(actions, mask=None):
            active = np.ones(len(self.episodes), dtype=bool) if mask is None else mask
            for index in np.flatnonzero(active):
                self.actions[index].append(np.asarray(actions[index]).copy())
            result = self._original_step(actions, mask=mask)
            infos = result[-1]
            for index in np.flatnonzero(active):
                self.frames[index].append(_frame(infos["pixels"][index]))
            return result

        self.world.envs.step = recording_step

    def restore(self) -> None:
        """Restore the environment step method after success or failure."""
        self.world.envs.step = self._original_step

    def save(self, metrics: dict) -> None:
        self.restore()
        if "episode_successes" in metrics:
            episode_successes = metrics["episode_successes"]
        elif "original_cube_criterion" in metrics:
            episode_successes = metrics["original_cube_criterion"]["episode_successes"]
        elif "dataset_goal_criterion" in metrics:
            episode_successes = metrics["dataset_goal_criterion"]["episode_successes"]
        else:
            raise KeyError("No per-episode success criterion found in planning metrics.")
        successes = np.asarray(episode_successes, dtype=bool)
        chunks = self.dataset.load_chunk(
            np.asarray(self.episodes),
            np.asarray(self.start_steps),
            np.asarray(self.start_steps) + self.goal_offset + 1,
        )
        manifest = []
        for index, (episode, start_step, actions, frames, chunk) in enumerate(
            zip(self.episodes, self.start_steps, self.actions, self.frames, chunks)
        ):
            stem = (
                f"seed_{self.seed}_chunk_{index:03d}_episode_{episode}_"
                f"start_{start_step}"
            )
            outcome = "success" if successes[index] else "fail"
            rollout_name = f"{stem}_{outcome}.gif"
            _save_gif(self.output / outcome / rollout_name, frames)
            _save_gif(self.output / "gt" / f"{stem}.gif", list(chunk["pixels"]))
            action_array = np.asarray(actions, dtype=np.float32)
            action_path = self.output / "actions" / f"{stem}.npy"
            action_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(action_path, action_array)
            manifest.append(
                {
                    "index": index,
                    "episode": episode,
                    "start_step": start_step,
                    "success": bool(successes[index]),
                    "num_actions": len(action_array),
                    "actions": action_array.tolist(),
                    "action_path": str(action_path),
                    "rollout_gif": str(self.output / outcome / rollout_name),
                    "ground_truth_gif": str(self.output / "gt" / f"{stem}.gif"),
                }
            )
        (self.output / "actions.json").write_text(json.dumps(manifest, indent=2) + "\n")
