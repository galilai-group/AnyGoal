"""Planning policy integration for DriftWorld."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy

from .actions import DriftWorldActionProcessor


class DriftWorldWorldModelPolicy(WorldModelPolicy):
    """Accumulate frameskip-aligned visual history for DriftWorld planning."""

    def __init__(
        self,
        *,
        action_processor: DriftWorldActionProcessor | None = None,
        **kwargs,
    ) -> None:
        self.action_processor = action_processor or DriftWorldActionProcessor()
        process = dict(kwargs.pop("process", {}) or {})
        process["action"] = self.action_processor
        super().__init__(process=process, **kwargs)
        model = self.solver.model
        self.action_frameskip = int(model.action_frameskip)
        self.num_history_frames = int(model.num_history_frames)
        if self.cfg.action_block != self.action_frameskip:
            raise ValueError(
                "plan_config.action_block must match DriftWorld's training "
                f"frameskip ({self.action_frameskip})."
            )
        if self.cfg.history_len != self.num_history_frames:
            raise ValueError(
                "plan_config.history_len must match DriftWorld's fixed visual "
                f"history ({self.num_history_frames})."
            )
        self._pixel_history: list[deque[torch.Tensor]] | None = None
        self._pending_actions: list[deque[torch.Tensor]] | None = None

    def set_env(self, env) -> None:
        super().set_env(env)
        self._pixel_history = [
            deque(maxlen=self.num_history_frames) for _ in range(env.num_envs)
        ]
        self._pending_actions = [
            deque(maxlen=self.action_frameskip) for _ in range(env.num_envs)
        ]

    @staticmethod
    def _latest(values: torch.Tensor) -> torch.Tensor:
        return values[:, -1] if values.ndim >= 3 else values

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        if self._pixel_history is None or self._pending_actions is None:
            raise RuntimeError("Set the policy environment before requesting actions.")
        prepared = super()._prepare_info(info_dict)
        if "pixels" not in prepared or "goal" not in prepared:
            raise KeyError("DriftWorld planning requires pixels and goal observations.")
        if "action" not in prepared:
            raise KeyError("DriftWorld history alignment requires the last action.")

        current_pixels = self._latest(prepared["pixels"])
        current_actions = torch.nan_to_num(
            self._latest(prepared["action"]).float(), nan=0.0
        )
        needs_flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(len(current_pixels), dtype=bool))
        ).reshape(len(current_pixels), -1)[:, 0]

        pixel_batches = []
        for index, (pixel, action) in enumerate(zip(current_pixels, current_actions)):
            if needs_flush[index]:
                self._pixel_history[index].clear()
                self._pending_actions[index].clear()
            if not self._pixel_history[index]:
                self._pixel_history[index].append(pixel)
            else:
                pending = self._pending_actions[index]
                pending.append(action)
                if len(pending) == self.action_frameskip:
                    self._pixel_history[index].append(pixel)
                    pending.clear()
            pixels = list(self._pixel_history[index])
            pixels = [pixels[0]] * (self.num_history_frames - len(pixels)) + pixels
            pixel_batches.append(torch.stack(pixels))

        output = {
            "pixels": torch.stack(pixel_batches),
            "goal": prepared["goal"][:, -1:],
        }
        if "proprio" in prepared:
            output["proprio"] = self._latest(prepared["proprio"])
        if "_needs_flush" in info_dict:
            output["_needs_flush"] = torch.as_tensor(needs_flush)
        if "terminated" in prepared:
            output["terminated"] = prepared["terminated"]
        return output
