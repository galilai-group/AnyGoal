"""Planning policy integration for IRASim."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy

from .actions import IRASimActionProcessor


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
