"""Planning adapters for encoder-free true-state predictors."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy


class FixedStandardScaler:
    """Minimal fixed z-score processor compatible with SWM policies."""

    def __init__(self, mean, std) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != self.std.shape:
            raise ValueError("Normalization mean and std must have matching shapes.")
        if np.any(self.std <= 0):
            raise ValueError("Normalization std must be positive.")

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


class FixedMinMaxScaler:
    """Minimal fixed [0, 1] processor compatible with SWM policies."""

    def __init__(self, minimum, maximum) -> None:
        self.minimum = np.asarray(minimum, dtype=np.float32)
        self.maximum = np.asarray(maximum, dtype=np.float32)
        if self.minimum.shape != self.maximum.shape:
            raise ValueError("Normalization min and max must have matching shapes.")
        if np.any(self.maximum <= self.minimum):
            raise ValueError("Normalization max must exceed min in every dimension.")
        self.scale = self.maximum - self.minimum

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.minimum) / self.scale

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale + self.minimum


class ToyWorldModelPolicy(WorldModelPolicy):
    """SWM planning policy that builds canonical true-state history."""

    def __init__(
        self,
        *,
        adapter,
        state_processor: FixedStandardScaler,
        action_processor: FixedStandardScaler,
        **kwargs,
    ) -> None:
        super().__init__(process={"action": action_processor}, **kwargs)
        self.adapter = adapter
        self.state_processor = state_processor
        self.action_processor = action_processor
        self._state_history = None
        self._transition_history = None
        self._pending_actions = None

    def set_env(self, env) -> None:
        super().set_env(env)
        self._state_history = [
            deque(maxlen=self.cfg.history_len) for _ in range(env.num_envs)
        ]
        self._transition_history = [
            deque(maxlen=max(self.cfg.history_len - 1, 1)) for _ in range(env.num_envs)
        ]
        self._pending_actions = [
            deque(maxlen=self.cfg.action_block) for _ in range(env.num_envs)
        ]

    @staticmethod
    def _latest(values) -> np.ndarray:
        values = np.asarray(values)
        return values[:, -1] if values.ndim >= 3 else values

    def _state_input(self, info_dict: dict, *, goal: bool):
        prefix = "goal_" if goal else ""
        resolved_keys = {}
        for key in self.adapter.state_keys:
            dataset_key = f"{prefix}{key}"
            environment_key = getattr(
                self.adapter, "environment_state_keys", {}
            ).get(key)
            if dataset_key in info_dict:
                resolved_keys[key] = dataset_key
            elif not goal and environment_key in info_dict:
                resolved_keys[key] = environment_key
            elif (
                not goal
                and key in getattr(self.adapter, "static_goal_state_keys", ())
                and f"goal_{key}" in info_dict
            ):
                resolved_keys[key] = f"goal_{key}"

        missing = [key for key in self.adapter.state_keys if key not in resolved_keys]
        if missing:
            raise KeyError(
                "Toy planning is missing state columns: " + ", ".join(missing)
            )

        columns = {
            key: torch.as_tensor(self._latest(info_dict[resolved_keys[key]]))
            for key in resolved_keys
        }
        if self.adapter.state_keys == ("state",):
            return columns["state"]
        return columns

    def _encode_states(self, values) -> torch.Tensor:
        encoded = self.adapter.encode_state(values).cpu().numpy()
        return torch.from_numpy(self.state_processor.transform(encoded))

    def _encode_actions(self, values) -> torch.Tensor:
        encoded = self.adapter.encode_action(torch.as_tensor(values)).cpu().numpy()
        return torch.from_numpy(self.action_processor.transform(encoded))

    def decode_states(self, values) -> np.ndarray:
        """Denormalize model predictions before sending them to a renderer."""
        canonical = self.state_processor.inverse_transform(values)
        decoded = self.adapter.decode_state(torch.as_tensor(canonical))
        return decoded.cpu().numpy()

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        if (
            self._state_history is None
            or self._transition_history is None
            or self._pending_actions is None
        ):
            raise RuntimeError("Set the policy environment before requesting actions.")
        current_states = self._encode_states(
            self._state_input(info_dict, goal=False)
        )
        goal_states = self._encode_states(self._state_input(info_dict, goal=True))
        current_actions = None
        if "action" in info_dict:
            current_actions = self._encode_actions(self._latest(info_dict["action"]))

        needs_flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(len(current_states), dtype=bool))
        ).reshape(len(current_states), -1)[:, 0]
        state_batches = []
        action_batches = []
        for index, current_state in enumerate(current_states):
            if needs_flush[index]:
                self._state_history[index].clear()
                self._transition_history[index].clear()
                self._pending_actions[index].clear()

            if not self._state_history[index]:
                self._state_history[index].append(current_state)
            elif current_actions is not None:
                pending_actions = self._pending_actions[index]
                pending_actions.append(current_actions[index])
                if len(pending_actions) == self.cfg.action_block:
                    self._transition_history[index].append(
                        torch.cat(tuple(pending_actions), dim=-1)
                    )
                    self._state_history[index].append(current_state)
                    pending_actions.clear()

            states = list(self._state_history[index])
            states = [states[0]] * (self.cfg.history_len - len(states)) + states
            state_batches.append(torch.stack(states))

            required_actions = self.cfg.history_len - 1
            actions = list(self._transition_history[index])[-required_actions:]
            zero_action = current_state.new_zeros(
                self.adapter.action_dim * self.cfg.action_block
            )
            actions = [zero_action] * (required_actions - len(actions)) + actions
            action_batches.append(torch.stack(actions))

        prepared = {
            "state": torch.stack(state_batches),
            "action": torch.stack(action_batches),
            "goal_state": goal_states[:, None],
        }
        if "_needs_flush" in info_dict:
            prepared["_needs_flush"] = torch.as_tensor(needs_flush)
        if "terminated" in info_dict:
            prepared["terminated"] = torch.as_tensor(info_dict["terminated"])
        return prepared


__all__ = ["FixedMinMaxScaler", "FixedStandardScaler", "ToyWorldModelPolicy"]
