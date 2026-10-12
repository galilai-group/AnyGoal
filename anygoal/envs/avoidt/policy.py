"""Policies for the Avoid-T environment."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np
from stable_worldmodel.policy import BasePolicy

from .env import resolve_goal


class AvoidTPPolicy(BasePolicy):
    """Move the circular agent to a corner while avoiding the physical T.

    The policy treats an expanded axis-aligned bounding box around the current
    T as an obstacle. A small visibility graph chooses a collision-free
    waypoint, and the action itself follows an Ornstein-Uhlenbeck process whose
    mean is the action pointing at that waypoint.
    """

    def __init__(
        self,
        goal: str | Sequence[float],
        speed: float = 0.55,
        clearance: float = 20.0,
        theta: float = 0.55,
        sigma: float = 0.06,
        ou_dt: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 < speed <= 1:
            raise ValueError('speed must be in (0, 1].')
        if clearance < 0:
            raise ValueError('clearance must be non-negative.')
        if theta <= 0 or sigma < 0 or ou_dt <= 0:
            raise ValueError('theta and ou_dt must be positive; sigma >= 0.')

        self.goal = resolve_goal(goal)
        self.speed = float(speed)
        self.clearance = float(clearance)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.ou_dt = float(ou_dt)
        self.seed = seed
        self.discrete = False
        self.rng = np.random.default_rng(seed)
        self.ou_state = np.empty((0, 2), dtype=np.float64)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if self.ou_state.size:
            self.ou_state.fill(0.0)

    def set_env(self, env) -> None:
        self.env = env
        envs = self._unwrapped_envs()
        for base_env in envs:
            spec = getattr(base_env, 'spec', None)
            supported_ids = (
                'swm/PushT',
                'anygoal/AvoidT',
                'anygoal/StackT',
                'anygoal/PushTAntigoal',
            )
            if spec is None or not any(
                prefix in spec.id for prefix in supported_ids
            ):
                raise ValueError(
                    'AvoidTPPolicy only supports PushT benchmark environments.'
                )
            if 'Discrete' in spec.id:
                raise ValueError(
                    'AvoidTPPolicy requires the continuous PushT environment.'
                )
        self.ou_state = np.zeros((len(envs), 2), dtype=np.float64)

    def get_action(self, info_dict, **kwargs) -> np.ndarray:
        if self.env is None:
            raise RuntimeError('Environment not set for the policy.')

        envs = self._unwrapped_envs()
        if self.ou_state.shape != (len(envs), 2):
            self.ou_state = np.zeros((len(envs), 2), dtype=np.float64)

        step_indices = self._step_indices(info_dict, len(envs))
        actions = np.zeros((len(envs), 2), dtype=np.float32)

        for env_idx, base_env in enumerate(envs):
            if step_indices is not None and step_indices[env_idx] == 0:
                self.ou_state[env_idx] = 0.0

            agent_position = np.asarray(base_env.agent.position, dtype=np.float64)
            obstacle_bounds = self._expanded_block_bounds(base_env)
            waypoint = self._next_waypoint(
                agent_position,
                self.goal,
                obstacle_bounds,
                base_env.window_size,
            )
            mean_action = self._mean_action(
                agent_position, waypoint, base_env.action_scale
            )

            noise = self.rng.normal(size=2)
            ou_action = self.ou_state[env_idx]
            ou_action += self.theta * (mean_action - ou_action) * self.ou_dt
            ou_action += self.sigma * np.sqrt(self.ou_dt) * noise
            ou_action = np.clip(ou_action, -1.0, 1.0)

            # OU momentum can lag behind a newly selected waypoint. Use the
            # safe mean immediately if that lag would enter the obstacle box.
            proposed_target = agent_position + ou_action * base_env.action_scale
            if self._segment_intersects_box(
                agent_position, proposed_target, obstacle_bounds
            ):
                ou_action = mean_action

            self.ou_state[env_idx] = ou_action
            actions[env_idx] = ou_action.astype(np.float32)

        return actions

    def _unwrapped_envs(self) -> list:
        if hasattr(self.env, 'envs'):
            return [env.unwrapped for env in self.env.envs]
        base_env = self.env.unwrapped
        if hasattr(base_env, 'envs'):
            return [env.unwrapped for env in base_env.envs]
        return [base_env]

    @staticmethod
    def _step_indices(info_dict, num_envs: int) -> np.ndarray | None:
        if not isinstance(info_dict, dict) or 'step_idx' not in info_dict:
            return None
        step_indices = np.asarray(info_dict['step_idx']).reshape(num_envs, -1)
        return step_indices[:, -1]

    def _expanded_block_bounds(self, env) -> tuple[float, float, float, float]:
        points = []
        for shape in env.block.shapes:
            if hasattr(shape, 'get_vertices'):
                points.extend(
                    tuple(env.block.local_to_world(vertex))
                    for vertex in shape.get_vertices()
                )

        if not points:
            block_position = np.asarray(env.block.position, dtype=np.float64)
            points = [block_position - 60.0, block_position + 60.0]

        points_array = np.asarray(points, dtype=np.float64)
        agent_radius = max(
            (
                float(shape.radius)
                for shape in env.agent.shapes
                if hasattr(shape, 'radius')
            ),
            default=0.0,
        )
        margin = agent_radius + self.clearance
        return (
            float(points_array[:, 0].min() - margin),
            float(points_array[:, 0].max() + margin),
            float(points_array[:, 1].min() - margin),
            float(points_array[:, 1].max() + margin),
        )

    def _mean_action(
        self, position: np.ndarray, waypoint: np.ndarray, action_scale: float
    ) -> np.ndarray:
        displacement = waypoint - position
        distance = np.linalg.norm(displacement)
        if distance < 1e-8:
            return np.zeros(2, dtype=np.float64)
        magnitude = min(self.speed, distance / float(action_scale))
        return displacement / distance * magnitude

    @classmethod
    def _next_waypoint(
        cls,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: tuple[float, float, float, float],
        window_size: float,
    ) -> np.ndarray:
        xmin, xmax, ymin, ymax = bounds
        waypoint_padding = 5.0
        if xmin < start[0] < xmax and ymin < start[1] < ymax:
            exits = np.array(
                [
                    (xmin - waypoint_padding, start[1]),
                    (xmax + waypoint_padding, start[1]),
                    (start[0], ymin - waypoint_padding),
                    (start[0], ymax + waypoint_padding),
                ],
                dtype=np.float64,
            )
            return exits[np.argmin(np.linalg.norm(exits - start, axis=1))]

        if not cls._segment_intersects_box(start, goal, bounds):
            return goal.copy()

        wall_margin = 20.0
        corners = [
            np.array(
                (xmin - waypoint_padding, ymin - waypoint_padding),
                dtype=np.float64,
            ),
            np.array(
                (xmin - waypoint_padding, ymax + waypoint_padding),
                dtype=np.float64,
            ),
            np.array(
                (xmax + waypoint_padding, ymin - waypoint_padding),
                dtype=np.float64,
            ),
            np.array(
                (xmax + waypoint_padding, ymax + waypoint_padding),
                dtype=np.float64,
            ),
        ]
        corners = [
            point
            for point in corners
            if np.all(point >= wall_margin)
            and np.all(point <= window_size - wall_margin)
        ]
        nodes = [start, goal, *corners]
        distances = [float('inf')] * len(nodes)
        previous = [-1] * len(nodes)
        distances[0] = 0.0
        queue = [(0.0, 0)]

        while queue:
            distance, node_idx = heapq.heappop(queue)
            if distance != distances[node_idx]:
                continue
            if node_idx == 1:
                break
            for next_idx, next_node in enumerate(nodes):
                if next_idx == node_idx or cls._segment_intersects_box(
                    nodes[node_idx], next_node, bounds
                ):
                    continue
                candidate = distance + float(
                    np.linalg.norm(next_node - nodes[node_idx])
                )
                if candidate < distances[next_idx]:
                    distances[next_idx] = candidate
                    previous[next_idx] = node_idx
                    heapq.heappush(queue, (candidate, next_idx))

        if not np.isfinite(distances[1]):
            # This only occurs if the expanded obstacle reaches a wall.
            return min(
                corners or [goal],
                key=lambda point: np.linalg.norm(point - start),
            ).copy()

        path = [1]
        while previous[path[-1]] != -1:
            path.append(previous[path[-1]])
        path.reverse()
        return nodes[path[1]].copy()

    @staticmethod
    def _segment_intersects_box(
        start: np.ndarray,
        end: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> bool:
        """Test intersection with the box interior, allowing tangent paths."""
        xmin, xmax, ymin, ymax = bounds
        epsilon = 1e-7
        xmin, xmax = xmin + epsilon, xmax - epsilon
        ymin, ymax = ymin + epsilon, ymax - epsilon
        direction = end - start
        t_min, t_max = 0.0, 1.0

        for origin, delta, lower, upper in (
            (start[0], direction[0], xmin, xmax),
            (start[1], direction[1], ymin, ymax),
        ):
            if abs(delta) < epsilon:
                if origin <= lower or origin >= upper:
                    return False
                continue
            near = (lower - origin) / delta
            far = (upper - origin) / delta
            if near > far:
                near, far = far, near
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return False

        return t_max > epsilon and t_min < 1.0 - epsilon


class SampledGoalAvoidTPolicy(AvoidTPPolicy):
    """Avoid the T while following each environment's sampled ball goal."""

    def get_action(self, info_dict, **kwargs) -> np.ndarray:
        if self.env is None:
            raise RuntimeError('Environment not set for the policy.')

        envs = self._unwrapped_envs()
        num_envs = len(envs)
        if self.ou_state.shape != (num_envs, 2):
            self.ou_state = np.zeros((num_envs, 2), dtype=np.float64)

        goal_states = np.asarray(info_dict['goal_state'])
        goal_states = goal_states.reshape(num_envs, -1, goal_states.shape[-1])
        ball_goals = goal_states[:, -1, :2]
        step_indices = self._step_indices(info_dict, num_envs)
        actions = np.zeros((num_envs, 2), dtype=np.float32)

        for env_idx, base_env in enumerate(envs):
            if step_indices is not None and step_indices[env_idx] == 0:
                self.ou_state[env_idx] = 0.0

            agent_position = np.asarray(base_env.agent.position, dtype=np.float64)
            obstacle_bounds = self._expanded_block_bounds(base_env)
            waypoint = self._next_waypoint(
                agent_position,
                ball_goals[env_idx],
                obstacle_bounds,
                base_env.window_size,
            )
            mean_action = self._mean_action(
                agent_position, waypoint, base_env.action_scale
            )
            ou_action = self.ou_state[env_idx]
            ou_action += self.theta * (mean_action - ou_action) * self.ou_dt
            ou_action += (
                self.sigma
                * np.sqrt(self.ou_dt)
                * self.rng.normal(size=2)
            )
            ou_action = np.clip(ou_action, -1.0, 1.0)

            proposed_target = agent_position + ou_action * base_env.action_scale
            if self._segment_intersects_box(
                agent_position, proposed_target, obstacle_bounds
            ):
                ou_action = mean_action

            self.ou_state[env_idx] = ou_action
            actions[env_idx] = ou_action.astype(np.float32)

        return actions


__all__ = ['AvoidTPPolicy', 'SampledGoalAvoidTPolicy']
