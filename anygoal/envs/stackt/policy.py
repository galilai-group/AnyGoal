"""Policies for the Stack-T environments."""

from __future__ import annotations

import numpy as np

from ..avoidt.policy import AvoidTPPolicy
from .env import (
    StackT,
    rotate,
    wrapped_angle_error,
)


class StackTPolicy(AvoidTPPolicy):
    """OU controller that approaches behind the T and pushes through its CoG."""

    def __init__(
        self,
        speed: float = 0.55,
        theta: float = 0.65,
        sigma: float = 0.025,
        ou_dt: float = 1.0,
        approach_distance: float = 105.0,
        push_through_distance: float = 60.0,
        alignment_tolerance: float = 24.0,
        torque_gain: float = 55.0,
        max_lateral_offset: float = 18.0,
        rotation_stage_threshold: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            goal=(0.0, 0.0),
            speed=speed,
            clearance=4.0,
            theta=theta,
            sigma=sigma,
            ou_dt=ou_dt,
            seed=seed,
            **kwargs,
        )
        if approach_distance <= 0.0 or push_through_distance <= 0.0:
            raise ValueError("StackT approach and push distances must be positive.")
        if alignment_tolerance <= 0.0 or torque_gain < 0.0:
            raise ValueError("Invalid StackT alignment or torque parameters.")
        if max_lateral_offset < 0.0:
            raise ValueError("max_lateral_offset must be non-negative.")
        if rotation_stage_threshold is not None and rotation_stage_threshold <= 0.0:
            raise ValueError("rotation_stage_threshold must be positive.")
        self.approach_distance = float(approach_distance)
        self.push_through_distance = float(push_through_distance)
        self.alignment_tolerance = float(alignment_tolerance)
        self.torque_gain = float(torque_gain)
        self.max_lateral_offset = float(max_lateral_offset)
        self.rotation_stage_threshold = rotation_stage_threshold

    def set_env(self, env) -> None:
        super().set_env(env)
        for base_env in self._unwrapped_envs():
            if not isinstance(base_env, StackT):
                raise ValueError("StackTPolicy requires a StackT environment.")

    @staticmethod
    def _approach_waypoint(
        agent_position: np.ndarray,
        block_cog: np.ndarray,
        rear_waypoint: np.ndarray,
        window_size: float,
    ) -> np.ndarray:
        """Move around the T before approaching a contact point."""
        orbit_radius = 110.0
        relative_agent = agent_position - block_cog
        agent_radius = float(np.linalg.norm(relative_agent))
        relative_rear = rear_waypoint - block_cog
        rear_angle = float(np.arctan2(relative_rear[1], relative_rear[0]))
        if agent_radius < 1e-8:
            agent_angle = rear_angle + np.pi
        else:
            agent_angle = float(
                np.arctan2(relative_agent[1], relative_agent[0])
            )

        angle_error = wrapped_angle_error(rear_angle, agent_angle)
        if abs(angle_error) <= 0.2:
            return rear_waypoint.copy()
        if agent_radius < orbit_radius - 5.0:
            waypoint_angle = agent_angle
        else:
            waypoint_angle = agent_angle + float(
                np.clip(angle_error, -0.35, 0.35)
            )

        waypoint = block_cog + orbit_radius * np.array(
            (np.cos(waypoint_angle), np.sin(waypoint_angle)),
            dtype=np.float64,
        )
        waypoint = np.clip(waypoint, 20.0, window_size - 20.0)
        if np.linalg.norm(waypoint - agent_position) < 5.0:
            orbit_direction = float(np.sign(angle_error))
            tangent = orbit_direction * np.array(
                (-np.sin(agent_angle), np.cos(agent_angle)),
                dtype=np.float64,
            )
            waypoint = np.clip(
                agent_position + 50.0 * tangent,
                20.0,
                window_size - 20.0,
            )
        return waypoint

    def get_action(self, info_dict, **kwargs) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Environment not set for the policy.")

        envs = self._unwrapped_envs()
        if self.ou_state.shape != (len(envs), 2):
            self.ou_state = np.zeros((len(envs), 2), dtype=np.float64)
        step_indices = self._step_indices(info_dict, len(envs))
        actions = np.zeros((len(envs), 2), dtype=np.float32)

        for env_idx, base_env in enumerate(envs):
            if step_indices is not None and step_indices[env_idx] == 0:
                self.ou_state[env_idx] = 0.0

            agent_position = np.asarray(
                base_env.agent.position, dtype=np.float64
            )
            block_position = np.asarray(
                base_env.block.position, dtype=np.float64
            )
            target_pose = np.asarray(base_env.goal_state[2:5], dtype=np.float64)
            displacement = target_pose[:2] - block_position
            position_error = float(np.linalg.norm(displacement))
            angle_error = wrapped_angle_error(
                target_pose[2], float(base_env.block.angle)
            )
            rotation_stage_threshold = (
                base_env.angle_tolerance * 0.9
                if self.rotation_stage_threshold is None
                else self.rotation_stage_threshold
            )
            block_cog = np.asarray(
                base_env.block.local_to_world(base_env.block.center_of_gravity),
                dtype=np.float64,
            )

            if (
                position_error < base_env.position_tolerance * 0.7
                and abs(angle_error) < base_env.angle_tolerance * 0.8
            ):
                self.ou_state[env_idx] = 0.0
                continue

            if abs(angle_error) >= rotation_stage_threshold:
                turn_sign = float(np.sign(angle_error))
                head_push_direction = rotate(
                    np.array((turn_sign, 0.0), dtype=np.float64),
                    float(base_env.block.angle),
                )
                use_head_end = np.dot(head_push_direction, displacement) >= 0.0
                if use_head_end:
                    push_direction = head_push_direction
                    contact_point = block_position + rotate(
                        np.array((-turn_sign * 60.0, 15.0), dtype=np.float64),
                        float(base_env.block.angle),
                    )
                    rear_waypoint = contact_point - push_direction * 38.0
                else:
                    push_direction = -head_push_direction
                    contact_point = block_position + rotate(
                        np.array((turn_sign * 15.0, 90.0), dtype=np.float64),
                        float(base_env.block.angle),
                    )
                    rear_waypoint = contact_point - push_direction * 38.0
                relative_agent = agent_position - contact_point
                longitudinal = float(np.dot(relative_agent, push_direction))
                lateral_direction = np.array(
                    (-push_direction[1], push_direction[0]), dtype=np.float64
                )
                lateral_error = float(
                    np.dot(relative_agent, lateral_direction)
                )
                aligned = (
                    -55.0 < longitudinal < 45.0
                    and abs(lateral_error) < self.alignment_tolerance
                )
                if aligned:
                    waypoint = contact_point + push_direction * 70.0
                else:
                    waypoint = self._approach_waypoint(
                        agent_position,
                        block_cog,
                        rear_waypoint,
                        base_env.window_size,
                    )
            elif position_error < 1e-8:
                push_direction = rotate(
                    np.array((1.0, 0.0)), target_pose[2]
                )
            else:
                push_direction = displacement / position_error
            if abs(angle_error) < rotation_stage_threshold:
                lateral_direction = np.array(
                    (-push_direction[1], push_direction[0]), dtype=np.float64
                )
                lateral_offset = float(
                    np.clip(
                        -self.torque_gain * angle_error,
                        -self.max_lateral_offset,
                        self.max_lateral_offset,
                    )
                )
                rear_waypoint = (
                    block_cog
                    - push_direction * self.approach_distance
                    + lateral_direction * lateral_offset
                )
                relative_agent = agent_position - block_cog
                longitudinal = float(np.dot(relative_agent, push_direction))
                lateral_error = float(
                    np.dot(relative_agent, lateral_direction) - lateral_offset
                )
                aligned = (
                    longitudinal < self.push_through_distance
                    and abs(lateral_error) < self.alignment_tolerance
                )

                if aligned:
                    waypoint = (
                        block_cog
                        + push_direction * self.push_through_distance
                        + lateral_direction * lateral_offset
                    )
                else:
                    waypoint = self._approach_waypoint(
                        agent_position,
                        block_cog,
                        rear_waypoint,
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

            if not aligned:
                proposed_target = (
                    agent_position + ou_action * base_env.action_scale
                )
                obstacle_bounds = self._expanded_block_bounds(base_env)
                if self._segment_intersects_box(
                    agent_position, proposed_target, obstacle_bounds
                ):
                    ou_action = mean_action

            self.ou_state[env_idx] = ou_action
            actions[env_idx] = ou_action.astype(np.float32)

        return actions


class StackTV2Policy(StackTPolicy):
    """Lower-noise, faster StackT controller for precise opposite rotation."""

    def __init__(self, speed: float = 0.7, **kwargs) -> None:
        kwargs.setdefault("theta", 0.85)
        kwargs.setdefault("sigma", 0.01)
        kwargs.setdefault("rotation_stage_threshold", np.deg2rad(10.0))
        super().__init__(speed=speed, **kwargs)


__all__ = ["StackTPolicy", "StackTV2Policy"]
