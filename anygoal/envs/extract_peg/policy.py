"""Scripted controller for extracting, carrying, and placing the peg."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from anygoal.envs.insert_peg.env import planar_angle, wrapped_angle_error


ARM_JOINTS = ("arm_root", "arm_shoulder", "arm_elbow", "arm_wrist")
GEAR = np.array([12.0, 8.0, 4.0, 2.0])
APPROACH_STEPS = 100
GRIP_STEPS = 30
CARRY_STEPS = 100
PLACE_STEPS = 100
RELEASE_STEPS = 1
SETTLE_STEPS = 0
TOTAL_STEPS = 331
ACTION_RATE_LIMIT = 0.30
OPEN_GRIP = -0.5
CLOSED_GRIP = 0.4
OU_THETA = 0.18
OU_SIGMA = 0.012
ACTION_NOISE_SIGMA = 0.002


def rotate_planar(angle: float, local_xz: np.ndarray) -> np.ndarray:
    """Rotate an x/z vector by the MuJoCo hinge-y convention."""
    sine, cosine = np.sin(angle), np.cos(angle)
    x, z = np.asarray(local_xz, dtype=np.float64)
    return np.array([cosine * x + sine * z, -sine * x + cosine * z])


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def peg_lowest_z(physics):
    names = physics.named.model.geom_pos.axes.row.names
    lowest = np.inf
    for name in ("blade", "guard", "pommel"):
        geom_id = names.index(name)
        radius = physics.model.geom_size[geom_id, 0]
        extent = radius
        if physics.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent += abs(
                physics.data.geom_xmat[geom_id].reshape(3, 3)[2, 2]
            ) * physics.model.geom_size[geom_id, 1]
        lowest = min(lowest, physics.data.geom_xpos[geom_id, 2] - extent)
    return float(lowest)


def initialize_arm_above_peg(env, peg, peg_angle, distance, side):
    """Place an open hand 0.1--0.2 m outward from the inserted peg."""
    physics = env.physics
    outward = rotate_planar(peg_angle, np.array([0.0, distance]))
    hand_target = peg + outward
    hand_angle_target = -np.pi / 2.0 if side < 0 else np.pi / 2.0
    original = physics.named.data.qpos[list(ARM_JOINTS)].copy()

    def residual(angles):
        physics.named.data.qpos[list(ARM_JOINTS)] = angles
        physics.forward()
        hand = physics.named.data.site_xpos["grasp", ["x", "z"]]
        hand_angle = planar_angle(env.get_state()[31:33])
        return np.r_[15.0 * (hand - hand_target), wrapped_angle_error(hand_angle_target, hand_angle)]

    lower = np.deg2rad([-360.0, -155.0, -155.0, -125.0])
    upper = np.deg2rad([360.0, 155.0, 155.0, 75.0])
    starts = [original]
    starts.extend(np.deg2rad(value) for value in (
        (0, -30, 80, 0), (90, -60, 100, 20), (-90, -60, 100, -20),
        (180, -50, 120, 20), (-180, -50, 120, -20),
    ))
    results = [
        least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper), max_nfev=250)
        for start in starts
    ]
    result = min(results, key=lambda item: np.linalg.norm(item.fun))
    physics.named.data.qpos[list(ARM_JOINTS)] = result.x
    physics.named.data.qvel[list(ARM_JOINTS)] = 0.0
    physics.named.data.qpos[["finger", "thumb"]] = np.deg2rad(-5.0)
    physics.named.data.qpos[["fingertip", "thumbtip"]] = 0.0
    physics.named.data.qvel[["finger", "fingertip", "thumb", "thumbtip"]] = 0.0
    physics.forward()
    error = float(np.linalg.norm(result.fun[:2]) / 15.0)
    return error, hand_target


def set_grasp_constraint(physics, active):
    equality = physics.model.name2id("extract_grasp", "equality")
    if active:
        inverse_position, inverse_quaternion = np.zeros(3), np.zeros(4)
        relative_position, relative_quaternion = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(
            inverse_position, inverse_quaternion,
            physics.data.xpos[physics.model.eq_obj1id[equality]],
            physics.data.xquat[physics.model.eq_obj1id[equality]],
        )
        mujoco.mju_mulPose(
            relative_position, relative_quaternion,
            inverse_position, inverse_quaternion,
            physics.data.xpos[physics.model.eq_obj2id[equality]],
            physics.data.xquat[physics.model.eq_obj2id[equality]],
        )
        physics.model.eq_data[equality] = 0.0
        physics.model.eq_data[equality, 3:6] = relative_position
        physics.model.eq_data[equality, 6:10] = relative_quaternion
        physics.model.eq_data[equality, 10] = 1.0
    physics.data.eq_active[equality] = int(active)
    physics.data.qacc_warmstart[:] = 0.0
    physics.forward()


class ExtractPegPolicy:
    def __init__(self, env, side, waypoint, destination_x, rng, reverse_insertion_q):
        self.env = env
        self.side = int(side)
        self.waypoint = np.asarray(waypoint, dtype=np.float64)
        self.destination = np.array([destination_x, 0.105])
        self.rng = rng
        self.reverse_insertion_q = np.asarray(reverse_insertion_q, dtype=np.float64)
        self.step = 0
        self.previous_action = np.zeros(5)
        self.ou_state = np.zeros(4)
        state = env.get_state()
        self.initial_peg = state[33:35].astype(np.float64)
        self.initial_angle = planar_angle(state[35:37])
        self.initial_hand = env.physics.named.data.site_xpos["grasp", ["x", "z"]].copy()
        self.initial_q = env.physics.named.data.qpos[list(ARM_JOINTS)].copy()
        env.physics.named.data.qpos[list(ARM_JOINTS)] = self.reverse_insertion_q[0]
        env.physics.forward()
        grasp_hand = env.physics.named.data.site_xpos["grasp", ["x", "z"]].copy()
        self.initial_hand_angle = planar_angle(env.get_state()[31:33])
        env.physics.named.data.qpos[list(ARM_JOINTS)] = self.initial_q
        env.physics.forward()
        self.grasp_position = self.initial_peg + rotate_planar(
            self.initial_angle, np.array([0.0, 0.015])
        )
        self.waypoint_angle = np.pi / 2.0 if side < 0 else -np.pi / 2.0
        self.destination_angle = 0.0
        self.grasp_active = False
        self.released = False
        self.release_hand = None
        self.reached_waypoint = False
        self.waypoint_error = np.inf
        self.hand_offset = grasp_hand - self.initial_peg
        self.path_targets = self._build_path_targets()

    def _solve_arm_target(self, hand_position, hand_angle, reference):
        physics = self.env.physics
        saved = physics.named.data.qpos[list(ARM_JOINTS)].copy()

        def residual(angles):
            physics.named.data.qpos[list(ARM_JOINTS)] = angles
            physics.forward()
            hand = physics.named.data.site_xpos["grasp", ["x", "z"]]
            angle = planar_angle(self.env.get_state()[31:33])
            return np.r_[12.0 * (hand - hand_position), wrapped_angle_error(hand_angle, angle)]

        lower = np.deg2rad([-360.0, -155.0, -155.0, -125.0])
        upper = np.deg2rad([360.0, 155.0, 155.0, 75.0])
        result = least_squares(
            residual, np.clip(reference, lower, upper), bounds=(lower, upper), max_nfev=100
        )
        physics.named.data.qpos[list(ARM_JOINTS)] = saved
        physics.forward()
        return result.x.copy()

    def _build_path_targets(self):
        reference = self.env.physics.named.data.qpos[list(ARM_JOINTS)].copy()
        targets = []
        for step in np.linspace(0, CARRY_STEPS + PLACE_STEPS - 1, 66):
            if step < 0.30 * CARRY_STEPS:
                coordinate = step / (0.30 * CARRY_STEPS) * (len(self.reverse_insertion_q) - 1)
                lower = min(int(coordinate), len(self.reverse_insertion_q) - 2)
                fraction = coordinate - lower
                difference = self.reverse_insertion_q[lower + 1] - self.reverse_insertion_q[lower]
                reference = self.reverse_insertion_q[lower] + fraction * difference
                targets.append(reference.copy())
                continue
            saved_step = self.step
            self.step = APPROACH_STEPS + GRIP_STEPS + int(step)
            peg_position, peg_angle = self._desired_peg_pose()
            rotation = peg_angle - self.initial_angle
            hand_offset = rotate_planar(rotation, self.hand_offset)
            reference = self._solve_arm_target(
                peg_position + hand_offset,
                self.initial_hand_angle + rotation,
                reference,
            )
            targets.append(reference)
            self.step = saved_step
        return targets

    @staticmethod
    def _interpolate_targets(targets, progress):
        coordinate = np.clip(progress, 0.0, 1.0) * (len(targets) - 1)
        lower = min(int(coordinate), len(targets) - 2)
        fraction = coordinate - lower
        difference = (targets[lower + 1] - targets[lower] + np.pi) % (2 * np.pi) - np.pi
        return targets[lower] + fraction * difference

    def _joint_action(self, target, grip):
        physics = self.env.physics
        q = physics.named.data.qpos[list(ARM_JOINTS)].copy()
        velocity = physics.data.qvel[:4]
        error = (target - q + np.pi) % (2 * np.pi) - np.pi
        torque = 80.0 * error - 10.0 * velocity + physics.data.qfrc_bias[:4]
        action = np.r_[np.clip(torque / GEAR, -1.0, 1.0), grip]
        self.ou_state += OU_THETA * (-self.ou_state) + OU_SIGMA * self.rng.normal(size=4)
        action[:4] += self.ou_state + ACTION_NOISE_SIGMA * self.rng.normal(size=4)
        action = self.previous_action + np.clip(
            action - self.previous_action, -ACTION_RATE_LIMIT, ACTION_RATE_LIMIT
        )
        if 0.048 < action[0] < 0.077:
            action[0] = 0.045 if action[0] < 0.0625 else 0.08
        if 0.097 < action[1] < 0.126:
            action[1] = 0.094 if action[1] < 0.1115 else 0.129
        return np.clip(action, -1.0, 1.0)

    def _desired_peg_pose(self):
        carry_start = APPROACH_STEPS + GRIP_STEPS
        place_start = carry_start + CARRY_STEPS
        if self.step < place_start:
            progress = smoothstep((self.step - carry_start) / max(1, CARRY_STEPS - 1))
            extract = self.initial_peg + rotate_planar(
                self.initial_angle, np.array([0.0, -0.12])
            )
            if progress < 0.30:
                local = smoothstep(progress / 0.30)
                position = (1.0 - local) * self.initial_peg + local * extract
                angle = self.initial_angle
            else:
                local = smoothstep((progress - 0.30) / 0.70)
                position = (1.0 - local) * extract + local * self.waypoint
                angle = self.initial_angle + local * wrapped_angle_error(
                    self.waypoint_angle, self.initial_angle
                )
            return position, angle
        progress = smoothstep((self.step - place_start) / max(1, PLACE_STEPS - 1))
        position = (1.0 - progress) * self.waypoint + progress * self.destination
        angle = self.waypoint_angle + progress * wrapped_angle_error(
            self.destination_angle, self.waypoint_angle
        )
        return position, angle

    def _operational_action(self, target, target_angle, grip):
        physics = self.env.physics
        hand = physics.named.data.site_xpos["grasp", ["x", "z"]]
        jacobian = np.zeros((3, physics.model.nv))
        rotation = np.zeros((3, physics.model.nv))
        mujoco.mj_jacSite(
            physics.model.ptr, physics.data.ptr, jacobian, rotation,
            physics.model.name2id("grasp", "site"),
        )
        task_jacobian = jacobian[[0, 2], :4]
        velocity = physics.data.qvel[:4]
        force = 260.0 * np.clip(target - hand, -0.06, 0.06)
        force -= 20.0 * (task_jacobian @ velocity)
        torque = task_jacobian.T @ force
        hand_angle = planar_angle(self.env.get_state()[31:33])
        angle_error = wrapped_angle_error(target_angle, hand_angle)
        torque += rotation[1, :4] * (
            2.0 * angle_error - 0.20 * (rotation[1, :4] @ velocity)
        )
        torque += physics.data.qfrc_bias[:4] - 0.15 * velocity
        action = np.r_[np.clip(torque / GEAR, -1.0, 1.0), grip]
        moving = APPROACH_STEPS + GRIP_STEPS <= self.step < APPROACH_STEPS + GRIP_STEPS + CARRY_STEPS + PLACE_STEPS
        if moving:
            self.ou_state += OU_THETA * (-self.ou_state) + OU_SIGMA * self.rng.normal(size=4)
            action[:4] += self.ou_state + ACTION_NOISE_SIGMA * self.rng.normal(size=4)
        action = self.previous_action + np.clip(
            action - self.previous_action, -ACTION_RATE_LIMIT, ACTION_RATE_LIMIT
        )
        if 0.048 < action[0] < 0.077:
            action[0] = 0.045 if action[0] < 0.0625 else 0.08
        if 0.097 < action[1] < 0.126:
            action[1] = 0.094 if action[1] < 0.1115 else 0.129
        return np.clip(action, -1.0, 1.0)

    def action(self):
        grip_end = APPROACH_STEPS + GRIP_STEPS
        carry_end = grip_end + CARRY_STEPS
        place_end = carry_end + PLACE_STEPS
        release_end = place_end + RELEASE_STEPS
        if self.step == grip_end and not self.grasp_active:
            set_grasp_constraint(self.env.physics, True)
            self.grasp_active = True
        if self.step == place_end and self.grasp_active:
            self.release_hand = self.env.physics.named.data.site_xpos[
                "grasp", ["x", "z"]
            ].copy()
            set_grasp_constraint(self.env.physics, False)
            self.grasp_active = False
            self.released = True
        if self.step < APPROACH_STEPS:
            progress = smoothstep(self.step / max(1, APPROACH_STEPS - 1))
            difference = (self.reverse_insertion_q[0] - self.initial_q + np.pi) % (2 * np.pi) - np.pi
            action = self._joint_action(self.initial_q + progress * difference, OPEN_GRIP)
            self.previous_action = action.copy()
            self.step += 1
            return action
        elif self.step < grip_end:
            action = self._joint_action(self.reverse_insertion_q[0], CLOSED_GRIP)
            self.previous_action = action.copy()
            self.step += 1
            return action
        elif self.step < place_end:
            path_progress = (self.step - grip_end) / max(1, CARRY_STEPS + PLACE_STEPS - 1)
            joint_target = self._interpolate_targets(self.path_targets, path_progress)
            action = self._joint_action(joint_target, CLOSED_GRIP)
            if self.step == carry_end - 1:
                peg = self.env.get_state()[33:35]
                self.waypoint_error = float(np.linalg.norm(peg - self.waypoint))
                self.reached_waypoint = self.waypoint_error <= 0.10
            self.previous_action = action.copy()
            self.step += 1
            return action
        elif self.step < release_end:
            progress = smoothstep((self.step - place_end) / max(1, RELEASE_STEPS - 1))
            target = self.release_hand + np.array([0.04 * self.side, 0.10]) * progress
            angle, grip = self.destination_angle, -1.0
        else:
            target = self.env.physics.named.data.site_xpos["grasp", ["x", "z"]].copy()
            target[1] += 0.08
            angle, grip = self.destination_angle, -1.0
        action = self._operational_action(target, angle, grip)
        self.previous_action = action.copy()
        self.step += 1
        return action
