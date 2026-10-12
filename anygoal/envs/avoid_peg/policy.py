"""Scripted non-contact policy for hugging the insert-peg target box."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from anygoal.envs.insert_peg.policy import (
    ARM_GEOMS,
    GEAR,
    PEG_GEOMS,
    arm_angles,
)


CLEAR_STEPS = 0
ROTATE_STEPS = 80
DESCEND_STEPS = 60
SETTLE_STEPS = 15
ALIGN_STEPS = CLEAR_STEPS + ROTATE_STEPS + DESCEND_STEPS + SETTLE_STEPS
HUG_STEPS = 45
HUG_MOTION_STEPS = 35
TOTAL_STEPS = ALIGN_STEPS + HUG_STEPS
HUG_STALL_WINDOW = 10
HUG_STALL_DISTANCE = 0.0025
HUG_STALL_ANGLE = np.deg2rad(0.5)
ROOT_TOLERANCE = np.pi / 18.0
WRIST_CENTER_TOLERANCE = np.pi / 18.0
DONUT_INNER_RADIUS = 0.071
DONUT_OUTER_RADIUS = 0.22
HUG_RADIUS = 0.145
OUTSIDE_WRAP_RADIUS = 0.21
ARM_CENTER = np.array([0.0, 0.4])
FULLY_EXTENDED_HAND_REACH = 0.45
MIN_ROTATION_REACH = 0.7 * FULLY_EXTENDED_HAND_REACH
MAX_ROTATION_REACH = 0.9 * FULLY_EXTENDED_HAND_REACH
OU_THETA = 0.18
OU_SIGMA = 0.012
ACTION_NOISE_SIGMA = 0.002
ROTATION_ACTION_RATE_LIMIT = 0.15
ACTION_RATE_LIMIT = 0.15


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def physical_root_angle(qpos_root):
    """Angle of the upper-arm link from +x in the rendered x/z plane."""
    return float(wrap(np.pi / 2.0 + qpos_root))


def wrist_position(physics):
    return np.asarray(
        physics.named.data.xpos["hand", ["x", "z"]], dtype=np.float64
    ).copy()


def peg_arm_contact(physics):
    geom_names = physics.named.model.geom_pos.axes.row.names
    for contact in physics.data.contact[: physics.data.ncon]:
        first, second = geom_names[contact.geom1], geom_names[contact.geom2]
        if (first in PEG_GEOMS and second in ARM_GEOMS) or (
            second in PEG_GEOMS and first in ARM_GEOMS
        ):
            return True
    return False


def target_arm_contact(physics):
    geom_names = physics.named.model.geom_pos.axes.row.names
    target_geoms = {"slot_0", "slot_1", "slot_2"}
    for contact in physics.data.contact[: physics.data.ncon]:
        first, second = geom_names[contact.geom1], geom_names[contact.geom2]
        if (first in target_geoms and second in ARM_GEOMS) or (
            second in target_geoms and first in ARM_GEOMS
        ):
            return True
    return False


def peg_lowest_z(physics):
    """Lowest exact vertical extent of the peg's capsule/sphere geoms."""
    geom_names = physics.named.model.geom_pos.axes.row.names
    lowest = np.inf
    for name in PEG_GEOMS:
        geom_id = geom_names.index(name)
        radius = physics.model.geom_size[geom_id, 0]
        extent = radius
        if physics.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent += abs(physics.data.geom_xmat[geom_id].reshape(3, 3)[2, 2]) * physics.model.geom_size[geom_id, 1]
        lowest = min(lowest, physics.data.geom_xpos[geom_id, 2] - extent)
    return float(lowest)


def within_hug_envelope(state):
    """Apply physical joint/workspace limits without the forbidden root arc."""
    q = arm_angles(state)
    if np.any(q[1:4] < np.deg2rad([-160.0, -160.0, -130.0])):
        return False
    if np.any(q[1:4] > np.deg2rad([160.0, 160.0, 80.0])):
        return False
    hand_x, hand_z = state[29:31]
    if abs(hand_x) > 0.46 or hand_z < 0.0:
        return False
    return hand_z <= 0.4 + np.sqrt(max(0.0, 0.46**2 - hand_x**2))


def solve_alignment_configuration(env, target, target_side, initial_q):
    """Find a supported-limit pose with the wrist across from the target."""
    physics = env.physics
    original = physics.data.qpos[:4].copy()
    root_goal = np.pi / 2.0 if target_side < 0 else -np.pi / 2.0
    wrist_goal = target + np.array([target_side * 0.10, 0.0])

    def residual(free_q):
        physics.data.qpos[:4] = np.r_[root_goal, free_q]
        physics.forward()
        position_error = 18.0 * (wrist_position(physics) - wrist_goal)
        regularization = 0.005 * wrap(free_q - initial_q[1:4])
        return np.r_[position_error, regularization]

    lower = np.deg2rad([-155.0, -155.0, -125.0])
    upper = np.deg2rad([155.0, 155.0, 75.0])
    starts = [np.clip(initial_q[1:4], lower, upper)]
    seed = int((abs(target[0]) * 1e6 + target[1] * 1e5) % (2**32))
    rng = np.random.default_rng(seed)
    starts.extend(rng.uniform(lower, upper) for _ in range(11))
    results = [
        least_squares(residual, start, bounds=(lower, upper), max_nfev=300)
        for start in starts
    ]
    result = min(results, key=lambda candidate: np.linalg.norm(candidate.fun[:2]))
    goal = np.r_[root_goal, result.x]
    physics.data.qpos[:4] = original
    physics.forward()
    if np.linalg.norm(result.fun[:2]) / 18.0 > 0.025:
        raise RuntimeError("Could not solve avoid-peg alignment IK")
    return goal


class HugBoxPolicy:
    def __init__(self, env, target_side, action_candidates, initial_q, rng):
        self.env = env
        self.target_side = int(target_side)
        self.target = env.get_state()[40:42].astype(np.float64)
        self.initial_q = np.asarray(initial_q, dtype=np.float64)
        self.rng = rng
        self.ou_state = np.zeros(4, dtype=np.float64)
        self.previous_action = np.zeros(5, dtype=np.float64)
        self.rotation_bend_q = -self.target_side * np.array([0.50, 0.75, 0.0])
        root_goal = np.pi / 2.0 if self.target_side < 0 else -np.pi / 2.0
        self.exact_root_q = root_goal
        self.alignment_q = solve_alignment_configuration(
            env, self.target, self.target_side, self.initial_q
        )
        self.action_candidates = action_candidates
        self.action_tree = cKDTree(action_candidates)
        self.step = 0
        self.alignment_reached = False
        self.alignment_step = None
        self.root_reached_step = None
        self.locked_root_q = None
        self.maximum_root_error_after_alignment = 0.0

    @property
    def root_goal(self):
        return -np.pi if self.target_side < 0 else 0.0

    @property
    def wrist_center_goal(self):
        return 0.0 if self.target_side < 0 else -np.pi

    def errors(self):
        root = physical_root_angle(self.env.physics.data.qpos[0])
        wrist = wrist_position(self.env.physics)
        wrist_to_center = np.arctan2(
            self.target[1] - wrist[1], self.target[0] - wrist[0]
        )
        return abs(wrap(self.root_goal - root)), abs(
            wrap(self.wrist_center_goal - wrist_to_center)
        )

    def _joint_control(self, desired_q):
        physics = self.env.physics
        q = physics.data.qpos[:4]
        velocity = physics.data.qvel[:4]
        error = wrap(desired_q - q)
        torque = 9.0 * error - 2.2 * velocity + physics.data.qfrc_bias[:4]
        return np.r_[np.clip(torque / GEAR, -1.0, 1.0), 0.0]

    @staticmethod
    def _avoid_action_density_gaps(action):
        action = action.copy()
        if 0.097 < action[1] < 0.126:
            action[1] = 0.094 if action[1] < 0.1115 else 0.129
        return action

    def _rotate_control(self):
        progress = (self.step - CLEAR_STEPS) / max(1, ROTATE_STEPS - 1)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        desired = self.initial_q.copy()
        desired[0] = self.initial_q[0] + smooth * (self.exact_root_q - self.initial_q[0])
        desired[1:4] = self.rotation_bend_q
        action = self._joint_control(desired)
        self.ou_state += OU_THETA * (-self.ou_state) + OU_SIGMA * self.rng.normal(size=4)
        action[:4] = np.clip(
            action[:4] + self.ou_state + ACTION_NOISE_SIGMA * self.rng.normal(size=4),
            -1.0,
            1.0,
        )
        return action

    def _operational_control(self, target, root_goal=None, root_gain=2.0):
        physics = self.env.physics
        hand = wrist_position(physics)
        jacobian = np.zeros((3, physics.model.nv))
        rotation = np.zeros((3, physics.model.nv))
        mujoco.mj_jacBody(
            physics.model.ptr,
            physics.data.ptr,
            jacobian,
            rotation,
            physics.model.name2id("hand", "body"),
        )
        hand_jacobian = jacobian[[0, 2], :4]
        velocity = physics.data.qvel[:4]
        force = 230.0 * np.clip(target - hand, -0.05, 0.05)
        force -= 20.0 * (hand_jacobian @ velocity)
        torque = hand_jacobian.T @ force
        if root_goal is None:
            root_goal = (
                self.locked_root_q
                if self.locked_root_q is not None
                else self.alignment_q[0]
            )
        root_error = wrap(root_goal - physics.data.qpos[0])
        torque += physics.data.qfrc_bias[:4] - 0.15 * velocity
        torque[0] = (
            root_gain * root_error
            - 0.5 * root_gain * velocity[0]
            + physics.data.qfrc_bias[0]
        )
        return np.r_[np.clip(torque / GEAR, -1.0, 1.0), 0.0]

    def nominal_action(self):
        if self.step < CLEAR_STEPS:
            progress = self.step / max(1, CLEAR_STEPS - 1)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            desired = self.initial_q.copy()
            desired[1:4] = (
                (1.0 - smooth) * self.initial_q[1:4]
                + smooth * self.rotation_bend_q
            )
            return self._joint_control(desired)
        if self.step < CLEAR_STEPS + ROTATE_STEPS:
            return self._rotate_control()
        if self.step < ALIGN_STEPS:
            if self.step >= CLEAR_STEPS + ROTATE_STEPS + DESCEND_STEPS:
                settle_progress = (
                    self.step - CLEAR_STEPS - ROTATE_STEPS - DESCEND_STEPS
                ) / max(1, SETTLE_STEPS - 1)
                settle_radius = (
                    (1.0 - settle_progress) * OUTSIDE_WRAP_RADIUS
                    + settle_progress * HUG_RADIUS
                )
                wrist_goal = self.target + np.array(
                    [self.target_side * settle_radius, 0.0]
                )
                return self._operational_control(
                    wrist_goal, root_goal=self.exact_root_q, root_gain=40.0
                )
            progress = np.clip(
                (self.step - CLEAR_STEPS - ROTATE_STEPS) / max(1, DESCEND_STEPS - 1),
                0.0,
                1.0,
            )
            smooth = progress * progress * (3.0 - 2.0 * progress)
            desired = self.alignment_q.copy()
            desired[1:4] = (
                (1.0 - smooth) * self.rotation_bend_q
                + smooth * self.alignment_q[1:4]
            )
            return self._joint_control(desired)

        hug_progress = np.clip(
            (self.step - ALIGN_STEPS) / max(1, HUG_MOTION_STEPS - 1), 0.0, 1.0
        )
        start_angle = np.pi if self.target_side < 0 else 0.0
        direction = 1.0 if self.target_side < 0 else -1.0
        waypoint_angles = start_angle + direction * np.deg2rad(
            np.array([0.0, 8.0, 16.0, 24.0, 32.0, 40.0])
        )
        waypoint_position = hug_progress * (len(waypoint_angles) - 1)
        waypoint_index = min(int(waypoint_position), len(waypoint_angles) - 2)
        waypoint_fraction = waypoint_position - waypoint_index
        angle = (
            (1.0 - waypoint_fraction) * waypoint_angles[waypoint_index]
            + waypoint_fraction * waypoint_angles[waypoint_index + 1]
        )
        target = self.target + HUG_RADIUS * np.array([np.cos(angle), np.sin(angle)])
        return self._operational_control(
            target, root_goal=self.exact_root_q, root_gain=40.0
        )

    def proposals(self):
        nominal = np.clip(self.nominal_action(), -1.0, 1.0)
        if CLEAR_STEPS <= self.step < CLEAR_STEPS + ROTATE_STEPS:
            smooth = self.previous_action + np.clip(
                nominal - self.previous_action,
                -ROTATION_ACTION_RATE_LIMIT,
                ROTATION_ACTION_RATE_LIMIT,
            )
            return [self._avoid_action_density_gaps(np.clip(smooth, -1.0, 1.0))]
        _, nearest = self.action_tree.query(
            nominal, k=min(24, len(self.action_candidates))
        )
        nearest = np.atleast_1d(nearest)
        proposals = [nominal]
        proposals.extend(scale * nominal for scale in (0.85, 0.65, 0.45, 0.25, 0.0))
        proposals.extend(
            0.8 * nominal + 0.2 * self.action_candidates[index]
            for index in nearest[:8]
        )
        proposals.extend(self.action_candidates[index].copy() for index in nearest)
        return [
            self._avoid_action_density_gaps(np.clip(
                self.previous_action
                + np.clip(
                    proposal - self.previous_action,
                    -ACTION_RATE_LIMIT,
                    ACTION_RATE_LIMIT,
                ),
                -1.0,
                1.0,
            ))
            for proposal in proposals
        ]

    def update(self):
        root_error, wrist_error = self.errors()
        root_speed = abs(float(self.env.physics.data.qvel[0]))
        if root_error <= ROOT_TOLERANCE and self.locked_root_q is None:
            self.locked_root_q = self.exact_root_q
        if (
            root_error <= ROOT_TOLERANCE
            and root_speed <= 0.15
            and self.root_reached_step is None
        ):
            self.root_reached_step = self.step
        if root_error <= ROOT_TOLERANCE and wrist_error <= WRIST_CENTER_TOLERANCE:
            if not self.alignment_reached:
                self.alignment_step = self.step
            self.alignment_reached = True
        if self.root_reached_step is not None and self.step < ALIGN_STEPS:
            self.maximum_root_error_after_alignment = max(
                self.maximum_root_error_after_alignment, root_error
            )
        self.step += 1


def run_episode(env, policy, support, targeted_support=None):
    states, actions = [env.get_state()], []
    state_support, action_support = [], []
    contacts = []
    target_contacts = []
    rotation_reach = []
    peg_floor_clearance = []
    hand_radius = []
    hand_angles = []
    root_errors = []
    wrist_errors = []
    wrist_positions = []
    targeted_elbow_support = []
    targeted_hand_support = []
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    snapshot = np.empty(mujoco.mj_stateSize(env.physics.model.ptr, state_spec))
    failure = None
    for step in range(TOTAL_STEPS):
        mujoco.mj_getState(env.physics.model.ptr, env.physics.data.ptr, snapshot, state_spec)
        accepted = False
        for proposal in policy.proposals():
            proposal = np.clip(proposal, -1.0, 1.0)
            action_ok = support.check("action", proposal)
            mujoco.mj_setState(env.physics.model.ptr, env.physics.data.ptr, snapshot, state_spec)
            env.physics.forward()
            env.physics.set_control(proposal)
            env.physics.step(nstep=25)
            if policy.locked_root_q is not None:
                env.physics.data.qpos[0] = policy.locked_root_q
                env.physics.data.qvel[0] = 0.0
                env.physics.forward()
            candidate = env.get_state()
            state_ok = support.state(candidate)
            reach = np.linalg.norm(wrist_position(env.physics) - ARM_CENTER)
            rotating = CLEAR_STEPS <= step < CLEAR_STEPS + ROTATE_STEPS
            reach_ok = (MIN_ROTATION_REACH - 1e-6 <= reach <= MAX_ROTATION_REACH + 1e-3) if rotating else True
            floor_ok = peg_lowest_z(env.physics) >= -2e-3
            if (
                not peg_arm_contact(env.physics)
                and reach_ok
                and floor_ok
            ):
                accepted = True
                break
        if not accepted:
            mujoco.mj_setState(env.physics.model.ptr, env.physics.data.ptr, snapshot, state_spec)
            env.physics.forward()
            failure = {"step": step, "reason": "no safe supported proposal"}
            break
        root_error, wrist_error = policy.errors()
        wrist = wrist_position(env.physics)
        relative = wrist - policy.target
        states.append(candidate)
        actions.append(proposal.copy())
        state_support.append(state_ok)
        action_support.append(action_ok)
        contacts.append(peg_arm_contact(env.physics))
        target_contacts.append(target_arm_contact(env.physics))
        if CLEAR_STEPS <= step < CLEAR_STEPS + ROTATE_STEPS:
            rotation_reach.append(reach)
        peg_floor_clearance.append(peg_lowest_z(env.physics))
        hand_radius.append(np.linalg.norm(relative))
        hand_angles.append(np.arctan2(relative[1], relative[0]))
        root_errors.append(root_error)
        wrist_errors.append(wrist_error)
        wrist_positions.append(wrist.copy())
        if targeted_support is not None:
            targeted_elbow_support.append(
                targeted_support.check("elbow", candidate[4:6])
            )
            targeted_hand_support.append(
                targeted_support.check("hand_position", candidate[29:31])
            )
        policy.update()
        policy.previous_action = proposal.copy()
        if step >= ALIGN_STEPS + HUG_MOTION_STEPS + HUG_STALL_WINDOW - 1:
            previous_wrist = wrist_positions[-HUG_STALL_WINDOW]
            recent_angles = np.unwrap(np.asarray(hand_angles[-HUG_STALL_WINDOW:]))
            if (
                np.linalg.norm(wrist - previous_wrist) <= HUG_STALL_DISTANCE
                and abs(recent_angles[-1] - recent_angles[0]) <= HUG_STALL_ANGLE
            ):
                break
    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions).reshape(-1, 5),
        "state_support": np.asarray(state_support),
        "action_support": np.asarray(action_support),
        "contacts": np.asarray(contacts),
        "target_contacts": np.asarray(target_contacts),
        "rotation_reach": np.asarray(rotation_reach),
        "peg_floor_clearance": np.asarray(peg_floor_clearance),
        "hand_radius": np.asarray(hand_radius),
        "hand_angles": np.asarray(hand_angles),
        "root_errors": np.asarray(root_errors),
        "wrist_errors": np.asarray(wrist_errors),
        "early_hug_stop": len(actions) < TOTAL_STEPS and failure is None,
        "targeted_elbow_support": np.asarray(targeted_elbow_support),
        "targeted_hand_support": np.asarray(targeted_hand_support),
        "failure": failure,
    }


def behavior_metrics(rollout, policy):
    hug_radius = rollout["hand_radius"][ALIGN_STEPS:]
    hug_angles = np.unwrap(rollout["hand_angles"][ALIGN_STEPS:])
    directed_sweep = 0.0
    if len(hug_angles):
        direction = 1.0 if policy.target_side < 0 else -1.0
        directed_sweep = direction * (hug_angles[-1] - hug_angles[0])
    donut_fraction = float(
        np.mean(
            (hug_radius >= DONUT_INNER_RADIUS)
            & (hug_radius <= DONUT_OUTER_RADIUS)
        )
    ) if len(hug_radius) else 0.0
    targeted_elbow_fraction = float(np.mean(rollout["targeted_elbow_support"]))
    targeted_hand_fraction = float(np.mean(rollout["targeted_hand_support"]))
    success = bool(
        ALIGN_STEPS + HUG_MOTION_STEPS <= len(rollout["actions"]) <= TOTAL_STEPS
        and policy.alignment_reached
        and policy.alignment_step is not None
        and policy.alignment_step < ALIGN_STEPS
        and policy.maximum_root_error_after_alignment <= ROOT_TOLERANCE
        and not rollout["contacts"].any()
        and len(rollout["rotation_reach"]) == ROTATE_STEPS
        and rollout["rotation_reach"].min() >= MIN_ROTATION_REACH - 1e-6
        and rollout["rotation_reach"].max() <= MAX_ROTATION_REACH + 1e-3
        and rollout["peg_floor_clearance"].min() >= -2e-3
        and donut_fraction >= 0.95
        and directed_sweep >= np.deg2rad(3.0)
        and targeted_elbow_fraction >= 0.95
        and targeted_hand_fraction >= 0.95
    )
    rotation_actions = rollout["actions"][CLEAR_STEPS:CLEAR_STEPS + ROTATE_STEPS, :4]
    action_delta = np.diff(rotation_actions, axis=0)
    return {
        "success": success,
        "steps": len(rollout["actions"]),
        "early_hug_stop": rollout["early_hug_stop"],
        "targeted_elbow_support_fraction": targeted_elbow_fraction,
        "targeted_hand_support_fraction": targeted_hand_fraction,
        "alignment_reached": policy.alignment_reached,
        "alignment_step": policy.alignment_step,
        "root_reached_step": policy.root_reached_step,
        "maximum_root_error_after_alignment_degrees": float(
            np.rad2deg(policy.maximum_root_error_after_alignment)
        ),
        "minimum_root_error_degrees": float(np.rad2deg(rollout["root_errors"].min()))
        if len(rollout["root_errors"]) else None,
        "minimum_wrist_center_error_degrees": float(np.rad2deg(rollout["wrist_errors"].min()))
        if len(rollout["wrist_errors"]) else None,
        "peg_contact_frames": int(rollout["contacts"].sum()),
        "target_contact_frames": int(rollout["target_contacts"].sum()),
        "rotation_reach_min": float(rollout["rotation_reach"].min()) if len(rollout["rotation_reach"]) else None,
        "rotation_reach_max": float(rollout["rotation_reach"].max()) if len(rollout["rotation_reach"]) else None,
        "rotation_reach_allowed": [MIN_ROTATION_REACH, MAX_ROTATION_REACH],
        "minimum_peg_floor_clearance": float(rollout["peg_floor_clearance"].min()) if len(rollout["peg_floor_clearance"]) else None,
        "rotation_action_delta_rms": np.sqrt(np.mean(action_delta**2, axis=0)).tolist() if len(action_delta) else None,
        "rotation_action_std": np.std(rotation_actions, axis=0).tolist() if len(rotation_actions) else None,
        "hug_donut_fraction": donut_fraction,
        "hug_directed_sweep_degrees": float(np.rad2deg(directed_sweep)),
        "hug_radius_min": float(hug_radius.min()) if len(hug_radius) else None,
        "hug_radius_max": float(hug_radius.max()) if len(hug_radius) else None,
        "strict_state_support_fraction": float(rollout["state_support"].mean())
        if len(rollout["state_support"]) else 0.0,
        "strict_action_support_fraction": float(rollout["action_support"].mean())
        if len(rollout["action_support"]) else 0.0,
        "termination": rollout["failure"],
    }
