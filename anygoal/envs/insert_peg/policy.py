"""Physics-aware non-greedy expert for sliding a peg on an outside ramp."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import cKDTree


LOW = np.deg2rad([-180, -75, -100, -130, -10, -45, -10, -45])
HIGH = np.deg2rad([180, 160, 160, 80, 35, 15, 35, 15])
GEAR = np.array([12.0, 8.0, 4.0, 2.0])
PEG_GEOMS = {"blade", "guard", "pommel"}
ARM_GEOMS = {
    "upper_arm", "middle_arm", "lower_arm", "hand", "palm1", "palm2",
    "thumb1", "thumb2", "thumbtip1", "thumbtip2", "finger1", "finger2",
    "fingertip1", "fingertip2",
}
# Indices into successful_reference[::4]. These anchors were replay-validated
# for all 500 steps under the strict late-contact rule. They come from the same
# immutable successful RL-Unplugged reference used by the collector.
VALIDATED_ANCHORS = {
    1: np.array([60095, 31394, 11806, 31329, 31330, 41713]),
    -1: np.array([41794]),
}


def arm_angles(state: np.ndarray) -> np.ndarray:
    return np.arctan2(state[:16:2], state[1:16:2])


def within_physical_envelope(state: np.ndarray) -> bool:
    q = arm_angles(state)
    if np.any(q < LOW) or np.any(q > HIGH):
        return False
    if np.deg2rad(-75) <= q[0] <= np.deg2rad(-50):
        return False
    hand_x, hand_z = state[29:31]
    if abs(hand_x) > 0.42:
        return False
    if hand_z > 0.4 + np.sqrt(max(0.0, 0.42**2 - hand_x**2)):
        return False
    peg_x, peg_z = state[33:35]
    if peg_z > 0.6:
        return False
    return not (
        0.3 <= abs(peg_x) <= 0.5
        and 0.4 <= peg_z <= 0.6
        and abs(peg_x) + peg_z >= 0.9
    )


def peg_arm_contact(physics) -> bool:
    geom_names = physics.named.model.geom_pos.axes.row.names
    for contact in physics.data.contact[: physics.data.ncon]:
        first, second = geom_names[contact.geom1], geom_names[contact.geom2]
        if (first in PEG_GEOMS and second in ARM_GEOMS) or (
            second in PEG_GEOMS and first in ARM_GEOMS
        ):
            return True
    return False


class SlidingPegExpert:
    def __init__(
        self, env, ramp_side, bank, actions, action_candidates,
        pose_tree, action_tree, anchor_index, anchor_state,
    ):
        self.env = env
        self.ramp_side = int(ramp_side)
        self.bank = bank
        self.actions = actions
        self.action_candidates = action_candidates
        self.pose_tree = pose_tree
        self.action_tree = action_tree
        self.anchor = anchor_state.copy()
        self.anchor_index = int(anchor_index)
        self.phase = "push"
        self.phase_steps = 0
        self.cycles = 0
        self.contact_events = 0
        self.last_contact = False
        self.push_start = self.anchor[29:31].copy()
        self.push_end = np.array(
            [self.ramp_side * 0.415, min(0.25, self.push_start[1] + 0.085)]
        )
        self.grip = float(actions[anchor_index, 4])

    def _reference_wrist_and_grip(self, state):
        features = np.r_[state[:6], state[8:16], state[29:31]]
        _, index = self.pose_tree.query(features)
        wrist = np.arctan2(self.bank[index, 6], self.bank[index, 7])
        return wrist, float(self.actions[index, 4])

    def _operational_control(self, target, hand_angle):
        state = self.env.get_state()
        hand = state[29:31]
        physics = self.env.physics
        jacobian = np.zeros((3, physics.model.nv))
        rotation = np.zeros((3, physics.model.nv))
        mujoco.mj_jacBody(
            physics.model.ptr, physics.data.ptr, jacobian, rotation,
            physics.model.name2id("hand", "body"),
        )
        hand_jacobian = jacobian[[0, 2], :4]
        velocity = physics.data.qvel[:4]
        error = np.clip(target - hand, -0.06, 0.06)
        torque = hand_jacobian.T @ (
            210.0 * error - 18.0 * (hand_jacobian @ velocity)
        )
        current_angle = 2.0 * np.arctan2(state[32], state[31])
        angle_error = (hand_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
        torque += rotation[1, :4] * (
            1.2 * angle_error - 0.16 * (rotation[1, :4] @ velocity)
        )
        torque += physics.data.qfrc_bias[:4] - 0.12 * velocity
        wrist_target, grip = self._reference_wrist_and_grip(state)
        wrist = arm_angles(state)[3]
        wrist_error = (wrist_target - wrist + np.pi) % (2 * np.pi) - np.pi
        torque[3] += 1.6 * wrist_error
        return np.r_[
            np.clip(torque / GEAR, -1, 1),
            np.clip(0.65 * self.grip + 0.35 * grip, -1, 1),
        ]

    def action(self):
        peg = self.env.get_state()[33:35]
        if self.phase == "push" and (peg[1] > 0.12 or self.phase_steps > 80):
            self.phase, self.phase_steps = "retract", 0
            self.cycles += 1
        elif self.phase == "retract" and self.phase_steps > 45:
            self.phase, self.phase_steps = "push", 0
        if self.phase == "push":
            progress = min(1.0, self.phase_steps / 55.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            target = self.push_start + smooth * (self.push_end - self.push_start)
        else:
            target = np.array([self.ramp_side * 0.29, 0.20])
        hand_angle = 2.0 * np.arctan2(self.anchor[32], self.anchor[31])
        return self._operational_control(target, hand_angle)

    def update(self, contact: bool) -> None:
        self.phase_steps += 1
        if contact and not self.last_contact:
            self.contact_events += 1
        self.last_contact = contact


def build_search_trees(bank, action_candidates):
    pose_features = np.c_[bank[:, :6], bank[:, 8:16], bank[:, 29:31]]
    return cKDTree(pose_features), cKDTree(action_candidates)


def initialize_episode(
    env, rng, bank, actions, action_candidates, target_pool, target_side,
    support, pose_tree, action_tree,
):
    """Initialize on the ramp opposite a balanced, source-dataset slot pose."""
    peg, hand = bank[:, 33:35], bank[:, 29:31]
    ramp_error = np.abs(peg[:, 1] - (np.abs(peg[:, 0]) - 0.4))
    ramp_side = -int(target_side)
    geometric_candidates = np.flatnonzero(
        (peg[:, 0] * ramp_side > 0.405)
        & (np.abs(peg[:, 0]) < 0.50)
        & (peg[:, 1] > 0.015)
        & (peg[:, 1] < 0.12)
        & (ramp_error < 0.03)
        & (np.linalg.norm(peg - hand, axis=1) < 0.13)
        & (bank[:, 24:29].max(axis=1) > 0.05)
    )
    validated = VALIDATED_ANCHORS[ramp_side]
    candidates = validated[np.isin(validated, geometric_candidates)]
    side_targets = target_pool[target_pool[:, 0] * target_side > 0.2]
    if not len(candidates) or not len(side_targets):
        raise RuntimeError(f"No initialization candidate for target side {target_side}")
    for _ in range(3000):
        index = int(rng.choice(candidates))
        state = bank[index].copy()
        state[16:24] = 0.0
        state[37:40] = 0.0
        state[40:44] = side_targets[int(rng.integers(len(side_targets)))]
        env.reset(seed=int(rng.integers(2**30)), options={"state": state})
        restored = env.get_state()
        if within_physical_envelope(restored) and support.state(restored):
            return SlidingPegExpert(
                env, ramp_side, bank, actions, action_candidates,
                pose_tree, action_tree, index, restored,
            )
    raise RuntimeError("Could not initialize a supported insert-peg episode")


def run_episode(env, policy, support, steps=500):
    states, actions, contacts = [env.get_state()], [], []
    supported_states, supported_actions = [], []
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    snapshot = np.empty(mujoco.mj_stateSize(env.physics.model.ptr, state_spec))
    failure = None
    for step in range(steps):
        nominal = policy.action()
        _, nearest = policy.action_tree.query(
            nominal, k=min(24, len(policy.action_candidates))
        )
        nearest = np.atleast_1d(nearest)
        proposals = [nominal]
        proposals.extend(scale * nominal for scale in (0.85, 0.65, 0.45, 0.25, 0.0))
        proposals.extend(
            0.75 * nominal + 0.25 * policy.action_candidates[index]
            for index in nearest[:6]
        )
        proposals.extend(policy.action_candidates[index].copy() for index in nearest)
        mujoco.mj_getState(
            env.physics.model.ptr, env.physics.data.ptr, snapshot, state_spec
        )
        accepted = False
        for proposal in proposals:
            proposal = np.clip(proposal, -1, 1)
            action_supported = support.check("action", proposal)
            action_budget = max(3, int(0.045 * (step + 1)))
            if not action_supported and supported_actions.count(False) >= action_budget:
                continue
            mujoco.mj_setState(
                env.physics.model.ptr, env.physics.data.ptr, snapshot, state_spec
            )
            env.physics.forward()
            env.physics.set_control(proposal)
            env.physics.step(nstep=10)
            candidate = env.get_state()
            state_supported = support.state(candidate)
            state_budget = max(3, int(0.045 * (step + 1)))
            if within_physical_envelope(candidate) and (
                state_supported or supported_states.count(False) < state_budget
            ):
                accepted = True
                break
        if not accepted:
            mujoco.mj_setState(
                env.physics.model.ptr, env.physics.data.ptr, snapshot, state_spec
            )
            env.physics.forward()
            failure = {"step": step, "physical_envelope": within_physical_envelope(env.get_state())}
            break
        contact = peg_arm_contact(env.physics)
        policy.update(contact)
        actions.append(proposal.copy())
        states.append(candidate)
        contacts.append(contact)
        supported_states.append(state_supported)
        supported_actions.append(action_supported)
    return (
        np.asarray(states), np.asarray(actions).reshape(-1, 5), np.asarray(contacts),
        failure, np.asarray(supported_states), np.asarray(supported_actions),
    )


def behavior_metrics(states, contacts, policy):
    peg = states[:, 33:35]
    delta_z = np.diff(peg[:, 1])
    contact_steps = np.flatnonzero(contacts)
    contact_bins = [int(contacts[start : start + 100].sum()) for start in range(0, 500, 100)]
    last_contact = int(contact_steps[-1]) if len(contact_steps) else -1
    late_path = float(np.linalg.norm(np.diff(peg[-101:], axis=0), axis=1).sum())
    excursion = float(np.ptp(peg[:, 1]))
    reversals = int(
        np.count_nonzero((delta_z[:-1] < -0.001) & (delta_z[1:] > 0.001))
    )
    path_length = float(np.linalg.norm(np.diff(peg, axis=0), axis=1).sum())
    ramp_error = np.abs(peg[:, 1] - np.maximum(0.0, np.abs(peg[:, 0]) - 0.4))
    same_side = float(np.mean(peg[:, 0] * policy.ramp_side > 0.38))
    ramp_fraction = float(np.mean(ramp_error < 0.045))
    success = (
        len(states) == 501
        and policy.contact_events >= 10
        and contacts.sum() >= 30
        and policy.cycles >= 2
        and excursion >= 0.055
        and reversals >= 3
        and path_length >= 0.30
        and same_side >= 0.90
        and ramp_fraction >= 0.75
        and last_contact >= 450
        and late_path >= 0.10
        and min(contact_bins) >= 1
    )
    return {
        "success": bool(success),
        "vertical_excursion": excursion,
        "upward_reversals": reversals,
        "object_path_length": path_length,
        "late_object_path_length": late_path,
        "contact_frames": int(contacts.sum()),
        "contact_events": policy.contact_events,
        "contact_frames_per_100_steps": contact_bins,
        "last_contact_step": last_contact,
        "completed_cycles": policy.cycles,
        "same_ramp_fraction": same_side,
        "ramp_occupancy_fraction": ramp_fraction,
    }
