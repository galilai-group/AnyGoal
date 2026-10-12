"""Reference-support and HDF5 helpers for generated Cube variants."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from anygoal.envs.cube_hop.env import CUBE_HOP_SIDES


REFERENCE_ACTION_SEED = 20260811
REFERENCE_ACTIONS = 200_000
CALIBRATION_ACTIONS = 20_000
SUPPORT_NEIGHBORS = 20
SUPPORT_QUANTILE = 0.95
CUBE_DRIFT_TOLERANCE = 2e-4
MAX_ATTEMPT_MULTIPLIER = 20

SOURCE_COLUMNS = (
    "control",
    "id",
    "observation",
    "pixels",
    "prev_qpos",
    "prev_qvel",
    "privileged/block_0_pos",
    "privileged/block_0_quat",
    "privileged/block_0_yaw",
    "privileged/target_block",
    "privileged/target_block_pos",
    "privileged/target_block_yaw",
    "privileged/target_task",
    "proprio/effector_pos",
    "proprio/effector_yaw",
    "proprio/gripper_contact",
    "proprio/gripper_opening",
    "proprio/gripper_vel",
    "proprio/joint_pos",
    "proprio/joint_vel",
    "qpos",
    "qvel",
    "render_time",
    "reward",
    "step_idx",
    "success",
    "target",
    "terminated",
    "time",
    "truncated",
)

CUBE_H5_COLUMNS = (
    "action",
    "control",
    "ep_idx",
    "id",
    "observation",
    "pixels",
    "prev_qpos",
    "prev_qvel",
    "privileged_block_0_pos",
    "privileged_block_0_quat",
    "privileged_block_0_yaw",
    "privileged_target_block",
    "privileged_target_block_pos",
    "privileged_target_block_yaw",
    "privileged_target_task",
    "proprio_effector_pos",
    "proprio_effector_yaw",
    "proprio_gripper_contact",
    "proprio_gripper_opening",
    "proprio_gripper_vel",
    "proprio_joint_pos",
    "proprio_joint_vel",
    "qpos",
    "qvel",
    "render_time",
    "reward",
    "step_idx",
    "success",
    "target",
    "terminated",
    "time",
    "truncated",
)

COLUMN_DTYPES = {
    "action": np.float32,
    "control": np.float64,
    "ep_idx": np.int32,
    "id": np.int64,
    "observation": np.float64,
    "pixels": np.uint8,
    "prev_qpos": np.float64,
    "prev_qvel": np.float64,
    "privileged_block_0_pos": np.float64,
    "privileged_block_0_quat": np.float64,
    "privileged_block_0_yaw": np.float64,
    "privileged_target_block": np.int64,
    "privileged_target_block_pos": np.float64,
    "privileged_target_block_yaw": np.float64,
    "proprio_effector_pos": np.float64,
    "proprio_effector_yaw": np.float64,
    "proprio_gripper_contact": np.float64,
    "proprio_gripper_opening": np.float64,
    "proprio_gripper_vel": np.float64,
    "proprio_joint_pos": np.float64,
    "proprio_joint_vel": np.float64,
    "qpos": np.float64,
    "qvel": np.float64,
    "render_time": np.float64,
    "reward": np.float64,
    "step_idx": np.int64,
    "success": np.bool_,
    "target": np.float64,
    "terminated": np.bool_,
    "time": np.float64,
    "truncated": np.bool_,
}


def flattened_column(source: str) -> str:
    return source.replace("/", "_")


def reference_episode_length(path: Path) -> int:
    """Read and validate the single fixed episode length in a Cube HDF5 file."""
    with h5py.File(path, "r") as dataset:
        lengths = dataset["ep_len"][:]
    unique_lengths = np.unique(lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Reference Cube episodes are not fixed-length: {unique_lengths.tolist()}"
        )
    episode_length = int(unique_lengths[0])
    if episode_length < 2:
        raise ValueError("Reference episode length must include at least one action.")
    return episode_length


class CubeActionSupport:
    """Calibrate an empirical 20-NN support guard on open-gripper Cube actions."""

    def __init__(
        self,
        path: Path,
        seed: int = REFERENCE_ACTION_SEED,
        reference_actions: int = REFERENCE_ACTIONS,
        calibration_actions: int = CALIBRATION_ACTIONS,
        neighbors: int = SUPPORT_NEIGHBORS,
        support_quantile: float = SUPPORT_QUANTILE,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        if min(reference_actions, calibration_actions, neighbors) <= 0:
            raise ValueError("Action support sample sizes must be positive.")
        if not 0.0 < support_quantile < 1.0:
            raise ValueError("support_quantile must be in (0, 1).")

        with h5py.File(path, "r") as dataset:
            actions = dataset["action"][:].astype(np.float32)
            gripper_opening = dataset["proprio_gripper_opening"][:, 0]
        valid = np.isfinite(actions).all(axis=1) & (gripper_opening < 0.05)
        open_actions = actions[valid]
        requested = reference_actions + calibration_actions
        if len(open_actions) < requested:
            raise ValueError(
                f"Need {requested:,} open actions, found {len(open_actions):,}."
            )

        rng = np.random.default_rng(seed)
        selected = rng.choice(len(open_actions), size=requested, replace=False)
        reference = open_actions[selected[:reference_actions]].astype(np.float64)
        calibration = open_actions[selected[reference_actions:]].astype(np.float64)
        self.tree = cKDTree(reference)
        self.neighbors = int(neighbors)
        calibration_distances = self._neighbor_radius(calibration)
        self.calibration_distances = np.sort(calibration_distances)
        self.support_radius = float(
            np.quantile(calibration_distances, support_quantile)
        )
        self.support_quantile = float(support_quantile)
        self.seed = int(seed)
        self.open_action_count = int(len(open_actions))
        self.reference_count = int(reference_actions)
        self.calibration_count = int(calibration_actions)
        self.generated_radii: list[float] = []
        self.generated_percentiles: list[float] = []
        self.shrunk_actions = 0

    def _neighbor_radius(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        distances = self.tree.query(
            actions,
            k=self.neighbors,
            workers=-1 if len(actions) >= 1000 else 1,
        )[0]
        return distances[..., -1]

    def constrain(self, action: np.ndarray) -> np.ndarray:
        """Shrink a candidate toward the supported idle action when necessary."""
        candidate = np.asarray(action, dtype=np.float64).copy()
        if candidate.shape != (5,) or not np.isfinite(candidate).all():
            raise ValueError(f"Cube action must be finite shape (5,), got {candidate}.")
        candidate = np.clip(candidate, -1.0, 1.0)

        for shrink_step in range(41):
            radius = float(self._neighbor_radius(candidate[None])[0])
            if radius <= self.support_radius:
                percentile = float(
                    np.searchsorted(self.calibration_distances, radius, side="right")
                    / len(self.calibration_distances)
                )
                self.generated_radii.append(radius)
                self.generated_percentiles.append(percentile)
                if shrink_step:
                    self.shrunk_actions += 1
                return candidate.astype(np.float32)
            candidate[:4] *= 0.9

        raise RuntimeError(
            "Could not project CubeHop action into calibrated Cube support."
        )

    def metadata(self) -> dict:
        radii = np.asarray(self.generated_radii, dtype=np.float64)
        percentiles = np.asarray(self.generated_percentiles, dtype=np.float64)
        return {
            "method": "raw normalized 5-D Euclidean 20-nearest-neighbor radius",
            "seed": self.seed,
            "open_gripper_action_count": self.open_action_count,
            "reference_action_count": self.reference_count,
            "calibration_action_count": self.calibration_count,
            "neighbors": self.neighbors,
            "support_quantile": self.support_quantile,
            "support_radius": self.support_radius,
            "generated_action_count": int(len(radii)),
            "generated_radius_max": float(radii.max()) if len(radii) else None,
            "generated_support_percentile_max": (
                float(percentiles.max()) if len(percentiles) else None
            ),
            "generated_support_percentile_q99": (
                float(np.quantile(percentiles, 0.99)) if len(percentiles) else None
            ),
            "shrunk_action_count": int(self.shrunk_actions),
        }


class CubeHDF5Writer:
    """Write the root-level schema and compression used by OGBench Cube."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.file: h5py.File | None = None
        self.initialized = False
        self.global_offset = 0
        self.episode_count = 0

    def __enter__(self) -> "CubeHDF5Writer":
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w", libver="latest")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def write_episode(self, episode: dict[str, list]) -> None:
        if self.file is None:
            raise RuntimeError("CubeHDF5Writer used outside its context.")
        missing = set(CUBE_H5_COLUMNS).difference(episode)
        extra = set(episode).difference(CUBE_H5_COLUMNS)
        if missing or extra:
            raise ValueError(
                f"Cube episode schema mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        arrays = {}
        for column in CUBE_H5_COLUMNS:
            if column == "privileged_target_task":
                arrays[column] = np.asarray(episode[column], dtype=object)
            else:
                arrays[column] = np.asarray(
                    episode[column], dtype=COLUMN_DTYPES[column]
                )
        episode_length = len(arrays["action"])
        if any(len(array) != episode_length for array in arrays.values()):
            raise ValueError("Cube episode columns have inconsistent lengths.")

        if not self.initialized:
            self._create_schema(arrays)
            self.initialized = True

        end = self.global_offset + episode_length
        for column, values in arrays.items():
            dataset = self.file[column]
            dataset.resize(end, axis=0)
            dataset[self.global_offset : end] = values

        self.file["ep_len"].resize(self.episode_count + 1, axis=0)
        self.file["ep_len"][self.episode_count] = episode_length
        self.file["ep_offset"].resize(self.episode_count + 1, axis=0)
        self.file["ep_offset"][self.episode_count] = self.global_offset
        self.global_offset = end
        self.episode_count += 1

    def _create_schema(self, arrays: dict[str, np.ndarray]) -> None:
        assert self.file is not None
        for column, values in arrays.items():
            step_shape = values.shape[1:]
            leading_chunk = 100 if column == "pixels" else 1000
            kwargs = {}
            if column == "pixels":
                kwargs.update(
                    hdf5plugin.Blosc(
                        cname="lz4",
                        clevel=5,
                        shuffle=hdf5plugin.Blosc.SHUFFLE,
                    )
                )
            dtype = (
                h5py.string_dtype(encoding="utf-8")
                if column == "privileged_target_task"
                else COLUMN_DTYPES[column]
            )
            self.file.create_dataset(
                column,
                shape=(0, *step_shape),
                maxshape=(None, *step_shape),
                chunks=(leading_chunk, *step_shape),
                dtype=dtype,
                **kwargs,
            )
        self.file.create_dataset(
            "ep_len",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.int32,
        )
        self.file.create_dataset(
            "ep_offset",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.int64,
        )


def _info_value(info_dict: dict, source: str, env_idx: int):
    value = info_dict[source]
    if isinstance(value, torch.Tensor):
        return value[env_idx, 0].detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        selected = value[env_idx, 0]
        return selected.copy() if isinstance(selected, np.ndarray) else selected
    return value[env_idx][0]


def _append_frame(
    buffers: list[defaultdict[str, list]],
    info_dict: dict,
    actions: np.ndarray,
    active_envs: int,
) -> None:
    for env_idx in range(active_envs):
        buffers[env_idx]["action"].append(actions[env_idx].copy())
        for source in SOURCE_COLUMNS:
            buffers[env_idx][flattened_column(source)].append(
                _info_value(info_dict, source, env_idx)
            )


def _balanced_batch_sides(
    accepted_by_side: dict[str, int], side_quota: int, num_envs: int
) -> list[str]:
    remaining = {
        side: max(side_quota - accepted_by_side[side], 0) for side in CUBE_HOP_SIDES
    }
    batch_sides = []
    for env_idx in range(num_envs):
        available = [side for side in CUBE_HOP_SIDES if remaining[side] > 0]
        if available:
            side = max(available, key=lambda name: remaining[name])
            remaining[side] -= 1
        else:
            side = CUBE_HOP_SIDES[env_idx % len(CUBE_HOP_SIDES)]
        batch_sides.append(side)
    return batch_sides


def collect_fixed_length_episodes(
    world,
    writer: CubeHDF5Writer,
    episodes: int,
    episode_length: int,
    seed: int,
) -> dict:
    """Collect fixed episodes, rejecting any that physically touch or move the cube."""
    if episode_length < 2:
        raise ValueError("episode_length must be at least 2.")
    if episodes <= 0 or episodes % len(CUBE_HOP_SIDES) != 0:
        raise ValueError(
            f"episodes must be positive and divisible by {len(CUBE_HOP_SIDES)}."
        )
    action_steps = episode_length - 1
    side_quota = episodes // len(CUBE_HOP_SIDES)
    accepted_by_side = {side: 0 for side in CUBE_HOP_SIDES}
    evaluated_by_side = {side: 0 for side in CUBE_HOP_SIDES}
    rejected_by_side = {side: 0 for side in CUBE_HOP_SIDES}
    accepted = 0
    evaluated = 0
    simulated = 0
    rejected = 0
    rejected_for_contact = 0
    rejected_for_drift = 0
    maximum_cube_drift = 0.0
    maximum_candidate_cube_drift = 0.0
    initial_effector_positions = []
    initial_cube_positions = []
    sampled_gripper_targets = []
    sampled_waypoints = []
    completed_gripper_targets = []
    completed_waypoints = []
    episode_sides = []
    initial_effector_positions_by_side = {side: [] for side in CUBE_HOP_SIDES}
    initial_cube_positions_by_side = {side: [] for side in CUBE_HOP_SIDES}
    sampled_gripper_targets_by_side = {side: [] for side in CUBE_HOP_SIDES}
    sampled_waypoints_by_side = {side: [] for side in CUBE_HOP_SIDES}
    completed_gripper_targets_by_side = {side: [] for side in CUBE_HOP_SIDES}
    completed_waypoints_by_side = {side: [] for side in CUBE_HOP_SIDES}
    maximum_simulated = episodes * MAX_ATTEMPT_MULTIPLIER

    policy = world.policy
    if policy is None or not hasattr(policy, "episode_summary"):
        raise TypeError("CubeHop collection requires a policy with episode_summary().")

    with (
        writer as active_writer,
        tqdm(total=episodes, desc="CubeHop episodes") as progress,
    ):
        while accepted < episodes:
            if simulated >= maximum_simulated:
                raise RuntimeError(
                    f"Accepted only {accepted}/{episodes} episodes after "
                    f"simulating {simulated}; cube-contact rejection is too high."
                )
            batch_seeds = [seed + simulated + idx for idx in range(world.num_envs)]
            batch_sides = _balanced_batch_sides(
                accepted_by_side, side_quota, world.num_envs
            )
            for env, side in zip(world.envs.envs, batch_sides):
                env.unwrapped.set_cube_hop_side(side)
            simulated += world.num_envs
            world.reset(seed=batch_seeds)
            buffers = [defaultdict(list) for _ in range(world.num_envs)]

            for step in range(action_steps):
                actions = world._get_actions()
                _append_frame(buffers, world.infos, actions, world.num_envs)
                (
                    _,
                    world.rewards,
                    world.terminateds,
                    world.truncateds,
                    world.infos,
                ) = world.envs.step(actions)
                if step < action_steps - 1 and (
                    world.terminateds.any() or world.truncateds.any()
                ):
                    raise RuntimeError(
                        f"CubeHop episode ended early at action step {step + 1}."
                    )

            if not np.all(world.truncateds):
                raise RuntimeError("CubeHop batch did not end at its time limit.")
            terminal_actions = np.full((world.num_envs, 5), np.nan, dtype=np.float32)
            _append_frame(buffers, world.infos, terminal_actions, world.num_envs)

            for env_idx, buffer in enumerate(buffers):
                if accepted >= episodes:
                    break
                side = batch_sides[env_idx]
                if accepted_by_side[side] >= side_quota:
                    continue
                evaluated += 1
                evaluated_by_side[side] += 1
                steps = np.asarray(buffer["step_idx"], dtype=np.int64)
                if not np.array_equal(steps, np.arange(episode_length)):
                    raise RuntimeError(
                        f"Invalid CubeHop step alignment: {steps.tolist()}"
                    )
                cube_positions = np.asarray(
                    buffer["privileged_block_0_pos"], dtype=np.float64
                )
                cube_drift = float(
                    np.linalg.norm(cube_positions - cube_positions[0], axis=1).max()
                )
                maximum_candidate_cube_drift = max(
                    maximum_candidate_cube_drift, cube_drift
                )
                base_env = world.envs.envs[env_idx].unwrapped
                touched_cube = bool(base_env._cube_hop_touched_cube)
                excessive_drift = cube_drift > CUBE_DRIFT_TOLERANCE
                if touched_cube or excessive_drift:
                    rejected += 1
                    rejected_by_side[side] += 1
                    rejected_for_contact += int(touched_cube)
                    rejected_for_drift += int(excessive_drift)
                    progress.set_postfix(rejected=rejected)
                    continue

                buffer["ep_idx"] = [np.int32(accepted)] * episode_length
                maximum_cube_drift = max(maximum_cube_drift, cube_drift)
                initial_cube_positions.append(cube_positions[0].copy())
                initial_effector_positions.append(
                    np.asarray(buffer["proprio_effector_pos"][0]).copy()
                )
                episode_summary = policy.episode_summary(env_idx)
                if episode_summary["side"] != side:
                    raise RuntimeError(
                        f"CubeHop side mismatch: {episode_summary['side']} != {side}."
                    )
                sampled_gripper_targets.extend(
                    episode_summary["sampled_gripper_targets"]
                )
                sampled_waypoints.extend(episode_summary["sampled_waypoints"])
                completed_gripper_targets.append(
                    episode_summary["completed_gripper_targets"]
                )
                completed_waypoints.append(episode_summary["completed_waypoints"])
                episode_sides.append(side)
                initial_effector_positions_by_side[side].append(
                    initial_effector_positions[-1]
                )
                initial_cube_positions_by_side[side].append(initial_cube_positions[-1])
                sampled_gripper_targets_by_side[side].extend(
                    episode_summary["sampled_gripper_targets"]
                )
                sampled_waypoints_by_side[side].extend(
                    episode_summary["sampled_waypoints"]
                )
                completed_gripper_targets_by_side[side].append(
                    episode_summary["completed_gripper_targets"]
                )
                completed_waypoints_by_side[side].append(
                    episode_summary["completed_waypoints"]
                )
                active_writer.write_episode(dict(buffer))
                accepted += 1
                accepted_by_side[side] += 1
                progress.update(1)

    if any(count != side_quota for count in accepted_by_side.values()):
        raise RuntimeError(f"Unbalanced CubeHop collection: {accepted_by_side}.")

    return {
        "episodes": accepted,
        "simulated_candidate_episodes": simulated,
        "evaluated_candidate_episodes": evaluated,
        "side_quota": side_quota,
        "accepted_episodes_by_side": accepted_by_side,
        "evaluated_candidate_episodes_by_side": evaluated_by_side,
        "rejected_candidate_episodes_by_side": rejected_by_side,
        "rejected_candidate_episodes": rejected,
        "rejected_for_robot_cube_contact": rejected_for_contact,
        "rejected_for_cube_drift": rejected_for_drift,
        "cube_drift_rejection_threshold_m": CUBE_DRIFT_TOLERANCE,
        "episode_length": episode_length,
        "finite_actions_per_episode": action_steps,
        "frames": accepted * episode_length,
        "maximum_cube_drift": maximum_cube_drift,
        "maximum_candidate_cube_drift": maximum_candidate_cube_drift,
        "initial_effector_positions": np.asarray(initial_effector_positions),
        "initial_cube_positions": np.asarray(initial_cube_positions),
        "sampled_gripper_targets": np.asarray(sampled_gripper_targets).reshape(-1, 3),
        "sampled_waypoints": np.asarray(sampled_waypoints).reshape(-1, 3),
        "completed_gripper_targets": np.asarray(
            completed_gripper_targets, dtype=np.int64
        ),
        "completed_waypoints": np.asarray(completed_waypoints, dtype=np.int64),
        "episode_sides": episode_sides,
        "initial_effector_positions_by_side": {
            side: np.asarray(initial_effector_positions_by_side[side]).reshape(-1, 3)
            for side in CUBE_HOP_SIDES
        },
        "initial_cube_positions_by_side": {
            side: np.asarray(initial_cube_positions_by_side[side]).reshape(-1, 3)
            for side in CUBE_HOP_SIDES
        },
        "sampled_gripper_targets_by_side": {
            side: np.asarray(sampled_gripper_targets_by_side[side]).reshape(-1, 3)
            for side in CUBE_HOP_SIDES
        },
        "sampled_waypoints_by_side": {
            side: np.asarray(sampled_waypoints_by_side[side]).reshape(-1, 3)
            for side in CUBE_HOP_SIDES
        },
        "completed_gripper_targets_by_side": {
            side: np.asarray(completed_gripper_targets_by_side[side], dtype=np.int64)
            for side in CUBE_HOP_SIDES
        },
        "completed_waypoints_by_side": {
            side: np.asarray(completed_waypoints_by_side[side], dtype=np.int64)
            for side in CUBE_HOP_SIDES
        },
    }


__all__ = [
    "CALIBRATION_ACTIONS",
    "CUBE_DRIFT_TOLERANCE",
    "CUBE_H5_COLUMNS",
    "CubeActionSupport",
    "CubeHDF5Writer",
    "REFERENCE_ACTIONS",
    "REFERENCE_ACTION_SEED",
    "SOURCE_COLUMNS",
    "SUPPORT_NEIGHBORS",
    "SUPPORT_QUANTILE",
    "_balanced_batch_sides",
    "collect_fixed_length_episodes",
    "reference_episode_length",
]
