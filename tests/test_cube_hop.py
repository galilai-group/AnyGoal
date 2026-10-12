from __future__ import annotations

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from anygoal.datasets.cube.support import (
    CUBE_H5_COLUMNS,
    COLUMN_DTYPES,
    CubeActionSupport,
    CubeHDF5Writer,
    _balanced_batch_sides,
)
from anygoal.envs.cube_hop.env import (
    CUBE_HOP_SIDES,
    GRIPPER_TARGET_HIGH,
    GRIPPER_TARGET_LOW,
    LEFT_CUBE_POSITION_HIGH,
    LEFT_CUBE_POSITION_LOW,
    LEFT_GRIPPER_TARGET_HIGH,
    LEFT_GRIPPER_TARGET_LOW,
    LEFT_WAYPOINT_HIGH,
    LEFT_WAYPOINT_LOW,
    MAX_TRANSLATION_ACTION_NORM,
    MAX_YAW_ACTION,
    SIDE_SAMPLING_BOUNDS,
    WAYPOINT_HIGH,
    WAYPOINT_LOW,
    cube_hop_success_criterion,
    wrapped_angle_error,
)
from anygoal.envs.cube_hop.policy import CubeHopPolicy


class IdentitySupport:
    def constrain(self, action):
        return np.asarray(action, dtype=np.float32)


def test_corrected_destination_sampling_bounds():
    assert np.array_equal(GRIPPER_TARGET_LOW, np.array((0.3, -0.15, 0.02)))
    assert np.array_equal(GRIPPER_TARGET_HIGH, np.array((0.55, 0.05, 0.02)))
    assert np.array_equal(WAYPOINT_LOW, np.array((0.3, -0.3, 0.2)))
    assert np.array_equal(WAYPOINT_HIGH, np.array((0.55, -0.15, 0.35)))
    assert MAX_TRANSLATION_ACTION_NORM == 0.8
    assert MAX_YAW_ACTION == 2.0 / 3.0


def test_left_destination_sampling_bounds_and_mirroring():
    assert CUBE_HOP_SIDES == ("right", "left")
    assert np.array_equal(LEFT_CUBE_POSITION_LOW, np.array((0.3, -0.3, 0.02)))
    assert np.array_equal(LEFT_CUBE_POSITION_HIGH, np.array((0.55, -0.1, 0.02)))
    assert np.array_equal(LEFT_GRIPPER_TARGET_LOW, np.array((0.3, -0.05, 0.02)))
    assert np.array_equal(LEFT_GRIPPER_TARGET_HIGH, np.array((0.55, 0.15, 0.02)))
    assert np.array_equal(LEFT_WAYPOINT_LOW, np.array((0.3, 0.15, 0.2)))
    assert np.array_equal(LEFT_WAYPOINT_HIGH, np.array((0.55, 0.3, 0.35)))
    cube_low, cube_high = SIDE_SAMPLING_BOUNDS["left"]["cube"]
    assert np.array_equal(cube_low, LEFT_CUBE_POSITION_LOW)
    assert np.array_equal(cube_high, LEFT_CUBE_POSITION_HIGH)


def test_balanced_batch_side_assignment_preserves_equal_accepted_quotas():
    first_batch = _balanced_batch_sides({"right": 0, "left": 0}, 4, 8)
    assert first_batch.count("right") == 4
    assert first_batch.count("left") == 4

    catchup_batch = _balanced_batch_sides({"right": 4, "left": 2}, 4, 4)
    assert catchup_batch[:2] == ["left", "left"]


def test_translation_action_uses_conservative_joint_norm():
    policy = CubeHopPolicy(action_support=IdentitySupport())
    action = policy._translation_action(np.array((1.0, 1.0, 1.0)), gripper_opening=0.12)
    assert np.isclose(np.linalg.norm(action[:3]), MAX_TRANSLATION_ACTION_NORM)
    assert action[3] == 0.0
    assert np.isclose(action[4], -0.12)


def test_clockwise_wrapped_yaw_error():
    assert np.isclose(wrapped_angle_error(-0.2, 0.0), -0.2)
    assert np.isclose(wrapped_angle_error(np.pi - 0.1, -np.pi + 0.1), -0.2)


def test_cube_hop_criterion_3_requires_all_three_errors_within_tolerance():
    tolerances = {
        "ee_position_tolerance": 0.04,
        "ee_yaw_tolerance": 0.10,
        "cube_position_tolerance": 0.04,
    }
    assert not cube_hop_success_criterion(
        ee_position_error=0.03,
        ee_yaw_error=0.20,
        cube_position_error=0.03,
        **tolerances,
    )
    assert not cube_hop_success_criterion(
        ee_position_error=0.03,
        ee_yaw_error=0.05,
        cube_position_error=0.05,
        **tolerances,
    )
    assert cube_hop_success_criterion(
        ee_position_error=0.03,
        ee_yaw_error=0.05,
        cube_position_error=0.03,
        **tolerances,
    )


def test_action_support_projects_to_calibrated_neighborhood(tmp_path):
    rng = np.random.default_rng(7)
    actions = rng.normal(0.0, 0.08, size=(500, 5)).astype(np.float32)
    openings = np.zeros((500, 1), dtype=np.float64)
    reference = tmp_path / "reference.h5"
    with h5py.File(reference, "w") as dataset:
        dataset.create_dataset("action", data=actions)
        dataset.create_dataset("proprio_gripper_opening", data=openings)

    support = CubeActionSupport(
        reference,
        seed=3,
        reference_actions=300,
        calibration_actions=100,
        neighbors=10,
        support_quantile=0.95,
    )
    candidate = np.ones(5, dtype=np.float64)
    candidate[4] = 0.0
    projected = support.constrain(candidate)
    assert np.all(np.abs(projected) <= 1.0)
    assert support.metadata()["generated_support_percentile_max"] <= 0.95
    assert support.shrunk_actions == 1


def test_cube_writer_uses_original_root_schema(tmp_path):
    step_shapes = {
        "action": (5,),
        "control": (7,),
        "observation": (28,),
        "pixels": (8, 8, 3),
        "prev_qpos": (21,),
        "prev_qvel": (20,),
        "privileged_block_0_pos": (3,),
        "privileged_block_0_quat": (4,),
        "privileged_block_0_yaw": (1,),
        "privileged_target_block_pos": (3,),
        "privileged_target_block_yaw": (1,),
        "proprio_effector_pos": (3,),
        "proprio_effector_yaw": (1,),
        "proprio_gripper_contact": (1,),
        "proprio_gripper_opening": (1,),
        "proprio_gripper_vel": (1,),
        "proprio_joint_pos": (6,),
        "proprio_joint_vel": (6,),
        "qpos": (21,),
        "qvel": (20,),
        "target": (28,),
        "time": (1,),
    }
    episode = {}
    for column in CUBE_H5_COLUMNS:
        if column == "privileged_target_task":
            episode[column] = ["cube_hop", "cube_hop"]
        else:
            shape = step_shapes.get(column, ())
            episode[column] = np.zeros((2, *shape), dtype=COLUMN_DTYPES[column])

    output = tmp_path / "cube.h5"
    with CubeHDF5Writer(output) as writer:
        writer.write_episode(episode)

    with h5py.File(output, "r") as dataset:
        assert set(dataset) == set(CUBE_H5_COLUMNS) | {"ep_len", "ep_offset"}
        assert dataset["pixels"].chunks == (100, 8, 8, 3)
        assert "32001" in dataset["pixels"]._filters
        assert dataset["ep_len"][:].tolist() == [2]
        assert dataset["ep_offset"][:].tolist() == [0]
