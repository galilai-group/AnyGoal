"""Render selected RL Unplugged Manipulator state sequences through dm_control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dm_control import suite


DEFAULT_DATASET = Path.home() / ".stable_worldmodel/datasets/rlu_manipulator_insert_peg_fragments.h5"
DEFAULT_SELECTION = Path("reports/rlu_manipulator_insert_peg_random_10_sequences_2x5.json")
ARM_JOINTS = ["arm_root", "arm_shoulder", "arm_elbow", "arm_wrist", "finger", "fingertip", "thumb", "thumbtip"]
OBJECT_JOINTS = ["peg_x", "peg_z", "peg_y"]


def planar_angle(quaternion_wy: np.ndarray) -> float:
    return float(2 * np.arctan2(quaternion_wy[1], quaternion_wy[0]))


def restore_state(physics, state: np.ndarray) -> None:
    arm_pose = state[:16].reshape(8, 2)
    physics.named.data.qpos[ARM_JOINTS] = np.arctan2(arm_pose[:, 0], arm_pose[:, 1])
    physics.named.data.qvel[ARM_JOINTS] = state[16:24]
    physics.named.data.qpos[OBJECT_JOINTS] = [state[33], state[34], planar_angle(state[35:37])]
    physics.named.data.qvel[OBJECT_JOINTS] = state[37:40]
    target_xz = state[40:42]
    target_wy = state[42:44]
    for body in ("target_peg", "slot"):
        physics.named.model.body_pos[body, ["x", "z"]] = target_xz
        physics.named.model.body_quat[body, ["qw", "qy"]] = target_wy
    physics.forward()


def observable_state(env) -> np.ndarray:
    observation = env.task.get_observation(env.physics)
    return np.concatenate([np.asarray(value).reshape(-1) for value in observation.values()])


def state_restoration_error(reconstructed: np.ndarray, recorded: np.ndarray) -> float:
    reconstructed = reconstructed.copy()
    for quaternion_slice in (slice(31, 33), slice(35, 37), slice(42, 44)):
        direct = np.linalg.norm(reconstructed[quaternion_slice] - recorded[quaternion_slice])
        flipped = np.linalg.norm(-reconstructed[quaternion_slice] - recorded[quaternion_slice])
        if flipped < direct:
            reconstructed[quaternion_slice] *= -1
    validated_fields = np.r_[0:24, 29:44]
    return float(np.max(np.abs(reconstructed[validated_fields] - recorded[validated_fields])))


def add_label(frame: np.ndarray, sequence_index: int, step: int) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    label = f"sequence {sequence_index}  step {step}/999"
    draw.rectangle((0, 0, 223, 16), fill=(255, 255, 255))
    draw.text((4, 3), label, fill=(0, 0, 0), font=ImageFont.load_default())
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=Path("reports/rlu_manipulator_insert_peg_rgb_random_10_sequences_2x5.gif"))
    parser.add_argument("--step-stride", type=int, default=10)
    parser.add_argument("--duration-ms", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())
    selected = selection["selected_sequence_indices"]
    if len(selected) != 10 or len(set(selected)) != 10:
        raise ValueError("Selection must contain 10 unique sequence indices")
    sampled_steps = list(range(0, 1000, args.step_stride))
    envs = [suite.load(domain_name="manipulator", task_name="insert_peg") for _ in selected]
    for env in envs:
        env.reset()
    with h5py.File(args.dataset.expanduser(), "r") as handle:
        lengths = handle["ep_len"][:]
        offsets = handle["ep_offset"][:]
        states = []
        for index in selected:
            if lengths[index] != 1000:
                raise ValueError(f"Expected sequence {index} to have 1,000 transitions")
            start = int(offsets[index])
            states.append(handle["observation"][start : start + 1000])

    # Validate the invertible, render-relevant observation fields on every frame.
    maximum_error = 0.0
    gif_frames = []
    for step in sampled_steps:
        canvas = Image.new("RGB", (5 * 224, 2 * 224))
        for panel, (env, sequence_index, sequence_states) in enumerate(zip(envs, selected, states)):
            restore_state(env.physics, sequence_states[step])
            reconstructed = observable_state(env)
            error = state_restoration_error(reconstructed, sequence_states[step])
            maximum_error = max(maximum_error, error)
            frame = env.physics.render(height=224, width=224, camera_id=0)
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise ValueError(f"Unexpected rendered frame: {frame.shape}, {frame.dtype}")
            row, column = divmod(panel, 5)
            canvas.paste(add_label(frame, sequence_index, step), (column * 224, row * 224))
        gif_frames.append(canvas)
    if maximum_error > 2e-5:
        raise ValueError(f"State restoration error is too large: {maximum_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(args.output, save_all=True, append_images=gif_frames[1:], duration=args.duration_ms, loop=0, optimize=True)
    metadata = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "selection": str(args.selection.resolve()),
        "selected_sequence_indices": selected,
        "task": "manipulator/insert_peg",
        "grid": [2, 5],
        "panel_resolution": [224, 224],
        "frames": len(gif_frames),
        "step_stride": args.step_stride,
        "duration_ms": args.duration_ms,
        "maximum_reconstruction_error": maximum_error,
        "validated_state_slices": ["arm_pos[0:16]", "arm_vel[16:24]", "hand_pos[29:33]", "object_pos[33:37]", "object_vel[37:40]", "target_pos[40:44]"],
        "touch_note": "Touch sensors are dynamic sensor outputs and are not used to reconstruct a static rendered frame.",
    }
    metadata_path = args.output.with_suffix(".json")
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    with metadata_path.open("x") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
