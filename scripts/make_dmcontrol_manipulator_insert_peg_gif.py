"""Roll out and render ten DeepMind Control Suite Manipulator episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from dm_control import suite
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/dmcontrol_manipulator_insert_peg_random_10_episodes_2x5.gif"))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--duration-ms", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = np.random.default_rng(args.seed)
    env = suite.load(domain_name="manipulator", task_name="insert_peg")
    episodes: list[list[np.ndarray]] = []
    lengths: list[int] = []
    for episode in range(10):
        env.reset()
        frames = []
        step = 0
        while True:
            if step % 10 == 0:
                frames.append(env.physics.render(height=224, width=224, camera_id=0))
            timestep = env.step(rng.uniform(env.action_spec().minimum, env.action_spec().maximum).astype(np.float32))
            step += 1
            if timestep.last():
                break
        episodes.append(frames)
        lengths.append(len(frames))
    env.close()
    max_length = max(lengths)
    canvas_frames = []
    for step in range(max_length):
        canvas = Image.new("RGB", (224 * 5, 224 * 2), "white")
        for index, frames in enumerate(episodes):
            frame = Image.fromarray(frames[min(step, len(frames) - 1)])
            canvas.paste(frame, ((index % 5) * 224, (index // 5) * 224))
        canvas_frames.append(canvas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas_frames[0].save(args.output, save_all=True, append_images=canvas_frames[1:], duration=args.duration_ms, loop=0, optimize=True)
    metadata = {"domain_name": "manipulator", "task_name": "insert_peg", "seed": args.seed, "episode_count": 10, "episode_lengths": lengths, "frame_stride": 10, "duration_ms": args.duration_ms, "render_shape": [224, 224, 3]}
    metadata_path = args.output.with_suffix(".json")
    with metadata_path.open("x") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
