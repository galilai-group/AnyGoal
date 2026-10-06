"""Minimal DeepMind Control Suite Manipulator insert_peg load/render check."""

from pathlib import Path
import os

os.environ.setdefault("MUJOCO_GL", "egl")
from dm_control import suite


def main() -> None:
    env = suite.load(domain_name="manipulator", task_name="insert_peg")
    timestep = env.reset()
    image = env.physics.render(height=224, width=224, camera_id=0)
    if image.shape != (224, 224, 3):
        raise ValueError(f"Unexpected RGB render shape: {image.shape}")
    output = Path("reports/dmcontrol_manipulator_insert_peg_reset.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(image).save(output)
    print(f"reset observation keys: {list(timestep.observation)}")
    print(f"render shape: {image.shape}, dtype: {image.dtype}")
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
