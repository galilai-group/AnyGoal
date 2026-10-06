"""Minimal headless render test for DMC Manipulator insert_peg."""

from pathlib import Path

import numpy as np
from PIL import Image
from dm_control import suite


env = suite.load(
    domain_name="manipulator",
    task_name="insert_peg",
)

timestep = env.reset()

image = env.physics.render(
    height=224,
    width=224,
    camera_id=0,
)

assert timestep.first()
assert image.shape == (224, 224, 3)
assert image.dtype == np.uint8
assert np.ptp(image) > 0

output = Path("reports/dm_control_manipulator_insert_peg_render_test.png")
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(image).save(output)
print(f"reset observation keys: {list(timestep.observation)}")
print(f"render shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
print(f"saved: {output.resolve()}")
