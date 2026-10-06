"""Test-only defaults for deterministic headless rendering."""

import os


os.environ.setdefault("MUJOCO_GL", "osmesa")
