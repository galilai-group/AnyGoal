"""Dataset visualizations that depend only on the logical ``pixels`` field."""

from __future__ import annotations

from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np


def render_dataset_gif(
    dataset: str | Path,
    output: str | Path,
    *,
    start: int = 0,
    frames: int = 100,
    fps: float = 10.0,
    overwrite: bool = False,
) -> Path:
    """Render a contiguous slice from any dataset exposing ``pixels[T,H,W,C]``."""

    source = Path(dataset).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if start < 0 or frames <= 0 or fps <= 0:
        raise ValueError("start must be non-negative; frames and fps must be positive.")
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError("dataset-gif requires `pip install anygoal[analysis]`.") from error

    with h5py.File(source, "r") as handle:
        if "pixels" not in handle:
            raise ValueError("Dataset has no top-level 'pixels' field.")
        pixels = handle["pixels"]
        if pixels.ndim != 4 or pixels.shape[-1] not in (3, 4):
            raise ValueError(f"Expected pixels[T,H,W,C], got {pixels.shape}.")
        stop = min(start + frames, len(pixels))
        if start >= stop:
            raise ValueError(f"Frame range starts beyond dataset length {len(pixels)}.")
        selected = np.asarray(pixels[start:stop])
    destination.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(destination, selected, duration=1000.0 / fps, loop=0)
    return destination
