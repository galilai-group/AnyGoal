"""Create a tiny random SD-style VAE used only by the IRASim smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusers.models import AutoencoderKL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke VAE: {output}")
    output.mkdir(parents=True)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=64,
    )
    vae.save_pretrained(output / "vae")
    print(f"wrote smoke VAE: {output / 'vae'}")


if __name__ == "__main__":
    main()
