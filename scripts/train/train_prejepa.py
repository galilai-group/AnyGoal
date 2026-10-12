"""Compatibility entry point for the installed PreJEPA trainer."""

import logging

import torch

from anygoal.runners.train.train_prejepa import main


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    torch.set_float32_matmul_precision("high")
    main()
