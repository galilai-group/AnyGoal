"""Compatibility entry point for ``anygoal train``."""

import sys

from anygoal.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
