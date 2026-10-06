"""Compatibility wrapper for ``anygoal train``."""

from __future__ import annotations

import sys

from anygoal.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
