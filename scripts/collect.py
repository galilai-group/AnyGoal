"""Compatibility wrapper for ``anygoal data collect``."""

from __future__ import annotations

import sys

from anygoal.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["data", "collect", *sys.argv[1:]]))
