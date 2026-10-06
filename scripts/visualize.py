"""Compatibility wrapper for ``anygoal visualize``."""

from __future__ import annotations

import sys

from anygoal.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["visualize", *sys.argv[1:]]))
