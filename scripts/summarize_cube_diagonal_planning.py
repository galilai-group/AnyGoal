"""Summarize completed Cube Diagonal planning runs."""

from pathlib import Path
import sys

import summarize_planning_results as summary


ROOT = Path("outputs/planning_cube_diagonal_20260910")
def main() -> None:
    sys.argv = [
        sys.argv[0],
        "--root",
        str(ROOT),
        "--flat-dataset",
        "cube_diagonal_test",
    ]
    summary.main()


if __name__ == "__main__":
    main()
