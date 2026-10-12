"""Compatibility entry point for the installed toy-model planner."""

import sys

from anygoal.runners.plan.planning_toy import run


if __name__ == "__main__":
    if "--save-video" in sys.argv:
        sys.argv.remove("--save-video")
        sys.argv.append("output.save_video=true")
    run()
