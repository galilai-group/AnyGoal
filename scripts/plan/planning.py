"""Compatibility entry point for the installed common planning runner."""

from anygoal.runners.plan.planning import normalize_cli_aliases, run


if __name__ == "__main__":
    normalize_cli_aliases()
    run()
