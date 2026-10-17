"""Keep published runtime pins aligned with the research baseline."""
from pathlib import Path
import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def test_runtime_pins_match_requirements():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    baseline = {}
    for line in (root / "requirements.txt").read_text().splitlines():
        if re.match(r"^[A-Za-z0-9_.-]+==", line):
            requirement = Requirement(line)
            baseline[canonicalize_name(requirement.name)] = next(
                iter(requirement.specifier)
            ).version
    groups = [project["dependencies"]]
    groups.extend(project["optional-dependencies"][extra] for extra in ("eval", "all"))
    for group in groups:
        for raw in group:
            requirement = Requirement(raw)
            name = canonicalize_name(requirement.name)
            if name == "anygoal":
                assert str(requirement.specifier) == f"=={project['version']}"
                continue
            pins = list(requirement.specifier)
            assert len(pins) == 1 and pins[0].operator == "==", raw
            assert requirement.specifier.contains(baseline[name]), raw


def test_eval_pins_cover_known_runtime_incompatibilities():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    names = {
        canonicalize_name(Requirement(raw).name)
        for raw in project["optional-dependencies"]["eval"]
    }
    assert {"datasets", "fsspec", "dill", "multiprocess", "wandb"} <= names
