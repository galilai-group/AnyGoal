#!/usr/bin/env bash
set -euo pipefail
export UV_LINK_MODE=copy

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON:-$(command -v python)}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/anygoal-dist-smoke.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

build_env="${work_dir}/build-env"
wheel_env="${work_dir}/wheel-env"
sdist_env="${work_dir}/sdist-env"
artifact_dir="${work_dir}/dist"
user_dir="${work_dir}/user"
mkdir -p "${artifact_dir}" "${user_dir}"

uv venv --python "${python_command}" "${build_env}"
uv pip install --python "${build_env}/bin/python" build twine

cd "${repository_root}"
"${build_env}/bin/python" -m build --outdir "${artifact_dir}"
"${build_env}/bin/python" -m twine check "${artifact_dir}"/*

mapfile -t wheels < <(find "${artifact_dir}" -maxdepth 1 -type f -name '*.whl')
mapfile -t sdists < <(find "${artifact_dir}" -maxdepth 1 -type f -name '*.tar.gz')
test "${#wheels[@]}" -eq 1
test "${#sdists[@]}" -eq 1
wheel="${wheels[0]}"
sdist="${sdists[0]}"

REPOSITORY_ROOT="${repository_root}" WHEEL="${wheel}" SDIST="${sdist}" \
    "${build_env}/bin/python" - <<'PY'
import os
import tarfile
import zipfile
from pathlib import Path

root = Path(os.environ["REPOSITORY_ROOT"])
required = {
    path.relative_to(root).as_posix()
    for path in (root / "anygoal").rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix in {".py", ".yaml", ".json"}
}

with zipfile.ZipFile(os.environ["WHEEL"]) as archive:
    wheel_names = set(archive.namelist())
    wheel_contents = {
        name: archive.read(name)
        for name in wheel_names
        if name.startswith("anygoal/")
        and Path(name).suffix in {".py", ".yaml", ".json"}
    }
    metadata_name, = (
        name for name in wheel_names if name.endswith(".dist-info/METADATA")
    )
    wheel_metadata = archive.read(metadata_name).decode()
with tarfile.open(os.environ["SDIST"], "r:gz") as archive:
    sdist_contents = {
        "/".join(Path(member.name).parts[1:]): archive.extractfile(member).read()
        for member in archive.getmembers()
        if member.isfile()
        and "/".join(Path(member.name).parts[1:]).startswith("anygoal/")
        and Path(member.name).suffix in {".py", ".yaml", ".json"}
    }
    sdist_files = {
        "/".join(Path(name).parts[1:]) for name in archive.getnames()
    }

for label, files in (("wheel", wheel_contents), ("sdist", sdist_contents)):
    missing = sorted(required - set(files))
    if missing:
        raise SystemExit(f"{label} is missing package files: {missing}")

if set(wheel_contents) != set(sdist_contents):
    raise SystemExit("wheel and sdist contain different package file sets")
different = sorted(
    name for name in wheel_contents if wheel_contents[name] != sdist_contents[name]
)
if different:
    raise SystemExit(f"wheel and sdist package files differ: {different}")

license_files = ("LICENSE", "docs/NOTICE", "docs/THIRD_PARTY_NOTICES.md")
for required_sdist_file in ("pyproject.toml", "README.md", *license_files):
    if required_sdist_file not in sdist_files:
        raise SystemExit(f"sdist is missing {required_sdist_file}")
for license_file in license_files:
    expected_suffix = f".dist-info/licenses/{license_file}"
    if not any(name.endswith(expected_suffix) for name in wheel_names):
        raise SystemExit(f"wheel is missing {license_file}")
if "License-Expression: Apache-2.0" not in wheel_metadata:
    raise SystemExit("wheel metadata is missing Apache-2.0 license expression")

print(f"archive contents: PASS ({len(required)} package files checked)")
PY

smoke_install() {
    local distribution="$1"
    local environment="$2"
    local label="$3"

    uv venv --python "${python_command}" "${environment}"
    uv pip install --python "${environment}/bin/python" "${distribution}"
    uv pip check --python "${environment}/bin/python"

    cd "${user_dir}"
    REPOSITORY_ROOT="${repository_root}" \
        "${environment}/bin/python" - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

import anygoal
from anygoal.config import load_config
from anygoal.datasets.validation import validate_hdf5
from anygoal.resources import resource_path

package_path = Path(anygoal.__file__).resolve()
repository_root = Path(os.environ["REPOSITORY_ROOT"]).resolve()
if anygoal.__version__ != version("anygoal"):
    raise SystemExit(
        f"version mismatch: API={anygoal.__version__}, metadata={version('anygoal')}"
    )
if package_path.is_relative_to(repository_root):
    raise SystemExit(f"anygoal imported from the source tree: {package_path}")
if "site-packages" not in package_path.parts:
    raise SystemExit(f"anygoal did not import from site-packages: {package_path}")

for name in (
    "collect",
    "train",
    "plan",
    "evaluate_rollout",
    "get_registry",
):
    if not callable(getattr(anygoal, name, None)):
        raise SystemExit(f"public API is missing callable anygoal.{name}")

registry = anygoal.get_registry()
assert registry.dataset("cube_diagonal").id == "cubediagonal"
assert registry.model("adaln").id == "toy-adaln"
assert load_config()["wandb"]["mode"]
assert resource_path("plan", "driftworld", "pusht_lewm.yaml").is_file()

dataset = Path("tiny.h5")
with h5py.File(dataset, "w") as handle:
    handle.create_dataset("action", data=np.zeros((3, 2), dtype=np.float32))
    handle.create_dataset("pixels", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
    handle.create_dataset("ep_len", data=np.array([3], dtype=np.int32))
report = validate_hdf5(dataset, dataset_id="pusht")
assert report.valid and report.rows == 3 and report.episodes == 1
video = Path("tiny.mp4")
imageio.mimsave(video, np.zeros((2, 16, 16, 3), dtype=np.uint8), fps=1)
assert video.stat().st_size > 0
print(f"Python API: PASS ({package_path})")
PY

    test -x "${environment}/bin/anygoal"
    "${environment}/bin/anygoal" --help >/dev/null
    installed_version="$(
        "${environment}/bin/python" -c \
            'from importlib.metadata import version; print(version("anygoal"))'
    )"
    test "$("${environment}/bin/anygoal" --version)" = \
        "anygoal ${installed_version}"
    "${environment}/bin/anygoal" list datasets | grep -F "pusht"
    "${environment}/bin/anygoal" list models | grep -F "driftworld"
    "${environment}/bin/anygoal" config show >/dev/null
    "${environment}/bin/anygoal" data validate tiny.h5 --dataset pusht
    "${environment}/bin/anygoal" eval rollout \
        --model driftworld --dataset pusht --dry-run | grep -F -- "-m anygoal.runners.rollout"
    "${environment}/bin/anygoal" eval planning \
        --model driftworld --dataset pusht --dry-run | grep -F "pusht_lewm"
    echo "${label} installation, dependencies, API, and CLI: PASS"
}

smoke_install "${wheel}" "${wheel_env}" "wheel"
rm -rf "${wheel_env}"
smoke_install "${sdist}" "${sdist_env}" "sdist"

echo "distribution smoke test: PASS"
