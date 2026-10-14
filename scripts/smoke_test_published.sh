#!/usr/bin/env bash
set -euo pipefail
export UV_LINK_MODE=copy

index="${1:?usage: smoke_test_published.sh <testpypi|pypi> <version>}"
version="${2:?usage: smoke_test_published.sh <testpypi|pypi> <version>}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/anygoal-published-smoke.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

uv venv --python "${PYTHON:-python}" "${work_dir}/venv"
case "${index}" in
    testpypi)
        uv pip install --python "${work_dir}/venv/bin/python" \
            --index https://test.pypi.org/simple --no-deps "anygoal==${version}"
        mapfile -t requirements < <(
            "${work_dir}/venv/bin/python" -c \
                'from importlib.metadata import requires; print(*requires("anygoal") or (), sep="\n")'
        )
        uv pip install --python "${work_dir}/venv/bin/python" "${requirements[@]}"
        ;;
    pypi)
        uv pip install --python "${work_dir}/venv/bin/python" "anygoal==${version}"
        ;;
    *)
        echo "index must be testpypi or pypi" >&2
        exit 2
        ;;
esac
uv pip check --python "${work_dir}/venv/bin/python"

cd "${work_dir}"
EXPECTED_VERSION="${version}" "${work_dir}/venv/bin/python" - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path

import anygoal

expected = os.environ["EXPECTED_VERSION"]
package_path = Path(anygoal.__file__).resolve()
assert anygoal.__version__ == expected
assert version("anygoal") == expected
assert "site-packages" in package_path.parts
assert anygoal.get_registry().dataset("pusht").filename == "pusht_expert_train.h5"
print(f"published API: PASS ({package_path})")
PY

test "$("${work_dir}/venv/bin/anygoal" --version)" = "anygoal ${version}"
"${work_dir}/venv/bin/anygoal" --help >/dev/null
"${work_dir}/venv/bin/anygoal" list datasets | grep -F "pusht"
"${work_dir}/venv/bin/anygoal" list models | grep -F "driftworld"
echo "${index} installation smoke test: PASS"
