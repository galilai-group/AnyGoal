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
        index_args=(
            --index https://test.pypi.org/simple
            --default-index https://pypi.org/simple
            --index-strategy unsafe-best-match
        )
        ;;
    pypi)
        index_args=()
        ;;
    *)
        echo "index must be testpypi or pypi" >&2
        exit 2
        ;;
esac

uv pip install --python "${work_dir}/venv/bin/python" \
    "${index_args[@]}" "anygoal==${version}"
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
