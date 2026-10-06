#!/bin/bash
set -euo pipefail

if [[ -n "${ANYGOAL_PYTHON_ENV:-}" ]]; then
  source "$ANYGOAL_PYTHON_ENV/bin/activate"
fi

exec "$@"
