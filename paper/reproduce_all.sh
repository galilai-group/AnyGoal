#!/bin/bash
set -euo pipefail

exec anygoal reproduce "$(dirname "$0")/manifest.yaml" "$@"
