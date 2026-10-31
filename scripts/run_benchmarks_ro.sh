#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TYPES=RO
exec "$root/scripts/run_benchmarks.sh" "$@"
