#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Fixed-response forward KL; the input must contain a response column.
exec bash "$root/scripts/train.sh" kd "$@"
