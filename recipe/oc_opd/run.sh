#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# OC-OPD config; all algorithm defaults remain in oc_opd/trainer/config/.
exec bash "$root/scripts/train.sh" oc_opd "$@"
