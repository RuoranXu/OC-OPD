#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# On-policy reverse KL with the opd Hydra config.
exec bash "$root/scripts/train.sh" opd "$@"
