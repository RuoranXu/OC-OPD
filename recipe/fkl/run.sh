#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# On-policy forward KL with the fkl Hydra config.
exec bash "$root/scripts/train.sh" fkl "$@"
