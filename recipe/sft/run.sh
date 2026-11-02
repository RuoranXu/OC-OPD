#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Fixed-response cross-entropy; no teacher model is allocated.
exec bash "$root/scripts/train.sh" sft "$@"
