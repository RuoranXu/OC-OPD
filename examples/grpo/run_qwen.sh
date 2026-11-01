#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Compatibility entry; the canonical baseline example lives alongside OPD/KD.
exec bash "$root/examples/baselines/grpo/run_qwen.sh" "$@"
