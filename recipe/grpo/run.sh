#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# veRL GRPO config; MODEL/DATA/RUN paths are handled by the shared launcher.
exec bash "$root/scripts/train.sh" grpo "$@"
