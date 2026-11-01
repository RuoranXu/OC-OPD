#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Fixed-response cross-entropy, without a teacher or student rollout engine.
# Each row needs prompt and a nonempty response; TRAIN_FILE overrides GSM8K.
export PROMPT_MODE="${PROMPT_MODE:-chat}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
exec bash "$root/recipe/sft/run.sh" "$@"
