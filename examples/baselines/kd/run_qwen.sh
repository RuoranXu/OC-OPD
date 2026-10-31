#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Fixed-response KD: forward KL with an online teacher, without student rollout.
# Each row needs prompt and a nonempty response. Default GSM8K responses are
# official reference solutions; supply TRAIN_FILE for a teacher-generated corpus.
export PROMPT_MODE="${PROMPT_MODE:-chat}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
exec bash "$root/recipe/kd/run.sh" "$@"
