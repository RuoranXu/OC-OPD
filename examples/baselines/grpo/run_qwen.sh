#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Native veRL GRPO: prompt messages + data_source + reward_model.ground_truth.
# Default input: DATA_ROOT/gsm8k/chat/train.parquet from examples/data_preprocess.
# Override STUDENT_PATH, TRAIN_FILE, RUN_ROOT/RUN_NAME or append Hydra arguments.
export PROMPT_MODE="${PROMPT_MODE:-chat}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
exec bash "$root/recipe/grpo/run.sh" "$@"
