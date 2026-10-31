#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# On-policy forward KL on the same generated contexts as the OPD recipe.
# TRAIN_FILE accepts prepared JSONL/Parquet; RUN_NAME separates checkpoints.
export PROMPT_MODE="${PROMPT_MODE:-chat}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
exec bash "$root/recipe/fkl/run.sh" "$@"
