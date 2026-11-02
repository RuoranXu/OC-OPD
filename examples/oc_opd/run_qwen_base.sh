#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Prepare GSM8K with --prompt-mode base, or set TRAIN_FILE to plain-string prompts.
export PROMPT_MODE=base
export ENABLE_THINKING=false
export STUDENT_PATH="${STUDENT_PATH:-${MODELS:-Qwen}/Qwen3-1.7B-Base}"
export TEACHER_PATH="${TEACHER_PATH:-${MODELS:-Qwen}/Qwen3-14B-Base}"
exec bash "$root/recipe/oc_opd/run.sh" "$@"
