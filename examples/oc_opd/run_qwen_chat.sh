#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Chat examples use role/content messages, rendered once by the training dataset.
export PROMPT_MODE=chat
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export STUDENT_PATH="${STUDENT_PATH:-${MODELS:-Qwen}/Qwen3-4B}"
export TEACHER_PATH="${TEACHER_PATH:-${MODELS:-Qwen}/Qwen3-30B-A3B-Instruct-2507}"
exec bash "$root/recipe/oc_opd/run.sh" "$@"
