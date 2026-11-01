#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# On-policy reverse KL: student rollouts and online frozen-teacher logits.
# Defaults: Qwen3-4B -> Qwen3-30B-A3B-Instruct-2507, chat without thinking.
# Set MODELS for local model roots, or STUDENT_PATH/TEACHER_PATH individually.
export PROMPT_MODE="${PROMPT_MODE:-chat}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
exec bash "$root/recipe/opd/run.sh" "$@"
