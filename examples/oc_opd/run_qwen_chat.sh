#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STUDENT_PATH="${STUDENT_PATH:-Qwen/Qwen3-4B}"
export TEACHER_PATH="${TEACHER_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
exec bash "$root/recipe/oc_opd/run.sh" data.prompt_mode=chat data.enable_thinking=false "$@"
