#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STUDENT_PATH="${STUDENT_PATH:-Qwen/Qwen3-4B}"
exec bash "$root/recipe/grpo/run.sh" data.prompt_mode=chat data.enable_thinking=false "$@"
