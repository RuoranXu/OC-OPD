#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STUDENT_PATH="${STUDENT_PATH:-Qwen/Qwen3-1.7B-Base}"
export TEACHER_PATH="${TEACHER_PATH:-Qwen/Qwen3-14B-Base}"
exec bash "$root/recipe/oc_opd/run.sh" data.prompt_mode=base "$@"
