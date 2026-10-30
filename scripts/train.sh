#!/usr/bin/env bash
set -euo pipefail

method="${1:?Usage: bash scripts/train.sh METHOD [Hydra overrides...]}"
shift
case "$method" in
  oc_opd|opd|fkl|kd|sft|grpo) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

: "${STUDENT_PATH:?Set STUDENT_PATH to a Hugging Face model or local checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a training JSONL or Parquet file}"
args=(
  "actor_rollout_ref.model.path=$STUDENT_PATH"
  "data.train_files=['$TRAIN_FILE']"
  "trainer.n_gpus_per_node=${N_GPUS:-4}"
)
case "$method" in
  oc_opd|opd|fkl|kd)
    : "${TEACHER_PATH:?Set TEACHER_PATH for distillation}"
    args+=("actor_rollout_ref.distillation.teacher_path=$TEACHER_PATH")
    ;;
esac

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec python -m oc_opd.trainer.main --config-name "$method" "${args[@]}" "$@"
