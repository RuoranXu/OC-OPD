#!/usr/bin/env bash
set -euo pipefail

method="${1:?Usage: bash scripts/train.sh METHOD [Hydra overrides...]}"
shift
case "$method" in
  oc_opd|opd|fkl|kd|sft|grpo) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"
TOTAL_STEPS="${TOTAL_STEPS:-}"

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
prompt_mode="${PROMPT_MODE:-chat}"
train_file_override=false
training_budget_override=false
inspect_config=false
for argument in "$@"; do
  case "$argument" in
    data.prompt_mode=*|+data.prompt_mode=*|++data.prompt_mode=*)
      prompt_mode="${argument#*=}"
      prompt_mode="${prompt_mode#\"}"; prompt_mode="${prompt_mode%\"}"
      prompt_mode="${prompt_mode#\'}"; prompt_mode="${prompt_mode%\'}"
      ;;
    data.train_files=*|+data.train_files=*|++data.train_files=*) train_file_override=true ;;
    trainer.total_epochs=*|+trainer.total_epochs=*|++trainer.total_epochs=*|\
    trainer.total_training_steps=*|+trainer.total_training_steps=*|++trainer.total_training_steps=*)
      training_budget_override=true
      ;;
    --cfg|--cfg=*|--help|-h|--info|--info=*) inspect_config=true ;;
  esac
done
if [[ "$inspect_config" == false && "$training_budget_override" == false \
      && -z "$TOTAL_EPOCHS" && -z "$TOTAL_STEPS" ]]; then
  echo 'Set exactly one training budget: TOTAL_EPOCHS or TOTAL_STEPS (left blank in scripts/train.sh).' >&2
  echo 'Alternatively provide trainer.total_epochs or trainer.total_training_steps as a Hydra override.' >&2
  exit 2
fi
case "$prompt_mode" in
  base|chat) ;;
  *) echo "PROMPT_MODE / data.prompt_mode must be base or chat" >&2; exit 2 ;;
esac
thinking="${ENABLE_THINKING:-false}"
case "$thinking" in
  true|false) ;;
  *) echo "ENABLE_THINKING must be true or false" >&2; exit 2 ;;
esac
n_gpus="${N_GPUS:-4}"
if [[ ! "$n_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_GPUS must be a positive integer" >&2
  exit 2
fi

# MODELS optionally selects a directory containing original Hugging Face models.
# STUDENT_PATH / TEACHER_PATH take precedence and may also be Hub model IDs.
model_name=Qwen3-4B
teacher_name=Qwen3-30B-A3B-Instruct-2507
if [[ "$prompt_mode" == base ]]; then
  model_name=Qwen3-4B-Base
  teacher_name=Qwen3-14B-Base
fi
model_root="${MODELS:-Qwen}"
student="${STUDENT_PATH:-${model_root%/}/$model_name}"
teacher="${TEACHER_PATH:-${model_root%/}/$teacher_name}"
data_root="${DATA_ROOT:-$root/data/processed}"
train_file="${TRAIN_FILE:-${data_root%/}/gsm8k/$prompt_mode/train.parquet}"
run_root="${RUN_ROOT:-$root/outputs}"
run_name="${RUN_NAME:-${method}_${prompt_mode}}"
run_dir="${RUN_DIR:-${run_root%/}/$run_name}"

if [[ "$train_file_override" == false && "$inspect_config" == false && ! -f "$train_file" ]]; then
  printf 'Training file does not exist: %s\n' "$train_file" >&2
  printf 'Prepare it with: python "%s/examples/data_preprocess/gsm8k.py" --prompt-mode %s --output-dir "%s/gsm8k/%s"\n' \
    "$root" "$prompt_mode" "${data_root%/}" "$prompt_mode" >&2
  echo 'Alternatively set TRAIN_FILE or override data.train_files with your prepared JSONL/Parquet.' >&2
  exit 2
fi

# Quote a Hydra string value independently of shell quoting. This preserves
# spaces, commas, brackets, quotes and backslashes in checkpoint/data paths.
hydra_string() {
  local value="$1" result='"' slashes='' character index
  for ((index = 0; index < ${#value}; index++)); do
    character="${value:index:1}"
    if [[ "$character" == '\' ]]; then
      slashes+='\'
    elif [[ "$character" == '"' ]]; then
      result+="${slashes}${slashes}\\\""
      slashes=''
    else
      result+="${slashes}${character}"
      slashes=''
    fi
  done
  # Hydra doubles only backslashes immediately before a quote (including the
  # closing quote), unlike JSON escaping which would change ordinary paths.
  printf '%s%s%s"' "$result" "$slashes" "$slashes"
}

args=(
  "actor_rollout_ref.model.path=$(hydra_string "$student")"
  "data.train_files=[$(hydra_string "$train_file")]"
  "data.prompt_mode=$prompt_mode"
  "data.enable_thinking=$thinking"
  "trainer.n_gpus_per_node=$n_gpus"
  "trainer.experiment_name=$(hydra_string "$run_name")"
  "trainer.default_local_dir=$(hydra_string "$run_dir")"
  "trainer.total_epochs=${TOTAL_EPOCHS:-null}"
  "trainer.total_training_steps=${TOTAL_STEPS:-null}"
)
case "$method" in
  oc_opd|opd|fkl|kd)
    args+=("actor_rollout_ref.distillation.teacher_path=$(hydra_string "$teacher")")
    ;;
esac

# KD/SFT: prompt + nonempty response fields. GRPO: prompt + data_source and
# reward_model.ground_truth compatible with the selected veRL reward function.
# The GSM8K preprocessing entry supplies all of these fields for its TRAIN split.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "${PYTHON:-python}" -m oc_opd.trainer.main --config-name "$method" "${args[@]}" "$@"
