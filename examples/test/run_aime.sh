#!/usr/bin/env bash
set -euo pipefail

# Linux / NVIDIA GPU. Keep this historical vLLM 0.8.5 evaluation environment
# separate from the repository's training environment; no training install needed.
# python3.10 -m venv .venv-aime
# source .venv-aime/bin/activate
# python -m pip install -r examples/test/requirements.txt
#
# Complete Hugging Face checkpoint (Base prompt by default):
# bash examples/test/run_aime.sh --model /path/to/merged_hf_model
#
# LoRA checkpoint: --model must be the ORIGINAL model used to train the adapter.
# bash examples/test/run_aime.sh --model Qwen/Qwen3-4B-Base \
#   --adapter /path/to/checkpoints/step_000250/adapter --mode base
# Also accepts checkpoint/actor/lora_adapter/ or checkpoint/lora_adapter/.
#
# Export full FSDP weights in the TRAINING environment before evaluation:
# python -m verl.model_merger merge --backend fsdp \
#   --local_dir /path/to/global_step_250/actor --target_dir /path/to/merged_hf_model
# If the export omits its tokenizer, also pass --tokenizer ORIGINAL_MODEL.
# Do not use an adapter with lora_alpha=0; it would disable the learned update.
#
# Defaults reproduce n=32/question, seed=604+year*1000+question_number, T=1,
# top_p=1, top_k=-1, 32768 context, 4 independent BF16 engines, TP=1.
# CUDA_VISIBLE_DEVICES is respected: --gpus 0,1 indexes that visible mask.
# --gpus 0 selects one GPU; --output-dir must change when the protocol changes.
# The original --swap-space 0 can fail for long n=32 sequence groups. Even CPU
# swapping cannot make a single request group larger than GPU KV capacity fit.
# For longer generations, explicitly choose --tensor-parallel-size 4 --inflight 1
# (one engine using all four GPUs), or TP=2 with two engines. These are recorded
# runtime changes; they retain seeds, n and length limits, but are not the old
# execution layout. --swap-space 16 is optional CPU KV swap GiB per GPU and is not
# a guarantee against OOM. Failures retain complete questions; nothing is silently
# shortened or skipped. Resume with the SAME command. Hardware/scheduling can
# still change draws, so a matched seed is not a promise of identical old scores.
#
# CPU preparation (requires tokenizer/runtime packages, no engine allocation):
# bash examples/test/run_aime.sh --model MODEL --prepare-only
# Rescore all saved responses with no GPU/model load:
# bash examples/test/run_aime.sh --output-dir outputs/aime24_25 --score-only

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -u "${SCRIPT_DIR}/eval_aime.py" "$@"
