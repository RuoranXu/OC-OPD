# OC-OPD

On-policy distillation with OC-OPD, built on the veRL distributed training runtime.

## Quick start

### 1. Install

Use Linux, NVIDIA GPUs with a compatible CUDA driver, and Python 3.10 or 3.11.
The examples default to four GPUs; memory requirements depend on model sizes and sequence lengths.

```bash
git clone https://github.com/RuoranXu/OC-OPD.git
cd OC-OPD
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,vllm]'
```

Training dependencies are defined in `requirements-train.txt` and `requirements-vllm.txt`.
For the core tensor objectives only, install with `python -m pip install -e .`.

### 2. Prepare data

Download and prepare the GSM8K training split for a chat student:

```bash
python examples/data_preprocess/gsm8k.py --prompt-mode chat
```

This writes `data/processed/gsm8k/chat/train.parquet`. For your own JSONL or Parquet
data, use [normalize.py](examples/data_preprocess/normalize.py) and set `TRAIN_FILE`
to the prepared file. Chat prompts are role/content messages; Base prompts are
plain text. Chat templates are applied once by the training dataset.

### 3. Train OC-OPD

Choose a positive training iteration count. The launchers have no preset total
training budget; use either `TOTAL_STEPS` or `TOTAL_EPOCHS`, never both.

```bash
unset TOTAL_EPOCHS
read -rp "Training iterations: " TOTAL_STEPS
export TOTAL_STEPS

STUDENT_PATH=Qwen/Qwen3-4B \
TEACHER_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
N_GPUS=4 \
bash examples/oc_opd/run_qwen_chat.sh
```

Model paths may be Hugging Face IDs or local directories. Student and teacher
must share the same vocabulary and special-token IDs. The chat example defaults
to non-thinking mode; set `ENABLE_THINKING=true` for a compatible thinking student.

Inspect the resolved configuration without starting training:

```bash
bash examples/oc_opd/run_qwen_chat.sh --cfg job --resolve
```

Checkpoints are written under `outputs/oc_opd_chat/global_step_*/actor/`.
Set `RUN_NAME` or `RUN_DIR` to choose another output location. Training saves
checkpoints at the configured interval and at completion, without selecting a
best checkpoint or automatically starting evaluation. The latest two checkpoints
are retained by default.

## Base students

Prepare a separate Base dataset and launch with the training budget set above:

```bash
python examples/data_preprocess/gsm8k.py --prompt-mode base
STUDENT_PATH=Qwen/Qwen3-1.7B-Base \
TEACHER_PATH=Qwen/Qwen3-14B-Base \
bash examples/oc_opd/run_qwen_base.sh
```

The Base entry uses plain-text prompts and disables chat thinking. To train a
4B Base student, set `STUDENT_PATH=Qwen/Qwen3-4B-Base`.

## Common overrides

Append Hydra overrides to a training command:

| Setting | Default | Override |
| --- | --- | --- |
| Fine-tuning | Full parameters | `finetuning_type=lora lora_rank=16` |
| Vocabulary support | Full vocabulary | `actor_rollout_ref.distillation.controller.vocabulary_mode=teacher_topk actor_rollout_ref.distillation.controller.teacher_top_k=16` |
| Learning rate | `1e-6` | `actor_rollout_ref.actor.optim.lr=1e-6` |
| Checkpoint interval | 30 iterations | `trainer.save_freq=30` |

LoRA covers attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP
projections (`gate_proj`, `up_proj`, `down_proj`). The top-k option renormalizes
both models over the teacher-selected support and changes the KL objective.
Full training defaults are in [training.yaml](oc_opd/trainer/config/training.yaml),
with OC-OPD batch settings in [oc_opd.yaml](oc_opd/trainer/config/oc_opd.yaml).

## Baselines and evaluation

Baseline launchers are in [examples/baselines](examples/baselines): `grpo`, `opd`
(reverse KL), `fkl`, `kd`, and `sft`. They accept the same model, data, and training
budget environment variables. KD/SFT additionally require a nonempty `response`
field; the GSM8K preprocessing entry supplies reference solutions.

For manual AIME 2024/2025 evaluation with avg@32 and pass@32, follow
[run_aime.sh](examples/test/run_aime.sh). Use its separate evaluation environment,
export full FSDP checkpoints to Hugging Face format first, and choose `--mode`
explicitly (`nonthinking` for the chat example above, `thinking`, or `base`).
