# OC-OPD



## Installation

The core objective needs PyTorch only:

```bash
pip install -e .
```

Distributed recipes target Linux with NVIDIA GPUs, PyTorch 2.8, transformers
4.56.2, vLLM 0.10.2 and the vendored veRL v0.6.1. Use a separate environment;
do not install another `verl` distribution into it.

```bash
pip install -r requirements-train.txt
# Alternatively:
docker build -f docker/Dockerfile -t oc-opd .
```

## Methods

| Recipe | Training prefixes | Objective |
| --- | --- | --- |
| `oc_opd` | Current student rollout | Frozen OC-OPD full-vocabulary signal |
| `opd` | Current student rollout | Exact reverse KL: student relative to teacher |
| `fkl` | Current student rollout | Exact forward KL: teacher relative to student |
| `kd` | Fixed dataset responses | Forward KL on teacher-forced prefixes |
| `sft` | Fixed dataset responses | Token cross-entropy |
| `grpo` | Groups of student rollouts | Native veRL group advantages and PPO clipped loss |

KD is token-level distribution distillation on fixed demonstrations, rather than
on-policy FKL. OPD uses exact full-vocabulary RKL. GRPO's optional reference
penalty is veRL's sampled KL estimator. GRPO requires a training reward: a
supported `data_source` and `reward_model.ground_truth`, or a custom function
configured with `custom_reward_function.path`.

## Training

```bash
python scripts/prepare_gsm8k.py --output data/gsm8k_chat.jsonl
export TRAIN_FILE="$PWD/data/gsm8k_chat.jsonl"
export STUDENT_PATH=Qwen/Qwen3-4B
export TEACHER_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507

bash recipe/oc_opd/run.sh trainer.total_training_steps=360
# Other independently launched methods:
bash recipe/opd/run.sh
bash recipe/kd/run.sh
bash recipe/grpo/run.sh
```

These are launch templates, not reported experiment settings or results. The
preparer writes only the GSM8K training split. Custom JSONL or Parquet data use
`prompt` (chat message list or plain Base string), plus `response` for KD/SFT.
Fixed responses can be teacher demonstrations or a supervised corpus. Overlong
prompts/responses raise errors instead of silently truncating the data.

Base models require `data.prompt_mode=base` and explicit completion prompts.
Chat models use the student's chat template; Qwen examples explicitly set
`data.enable_thinking=false`. Teacher and student see exactly the same tokenized
prefix. Vocabulary mappings and special-token IDs must match.

```bash
# Resolve configuration without allocating GPUs:
python -m oc_opd.trainer.main --config-name oc_opd --cfg job --resolve
# Prepare Base prompts:
python scripts/prepare_gsm8k.py --prompt-mode base --output data/gsm8k_base.jsonl
TRAIN_FILE="$PWD/data/gsm8k_base.jsonl" bash examples/oc_opd/run_qwen_base.sh
```

Controller defaults: `h0=1`, `epsilon=0.05`, `tau=0.01`, with `min(B/tau, 1)`
gating and per-token outputs of shape `(..., 1)`. The controller and teacher are
detached. Frozen teacher inference is online; full-vocabulary logits are
transient, not written to disk. All losses exclude prompt and padding positions.
Loss temperature is 1, independently of rollout sampling temperature.

Each iteration generates `data.train_batch_size * rollout.n` responses. The
provided one-pass recipes perform `train_batch_size / ppo_mini_batch_size`
optimizer updates per iteration. veRL expresses minibatches in prompts before
multiplying by `rollout.n`; the LR scheduler advances per iteration.
Distillation uses the global valid-response-token mean across GPUs.
Checkpoints include student, optimizer/scheduler and loader state. Resume with
`trainer.resume_mode=auto` and the same configuration and output directory.

The distillation adapter supports FSDP1, text-only inputs, sequence parallel
size 1, fixed microbatches and full student fine-tuning. It uses gradient
checkpointing, teacher CPU offload and chunked FP32 loss work. Output logits
still take memory proportional to context length and vocabulary size. Reduce
length or batch size to fit the hardware. Distributed OOM fails the job instead
of silently skipping a rank. GPU training must be verified in the target
Linux/CUDA environment. No benchmark results or SEED selection logic are included.

## Core API

```python
from oc_opd import OCOPDConfig, surrogate

loss, diagnostics = surrogate(
    student_logits, teacher_logits, response_mask,
    **OCOPDConfig(h0=1.0, epsilon=0.05, tau=0.01).as_kwargs(),
)
loss.backward()
```

`backward_chunked` is a separate API for frozen output heads; the veRL worker
trains the full student and does not use that shortcut.

The layout follows the method/runtime/recipe organization of
[SEED](https://github.com/jinyangwu/SEED). The runtime is pinned to
[veRL v0.6.1](https://github.com/verl-project/verl/tree/v0.6.1), with its upstream
copyright and Apache-2.0 notices retained in `third_party/verl/`.
