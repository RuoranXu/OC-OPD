# OC-OPD

## Quick start

### 1. Install

```bash
git clone https://github.com/RuoranXu/OC-OPD.git
cd OC-OPD
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,vllm]'
```

### 2. Prepare data

```bash
hf download BytedTsinghua-SIA/DAPO-Math-17k data/dapo-math-17k.parquet \
  --repo-type dataset --local-dir data/raw/dapo-math-17k
```

### 3. Train OC-OPD

```bash
unset TOTAL_EPOCHS
read -rp "Training iterations: " TOTAL_STEPS
export TOTAL_STEPS
export TRAIN_FILE="$PWD/data/raw/dapo-math-17k/data/dapo-math-17k.parquet"

STUDENT_PATH=Qwen/Qwen3-4B \
TEACHER_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
N_GPUS=4 \
bash examples/oc_opd/run_qwen_chat.sh
```

```bash
bash examples/oc_opd/run_qwen_chat.sh --cfg job --resolve
```

## Evaluation

For manual AIME 2024/2025 evaluation with avg@32 and pass@32, follow [run_aime.sh](examples/test/run_aime.sh).
