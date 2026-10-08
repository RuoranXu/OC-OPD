# OC-OPD

Core PyTorch implementation of OC-OPD for on-policy language-model
distillation. The package computes the full-vocabulary OC-OPD logit-gradient
signal and provides direct and position-chunked training integration.

## Installation

```bash
pip install -e .
```

## Usage

```python
from oc_opd import surrogate

loss, diagnostics = surrogate(
    student_logits,
    teacher_logits,
    response_mask,
    h0=1.0,
    epsilon=0.05,
    tau=0.01,
)
loss.backward()
```

`student_logits` and `teacher_logits` must be aligned full-vocabulary logits.
`response_mask` selects response prediction positions and excludes prompt and
padding tokens. The controller and teacher signal are detached before backward.

## Structure

```text
oc_opd/
  objective.py  OC-OPD signal and masked surrogate
  chunked.py    Exact position-chunked backward for frozen output heads
```
