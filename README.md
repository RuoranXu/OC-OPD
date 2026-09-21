# OC-OPD

PyTorch implementation of the OC-OPD gradient controller for on-policy
language-model distillation. The student learns on its own sampled responses;
the frozen teacher scores those responses at aligned prediction positions.

This repository contains the method implementation, integration documentation,
a synthetic optimizer-update example, and correctness tests. It has no benchmark
datasets, evaluation pipelines, experimental results, model weights, or remote
infrastructure dependencies.

## Installation

Python 3.10+ and PyTorch 2.3+ are required.

```bash
pip install -e .
python examples/minimal_update.py
python -m unittest discover -s tests -v
```

## Method

At each response position, let $p$ denote the teacher distribution and $q$ the
student distribution over the same vocabulary. OC-OPD combines the reverse- and
forward-KL gradient directions using entropy-informed weights and a closed-form
Nash direction. The resulting signal is frozen before the student backward pass.

The gate in this release is **$w=\min(B/\tau,1)$**, with **$\tau=0.01$**, where

$$B=\frac12\sum_v(\sqrt{p_v}-\sqrt{q_v})^2.$$

The default threshold changes from **0.05 to 0.01**. The gate reaches 1 at
$B\ge0.01$; below that threshold, the update is scaled by $B/0.01$.
This weakens the attenuation relative to the earlier threshold of 0.05.
Historical results produced with the earlier gate do not validate this version.
Use `tau=0.05` explicitly to recover the previous gate configuration.

See [the equations](docs/method.md) and [training integration](docs/integration.md).

## API

```python
from oc_opd import surrogate

# logits: [batch, sequence, vocabulary], aligned at causal prediction positions
# response_mask: [batch, sequence], True only for valid response positions
loss, diagnostics = surrogate(student_logits, teacher_logits, response_mask)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

`surrogate` averages over valid response tokens. Its scalar value is a gradient
carrier, not a KL value; log `forward_kl` and `reverse_kl` from diagnostics instead.
The teacher and controller receive no gradients.

`signal` exposes the detached logit-gradient signal. `backward_chunked` computes
the same update in position chunks when the student's output projection is frozen;
it does not approximate the vocabulary with top-k entries.

## Layout

```text
src/oc_opd/objective.py   OC-OPD signal and masked surrogate
src/oc_opd/chunked.py     Position-chunked backward for frozen output heads
examples/                Synthetic usage example
tests/                   Numerical and gradient correctness tests
docs/                    Method equations and integration requirements
```
