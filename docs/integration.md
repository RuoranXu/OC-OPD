# Integrating OC-OPD into training

## Align the response positions

Generate responses from the current student, then score the same responses
with student and teacher. Both probability vectors must represent the same next
token after the same response prefix. Teacher and student may use different
native prompt templates; account for each prefix length independently.

For a sequence with prompt length P, response token IDs begin at `ids[:, P:]`.
With causal models scored on `ids[:, :-1]`, prediction positions begin at `P-1`.
Exclude prompts and padding from the loss. Define EOS inclusion consistently.
Matching vocabulary size alone is insufficient: token-ID mappings must agree.

Choose the actual model's native prompt format. Base models and chat models are
not interchangeable. Render and inspect any thinking-mode switch rather than
assuming a flag changes every model's behavior.

## Ordinary integration

Compute teacher logits under `torch.no_grad()`. Pass aligned student logits,
teacher logits and a boolean response mask to `surrogate`. It supplies the
OC-OPD gradient; rollout generation, optimizer/scheduler choice and checkpoint
management belong to the surrounding trainer.

## Position-chunked integration

```python
from oc_opd import backward_chunked

optimizer.zero_grad(set_to_none=True)
metrics = backward_chunked(
    student_hidden, teacher_hidden,
    student_lm_head, teacher_lm_head,
    response_mask, chunk_size=128,
)
optimizer.step()
```

The student head must be frozen; the student backbone can be trainable through
LoRA or ordinary parameters. The helper obtains gradients for detached hidden
state chunks, then calls backward once on the original student hidden states.
It retains full vocabulary computations within each chunk. Teacher hidden states
may reside on the teacher device. Each hidden tensor must match its own head's
input device and dtype.

This helper performs backward internally; do not call backward again on its
diagnostics. When a head is trainable, use the ordinary surrogate or provide an
integration that also accumulates head gradients. Mixed-precision loss scaling
and distributed synchronization are not provided by this helper.

## Reduction and distributed training

The default objective is a mean over all valid response tokens. With unequal
token counts per rank, averaging each rank's local mean is not the same objective
as a global token mean. Scale or reduce according to the trainer's actual DDP
averaging convention. Handle empty local masks at the trainer level without
desynchronizing collectives; these core APIs reject an empty valid set.

## Correctness checks

The tests verify the new gate, frozen teacher/controller behavior, masked
logit gradients, identical-distribution zero updates and agreement between
direct and chunked backbone gradients. They are synthetic method tests, not
benchmark evaluation or claims about downstream performance.
