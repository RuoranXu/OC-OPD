"""Masked, response-mean OC-OPD surrogate."""

from torch import Tensor

from ..controller import signal
from ..distributions import select_support
from ..utils.validation import validate_response_mask
from .frozen_signal import frozen_signal_loss
from .reduction import response_weights


def surrogate(
    student_logits: Tensor,
    teacher_logits: Tensor,
    valid_mask: Tensor,
    *,
    reduction: str = "sequence_mean",
    **kwargs,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average per-response token means; token_mean is an explicit legacy option."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    validate_response_mask(student_logits, valid_mask)
    selected = student_logits[valid_mask]
    options = dict(kwargs)
    vocabulary = {
        name: options.pop(name) for name in ("vocabulary_mode", "teacher_top_k") if name in options
    }
    selected, teacher, _ = select_support(selected, teacher_logits[valid_mask], **vocabulary)
    gradient, diagnostics = signal(selected, teacher, vocabulary_mode="full", **options)
    if reduction == "sequence_mean":
        weights = response_weights(valid_mask, dtype=gradient.dtype)[valid_mask]
        return frozen_signal_loss(
            selected, gradient * weights.unsqueeze(-1), token_count=1
        ), diagnostics
    if reduction == "token_mean":
        return frozen_signal_loss(selected, gradient), diagnostics
    raise ValueError("reduction must be 'sequence_mean' or 'token_mean'.")
