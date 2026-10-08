"""Masked response-token mean OC-OPD surrogate."""

from torch import Tensor

from ..controller import signal
from ..utils.validation import validate_response_mask
from .frozen_signal import frozen_signal_loss


def surrogate(
    student_logits: Tensor,
    teacher_logits: Tensor,
    valid_mask: Tensor,
    **kwargs,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a frozen-signal loss and diagnostics for selected response tokens."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    validate_response_mask(student_logits, valid_mask)
    selected = student_logits[valid_mask]
    gradient, diagnostics = signal(selected, teacher_logits[valid_mask], **kwargs)
    return frozen_signal_loss(selected, gradient), diagnostics
