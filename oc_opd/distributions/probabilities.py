"""Full-support distributions with explicit probability underflow handling."""

import torch
from torch import Tensor

from ..utils.validation import validate_logits


def from_logits(
    student_logits: Tensor, teacher_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return student/teacher log probabilities followed by their probabilities.

    Half-precision inputs use float32. Probability underflow retries in float64
    rather than truncating the vocabulary or silently removing support.
    """
    validate_logits(student_logits, teacher_logits)
    dtype = (
        torch.float64
        if torch.float64 in (student_logits.dtype, teacher_logits.dtype)
        else torch.float32
    )
    log_student = student_logits.to(dtype).log_softmax(-1)
    log_teacher = teacher_logits.to(dtype).log_softmax(-1)
    if not (torch.isfinite(log_student).all() and torch.isfinite(log_teacher).all()):
        raise ValueError("Finite full-support logits are required.")

    student_prob = log_student.exp()
    teacher_prob = log_teacher.exp()
    if (student_prob == 0).any() or (teacher_prob == 0).any():
        if dtype == torch.float32:
            del log_student, log_teacher, student_prob, teacher_prob
            return from_logits(student_logits.double(), teacher_logits.double())
        raise ValueError("Probabilities underflow even in float64.")
    return log_student, log_teacher, student_prob, teacher_prob
