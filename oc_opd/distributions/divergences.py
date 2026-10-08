"""Exact KL divergences over the entire aligned token vocabulary."""

from torch import Tensor


def reverse_kl(student_prob: Tensor, log_student: Tensor, log_teacher: Tensor) -> Tensor:
    """Compute KL(student || teacher), retaining the singleton vocabulary axis."""
    return (student_prob * (log_student - log_teacher)).sum(-1, keepdim=True)


def forward_kl(teacher_prob: Tensor, log_teacher: Tensor, log_student: Tensor) -> Tensor:
    """Compute KL(teacher || student), retaining the singleton vocabulary axis."""
    return (teacher_prob * (log_teacher - log_student)).sum(-1, keepdim=True)
