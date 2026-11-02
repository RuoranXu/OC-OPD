"""Analytic gradients of the two KL objectives with respect to student logits."""

from torch import Tensor


def reverse_kl_gradient(
    student_prob: Tensor,
    log_student: Tensor,
    log_teacher: Tensor,
    reverse_kl: Tensor,
) -> Tensor:
    """Compute q * (log(q) - log(p) - KL(q || p))."""
    return student_prob * (log_student - log_teacher - reverse_kl)


def forward_kl_gradient(student_prob: Tensor, teacher_prob: Tensor) -> Tensor:
    """Compute q - p."""
    return student_prob - teacher_prob
