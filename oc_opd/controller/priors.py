"""Entropy-informed objective weights."""

from torch import Tensor


def entropy_priors(
    student_entropy: Tensor,
    teacher_entropy: Tensor,
    *,
    h0: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Return reverse/forward priors normalized to sum to one per position."""
    student_entropy = student_entropy / (student_entropy + h0)
    teacher_entropy = teacher_entropy / (teacher_entropy + h0)
    reverse_prior = epsilon + (1 - teacher_entropy) * student_entropy
    forward_prior = epsilon + teacher_entropy
    total = reverse_prior + forward_prior
    return reverse_prior / total, forward_prior / total
