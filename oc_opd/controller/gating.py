"""Squared-Hellinger distance gate."""

from torch import Tensor


def hellinger_gate(
    student_prob: Tensor, teacher_prob: Tensor, *, tau: float
) -> tuple[Tensor, Tensor]:
    """Return min(B / tau, 1) and B = 0.5 * sum((sqrt(p) - sqrt(q))**2)."""
    distance = 0.5 * (
        teacher_prob.sqrt() - student_prob.sqrt()
    ).square().sum(-1, keepdim=True)
    return (distance / tau).clamp(0, 1), distance
