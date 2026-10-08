"""Frozen full-vocabulary OC-OPD logit-gradient controller."""

import torch
from torch import Tensor

from ..config import DEFAULT_EPSILON, DEFAULT_H0, DEFAULT_TAU, OCOPDConfig
from ..distributions import entropy, forward_kl, from_logits, reverse_kl
from ..utils.tensors import normalize_direction
from .gating import hellinger_gate
from .gradients import forward_kl_gradient, reverse_kl_gradient
from .nash import nash_direction
from .priors import entropy_priors


@torch.no_grad()
def signal(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    h0: float = DEFAULT_H0,
    epsilon: float = DEFAULT_EPSILON,
    tau: float = DEFAULT_TAU,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the detached signal and per-position diagnostics.

    Logits must have identical (..., vocabulary) shapes and token mappings.
    Diagnostics retain shape (..., 1); the controller uses at least float32.
    """
    config = OCOPDConfig(h0=h0, epsilon=epsilon, tau=tau)
    log_student, log_teacher, student_prob, teacher_prob = from_logits(
        student_logits, teacher_logits
    )
    reverse = reverse_kl(student_prob, log_student, log_teacher)
    forward = forward_kl(teacher_prob, log_teacher, log_student)
    reverse_gradient = reverse_kl_gradient(student_prob, log_student, log_teacher, reverse)
    forward_gradient = forward_kl_gradient(student_prob, teacher_prob)

    reverse_norm = reverse_gradient.norm(dim=-1, keepdim=True)
    forward_norm = forward_gradient.norm(dim=-1, keepdim=True)
    reverse_direction = normalize_direction(reverse_gradient, reverse_norm)
    forward_direction = normalize_direction(forward_gradient, forward_norm)
    omega_reverse, omega_forward = entropy_priors(
        entropy(student_prob, log_student),
        entropy(teacher_prob, log_teacher),
        h0=config.h0,
        epsilon=config.epsilon,
    )
    direction, cosine = nash_direction(
        reverse_direction, forward_direction, omega_reverse, omega_forward
    )
    gate, distance = hellinger_gate(student_prob, teacher_prob, tau=config.tau)
    gradient = gate * forward_norm * direction
    gradient = gradient - gradient.mean(-1, keepdim=True)

    return gradient, {
        "reverse_kl": reverse,
        "forward_kl": forward,
        "hellinger_squared": distance,
        "gate": gate,
        "omega_R": omega_reverse,
        "omega_F": omega_forward,
        "cosine": cosine,
        "gain_R": (reverse_direction * direction).sum(-1, keepdim=True),
        "gain_F": (forward_direction * direction).sum(-1, keepdim=True),
    }
