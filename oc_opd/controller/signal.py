"""Frozen OC-OPD controller on teacher top-k or full vocabulary support."""

import torch
from torch import Tensor

from ..config import (
    DEFAULT_EPSILON,
    DEFAULT_H0,
    DEFAULT_TAU,
    DEFAULT_TEACHER_TOP_K,
    DEFAULT_VOCABULARY_MODE,
    OCOPDConfig,
)
from ..distributions import entropy, forward_kl, from_logits, reverse_kl, select_support
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
    vocabulary_mode: str = DEFAULT_VOCABULARY_MODE,
    teacher_top_k: int = DEFAULT_TEACHER_TOP_K,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the detached signal and per-position diagnostics.

    Logits must have identical (..., vocabulary) shapes and token mappings.
    The default uses the exact original full-vocabulary controller. Set
    vocabulary_mode='teacher_topk' to renormalize BOTH distributions over teacher
    top-16 IDs (configurable with teacher_top_k). Signal entries outside this
    support are zero; diagnostics describe the conditional distributions.
    Diagnostics retain shape (..., 1).
    """
    config = OCOPDConfig(
        h0=h0,
        epsilon=epsilon,
        tau=tau,
        vocabulary_mode=vocabulary_mode,
        teacher_top_k=teacher_top_k,
    )
    selected_student, selected_teacher, indices = select_support(
        student_logits,
        teacher_logits,
        vocabulary_mode=config.vocabulary_mode,
        teacher_top_k=config.teacher_top_k,
    )
    log_student, log_teacher, student_prob, teacher_prob = from_logits(
        selected_student, selected_teacher
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
    if indices is not None:
        gradient = gradient.new_zeros(student_logits.shape).scatter_(-1, indices, gradient)

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
