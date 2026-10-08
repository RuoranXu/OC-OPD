"""Frozen, full-vocabulary OC-OPD controller with gate min(B / tau, 1)."""

import math

import torch
from torch import Tensor


@torch.no_grad()
def signal(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    h0: float = 1.0,
    epsilon: float = 0.05,
    tau: float = 0.01,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the detached logit-gradient signal and per-position diagnostics.

    Inputs have shape ``(..., vocabulary)`` with identical token mappings.
    Diagnostics have shape ``(..., 1)``. Float16 and bfloat16 inputs are
    promoted to float32; float64 inputs or float32 probability underflow use
    float64.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 1:
        raise ValueError("Logits must have identical (..., vocabulary) shapes.")
    if student_logits.shape[-1] < 2:
        raise ValueError("At least two vocabulary entries are required.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("Logits must be on the same device.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("Logits must be floating-point tensors.")
    if not all(math.isfinite(value) and value > 0 for value in (h0, epsilon, tau)):
        raise ValueError("h0, epsilon and tau must be finite and positive.")

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
            return signal(
                student_logits.double(),
                teacher_logits.double(),
                h0=h0,
                epsilon=epsilon,
                tau=tau,
            )
        raise ValueError("Probabilities underflow even in float64.")

    reverse_kl = (
        student_prob * (log_student - log_teacher)
    ).sum(-1, keepdim=True)
    forward_kl = (
        teacher_prob * (log_teacher - log_student)
    ).sum(-1, keepdim=True)
    reverse_gradient = student_prob * (log_student - log_teacher - reverse_kl)
    forward_gradient = student_prob - teacher_prob

    tiny = torch.finfo(dtype).tiny
    reverse_norm = reverse_gradient.norm(dim=-1, keepdim=True)
    forward_norm = forward_gradient.norm(dim=-1, keepdim=True)
    reverse_direction = reverse_gradient / reverse_norm.clamp_min(tiny)
    forward_direction = forward_gradient / forward_norm.clamp_min(tiny)
    cosine = (
        reverse_direction * forward_direction
    ).sum(-1, keepdim=True).clamp(-1, 1)

    student_entropy = -(student_prob * log_student).sum(-1, keepdim=True)
    teacher_entropy = -(teacher_prob * log_teacher).sum(-1, keepdim=True)
    student_entropy = student_entropy / (student_entropy + h0)
    teacher_entropy = teacher_entropy / (teacher_entropy + h0)
    reverse_prior = epsilon + (1 - teacher_entropy) * student_entropy
    forward_prior = epsilon + teacher_entropy
    omega_reverse = reverse_prior / (reverse_prior + forward_prior)
    omega_forward = forward_prior / (reverse_prior + forward_prior)

    interaction = cosine * (omega_reverse - omega_forward)
    root = (interaction.square() + 4 * omega_reverse * omega_forward).sqrt()
    ratio = torch.where(
        interaction >= 0,
        (interaction + root) / (2 * omega_forward),
        2 * omega_reverse / (root - interaction),
    )
    direction = ratio * reverse_direction + forward_direction
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(tiny)

    distance = 0.5 * (
        teacher_prob.sqrt() - student_prob.sqrt()
    ).square().sum(-1, keepdim=True)
    gate = (distance / tau).clamp(0, 1)
    gradient = gate * forward_norm * direction
    gradient = gradient - gradient.mean(-1, keepdim=True)

    diagnostics = {
        "reverse_kl": reverse_kl,
        "forward_kl": forward_kl,
        "hellinger_squared": distance,
        "gate": gate,
        "omega_R": omega_reverse,
        "omega_F": omega_forward,
        "cosine": cosine,
        "gain_R": (reverse_direction * direction).sum(-1, keepdim=True),
        "gain_F": (forward_direction * direction).sum(-1, keepdim=True),
    }
    return gradient, diagnostics


def surrogate(
    student_logits: Tensor,
    teacher_logits: Tensor,
    valid_mask: Tensor,
    **kwargs,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a token-mean frozen-signal surrogate and token diagnostics."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    if valid_mask.dtype != torch.bool or valid_mask.shape != student_logits.shape[:-1]:
        raise ValueError("Use a boolean mask matching all non-vocabulary dimensions.")
    if valid_mask.device != student_logits.device:
        raise ValueError("Mask and student logits must be on the same device.")
    if not valid_mask.any():
        raise ValueError("At least one valid response position is required.")

    selected = student_logits[valid_mask]
    gradient, diagnostics = signal(selected, teacher_logits[valid_mask], **kwargs)
    loss = (
        gradient * selected.to(gradient.dtype).log_softmax(-1)
    ).sum(-1).mean()
    return loss, diagnostics
