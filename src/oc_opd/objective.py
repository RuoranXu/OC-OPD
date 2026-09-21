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

    Inputs have shape (..., vocabulary), identical token mappings and device.
    Diagnostics have shape (..., 1). Float16/bfloat16 inputs are promoted to
    float32; float64 inputs or float32 probability underflow use float64.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 1:
        raise ValueError("Logits must have identical (..., vocabulary) shapes.")
    if student_logits.shape[-1] < 2:
        raise ValueError("At least two vocabulary entries are required.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("Logits must be on the same device.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("Logits must be floating-point tensors.")
    if not all(math.isfinite(x) and x > 0 for x in (h0, epsilon, tau)):
        raise ValueError("h0, epsilon and tau must be finite and positive.")
    dtype = (torch.float64 if torch.float64 in
             (student_logits.dtype, teacher_logits.dtype) else torch.float32)
    lq = student_logits.to(dtype).log_softmax(-1)
    lp = teacher_logits.to(dtype).log_softmax(-1)
    if not (torch.isfinite(lq).all() and torch.isfinite(lp).all()):
        raise ValueError("Finite full-support logits are required.")
    q, p = lq.exp(), lp.exp()
    if (q == 0).any() or (p == 0).any():
        if dtype == torch.float32:
            del lq, lp, q, p
            return signal(student_logits.double(), teacher_logits.double(),
                          h0=h0, epsilon=epsilon, tau=tau)
        raise ValueError("Probabilities underflow even in float64.")

    reverse_kl = (q * (lq - lp)).sum(-1, keepdim=True)
    forward_kl = (p * (lp - lq)).sum(-1, keepdim=True)
    gr = q * (lq - lp - reverse_kl)
    gf = q - p
    tiny = torch.finfo(dtype).tiny
    nr = gr.norm(dim=-1, keepdim=True)
    nf = gf.norm(dim=-1, keepdim=True)
    er, ef = gr / nr.clamp_min(tiny), gf / nf.clamp_min(tiny)
    cosine = (er * ef).sum(-1, keepdim=True).clamp(-1, 1)

    hs = -(q * lq).sum(-1, keepdim=True)
    ht = -(p * lp).sum(-1, keepdim=True)
    hs, ht = hs / (hs + h0), ht / (ht + h0)
    ar, af = epsilon + (1 - ht) * hs, epsilon + ht
    wr, wf = ar / (ar + af), af / (ar + af)
    x = cosine * (wr - wf)
    root = (x.square() + 4 * wr * wf).sqrt()
    # Equivalent positive root, avoiding cancellation when x is negative.
    k = torch.where(x >= 0, (x + root) / (2 * wf), 2 * wr / (root - x))
    direction = k * er + ef
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(tiny)

    distance = 0.5 * (p.sqrt() - q.sqrt()).square().sum(-1, keepdim=True)
    gate = (distance / tau).clamp(0, 1)
    gradient = gate * nf * direction
    gradient = gradient - gradient.mean(-1, keepdim=True)
    return gradient, {
        "reverse_kl": reverse_kl, "forward_kl": forward_kl,
        "hellinger_squared": distance, "gate": gate,
        "omega_R": wr, "omega_F": wf, "cosine": cosine,
        "gain_R": (er * direction).sum(-1, keepdim=True),
        "gain_F": (ef * direction).sum(-1, keepdim=True),
    }


def surrogate(
    student_logits: Tensor, teacher_logits: Tensor, valid_mask: Tensor, **kwargs
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a token-mean frozen-signal surrogate and selected-token diagnostics.

    valid_mask selects response prediction positions, excluding prompts/padding.
    The scalar is a gradient carrier, not the value of a KL divergence.
    """
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
    loss = (gradient * selected.to(gradient.dtype).log_softmax(-1)).sum(-1).mean()
    return loss, diagnostics
