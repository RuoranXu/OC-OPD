"""Exact position-chunked gradient injection for frozen output projections."""

import torch
from torch import Tensor, nn

from ..controller import signal
from ..losses.frozen_signal import frozen_signal_loss


def backward_chunked(
    hidden: Tensor,
    teacher_hidden: Tensor,
    head: nn.Module,
    teacher_head: nn.Module,
    valid_mask: Tensor,
    *,
    chunk_size: int = 128,
    **kwargs,
) -> dict[str, Tensor]:
    """Backpropagate OC-OPD with full vocabulary computations in position chunks.

    Student and teacher prediction positions must align. Each hidden tensor must
    match its own head's device and dtype; token mappings must agree. The student
    head must be frozen. This calls hidden.backward once, without optimizer steps,
    mixed-precision loss scaling, or distributed synchronization.
    """
    if hidden.ndim < 2 or hidden.shape[:-1] != teacher_hidden.shape[:-1]:
        raise ValueError("Student and teacher prediction positions must align.")
    if valid_mask.dtype != torch.bool or valid_mask.shape != hidden.shape[:-1]:
        raise ValueError("A boolean position mask is required.")
    if valid_mask.device != hidden.device or not hidden.requires_grad:
        raise ValueError("Student hidden states require grad and a same-device mask.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("The student output projection must be frozen.")

    count = int(valid_mask.sum())
    if count == 0:
        raise ValueError("At least one valid response position is required.")
    selected = hidden.detach()[valid_mask]
    teacher_selected = teacher_hidden.detach()[valid_mask.to(teacher_hidden.device)]
    selected_gradient = torch.zeros_like(selected)
    records: dict[str, list[Tensor]] = {}

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_selected[start:stop]).to(hidden.device)
        with torch.enable_grad():
            proxy = selected[start:stop].requires_grad_(True)
            logits = head(proxy)
            gradient, diagnostics = signal(logits, teacher_logits, **kwargs)
            loss = frozen_signal_loss(logits, gradient, token_count=count)
            selected_gradient[start:stop] = torch.autograd.grad(loss, proxy)[0]
        for key, value in diagnostics.items():
            records.setdefault(key, []).append(value.detach())

    hidden_gradient = torch.zeros_like(hidden)
    hidden_gradient[valid_mask] = selected_gradient
    hidden.backward(hidden_gradient)
    return {key: torch.cat(values, dim=0) for key, values in records.items()}
