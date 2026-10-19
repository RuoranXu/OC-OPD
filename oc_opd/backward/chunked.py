"""Position-chunked OC backward, including trainable or tied output projections."""

import torch
from torch import Tensor, nn

from ..controller import signal
from ..distributions import select_support
from ..losses.frozen_signal import frozen_signal_loss
from ..losses.reduction import response_weights


def backward_chunked(
    hidden: Tensor,
    teacher_hidden: Tensor,
    head: nn.Module,
    teacher_head: nn.Module,
    valid_mask: Tensor,
    *,
    chunk_size: int = 128,
    reduction: str = "sequence_mean",
    **kwargs,
) -> dict[str, Tensor]:
    """Backpropagate OC-OPD in position chunks with equal response weights.

    Student and teacher prediction positions must align. Each hidden tensor must
    match its own head's device and dtype; token mappings must agree. The student
    head may be trainable, including weights tied to the backbone embedding.
    Gradients for head parameters accumulate alongside one hidden.backward call.
    No optimizer steps, mixed-precision scaling, or synchronization are performed.
    Defaults to teacher top-16; vocabulary_mode='full' selects the full objective.
    """
    if hidden.ndim < 2 or hidden.shape[:-1] != teacher_hidden.shape[:-1]:
        raise ValueError("Student and teacher prediction positions must align.")
    if valid_mask.dtype != torch.bool or valid_mask.shape != hidden.shape[:-1]:
        raise ValueError("A boolean position mask is required.")
    if valid_mask.device != hidden.device or not hidden.requires_grad:
        raise ValueError("Student hidden states require grad and a same-device mask.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if reduction not in {"sequence_mean", "token_mean"}:
        raise ValueError("reduction must be 'sequence_mean' or 'token_mean'.")

    count = int(valid_mask.sum())
    if count == 0:
        raise ValueError("At least one valid response position is required.")
    selected = hidden.detach()[valid_mask]
    teacher_selected = teacher_hidden.detach()[valid_mask.to(teacher_hidden.device)]
    selected_gradient = torch.zeros_like(selected)
    records: dict[str, list[Tensor]] = {}
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    weights = (
        response_weights(valid_mask, dtype=dtype)[valid_mask]
        if reduction == "sequence_mean"
        else torch.full((count,), 1.0 / count, device=hidden.device, dtype=dtype)
    )
    options = dict(kwargs)
    vocabulary = {
        key: options.pop(key) for key in ("vocabulary_mode", "teacher_top_k") if key in options
    }

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_selected[start:stop]).to(hidden.device)
        with torch.enable_grad():
            proxy = selected[start:stop].requires_grad_(True)
            logits = head(proxy)
            logits, teacher_logits, _ = select_support(logits, teacher_logits, **vocabulary)
            gradient, diagnostics = signal(
                logits, teacher_logits, vocabulary_mode="full", **options
            )
            loss = frozen_signal_loss(
                logits, gradient * weights[start:stop].unsqueeze(-1), token_count=1
            )
            derivatives = torch.autograd.grad(loss, (proxy, *head_parameters), allow_unused=True)
            selected_gradient[start:stop] = derivatives[0]
            # Explicitly accumulate the projection contribution. If weights are
            # tied, hidden.backward adds the independent backbone contribution.
            for parameter, derivative in zip(head_parameters, derivatives[1:], strict=True):
                if derivative is not None:
                    if parameter.grad is None:
                        parameter.grad = derivative.detach()
                    else:
                        parameter.grad.add_(derivative.detach())
        for key, value in diagnostics.items():
            records.setdefault(key, []).append(value.detach())

    hidden_gradient = torch.zeros_like(hidden)
    hidden_gradient[valid_mask] = selected_gradient
    hidden.backward(hidden_gradient)
    return {key: torch.cat(values, dim=0) for key, values in records.items()}
