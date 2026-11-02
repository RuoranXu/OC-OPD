"""Gradient-carrying surrogate with explicit response-token normalization."""

from __future__ import annotations

from torch import Tensor


def frozen_signal_loss(
    logits: Tensor, gradient: Tensor, *, token_count: int | None = None
) -> Tensor:
    """Inject a detached logit signal through log_softmax.

    Without token_count, average over the supplied prediction positions. For
    position chunks, pass the entire response-token count as the denominator.
    The returned scalar carries gradients; it is not a KL divergence value.
    """
    if logits.ndim < 1 or logits.shape != gradient.shape:
        raise ValueError("Logits and signal must have identical (..., vocabulary) shapes.")
    if not logits.is_floating_point() or not gradient.is_floating_point():
        raise ValueError("Logits and signal must be floating-point tensors.")
    if logits.device != gradient.device:
        raise ValueError("Logits and signal must be on the same device.")
    if logits.numel() == 0:
        raise ValueError("At least one prediction position and vocabulary entry are required.")
    if token_count is not None and (
        isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1
    ):
        raise ValueError("token_count must be a positive integer.")
    contributions = gradient.detach() * logits.to(gradient.dtype).log_softmax(-1)
    if token_count is None:
        return contributions.sum(-1).mean()
    return contributions.sum() / token_count
