"""Token-distribution entropy in nats."""

from torch import Tensor


def entropy(probabilities: Tensor, log_probabilities: Tensor) -> Tensor:
    """Reduce the vocabulary axis and keep one entropy per prediction position."""
    return -(probabilities * log_probabilities).sum(-1, keepdim=True)
