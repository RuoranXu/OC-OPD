"""Numerically safe operations for logit-space directions."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_direction(vector: Tensor, norm: Tensor | None = None) -> Tensor:
    """Normalize the vocabulary axis, mapping an exactly zero vector to zero."""
    if norm is None:
        norm = vector.norm(dim=-1, keepdim=True)
    return vector / norm.clamp_min(torch.finfo(vector.dtype).tiny)
