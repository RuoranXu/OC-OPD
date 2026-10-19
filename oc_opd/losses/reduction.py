"""Per-response normalization shared by direct and position-chunked losses."""

import torch
from torch import Tensor


def response_weights(mask: Tensor, *, dtype: torch.dtype, average: bool = True) -> Tensor:
    """Give equal weight to nonempty responses, averaging their valid positions.

    The last mask axis is response length; preceding axes index responses. A
    one-dimensional mask represents one response. Empty responses contribute
    zero and are excluded from the response denominator.
    """
    if mask.ndim < 1 or mask.dtype != torch.bool:
        raise ValueError("A boolean mask with a response-position axis is required.")
    counts = mask.sum(-1, keepdim=True)
    weights = mask.to(dtype) / counts.clamp_min(1).to(dtype)
    if average:
        weights = weights / (counts > 0).sum().clamp_min(1).to(dtype)
    return weights
