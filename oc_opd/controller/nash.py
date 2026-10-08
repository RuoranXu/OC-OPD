"""Closed-form Nash synthesis of normalized KL gradient directions."""

import torch
from torch import Tensor

from ..utils.tensors import normalize_direction


def nash_direction(
    reverse_direction: Tensor,
    forward_direction: Tensor,
    omega_reverse: Tensor,
    omega_forward: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the joint unit direction and the two input directions' cosine."""
    cosine = (
        reverse_direction * forward_direction
    ).sum(-1, keepdim=True).clamp(-1, 1)
    interaction = cosine * (omega_reverse - omega_forward)
    root = (interaction.square() + 4 * omega_reverse * omega_forward).sqrt()
    # The equivalent negative-interaction branch avoids subtractive cancellation.
    ratio = torch.where(
        interaction >= 0,
        (interaction + root) / (2 * omega_forward),
        2 * omega_reverse / (root - interaction),
    )
    direction = normalize_direction(ratio * reverse_direction + forward_direction)
    return direction, cosine
