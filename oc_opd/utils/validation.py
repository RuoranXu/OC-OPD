"""Alignment requirements for logits and response prediction masks."""

import torch
from torch import Tensor


def validate_logits(student_logits: Tensor, teacher_logits: Tensor) -> None:
    """Check shared shape, vocabulary axis, device, and floating-point inputs."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 1:
        raise ValueError("Logits must have identical (..., vocabulary) shapes.")
    if student_logits.shape[-1] < 2:
        raise ValueError("At least two vocabulary entries are required.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("Logits must be on the same device.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("Logits must be floating-point tensors.")


def validate_response_mask(positions: Tensor, valid_mask: Tensor) -> None:
    """Require a nonempty boolean mask over all non-feature dimensions."""
    if valid_mask.dtype != torch.bool or valid_mask.shape != positions.shape[:-1]:
        raise ValueError("Use a boolean mask matching all non-vocabulary dimensions.")
    if valid_mask.device != positions.device:
        raise ValueError("Mask and student logits must be on the same device.")
    if not valid_mask.any():
        raise ValueError("At least one valid response position is required.")
