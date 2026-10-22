"""Shared vocabulary support for conditional top-k or full-vocabulary losses."""

from __future__ import annotations

from torch import Tensor

from ..config import DEFAULT_TEACHER_TOP_K, DEFAULT_VOCABULARY_MODE, OCOPDConfig
from ..utils.validation import validate_logits


def select_support(
    student: Tensor,
    teacher: Tensor,
    *,
    vocabulary_mode: str = DEFAULT_VOCABULARY_MODE,
    teacher_top_k: int = DEFAULT_TEACHER_TOP_K,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Keep the full vocabulary by default; optionally select teacher top-k IDs.

    In top-k mode, subsequent softmax renormalizes BOTH models over that support.
    This defines distributions conditional on the same teacher-selected support.
    It is not full-vocabulary KL and does not preserve a full-vocabulary student
    denominator. Selection and teacher values are detached; student gather keeps
    gradients. Full mode, or k >= vocabulary size, keeps the original support.
    """
    validate_logits(student, teacher)
    OCOPDConfig(vocabulary_mode=vocabulary_mode, teacher_top_k=teacher_top_k)
    teacher = teacher.detach()
    if vocabulary_mode == "full" or teacher_top_k >= teacher.shape[-1]:
        return student, teacher, None
    values, indices = teacher.topk(teacher_top_k, dim=-1, sorted=False)
    return student.gather(-1, indices), values, indices
