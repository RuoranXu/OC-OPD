"""KL baselines on shared teacher top-k or full-vocabulary distributions."""

import torch
from torch import Tensor

from oc_opd.config import DEFAULT_TEACHER_TOP_K, DEFAULT_VOCABULARY_MODE
from oc_opd.distributions import select_support


def kl_terms(
    student: Tensor,
    teacher: Tensor,
    *,
    vocabulary_mode: str = DEFAULT_VOCABULARY_MODE,
    teacher_top_k: int = DEFAULT_TEACHER_TOP_K,
) -> tuple[Tensor, Tensor]:
    """Return RKL/FKL, conditionally normalized on the same selected support."""
    student, teacher, _ = select_support(
        student, teacher, vocabulary_mode=vocabulary_mode, teacher_top_k=teacher_top_k
    )
    dtype = torch.float64 if torch.float64 in (student.dtype, teacher.dtype) else torch.float32
    log_q = torch.log_softmax(student.to(dtype), dim=-1)
    log_p = torch.log_softmax(teacher.to(dtype), dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(-1), (log_p.exp() * (log_p - log_q)).sum(-1)
