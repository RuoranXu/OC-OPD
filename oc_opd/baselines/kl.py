"""Exact full-vocabulary KL objectives; teacher logits are always detached."""

import torch
from torch import Tensor


def kl_terms(student: Tensor, teacher: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-token (D_KL(student || teacher), D_KL(teacher || student))."""
    if student.shape != teacher.shape:
        raise ValueError("Full-vocabulary KL requires identical, aligned logit shapes.")
    log_q = torch.log_softmax(student.float(), dim=-1)
    log_p = torch.log_softmax(teacher.detach().float(), dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(-1), (log_p.exp() * (log_p - log_q)).sum(-1)
