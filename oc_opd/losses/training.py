"""Response-normalized objectives for globally averaged distributed training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from oc_opd.baselines.kl import kl_terms
from oc_opd.controller import signal
from oc_opd.distributions import select_support

from .frozen_signal import frozen_signal_loss
from .reduction import response_weights

METHODS = {"oc_opd", "opd", "fkl", "kd", "sft"}


def response_loss_sum(
    student: Tensor,
    teacher: Tensor | None,
    responses: Tensor,
    response_mask: Tensor,
    *,
    method: str,
    chunk_size: int = 128,
    controller: dict | None = None,
    reduction: str = "sequence_mean",
) -> tuple[Tensor, Tensor]:
    """Return sum of response token-means, plus token-summed RKL/FKL/count.

    sequence_mean (default) requires the caller to divide by the GLOBAL number
    of nonempty responses. token_sum instead requires global valid-token count.
    Empty responses never change either denominator. Distillation defaults to
    conditional teacher top-16 distributions; controller.vocabulary_mode='full'
    selects full-vocabulary OC and KL baselines.
    Inputs are already shifted to predict response tokens. Chunk checkpointing
    bounds the additional FP32 vocabulary work, not the model's output logits.
    All methods use temperature 1; rollout temperature is a separate setting.
    """
    if method not in METHODS or chunk_size < 1:
        raise ValueError("Unknown method or non-positive loss chunk size.")
    if student.ndim != 3 or student.shape[:2] != responses.shape:
        raise ValueError("Expected [batch, response_length, vocabulary] logits.")
    if responses.shape != response_mask.shape:
        raise ValueError("Response mask must match response token IDs.")
    if method != "sft" and (teacher is None or teacher.shape != student.shape):
        raise ValueError("Distillation needs aligned full-vocabulary teacher logits.")
    if reduction not in {"sequence_mean", "token_sum"}:
        raise ValueError("reduction must be 'sequence_mean' or 'token_sum'.")
    mask = response_mask.bool()
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    weights = (
        response_weights(mask, dtype=dtype, average=False)
        if reduction == "sequence_mean"
        else mask.to(dtype)
    )
    options = dict(controller or {})
    vocabulary = {
        key: options.pop(key) for key in ("vocabulary_mode", "teacher_top_k") if key in options
    }

    def chunk_loss(z, p, labels, mask, position_weights):
        # Keep the graph connected even on a rank with no valid response tokens.
        if not mask.any():
            return z.sum() * 0, z.new_zeros(3, dtype=torch.float32)
        z, labels, position_weights = z[mask], labels[mask], position_weights[mask]
        count = mask.sum().float()
        if method == "sft":
            return (
                F.cross_entropy(z.to(dtype), labels, reduction="none") * position_weights
            ).sum(), torch.stack((count * 0, count * 0, count))
        z, p, _ = select_support(z, p[mask], **vocabulary)
        if method == "oc_opd":
            gradient, diagnostics = signal(z, p, vocabulary_mode="full", **options)
            loss = frozen_signal_loss(z, gradient * position_weights.unsqueeze(-1), token_count=1)
            rkl = diagnostics["reverse_kl"].sum()
            fkl = diagnostics["forward_kl"].sum()
        else:
            reverse, forward = kl_terms(z, p, vocabulary_mode="full")
            loss = ((reverse if method == "opd" else forward) * position_weights).sum()
            rkl, fkl = reverse.detach().sum(), forward.detach().sum()
        return loss, torch.stack((rkl, fkl, count)).detach()

    loss = student.new_zeros((), dtype=torch.float32)
    stats = student.new_zeros(3, dtype=torch.float32)
    # Slice along positions to avoid making a full FP32 copy of the vocabulary.
    for start in range(0, responses.shape[1], chunk_size):
        stop = start + chunk_size
        args = (
            student[:, start:stop],
            teacher[:, start:stop] if teacher is not None else None,
            responses[:, start:stop],
            response_mask[:, start:stop].bool(),
            weights[:, start:stop],
        )
        value, diagnostic = checkpoint(chunk_loss, *args, use_reentrant=False)
        loss = loss + value
        stats = stats + diagnostic
    return loss, stats
