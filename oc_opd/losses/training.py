"""Token-summed objectives for globally normalized distributed training."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from oc_opd.baselines.kl import kl_terms
from oc_opd.controller import signal

from .frozen_signal import frozen_signal_loss

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
) -> tuple[Tensor, Tensor]:
    """Return loss sum and detached [RKL sum, FKL sum, valid-token count].

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

    def chunk_loss(z, p, labels, mask):
        # Keep the graph connected even on a rank with no valid response tokens.
        if not mask.any():
            return z.sum() * 0, z.new_zeros(3, dtype=torch.float32)
        z, labels = z[mask], labels[mask]
        count = mask.sum().float()
        if method == "sft":
            return F.cross_entropy(z.float(), labels, reduction="sum"), torch.stack(
                (count * 0, count * 0, count)
            )
        p = p[mask].detach()
        if method == "oc_opd":
            gradient, diagnostics = signal(z, p, **(controller or {}))
            loss = frozen_signal_loss(z, gradient, token_count=1)
            rkl = diagnostics["reverse_kl"].sum()
            fkl = diagnostics["forward_kl"].sum()
        else:
            reverse, forward = kl_terms(z, p)
            loss = reverse.sum() if method == "opd" else forward.sum()
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
        )
        value, diagnostic = checkpoint(chunk_loss, *args, use_reentrant=False)
        loss = loss + value
        stats = stats + diagnostic
    return loss, stats
