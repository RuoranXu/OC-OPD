"""Validated, immutable controller hyperparameters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

DEFAULT_H0 = 1.0
DEFAULT_EPSILON = 0.05
DEFAULT_TAU = 0.01
DEFAULT_VOCABULARY_MODE = "full"
DEFAULT_TEACHER_TOP_K = 16


@dataclass(frozen=True)
class OCOPDConfig:
    """Controller parameters and shared teacher-selected vocabulary support."""

    h0: float = DEFAULT_H0
    epsilon: float = DEFAULT_EPSILON
    tau: float = DEFAULT_TAU
    vocabulary_mode: str = DEFAULT_VOCABULARY_MODE
    teacher_top_k: int = DEFAULT_TEACHER_TOP_K

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0 for value in (self.h0, self.epsilon, self.tau)
        ):
            raise ValueError("h0, epsilon and tau must be finite and positive.")
        if self.vocabulary_mode not in {"teacher_topk", "full"}:
            raise ValueError("vocabulary_mode must be 'teacher_topk' or 'full'.")
        if (
            isinstance(self.teacher_top_k, bool)
            or not isinstance(self.teacher_top_k, int)
            or self.teacher_top_k < 2
        ):
            raise ValueError("teacher_top_k must be an integer of at least two.")

    def as_kwargs(self) -> dict[str, float | int | str]:
        """Return parameters accepted by signal, surrogate, and backward_chunked."""
        return asdict(self)
