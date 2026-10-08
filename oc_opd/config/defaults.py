"""Validated, immutable controller hyperparameters."""

import math
from dataclasses import asdict, dataclass

DEFAULT_H0 = 1.0
DEFAULT_EPSILON = 0.05
DEFAULT_TAU = 0.01


@dataclass(frozen=True)
class OCOPDConfig:
    """Entropy scale, prior floor, and Hellinger gate threshold."""

    h0: float = DEFAULT_H0
    epsilon: float = DEFAULT_EPSILON
    tau: float = DEFAULT_TAU

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.h0, self.epsilon, self.tau)
        ):
            raise ValueError("h0, epsilon and tau must be finite and positive.")

    def as_kwargs(self) -> dict[str, float]:
        """Return parameters accepted by signal, surrogate, and backward_chunked."""
        return asdict(self)
