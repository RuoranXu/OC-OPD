"""Detached-signal losses for student optimization."""

from .frozen_signal import frozen_signal_loss
from .surrogate import surrogate

__all__ = ["frozen_signal_loss", "surrogate"]
