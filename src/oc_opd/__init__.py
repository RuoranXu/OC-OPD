"""OC-OPD core objective and memory-efficient backward integration."""

from .objective import signal, surrogate
from .chunked import backward_chunked

__all__ = ["signal", "surrogate", "backward_chunked"]
