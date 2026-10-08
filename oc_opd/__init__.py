"""OC-OPD core objective and memory-efficient backward integration."""

from .chunked import backward_chunked
from .objective import signal, surrogate

__version__ = "0.1.0"
__all__ = ["signal", "surrogate", "backward_chunked"]
