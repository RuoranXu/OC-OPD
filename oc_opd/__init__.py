"""OC-OPD core objective and memory-efficient backward integration."""

from .backward import backward_chunked
from .config import OCOPDConfig
from .controller import signal
from .losses import surrogate

__version__ = "0.1.0"
__all__ = ["OCOPDConfig", "signal", "surrogate", "backward_chunked"]
