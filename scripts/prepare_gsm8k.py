"""Compatibility entry point for examples/data_preprocess/gsm8k.py."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[1] / "examples" / "data_preprocess" / "gsm8k.py"
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
