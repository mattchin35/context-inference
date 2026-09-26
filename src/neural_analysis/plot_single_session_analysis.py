"""Compatibility entry for the retained empty plotting placeholder."""

from __future__ import annotations

import runpy
import sys


_LEGACY_MODULE = "src.neural_analysis.legacy.plot_single_session_analysis"

if __name__ == "__main__":
    runpy.run_module(_LEGACY_MODULE, run_name="__main__")
else:
    from src.neural_analysis.legacy import plot_single_session_analysis as _legacy_module

    sys.modules[__name__] = _legacy_module
