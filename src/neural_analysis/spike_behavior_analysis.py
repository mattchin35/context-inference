"""Compatibility entry for the retained exploratory spike-analysis script."""

from __future__ import annotations

import runpy
import sys


_LEGACY_MODULE = "src.neural_analysis.legacy.spike_behavior_analysis"

if __name__ == "__main__":
    runpy.run_module(_LEGACY_MODULE, run_name="__main__")
else:
    from src.neural_analysis.legacy import spike_behavior_analysis as _legacy_module

    sys.modules[__name__] = _legacy_module
