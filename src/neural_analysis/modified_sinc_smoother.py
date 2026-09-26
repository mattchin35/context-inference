"""Compatibility entry for the retained modified-sinc prototype."""

from __future__ import annotations

import runpy
import sys


_LEGACY_MODULE = "src.neural_analysis.legacy.modified_sinc_smoother"

if __name__ == "__main__":
    runpy.run_module(_LEGACY_MODULE, run_name="__main__")
else:
    from src.neural_analysis.legacy import modified_sinc_smoother as _legacy_module

    sys.modules[__name__] = _legacy_module
