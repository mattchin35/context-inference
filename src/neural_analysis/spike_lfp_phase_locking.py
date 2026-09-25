"""Compatibility alias for :mod:`src.neural_analysis.spike_lfp.phase_locking`."""

from __future__ import annotations

import sys

from src.neural_analysis.spike_lfp import phase_locking as _canonical_module


sys.modules[__name__] = _canonical_module
