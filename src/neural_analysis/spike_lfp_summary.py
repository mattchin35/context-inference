"""Compatibility alias for :mod:`src.neural_analysis.spike_lfp.ppc`."""

from __future__ import annotations

import sys

from src.neural_analysis.spike_lfp import ppc as _canonical_module


sys.modules[__name__] = _canonical_module
