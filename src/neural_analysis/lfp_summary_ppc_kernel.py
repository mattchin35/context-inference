"""Compatibility alias for :mod:`src.neural_analysis.spike_lfp.ppc_kernel`."""

from __future__ import annotations

import sys

from src.neural_analysis.spike_lfp import ppc_kernel as _canonical_module


sys.modules[__name__] = _canonical_module
