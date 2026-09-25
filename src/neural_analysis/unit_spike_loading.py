"""Compatibility alias for :mod:`src.neural_analysis.spike_behavior.loading`."""

from __future__ import annotations

import sys

from src.neural_analysis.spike_behavior import loading as _canonical_module


sys.modules[__name__] = _canonical_module
