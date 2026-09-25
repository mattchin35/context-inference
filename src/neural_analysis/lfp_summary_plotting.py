"""Compatibility alias for :mod:`src.neural_analysis.lfp_summary.plotting`."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import plotting as _canonical_module


sys.modules[__name__] = _canonical_module
