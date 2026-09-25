"""Compatibility alias for :mod:`src.neural_analysis.lfp_summary.models`."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import models as _canonical_module


sys.modules[__name__] = _canonical_module
