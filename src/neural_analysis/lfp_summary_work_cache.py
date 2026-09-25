"""Compatibility alias for :mod:`src.neural_analysis.lfp_summary.work_cache`."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import work_cache as _canonical_module


sys.modules[__name__] = _canonical_module
