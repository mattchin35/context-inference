"""Compatibility alias for :mod:`src.neural_analysis.lfp_summary.spike_phase_validation`."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import spike_phase_validation as _canonical_module


sys.modules[__name__] = _canonical_module
