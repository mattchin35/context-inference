"""Compatibility alias for :mod:`src.neural_analysis.population.pca`."""

from __future__ import annotations

import sys

from src.neural_analysis.population import pca as _canonical_module


sys.modules[__name__] = _canonical_module
