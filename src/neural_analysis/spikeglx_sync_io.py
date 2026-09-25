"""Compatibility alias for :mod:`src.neural_analysis.synchronization.spikeglx`."""

from __future__ import annotations

import sys

from src.neural_analysis.synchronization import spikeglx as _canonical_module


sys.modules[__name__] = _canonical_module
