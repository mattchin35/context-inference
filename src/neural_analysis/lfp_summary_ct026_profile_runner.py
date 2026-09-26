"""Compatibility alias for the canonical CT026 profile runner."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import ct026_profile_runner as _canonical_module


sys.modules[__name__] = _canonical_module
