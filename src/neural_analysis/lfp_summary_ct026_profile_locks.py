"""Compatibility alias for the canonical CT026 profile locks."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import ct026_profile_locks as _canonical_module


sys.modules[__name__] = _canonical_module
