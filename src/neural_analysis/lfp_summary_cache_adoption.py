"""Compatibility command for cluster-side LFP-summary cache adoption."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import cache_relocation as _canonical_module


if __name__ == "__main__":
    raise SystemExit(_canonical_module.adoption_main())
else:
    sys.modules[__name__] = _canonical_module
