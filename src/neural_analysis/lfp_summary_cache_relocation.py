"""Compatibility command for canonical LFP-summary cache relocation."""

from __future__ import annotations

import sys

from src.neural_analysis.lfp_summary import cache_relocation as _canonical_module


if __name__ == "__main__":
    raise SystemExit(_canonical_module.main())
else:
    sys.modules[__name__] = _canonical_module
