"""Compatibility entry point for the canonical manual synchronization module."""

from __future__ import annotations

import sys

from src.neural_analysis.synchronization import manual as _canonical_module


if __name__ == "__main__":
    _canonical_module.main()
else:
    sys.modules[__name__] = _canonical_module
