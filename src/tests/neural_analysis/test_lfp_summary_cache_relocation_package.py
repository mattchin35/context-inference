"""Canonical ownership contracts for LFP-summary cache relocation."""

from __future__ import annotations

import importlib


LEGACY_MODULE = "src.neural_analysis.lfp_summary_cache_relocation"
CANONICAL_MODULE = "src.neural_analysis.lfp_summary.cache_relocation"


def test_cache_relocation_has_one_canonical_module() -> None:
    """Preserve every established import through one canonical module object."""
    legacy = importlib.import_module(LEGACY_MODULE)
    canonical = importlib.import_module(CANONICAL_MODULE)

    assert legacy is canonical
    for name in (
        "CacheRelocationRequest",
        "RelocationRuntime",
        "RelocationResult",
        "relocate_power_synchrony_cache",
        "main",
    ):
        value = getattr(canonical, name)
        assert getattr(legacy, name) is value
        assert value.__module__ == CANONICAL_MODULE
