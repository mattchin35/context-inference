"""Canonical LFP-summary runtime ownership and compatibility contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_RUNTIME_SURFACES = (
    (
        "src.neural_analysis.lfp_summary.runtime_common",
        (
            "PreparedPowerRun",
            "PreparedPhaseRun",
            "PreparedSpikeRun",
            "load_configured_trial_table",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary.power_runtime",
        (
            "prepare_power_run",
            "build_power_payload",
            "make_power_pipeline_dependencies",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary.synchrony_runtime",
        (
            "prepare_phase_run",
            "build_synchrony_payload",
            "make_synchrony_pipeline_dependencies",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary.spike_phase_runtime",
        (
            "load_configured_unit_spikes",
            "prepare_spike_run",
            "build_spike_phase_payload",
            "make_spike_phase_pipeline_dependencies",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary.runtime",
        ("make_lfp_summary_pipeline_dependencies",),
    ),
)


def _imported_module_names(module: ModuleType) -> set[str]:
    """Return absolute names used by one module's import statements."""

    module_path = Path(module.__file__ or "")
    syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    return imported_names


@pytest.mark.parametrize("module_name, _public_symbols", _RUNTIME_SURFACES)
def test_component_runtime_modules_are_importable(
    module_name: str,
    _public_symbols: tuple[str, ...],
) -> None:
    """Each approved runtime owner must be directly importable."""

    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


@pytest.mark.parametrize("module_name, public_symbols", _RUNTIME_SURFACES)
def test_runtime_callables_have_their_canonical_owner(
    module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Prepared records and component entry points have one inspectable owner."""

    module = importlib.import_module(module_name)

    for symbol_name in public_symbols:
        assert getattr(module, symbol_name).__module__ == module_name


def test_legacy_runtime_aliases_the_composed_canonical_facade() -> None:
    """The old module preserves public and tested private object identity."""

    legacy = importlib.import_module("src.neural_analysis.lfp_summary_runtime")
    facade = importlib.import_module("src.neural_analysis.lfp_summary.runtime")

    assert legacy is facade
    for owner_name, symbol_names in _RUNTIME_SURFACES:
        owner = importlib.import_module(owner_name)
        for symbol_name in symbol_names:
            assert getattr(facade, symbol_name) is getattr(owner, symbol_name)


@pytest.mark.parametrize(
    "module_name",
    (
        "src.neural_analysis.lfp_summary.runtime_common",
        "src.neural_analysis.lfp_summary.power_runtime",
        "src.neural_analysis.lfp_summary.synchrony_runtime",
        "src.neural_analysis.lfp_summary.spike_phase_runtime",
    ),
)
def test_component_runtimes_do_not_back_import_the_facade(module_name: str) -> None:
    """Component owners remain usable without the composed compatibility surface."""

    imported_names = _imported_module_names(importlib.import_module(module_name))

    assert "src.neural_analysis.lfp_summary_runtime" not in imported_names
    assert "src.neural_analysis.lfp_summary.runtime" not in imported_names


def test_ppc_runtime_uses_shared_validation_without_runtime_back_import() -> None:
    """PPC execution depends on common validation, not the composed runtime."""

    ppc_runtime = importlib.import_module("src.neural_analysis.lfp_summary_ppc_runtime")
    imported_names = _imported_module_names(ppc_runtime)

    assert "src.neural_analysis.lfp_summary.runtime_common" in imported_names
    assert "src.neural_analysis.lfp_summary_runtime" not in imported_names
    assert "src.neural_analysis.lfp_summary.runtime" not in imported_names


def test_shared_ppc_validation_objects_are_exact_facade_forwards() -> None:
    """Existing private imports keep identity while PPC uses the cycle-free owner."""

    common = importlib.import_module("src.neural_analysis.lfp_summary.runtime_common")
    facade = importlib.import_module("src.neural_analysis.lfp_summary.runtime")

    for symbol_name in (
        "_validate_prepared_phase_run",
        "_validate_prepared_spike_run",
        "_analysis_condition_membership",
    ):
        assert getattr(facade, symbol_name) is getattr(common, symbol_name)
