"""Canonical LFP-summary validation, plotting, and snapshot contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_MOVED_MODULES = (
    (
        "src.neural_analysis.lfp_power_validation",
        "src.neural_analysis.lfp_summary.power_validation",
        (
            "PowerValidationDependencies",
            "PowerValidationResult",
            "build_ct026_power_config",
            "run_power_validation",
            "render_cached_power_validation",
            "make_production_power_validation_dependencies",
        ),
    ),
    (
        "src.neural_analysis.lfp_synchrony_validation",
        "src.neural_analysis.lfp_summary.synchrony_validation",
        (
            "SynchronyValidationDependencies",
            "SynchronyValidationResult",
            "build_ct026_synchrony_config",
            "run_synchrony_validation",
            "render_cached_synchrony_validation",
            "make_production_synchrony_validation_dependencies",
        ),
    ),
    (
        "src.neural_analysis.lfp_spike_phase_validation",
        "src.neural_analysis.lfp_summary.spike_phase_validation",
        (
            "SpikePhasePreviewValidationDependencies",
            "SpikePhaseFilterBenchmark",
            "SpikePhaseReportMeasurements",
            "SpikePhasePreviewValidationResult",
            "run_spike_phase_preview_validation",
            "render_cached_spike_phase_preview_validation",
            "render_cached_spike_phase_report",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_plotting",
        "src.neural_analysis.lfp_summary.plotting",
        (
            "PlotContext",
            "PPCExemplarPanel",
            "plot_condition_psd",
            "plot_band_power_summary",
            "plot_phase_map",
            "plot_phase_band_summary",
            "plot_plv_distribution",
            "plot_plv_exemplar",
            "plot_unit_ppc_map",
            "plot_population_ppc_maps",
            "plot_ppc_band_summary",
            "plot_ppc_exemplar",
            "plot_ppc_exemplar_pair",
            "build_summary_figure_filename",
        ),
    ),
)

_SNAPSHOT_PUBLIC = (
    "SnapshotInspection",
    "SummarySource",
    "SnapshotPopulationStatus",
    "SnapshotPlotSelection",
    "SpikeSnapshotSlice",
    "SnapshotComponentCache",
    "validate_cache_snapshot",
    "resolve_summary_source",
    "snapshot_component_population_status",
    "snapshot_component_source_value_semantics",
    "select_spike_snapshot_slice",
    "plot_cached_snapshot_component",
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


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, public_symbols",
    _MOVED_MODULES,
)
def test_reporting_modules_have_exact_canonical_owners_and_legacy_aliases(
    legacy_module_name: str,
    canonical_module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Moved validation and plotting modules retain one live module state."""

    legacy = importlib.import_module(legacy_module_name)
    canonical = importlib.import_module(canonical_module_name)

    assert legacy is canonical
    for symbol_name in public_symbols:
        symbol = getattr(canonical, symbol_name)
        assert getattr(legacy, symbol_name) is symbol
        assert symbol.__module__ == canonical_module_name


def test_snapshot_reader_has_one_non_streamlit_owner_and_webapp_forwards() -> None:
    """Immutable snapshot inspection is independent of Streamlit composition."""

    snapshot = importlib.import_module("src.neural_analysis.lfp_summary.snapshot")
    webapp = importlib.import_module("src.neural_analysis.lfp_summary_webapp")

    for symbol_name in _SNAPSHOT_PUBLIC:
        symbol = getattr(snapshot, symbol_name)
        assert getattr(webapp, symbol_name) is symbol
        assert symbol.__module__ == snapshot.__name__


@pytest.mark.parametrize(
    "module_name",
    (
        "src.neural_analysis.lfp_summary.snapshot",
        "src.neural_analysis.lfp_summary.plotting",
    ),
)
def test_snapshot_and_plotting_do_not_import_streamlit(module_name: str) -> None:
    """Cache inspection and plotting remain usable outside the web process."""

    imported_names = _imported_module_names(importlib.import_module(module_name))

    assert "streamlit" not in imported_names


@pytest.mark.parametrize(
    "module_name, helper_name, validation_filename, has_pipeline",
    (
        (
            "src.neural_analysis.lfp_summary.power_validation",
            "_production_source_identifiers",
            "lfp_power_validation.py",
            True,
        ),
        (
            "src.neural_analysis.lfp_summary.synchrony_validation",
            "_production_source_identifiers",
            "lfp_synchrony_validation.py",
            True,
        ),
        (
            "src.neural_analysis.lfp_summary.spike_phase_validation",
            "_source_identifiers",
            "lfp_spike_phase_validation.py",
            False,
        ),
    ),
)
def test_validation_source_identifiers_preserve_legacy_module_paths(
    module_name: str,
    helper_name: str,
    validation_filename: str,
    has_pipeline: bool,
) -> None:
    """Report provenance keeps historical root module path identities."""

    module = importlib.import_module(module_name)
    models = importlib.import_module("src.neural_analysis.lfp_summary.models")

    identifiers = getattr(module, helper_name)(models.default_lfp_summary_config())

    assert Path(identifiers["runtime_module"]).name == "lfp_summary_runtime.py"
    assert Path(identifiers["validation_module"]).name == validation_filename
    if has_pipeline:
        assert Path(identifiers["pipeline_module"]).name == "lfp_summary_pipeline.py"
