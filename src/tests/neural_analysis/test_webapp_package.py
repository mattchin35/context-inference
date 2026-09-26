"""Ownership and compatibility contracts for the neural-analysis webapp package."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from src.neural_analysis import psth_webapp as legacy_webapp


PACKAGE_ROOT = Path(__file__).parents[2] / "neural_analysis" / "webapp"


def _canonical_module(module_name: str):
    """Import one canonical webapp module by its short, unitless module name."""
    return importlib.import_module(f"src.neural_analysis.webapp.{module_name}")


def _assert_exact_owner(module_name: str, object_names: tuple[str, ...]) -> None:
    """Require legacy exports to be the exact objects defined by one canonical owner."""
    canonical = _canonical_module(module_name)
    for object_name in object_names:
        canonical_object = getattr(canonical, object_name)
        assert getattr(legacy_webapp, object_name) is canonical_object
        assert canonical_object.__module__ == canonical.__name__


def test_root_webapp_import_is_the_canonical_app_module() -> None:
    """The documented root import remains one live module with the canonical app owner."""
    app = _canonical_module("app")

    assert legacy_webapp is app
    assert app.parse_webapp_arguments.__module__ == app.__name__
    assert app.main.__module__ == app.__name__


def test_session_input_helpers_have_one_canonical_owner() -> None:
    """Version-2 session resolution and channel selection live in session_inputs."""
    _assert_exact_owner(
        "session_inputs",
        (
            "MetadataLFPSiteInputs",
            "MetadataViewAvailability",
            "load_webapp_session",
            "metadata_lfp_site_inputs",
            "metadata_view_availability",
            "resolve_region_channels_for_source",
        ),
    )


def test_compatibility_only_population_adapter_is_not_made_canonical() -> None:
    """The obsolete population-shaped adapter stays only at the compatibility entry point."""
    session_inputs = _canonical_module("session_inputs")
    app = _canonical_module("app")

    assert not hasattr(session_inputs, "MetadataPopulationInputs")
    assert not hasattr(session_inputs, "metadata_population_inputs")
    assert app.MetadataPopulationInputs is legacy_webapp.MetadataPopulationInputs
    assert app.metadata_population_inputs is legacy_webapp.metadata_population_inputs


def test_cached_loading_helpers_have_one_canonical_owner() -> None:
    """File loading and Streamlit-cached source access live in data_loading."""
    _assert_exact_owner(
        "data_loading",
        (
            "OpenEphysCacheToken",
            "build_open_ephys_cache_token",
            "load_viewer_data_cached",
            "load_phase_clustering_session_cached",
            "load_metadata_behavior_session_cached",
            "load_metadata_viewer_data_cached",
            "load_channel_quality_cached",
            "load_cluster_info_cached",
            "load_summary_cluster_metadata",
            "load_summary_channel_metadata",
            "decode_lfp_sync_cached",
            "load_trial_lfp_trace_cached",
            "load_trial_lfp_trace_for_format",
            "load_trial_lfp_trace_for_format_with_sample_rate",
        ),
    )


def test_lfp_view_helpers_have_one_canonical_owner() -> None:
    """LFP computations, controls, and rendered views live together in lfp_views."""
    _assert_exact_owner(
        "lfp_views",
        (
            "compute_trial_lfp_spectrogram_cached",
            "compute_shared_lfp_power_limits_cached",
            "compute_lfp_phase_site_cached",
            "compute_single_trial_relative_phase_cached",
            "render_single_trial_relative_phase_view",
            "render_lfp_phase_clustering_view",
            "build_lfp_dropdown_options",
            "resolve_lfp_filter_band",
        ),
    )


def test_spike_lfp_view_helpers_have_one_canonical_owner() -> None:
    """Spike-LFP cached calculations and rendered views share one direct owner."""
    _assert_exact_owner(
        "spike_lfp_views",
        (
            "compute_spike_lfp_phase_locking_cached",
            "compute_single_trial_spike_lfp_hilbert_cached",
            "render_spike_lfp_phase_locking_view",
            "render_single_trial_spike_lfp_hilbert_view",
        ),
    )


def test_population_view_helpers_have_one_canonical_owner() -> None:
    """Population display selection and cached PCA calculations live together."""
    _assert_exact_owner(
        "population_views",
        (
            "is_population_pca_display",
            "resolve_pca_decoding_component_minimum",
            "select_visible_concatenated_trial_indices",
            "select_concatenated_trial_viewport",
            "resolve_population_pca_unit_ids",
            "filter_trial_indices_for_valid_alignment",
            "compute_population_pca_cached",
            "compute_population_pca_decoding_cached",
            "compute_population_pca_switch_trajectories_cached",
        ),
    )


def test_unit_and_summary_helpers_have_direct_canonical_owners() -> None:
    """Unit presentation and summary composition are not retained in the app module."""
    _assert_exact_owner(
        "unit_views",
        (
            "build_trial_view_plot_save_path",
            "format_metadata_row_for_display",
        ),
    )
    _assert_exact_owner(
        "summary_view",
        (
            "render_metadata_lfp_summary_view",
            "render_lfp_summary_view",
        ),
    )


def test_webapp_modules_do_not_import_the_legacy_root_module() -> None:
    """Canonical modules form a shallow package without a cycle through psth_webapp."""
    expected_modules = {
        "app.py",
        "data_loading.py",
        "lfp_views.py",
        "population_views.py",
        "session_inputs.py",
        "spike_lfp_views.py",
        "summary_view.py",
        "unit_views.py",
    }
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} == expected_modules | {"__init__.py"}

    for path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "src.neural_analysis.psth_webapp" not in imported_modules


def test_root_module_keeps_direct_streamlit_dispatch() -> None:
    """Running the documented root module still delegates argv to the canonical main."""
    source = (PACKAGE_ROOT.parent / "psth_webapp.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(isinstance(child, ast.Name) and child.id == "__name__" for child in ast.walk(node.test))
        for node in tree.body
    )

