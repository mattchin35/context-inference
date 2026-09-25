"""Canonical spike-behavior ownership and compatibility contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_CANONICAL_MODULE_NAMES = (
    "src.neural_analysis.spike_behavior.loading",
    "src.neural_analysis.spike_behavior.trials",
    "src.neural_analysis.spike_behavior.binning",
    "src.neural_analysis.spike_behavior.decoding",
    "src.neural_analysis.spike_behavior.psth",
    "src.neural_analysis.spike_behavior.publication",
)

_MODULE_SURFACES = (
    (
        "src.neural_analysis.spike_behavior_pynapple",
        "src.neural_analysis.spike_behavior.loading",
        (
            "Session",
            "parse_session_id",
            "load_session_tables",
            "load_session_info",
            "load_sorter_metadata",
            "load_aligned_spikes",
            "validate_aligned_spike_inputs",
            "build_spike_tsgroup",
            "build_lick_time_dict",
            "normalize_region_channels",
            "select_units_by_channels",
        ),
    ),
    (
        "src.neural_analysis.unit_spike_loading",
        "src.neural_analysis.spike_behavior.loading",
        (
            "ProbeDataPaths",
            "list_path_browser_entries",
            "resolve_channel_quality_path",
            "load_channel_quality",
            "select_channels_from_quality",
            "infer_probe_derived_dir",
            "get_probe_label_for_region",
            "validate_probe_paths_for_spike_plot",
            "get_ct014_region_channel_presets",
            "parse_channel_list",
            "format_channel_list",
            "normalize_quality_labels",
            "filter_cluster_metadata",
            "build_session_from_inputs",
            "build_session_from_probe",
            "load_viewer_data_for_probe",
            "load_viewer_data",
            "get_unit_spike_times",
        ),
    ),
    (
        "src.neural_analysis.spike_behavior_pynapple",
        "src.neural_analysis.spike_behavior.trials",
        (
            "resolve_trial_end",
            "make_trial_type_masks",
            "summarize_trial_masks",
        ),
    ),
    (
        "src.neural_analysis.psth_behavior",
        "src.neural_analysis.spike_behavior.trials",
        (
            "select_valid_lick_peth_trials",
            "make_lick_peth_trial_type_masks",
        ),
    ),
    (
        "src.neural_analysis.spike_behavior_pynapple",
        "src.neural_analysis.spike_behavior.binning",
        (
            "build_bin_edges",
            "bin_spikes_to_trial_pynapple",
            "bin_licks_to_trial_pynapple",
            "bin_region_trials",
            "make_classifier_bins",
            "collect_condition_classifier_bins",
        ),
    ),
    (
        "src.neural_analysis.spike_behavior_pynapple",
        "src.neural_analysis.spike_behavior.decoding",
        (
            "get_decode_target",
            "cv_decodeability_score",
            "summarize_decoding_results",
            "train_single_decoder_with_shuffle_null",
            "evaluate_decoder_on_condition",
            "run_base_condition_decoding",
            "run_repeated_correct_rewarded_decoder",
        ),
    ),
    (
        "src.neural_analysis.psth_behavior",
        "src.neural_analysis.spike_behavior.psth",
        (
            "LickPethData",
            "SpikePethData",
            "select_first_valid_unit_cluster_id",
            "build_lick_peth_data",
            "build_single_unit_spike_peth_data",
        ),
    ),
    (
        "src.neural_analysis.spike_behavior_pynapple",
        "src.neural_analysis.spike_behavior.publication",
        (
            "normalize_region_name_for_filename",
            "make_region_analysis_filename",
            "build_state_decodability_session_table",
            "save_state_decodability_session_csv",
            "build_correct_rewarded_decoding_performance_session_table",
            "save_correct_rewarded_decoding_performance_session_csv",
        ),
    ),
)

_HELD_BACK_LEGACY_FUNCTIONS = {
    "src.neural_analysis.spike_behavior_pynapple": ("plot_trial_raster", "main"),
    "src.neural_analysis.psth_behavior": (
        "plot_lick_peth",
        "plot_single_unit_spike_peth",
        "save_figure_with_message",
        "plot_peth",
        "main",
    ),
}


def _imported_module_names(module: ModuleType) -> set[str]:
    """Return absolute module names in one canonical module's imports."""

    module_path = Path(module.__file__ or "")
    syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    return imported_names


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_spike_behavior_modules_are_importable(
    canonical_module_name: str,
) -> None:
    """Each approved spike-behavior responsibility must be importable."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, public_symbols",
    _MODULE_SURFACES,
)
def test_legacy_spike_behavior_paths_forward_exact_public_objects(
    legacy_module_name: str,
    canonical_module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Existing imports must expose the exact canonical records and functions."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    for symbol_name in public_symbols:
        assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)


def test_portable_probe_routing_constants_have_one_loading_owner() -> None:
    """Probe routing stays canonical without freezing local example paths."""

    legacy_loading = importlib.import_module("src.neural_analysis.unit_spike_loading")
    canonical_loading = importlib.import_module("src.neural_analysis.spike_behavior.loading")

    for constant_name in (
        "PROBE_LABEL_HPC_V1",
        "PROBE_LABEL_PFC",
        "SUPPORTED_PROBE_LABELS",
        "REGION_TO_PROBE_LABEL",
    ):
        assert getattr(legacy_loading, constant_name) is getattr(canonical_loading, constant_name)


def test_unit_loading_alias_preserves_the_exercised_sbp_monkeypatch_seam() -> None:
    """Patching ``unit_spike_loading.sbp`` must reach canonical loading state."""

    legacy_loading = importlib.import_module("src.neural_analysis.unit_spike_loading")
    canonical_loading = importlib.import_module("src.neural_analysis.spike_behavior.loading")

    assert legacy_loading is canonical_loading
    assert legacy_loading.sbp is canonical_loading


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_spike_behavior_modules_do_not_import_legacy_mixed_modules(
    canonical_module_name: str,
) -> None:
    """Canonical responsibility modules must own code rather than wrap old mixtures."""

    canonical_module = importlib.import_module(canonical_module_name)
    imported_names = _imported_module_names(canonical_module)

    assert imported_names.isdisjoint(
        {
            "src.neural_analysis.spike_behavior_pynapple",
            "src.neural_analysis.unit_spike_loading",
            "src.neural_analysis.psth_behavior",
        }
    )


def test_plotting_remains_implemented_only_at_legacy_paths() -> None:
    """R2B moves calculations and publication while R3 retains plotting."""

    for legacy_module_name, held_names in _HELD_BACK_LEGACY_FUNCTIONS.items():
        legacy_module = importlib.import_module(legacy_module_name)
        for held_name in held_names:
            assert getattr(legacy_module, held_name).__module__ == legacy_module_name

    for canonical_module_name in _CANONICAL_MODULE_NAMES:
        canonical_module = importlib.import_module(canonical_module_name)
        imported_names = _imported_module_names(canonical_module)
        assert "matplotlib.pyplot" not in imported_names
        for held_names in _HELD_BACK_LEGACY_FUNCTIONS.values():
            for held_name in held_names:
                assert not hasattr(canonical_module, held_name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.neural_analysis.spike_behavior.plotting")
