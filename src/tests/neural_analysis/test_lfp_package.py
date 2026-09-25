"""Canonical LFP package, compatibility, and dependency contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

from src.neural_analysis import lfp_summary_models


_MODULE_SURFACES = (
    (
        "src.neural_analysis.lfp_loading",
        "src.neural_analysis.lfp.loading",
        (
            "load_lfp_metadata",
            "load_open_ephys_lfp_metadata",
            "sha256_file_content",
            "validate_open_ephys_site_metadata",
            "validate_lfp_saved_channel",
            "read_open_ephys_lfp_channel_window",
            "decode_lfp_sync",
            "build_open_ephys_lfp_irig_df",
            "map_lfp_time_window_to_samples",
            "read_lfp_saved_channel_window",
            "filter_lfp_trace",
            "load_trial_lfp_trace",
            "load_trial_lfp_trace_with_sample_rate",
            "load_open_ephys_trial_lfp_trace",
        ),
    ),
    (
        "src.neural_analysis.lfp_power_summary",
        "src.neural_analysis.lfp.power",
        (
            "TrialEpochPSDResult",
            "compute_welch_psd",
            "compute_trial_epoch_psds",
            "interpolate_linear_psd_to_canonical_grid",
            "compute_session_reference_psd",
            "compute_presession_reference_psd",
            "normalize_psd_db",
            "mean_band_power_linear",
            "summarize_trial_median_iqr",
        ),
    ),
    (
        "src.neural_analysis.lfp_spectrogram",
        "src.neural_analysis.lfp.spectrogram",
        (
            "LFPSpectrogramResult",
            "compute_wavelet_padding_s",
            "compute_decimation_factor",
            "apply_optional_60_hz_notch",
            "compute_morlet_log_power",
            "select_reference_trial_indices",
            "estimate_shared_log_power_limits",
        ),
    ),
    (
        "src.neural_analysis.lfp_phase_clustering",
        "src.neural_analysis.lfp.phase",
        (
            "ContinuousProcessingBlock",
            "WaveletCoefficientResult",
            "WaveletPhaseResult",
            "PhaseTrialTensor",
            "PhaseClusteringResult",
            "SingleTrialRelativePhaseResult",
            "WithinTrialPLVResult",
            "normalize_wavelet_phase",
            "compute_wavelet_coefficients",
            "compute_wavelet_phase",
            "plan_continuous_processing_blocks",
            "make_phase_trial_tensor",
            "compute_single_trial_relative_phase",
            "make_relative_phase_display_mask",
            "make_relative_phase_support_mask",
            "compute_plv_window_samples",
            "compute_within_trial_plv",
            "compute_itpc",
            "combine_phase_trial_tensors",
            "select_tensor_trial_mask",
            "compute_ispc",
            "make_phase_condition_masks",
            "save_phase_clustering_result",
            "save_single_trial_relative_phase_result",
            "save_single_trial_phase_analysis_result",
            "compute_site_phase_trial_tensor",
        ),
    ),
    (
        "src.neural_analysis.lfp_synchrony_summary",
        "src.neural_analysis.lfp.synchrony",
        (
            "PhaseClusteringSummary",
            "TrialPLVSummary",
            "BootstrapBandMean",
            "PhaseBandBootstrapSummary",
            "PLVExemplarSelection",
            "compute_phase_clustering_summary",
            "compute_trial_plv_by_frequency",
            "aggregate_trial_plv_bands",
            "bootstrap_phase_clustering_bands",
            "bootstrap_band_mean",
            "select_plv_exemplars",
            "make_synchrony_cache_arrays",
        ),
    ),
)

_CANONICAL_MODULE_NAMES = (
    "src.neural_analysis.lfp.config",
    *(canonical_name for _, canonical_name, _ in _MODULE_SURFACES),
)


def _imported_module_names(module: ModuleType) -> set[str]:
    """Return absolute module names in one canonical module's import statements."""

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
    "_legacy_module_name, canonical_module_name, _public_symbols",
    _MODULE_SURFACES,
)
def test_canonical_lfp_modules_are_importable(
    _legacy_module_name: str,
    canonical_module_name: str,
    _public_symbols: tuple[str, ...],
) -> None:
    """Each approved LFP scientific owner must be directly importable."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, public_symbols",
    _MODULE_SURFACES,
)
def test_legacy_lfp_paths_forward_exact_public_objects(
    legacy_module_name: str,
    canonical_module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Legacy LFP names must alias, rather than copy, canonical objects."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    for symbol_name in public_symbols:
        assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)


def test_lfp_config_is_canonical_without_changing_json_or_fingerprints() -> None:
    """Workflow config aliases must retain default, JSON, and hash identity."""

    lfp_config = importlib.import_module("src.neural_analysis.lfp.config")

    assert lfp_summary_models.AnalysisWindowConfig is lfp_config.AnalysisWindowConfig
    assert lfp_summary_models.FrequencyBandConfig is lfp_config.FrequencyBandConfig
    assert lfp_summary_models.PowerAnalysisConfig is lfp_config.PowerAnalysisConfig

    default_config = lfp_summary_models.default_lfp_summary_config()
    assert type(default_config.analysis_windows) is lfp_config.AnalysisWindowConfig
    assert type(default_config.power) is lfp_config.PowerAnalysisConfig
    assert all(
        type(band) is lfp_config.FrequencyBandConfig
        for band in (*default_config.power.bands, *default_config.phase.bands)
    )

    encoded_config = lfp_summary_models.canonical_config_json(default_config)
    restored_config = lfp_summary_models.lfp_summary_config_from_json(encoded_config)
    assert type(restored_config.analysis_windows) is lfp_config.AnalysisWindowConfig
    assert type(restored_config.power) is lfp_config.PowerAnalysisConfig
    assert all(
        type(band) is lfp_config.FrequencyBandConfig
        for band in (*restored_config.power.bands, *restored_config.phase.bands)
    )
    assert lfp_summary_models.canonical_config_json(restored_config) == encoded_config
    for component_name in ("power", "synchrony", "spike_phase"):
        assert lfp_summary_models.component_fingerprint(
            component_name,
            restored_config,
        ) == lfp_summary_models.component_fingerprint(component_name, default_config)


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_lfp_modules_do_not_import_lfp_summary(
    canonical_module_name: str,
) -> None:
    """Reusable LFP modules must not depend on the LFP-summary workflow."""

    canonical_module = importlib.import_module(canonical_module_name)
    imported_names = _imported_module_names(canonical_module)

    assert not any(
        imported_name == "src.neural_analysis.lfp_summary"
        or imported_name.startswith("src.neural_analysis.lfp_summary_")
        for imported_name in imported_names
    )


def test_phase_plotting_has_one_domain_owner_and_legacy_forward() -> None:
    """R3 owns phase plotting without adding it to the numerical module."""

    legacy_phase = importlib.import_module("src.neural_analysis.lfp_phase_clustering")
    canonical_phase = importlib.import_module("src.neural_analysis.lfp.phase")
    canonical_plotting = importlib.import_module("src.neural_analysis.lfp.plotting")

    assert not hasattr(canonical_phase, "plot_phase_clustering")
    assert legacy_phase.plot_phase_clustering is canonical_plotting.plot_phase_clustering
