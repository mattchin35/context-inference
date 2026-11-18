"""Canonical synchronization-package and legacy-import contracts."""

from __future__ import annotations

import importlib

import pytest


_MODULE_SURFACES = (
    (
        "src.neural_analysis.ephys_sync_utils",
        "src.neural_analysis.synchronization.alignment",
        (
            "OpenEphysContinuousSampleBounds",
            "save_stream_sync_npz",
            "apply_utc_hour_offset",
            "write_alignment_note",
            "convert_utc_series_to_new_york",
            "decode_binary_file_irig_utc",
            "decode_sync_line_to_irig_utc",
            "find_signal_edges",
            "map_digital_rising_edges_to_utc",
            "map_sample_indices_to_utc",
            "map_spike_times_to_utc",
            "pulse_lengths_from_edges",
            "load_open_ephys_continuous_sample_bounds",
            "convert_kilosort_samples_to_open_ephys_samples",
            "warn_if_samples_extrapolated",
            "sync_open_ephys_kilosort_spikes_to_utc",
            "sync_imec_spikes_to_utc",
            "sync_ni_rising_edges_to_utc",
            "decode_ni_irig_debug",
        ),
    ),
    (
        "src.neural_analysis.spikeglx_sync_io",
        "src.neural_analysis.synchronization.spikeglx",
        ("read_digital_lines", "read_digital_line"),
    ),
    (
        "src.neural_analysis.manual_session_synchronization",
        "src.neural_analysis.synchronization.manual",
        (
            "FlipperUtcMapper",
            "load_behavior_flipper_timestamps",
            "match_timestamps_by_intervals",
            "make_timebase_mapper",
            "apply_bounds_policy",
            "build_flipper_utc_mapper",
            "map_ni_times_to_utc",
            "map_imec_samples_to_utc",
            "write_manual_alignment_note",
            "load_spike_times_npy",
            "load_spike_clusters_npy",
            "validate_spike_times_and_clusters",
            "convert_spike_times_to_imec_seconds",
            "map_imec_spikes_to_utc",
            "sync_imec_spikes_to_manual_utc",
            "build_imec_to_ni_mapper",
            "map_ni_digital_rising_edges_to_utc",
            "get_flipper_events",
            "decode_flipper_barcodes",
            "main",
        ),
    ),
)


@pytest.mark.parametrize(
    "_legacy_module_name, canonical_module_name, _public_symbols",
    _MODULE_SURFACES,
)
def test_canonical_synchronization_modules_are_importable(
    _legacy_module_name: str,
    canonical_module_name: str,
    _public_symbols: tuple[str, ...],
) -> None:
    """Each approved synchronization owner must be directly importable."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, public_symbols",
    _MODULE_SURFACES,
)
def test_legacy_synchronization_paths_forward_exact_public_objects(
    legacy_module_name: str,
    canonical_module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Legacy imports must alias, rather than copy, canonical callables."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    for symbol_name in public_symbols:
        assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)
