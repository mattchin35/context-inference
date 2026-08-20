from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis import lfp_phase_clustering, psth_webapp


def test_build_lfp_dropdown_options_uses_only_explicit_probe_paths():
    """LFP dropdown choices should come directly from user-entered probe path fields."""
    options = psth_webapp.build_lfp_dropdown_options(
        hpc_v1_lfp_path="/data/hpc_v1/run0_g0_t0.imec1.lf.bin",
        pfc_lfp_path="/data/pfc/run0_g0_t0.imec0.lf.bin",
    )

    assert list(options.keys()) == ["HPC/V1 LFP", "PFC LFP"]
    assert options["HPC/V1 LFP"] == "/data/hpc_v1/run0_g0_t0.imec1.lf.bin"
    assert options["PFC LFP"] == "/data/pfc/run0_g0_t0.imec0.lf.bin"


def test_build_lfp_dropdown_options_preserves_empty_paths():
    """Empty LFP fields should stay selectable and be handled by later validation."""
    options = psth_webapp.build_lfp_dropdown_options(
        hpc_v1_lfp_path="",
        pfc_lfp_path="/data/pfc/run0_g0_t0.imec0.lf.bin",
    )

    assert options["HPC/V1 LFP"] == ""
    assert options["PFC LFP"] == "/data/pfc/run0_g0_t0.imec0.lf.bin"


def test_resolve_lfp_filter_band_maps_display_labels_to_cutoffs():
    """Displayed LFP filter labels should resolve to the intended frequency bands."""
    assert psth_webapp.resolve_lfp_filter_band("Default") is None
    assert psth_webapp.resolve_lfp_filter_band("Theta (5-10 Hz)") == (5.0, 10.0)
    assert psth_webapp.resolve_lfp_filter_band("Gamma (50-70 Hz)") == (50.0, 70.0)


def test_build_trial_view_plot_save_path_includes_plot_settings(tmp_path: Path):
    """Trial-view saved plots should be distinguished by the settings that define them."""
    save_path = psth_webapp.build_trial_view_plot_save_path(
        figure_path=tmp_path,
        session_id="CT014_2025-12-23_163505",
        region_name="HPC",
        trial_index=12,
        condition="correct_rewarded",
        action_label="left (1)",
        alignment_event="choice_time",
        window=(-2.0, 1.5),
        unit_page_index=3,
        population_psth_unit_scope="All selected units",
        population_psth_bin_size=0.05,
        lfp_label="HPC/V1 LFP, saved channel 12, Theta (5-10 Hz)",
        lfp_utc_offset_hours=1,
    )

    assert save_path.parent == tmp_path / "unit_spike_viewer"
    assert save_path.suffix == ".png"
    filename = save_path.name
    for expected_token in (
        "CT014_2025-12-23_163505",
        "HPC",
        "trial12",
        "correct_rewarded",
        "left-1",
        "choice_time",
        "window-2.0-to-1.5s",
        "unitpage3",
        "All-selected-units",
        "popbin0.05s",
        "HPC-V1-LFP-saved-channel-12-Theta-5-10-Hz",
        "lfputc1h",
    ):
        assert expected_token in filename


def test_format_metadata_row_for_display_keeps_mixed_values_arrow_safe():
    """Mixed numeric/string unit metadata should be shown as strings for Streamlit."""
    metadata_row = pd.Series(
        {
            "cluster_id": np.int64(12),
            "depth": 250.5,
            "group": "mua",
            "missing": np.nan,
        }
    )

    display_df = psth_webapp.format_metadata_row_for_display(metadata_row)

    assert display_df.index.tolist() == ["cluster_id", "depth", "group", "missing"]
    assert display_df.columns.tolist() == ["value"]
    assert display_df["value"].tolist() == ["12", "250.5", "mua", ""]
    assert all(isinstance(value, str) for value in display_df["value"].tolist())


def test_is_population_pca_display_includes_concatenated_mode():
    """Both PCA display modes should route through PCA computation and plotting."""
    assert psth_webapp.is_population_pca_display(psth_webapp.NEURAL_DISPLAY_POPULATION_PCA)
    assert psth_webapp.is_population_pca_display(psth_webapp.NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED)
    assert not psth_webapp.is_population_pca_display(psth_webapp.NEURAL_DISPLAY_SPIKE_RASTER)


def test_webapp_exposes_single_trial_lfp_spectrogram_defaults():
    """The dedicated LFP view should expose the approved scientific defaults."""
    assert psth_webapp.NEURAL_DISPLAY_LFP_SPECTROGRAM in psth_webapp.NEURAL_DISPLAY_OPTIONS
    assert psth_webapp.LFP_SPECTROGRAM_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.LFP_SPECTROGRAM_MAX_FREQUENCY_HZ == 80.0
    assert psth_webapp.LFP_SPECTROGRAM_FREQUENCY_COUNT == 40
    assert psth_webapp.LFP_SPECTROGRAM_REFERENCE_TRIAL_COUNT == 24
    assert psth_webapp.LFP_SPECTROGRAM_COLOR_PERCENTILES == (2.0, 98.0)
    assert psth_webapp.LFP_SPECTROGRAM_NOTCH_DEFAULT is False


def test_webapp_exposes_separate_lfp_phase_clustering_view_defaults():
    """ITPC/ISPC should use a dedicated view with the approved initial settings."""
    assert psth_webapp.PLOT_VIEW_LFP_PHASE_CLUSTERING in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.LFP_PHASE_CLUSTERING_ANALYSIS_OPTIONS == ("ITPC", "ISPC")
    assert psth_webapp.LFP_PHASE_CLUSTERING_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.LFP_PHASE_CLUSTERING_MAX_FREQUENCY_HZ == 100.0
    assert psth_webapp.LFP_PHASE_CLUSTERING_FREQUENCY_COUNT == 50
    assert psth_webapp.LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ == 500.0
    assert psth_webapp.LFP_PHASE_CLUSTERING_DEFAULT_WINDOW == (-1.0, 2.0)


def test_webapp_exposes_single_trial_relative_phase_view_defaults():
    """Relative phase should be a lightweight top-level single-trial view."""
    assert psth_webapp.PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.RELATIVE_PHASE_DEFAULT_WINDOW == (-1.0, 2.0)
    assert psth_webapp.RELATIVE_PHASE_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.RELATIVE_PHASE_MAX_FREQUENCY_HZ == 100.0
    assert psth_webapp.RELATIVE_PHASE_FREQUENCY_COUNT == 50
    assert psth_webapp.RELATIVE_PHASE_OUTPUT_SAMPLE_RATE_HZ == 500.0
    assert psth_webapp.RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS == (
        "Off",
        "Per-frequency percentile",
        "Absolute magnitude",
    )


def test_single_trial_relative_phase_cache_loads_only_two_padded_trial_segments(monkeypatch):
    """The lightweight viewer must not load or construct a full-session trial tensor."""
    load_calls = []

    def fake_load_trial_lfp_trace_for_format_with_sample_rate(**kwargs):
        """Record requested relative bounds and return one synthetic source-rate segment."""
        load_calls.append((kwargs["lfp_path"], kwargs["window_start_s"], kwargs["window_end_s"]))
        relative_time_s = np.arange(kwargs["window_start_s"], kwargs["window_end_s"], 0.002)
        phase_offset = 0.4 if kwargs["saved_channel_index"] == 1 else 0.0
        values = np.sin(2.0 * np.pi * 10.0 * relative_time_s + phase_offset)
        return relative_time_s, values, 500.0

    def fake_compute_wavelet_coefficients(time_s, lfp_values, sample_rate_hz, frequencies_hz, **_kwargs):
        """Return deterministic coefficients on the loaded absolute sample grid."""
        times = np.asarray(time_s, dtype=float)
        frequencies = np.asarray(frequencies_hz, dtype=float)
        coefficients = np.stack(
            [np.exp(1j * 2.0 * np.pi * frequency * times) for frequency in frequencies]
        ).astype(np.complex64)
        return lfp_phase_clustering.WaveletCoefficientResult(
            time_s=times,
            frequencies_hz=frequencies,
            coefficients=coefficients,
            sample_rate_hz=float(sample_rate_hz),
            source_time_s=times,
            source_lfp_values=np.asarray(lfp_values),
            source_sample_rate_hz=float(sample_rate_hz),
        )

    monkeypatch.setattr(
        psth_webapp,
        "load_trial_lfp_trace_for_format_with_sample_rate",
        fake_load_trial_lfp_trace_for_format_with_sample_rate,
    )
    monkeypatch.setattr(
        lfp_phase_clustering,
        "compute_wavelet_coefficients",
        fake_compute_wavelet_coefficients,
    )
    psth_webapp.compute_single_trial_relative_phase_cached.clear()

    result = psth_webapp.compute_single_trial_relative_phase_cached(
        lfp_format=psth_webapp.LFP_FORMAT_SPIKEGLX,
        lfp_path_a="a.lf.bin",
        lfp_mtime_ns_a=1,
        channel_a=1,
        site_a_label="A",
        aligned_sync_path_a=None,
        aligned_sync_mtime_ns_a=-1,
        lfp_path_b="b.lf.bin",
        lfp_mtime_ns_b=2,
        channel_b=2,
        site_b_label="B",
        aligned_sync_path_b=None,
        aligned_sync_mtime_ns_b=-1,
        trial_index=3,
        event_time_s=100.0,
        window_start_s=-1.0,
        window_end_s=2.0,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        frequencies_hz=(2.0, 10.0),
        gaussian_width=1.5,
        wavelet_window_length=1.0,
        precision=16,
        norm="l1",
        output_sample_rate_hz=500.0,
        notch_60_hz=False,
        notch_quality_factor=30.0,
        minimum_relative_magnitude=1e-12,
    )

    assert result.phase_angle_rad.shape == (2, 1500)
    assert load_calls == [("a.lf.bin", -5.0, 6.0), ("b.lf.bin", -5.0, 6.0)]


def test_pca_decoding_has_separate_plot_view_from_trial_filters():
    """PCA decoding should be a separate performance-only view, not a trial display mode."""
    assert psth_webapp.PLOT_VIEW_PCA_DECODING in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.PLOT_VIEW_PCA_DECODING != "Trial spikes/licks/choices"
    assert psth_webapp.PCA_DECODING_DEFAULT_COMPONENT_COUNT == 5
    assert psth_webapp.PCA_DECODING_DISPLAY_PERFORMANCE in psth_webapp.PCA_DECODING_DISPLAY_OPTIONS
    assert psth_webapp.PCA_DECODING_DISPLAY_AVERAGE_PC in psth_webapp.PCA_DECODING_DISPLAY_OPTIONS
    assert psth_webapp.PCA_DECODING_SHOW_RAW_PC_SCORES_DEFAULT is False


def test_pca_switch_trajectories_have_a_separate_plot_view():
    """Choice-switch trajectories should not reuse decoding or trial-view controls."""
    assert psth_webapp.PLOT_VIEW_PCA_SWITCH_TRAJECTORIES in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.PLOT_VIEW_PCA_SWITCH_TRAJECTORIES != psth_webapp.PLOT_VIEW_PCA_DECODING


def test_resolve_pca_decoding_component_minimum_requires_two_for_pc_score_plot():
    """Average PC score plots need PC1 and PC2 while performance decoding can use one PC."""
    assert (
        psth_webapp.resolve_pca_decoding_component_minimum(
            psth_webapp.PCA_DECODING_DISPLAY_AVERAGE_PC
        )
        == 2
    )
    assert (
        psth_webapp.resolve_pca_decoding_component_minimum(
            psth_webapp.PCA_DECODING_DISPLAY_PERFORMANCE
        )
        == 1
    )


def test_select_visible_concatenated_trial_indices_limits_page_size():
    """Concatenated PCA should display a bounded page of selected trials."""
    trial_indices = np.arange(25, dtype=int)

    visible_trial_indices, page_count = psth_webapp.select_visible_concatenated_trial_indices(
        trial_indices=trial_indices,
        page_index=1,
        visible_trial_count=10,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(10, 20, dtype=int))
    assert page_count == 3


def test_select_visible_concatenated_trial_indices_clamps_oversized_page():
    """Out-of-range pages should resolve to the last available visible trial page."""
    trial_indices = np.arange(23, dtype=int)

    visible_trial_indices, page_count = psth_webapp.select_visible_concatenated_trial_indices(
        trial_indices=trial_indices,
        page_index=999,
        visible_trial_count=10,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(20, 23, dtype=int))
    assert page_count == 3


def test_select_concatenated_trial_viewport_uses_start_position():
    """The fixed concatenated PCA viewport should slide through the selected trial list."""
    trial_indices = np.arange(30, dtype=int)

    visible_trial_indices, clamped_start_position = psth_webapp.select_concatenated_trial_viewport(
        trial_indices=trial_indices,
        start_position=12,
        visible_trial_count=8,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(12, 20, dtype=int))
    assert clamped_start_position == 12


def test_select_concatenated_trial_viewport_clamps_to_last_full_window():
    """Oversized start positions should keep the viewport filled when enough trials exist."""
    trial_indices = np.arange(30, dtype=int)

    visible_trial_indices, clamped_start_position = psth_webapp.select_concatenated_trial_viewport(
        trial_indices=trial_indices,
        start_position=999,
        visible_trial_count=8,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(22, 30, dtype=int))
    assert clamped_start_position == 22


def test_concatenated_pca_defaults_are_compact_for_fixed_viewport():
    """The default concatenated PCA view should fit as a compact on-screen viewport."""
    assert psth_webapp.DEFAULT_CONCATENATED_PCA_VISIBLE_TRIAL_COUNT == 20
    assert psth_webapp.DEFAULT_CONCATENATED_PCA_COMPONENT_COUNT == 3
    assert psth_webapp.CONCATENATED_PCA_FIGURE_SIZE == (11.0, 5.0)


def test_webapp_has_matplotlib_cleanup_dependency():
    """The Streamlit app should keep the pyplot handle used for figure cleanup."""
    assert callable(psth_webapp.plt.close)


def test_webapp_dataframe_width_argument_avoids_deprecated_streamlit_api():
    """Metadata tables should use Streamlit's current width argument."""
    webapp_source = Path(psth_webapp.__file__).read_text()

    assert "use_container_width" not in webapp_source
    assert 'width="stretch"' in webapp_source
