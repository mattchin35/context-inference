from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis import psth_webapp


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
