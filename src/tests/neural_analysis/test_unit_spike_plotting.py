from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src.neural_analysis import unit_spike_plotting


def test_hilbert_phase_trial_plot_has_exactly_one_public_definition():
    """The public Hilbert-phase trial plot must not be shadowed by a later definition."""
    source_path = Path(unit_spike_plotting.__file__)
    module_ast = ast.parse(source_path.read_text(encoding="utf-8"))
    matching_definitions = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "plot_trial_spike_lfp_hilbert_phase_and_behavior"
    ]

    assert len(matching_definitions) == 1


def make_trial_df() -> pd.DataFrame:
    """Build a compact trial table for unit-raster helper tests."""
    return pd.DataFrame(
        {
            "choice_time": [10.0, 20.0, 30.0, 40.0],
            "start_time": [9.0, 19.0, 29.0, 39.0],
            "led_on_time": [9.5, 20.25, np.nan, 40.5],
            "give_reward": [0, 0, 0, 0],
            "correct": [1, 0, 1, 1],
            "reward": [1, 0, 0, 1],
            "action": [0, 1, 1, 0],
        }
    )


def test_filter_trials_for_unit_plot_handles_condition_and_action():
    """Trial filtering should combine behavioral condition and action selectors."""
    trial_df = make_trial_df()

    selected_indices = unit_spike_plotting.filter_trials_for_unit_plot(
        trial_df,
        condition="omission",
        action=1,
    )

    np.testing.assert_array_equal(selected_indices, np.array([2], dtype=int))


def test_paginate_trial_indices_preserves_chronological_order():
    """Raster pages should keep the original chronological trial order."""
    trial_indices = np.array([0, 3, 5, 7, 9], dtype=int)

    page_indices = unit_spike_plotting.paginate_trial_indices(
        trial_indices,
        page_index=1,
        page_size=2,
    )

    np.testing.assert_array_equal(page_indices, np.array([5, 7], dtype=int))


def test_split_trial_indices_by_action_returns_left_and_right_choices():
    """Choice-comparison views should split filtered trials by project action codes."""
    trial_df = make_trial_df()

    left_trials, right_trials = unit_spike_plotting.split_trial_indices_by_action(
        trial_df=trial_df,
        trial_indices=np.array([0, 1, 2, 3], dtype=int),
    )

    np.testing.assert_array_equal(left_trials, np.array([1, 2], dtype=int))
    np.testing.assert_array_equal(right_trials, np.array([0, 3], dtype=int))


def test_extract_relative_unit_spikes_aligns_to_choice_time():
    """Spike times should be returned relative to the requested event."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.9, 10.1, 19.7, 20.4, 41.0], dtype=float)

    relative_spikes = unit_spike_plotting.extract_relative_unit_spikes(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="choice_time",
        window=(-0.5, 0.5),
    )

    assert len(relative_spikes) == 2
    np.testing.assert_allclose(relative_spikes[0], np.array([-0.1, 0.1]))
    np.testing.assert_allclose(relative_spikes[1], np.array([-0.3, 0.4]))


def test_compute_psth_hz_returns_firing_rate_not_raw_counts():
    """PSTH values should be normalized by trial count and bin width."""
    relative_spikes = [
        np.array([-0.4, -0.2, 0.2]),
        np.array([-0.3, 0.1, 0.3]),
    ]

    bin_centers, firing_rate_hz = unit_spike_plotting.compute_psth_hz(
        relative_spikes,
        window=(-0.5, 0.5),
        bin_size=0.5,
    )

    np.testing.assert_allclose(bin_centers, np.array([-0.25, 0.25]))
    np.testing.assert_allclose(firing_rate_hz, np.array([3.0, 3.0]))


def test_compute_binned_firing_rates_hz_returns_trial_by_bin_matrix():
    """Binned single-unit rates should be normalized per trial in Hz."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 9.15, 9.92, 19.05, 19.81], dtype=float)

    bin_centers, firing_rates_hz = unit_spike_plotting.compute_binned_firing_rates_hz(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.3),
        bin_size=0.1,
    )

    np.testing.assert_allclose(bin_centers, np.array([0.05, 0.15, 0.25]))
    assert firing_rates_hz.shape == (2, 3)
    np.testing.assert_allclose(firing_rates_hz[0], np.array([10.0, 10.0, 0.0]))
    np.testing.assert_allclose(firing_rates_hz[1], np.array([10.0, 0.0, 0.0]))


def test_compute_binned_firing_rates_hz_trims_incomplete_final_bin():
    """The final partial bin should be excluded rather than treated as 100 ms."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 9.15, 9.25], dtype=float)

    bin_centers, firing_rates_hz = unit_spike_plotting.compute_binned_firing_rates_hz(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=np.array([0], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.25),
        bin_size=0.1,
    )

    np.testing.assert_allclose(bin_centers, np.array([0.05, 0.15]))
    assert firing_rates_hz.shape == (1, 2)
    np.testing.assert_allclose(firing_rates_hz[0], np.array([10.0, 10.0]))


def test_plot_unit_binned_rate_trial_traces_draws_each_trial_and_mean():
    """Trial-trace rate plots should show all selected trials plus the mean."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 9.15, 19.05], dtype=float)

    figure, axis = unit_spike_plotting.plot_unit_binned_rate_trial_traces(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.3),
        bin_size=0.1,
        unit_id=12,
    )

    firing_rate_lines = [line for line in axis.lines if line.get_linestyle() != "--"]
    assert len(firing_rate_lines) == 3
    assert axis.get_ylabel() == "Firing rate (Hz)"
    assert "trial rates" in axis.get_title()
    plt.close(figure)


def test_plot_unit_binned_rate_mean_sd_draws_mean_line_and_sd_band():
    """Mean/SD rate plots should show the mean with one variability band."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 9.15, 19.05], dtype=float)

    figure, axis = unit_spike_plotting.plot_unit_binned_rate_mean_sd(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.3),
        bin_size=0.1,
        unit_id=12,
    )

    firing_rate_lines = [line for line in axis.lines if line.get_linestyle() != "--"]
    assert len(firing_rate_lines) == 1
    assert len(axis.collections) == 1
    assert axis.get_ylabel() == "Firing rate (Hz)"
    assert "mean +/- SD" in axis.get_title()
    plt.close(figure)


def test_compute_trial_event_offsets_relative_to_start_time():
    """Event markers should be expressed in seconds relative to trial start."""
    trial_df = make_trial_df()

    choice_offsets = unit_spike_plotting.compute_trial_event_offsets(
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        event_column="choice_time",
    )
    led_offsets = unit_spike_plotting.compute_trial_event_offsets(
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        event_column="led_on_time",
    )

    np.testing.assert_allclose(choice_offsets, np.array([1.0, 1.0]))
    np.testing.assert_allclose(led_offsets, np.array([0.5, 1.25]))


def test_compute_trial_event_offsets_relative_to_choice_time():
    """Event markers should support choice-aligned raster plots."""
    trial_df = make_trial_df()

    start_offsets = unit_spike_plotting.compute_trial_event_offsets(
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="choice_time",
        event_column="start_time",
    )
    led_offsets = unit_spike_plotting.compute_trial_event_offsets(
        trial_df=trial_df,
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="choice_time",
        event_column="led_on_time",
    )

    np.testing.assert_allclose(start_offsets, np.array([-1.0, -1.0]))
    np.testing.assert_allclose(led_offsets, np.array([-0.5, 0.25]))


def test_compute_trial_event_offsets_missing_values_are_nan():
    """Missing event times should remain missing rather than raising errors."""
    trial_df = make_trial_df()

    led_offsets = unit_spike_plotting.compute_trial_event_offsets(
        trial_df=trial_df,
        trial_indices=np.array([1, 2], dtype=int),
        alignment_event="start_time",
        event_column="led_on_time",
    )

    assert led_offsets[0] == pytest.approx(1.25)
    assert np.isnan(led_offsets[1])


def test_plot_unit_raster_and_psth_returns_two_axes():
    """Combined unit plots should show a raster and a PSTH together."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.9, 10.1, 19.7, 20.4, 30.2], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_raster_and_psth(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=np.array([0, 1], dtype=int),
        psth_trial_indices=np.array([0, 1, 2], dtype=int),
        alignment_event="choice_time",
        window=(-0.5, 0.5),
        bin_size=0.5,
        unit_id=10,
        title_suffix="HPC",
    )

    assert len(axes) == 2
    assert "Unit 10" in axes[0].get_title()
    assert axes[1].get_ylabel() == "Firing rate (Hz)"
    plt.close(figure)


def test_plot_unit_raster_and_psth_can_use_trial_trace_summary():
    """Trial-trace summaries should replace only the lower panel, not the raw raster."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 9.15, 19.05, 19.15, 30.05], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_raster_and_psth(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=np.array([0, 1], dtype=int),
        psth_trial_indices=np.array([0, 1, 2], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.3),
        bin_size=0.1,
        unit_id=10,
        summary_plot_type="binned-rate-trials-mean",
        binned_rate_bin_size=0.1,
    )

    assert len(axes) == 2
    assert any(collection.get_offsets().size for collection in axes[0].collections)
    firing_rate_lines = [line for line in axes[1].lines if line.get_linestyle() != "--"]
    assert len(firing_rate_lines) == 4
    assert len(axes[1].containers) == 0
    plt.close(figure)


def test_plot_unit_raster_and_psth_can_use_mean_sd_summary():
    """Mean/SD summaries should keep the top raster visible."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 9.15, 19.05, 19.15, 30.05], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_raster_and_psth(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=np.array([0, 1], dtype=int),
        psth_trial_indices=np.array([0, 1, 2], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.3),
        bin_size=0.1,
        unit_id=10,
        summary_plot_type="binned-rate-mean-sd",
        binned_rate_bin_size=0.1,
    )

    assert len(axes) == 2
    assert any(collection.get_offsets().size for collection in axes[0].collections)
    firing_rate_lines = [line for line in axes[1].lines if line.get_linestyle() != "--"]
    assert len(firing_rate_lines) == 1
    assert len(axes[1].collections) == 1
    assert len(axes[1].containers) == 0
    plt.close(figure)


def test_plot_unit_left_right_choice_comparison_returns_two_by_two_axes():
    """Choice comparison should show left and right rasters with their summaries."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 19.05, 29.05, 39.05], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_left_right_choice_comparison(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        left_raster_trial_indices=np.array([1], dtype=int),
        right_raster_trial_indices=np.array([0], dtype=int),
        left_summary_trial_indices=np.array([1, 2], dtype=int),
        right_summary_trial_indices=np.array([0, 3], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.2),
        bin_size=0.1,
        unit_id=10,
        summary_plot_type="psth",
        title_suffix="HPC; page 1/2",
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "Left choices"
    assert axes[0, 1].get_title() == "Right choices"
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Unit 10 aligned to start_time (HPC; page 1/2)"
    left_annotation = {text.get_text() for text in axes[0, 0].texts}
    right_annotation = {text.get_text() for text in axes[0, 1].texts}
    assert "raster n=1; summary n=2" in left_annotation
    assert "raster n=1; summary n=2" in right_annotation
    assert any(collection.get_offsets().size for collection in axes[0, 0].collections)
    assert any(collection.get_offsets().size for collection in axes[0, 1].collections)
    assert len(axes[1, 0].containers) == 1
    assert len(axes[1, 1].containers) == 1
    plt.close(figure)


def test_plot_unit_left_right_choice_comparison_summaries_use_each_side_trials():
    """Each comparison summary should use all trials for that side, not the raster page only."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.05, 19.05, 29.15, 39.15], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_left_right_choice_comparison(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        left_raster_trial_indices=np.array([1], dtype=int),
        right_raster_trial_indices=np.array([0], dtype=int),
        left_summary_trial_indices=np.array([1, 2], dtype=int),
        right_summary_trial_indices=np.array([0, 3], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.2),
        bin_size=0.1,
        unit_id=10,
        summary_plot_type="binned-rate-mean-sd",
        binned_rate_bin_size=0.1,
    )

    left_mean_line = [line for line in axes[1, 0].lines if line.get_linestyle() != "--"][0]
    right_mean_line = [line for line in axes[1, 1].lines if line.get_linestyle() != "--"][0]
    np.testing.assert_allclose(left_mean_line.get_ydata(), np.array([5.0, 5.0]))
    np.testing.assert_allclose(right_mean_line.get_ydata(), np.array([5.0, 5.0]))
    assert len(axes[1, 0].collections) == 1
    assert len(axes[1, 1].collections) == 1
    plt.close(figure)


def test_plot_unit_raster_and_psth_accepts_event_markers():
    """Optional event markers should be drawn without changing the PSTH API."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.9, 10.1, 19.7, 20.4, 30.2], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_raster_and_psth(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=np.array([0, 1], dtype=int),
        psth_trial_indices=np.array([0, 1, 2], dtype=int),
        alignment_event="start_time",
        window=(-0.5, 1.5),
        bin_size=0.5,
        unit_id=10,
        event_marker_columns=("choice_time", "led_on_time"),
        event_marker_styles={
            "choice_time": {"label": "choice", "color": "tab:purple"},
            "led_on_time": {"label": "LED", "color": "tab:green"},
        },
    )

    assert len(axes[0].collections) > 0
    legend_text = {text.get_text() for text in axes[0].get_legend().get_texts()}
    assert {"choice", "LED"}.issubset(legend_text)
    plt.close(figure)


def test_plot_unit_raster_and_psth_accepts_custom_figure_size():
    """Caller-provided figure size should control the rendered plot dimensions."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.9, 10.1, 19.7, 20.4, 30.2], dtype=float)

    figure, _axes = unit_spike_plotting.plot_unit_raster_and_psth(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=np.array([0, 1, 2], dtype=int),
        psth_trial_indices=np.array([0, 1, 2], dtype=int),
        alignment_event="choice_time",
        window=(-0.5, 0.5),
        bin_size=0.5,
        unit_id=10,
        figure_size=(12.0, 11.0),
    )

    np.testing.assert_allclose(figure.get_size_inches(), np.array([12.0, 11.0]))
    plt.close(figure)


def test_plot_unit_raster_row_spacing_changes_plotted_y_coordinates():
    """Row spacing should spread trial rows in raster data coordinates."""
    trial_df = make_trial_df()
    unit_spike_times = np.array([9.9, 10.1, 19.7, 20.4, 30.2], dtype=float)

    figure, axes = unit_spike_plotting.plot_unit_raster_and_psth(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=np.array([0, 1, 2], dtype=int),
        psth_trial_indices=np.array([0, 1, 2], dtype=int),
        alignment_event="choice_time",
        window=(-0.5, 0.5),
        bin_size=0.5,
        unit_id=10,
        raster_row_spacing=2.0,
    )

    plotted_y_coordinates = np.concatenate(
        [collection.get_offsets()[:, 1] for collection in axes[0].collections if collection.get_offsets().size]
    )
    np.testing.assert_allclose(np.unique(plotted_y_coordinates), np.array([0.0, 2.0, 4.0]))
    plt.close(figure)


def test_paginate_unit_ids_preserves_cluster_id_order():
    """Combined trial rasters should page units in the caller-provided order."""
    unit_ids = np.array([10, 12, 15, 20, 21], dtype=int)

    page_unit_ids = unit_spike_plotting.paginate_unit_ids(
        unit_ids,
        page_index=1,
        page_size=2,
    )

    np.testing.assert_array_equal(page_unit_ids, np.array([15, 20], dtype=int))


def test_extract_relative_events_for_trial_filters_to_window():
    """Single-trial event extraction should return relative times inside the window."""
    relative_events = unit_spike_plotting.extract_relative_events_for_trial(
        event_times=nap.Ts(t=np.array([9.2, 9.7, 10.1, 11.7], dtype=float)),
        reference_time=10.0,
        window=(-0.5, 1.0),
    )

    np.testing.assert_allclose(relative_events, np.array([-0.3, 0.1]))


def test_extract_relative_spikes_for_units_returns_one_array_per_unit():
    """Single-trial spike extraction should keep unit ids attached to relative spike times."""
    spike_group = nap.TsGroup(
        {
            10: nap.Ts(t=np.array([9.8, 10.1, 10.9], dtype=float)),
            11: nap.Ts(t=np.array([9.0, 10.2, 11.5], dtype=float)),
        }
    )

    relative_spikes = unit_spike_plotting.extract_relative_spikes_for_units(
        spike_group=spike_group,
        unit_ids=np.array([10, 11], dtype=int),
        reference_time=10.0,
        window=(-0.5, 1.0),
    )

    assert set(relative_spikes) == {10, 11}
    np.testing.assert_allclose(relative_spikes[10], np.array([-0.2, 0.1, 0.9]))
    np.testing.assert_allclose(relative_spikes[11], np.array([0.2]))


def test_compute_population_psth_hz_normalizes_by_units_and_bin_size():
    """Population PSTH should report firing rate per unit."""
    relative_spikes_by_unit = {
        10: np.array([-0.4, 0.1], dtype=float),
        11: np.array([0.2], dtype=float),
    }

    bin_centers, population_rate_hz = unit_spike_plotting.compute_population_psth_hz(
        relative_spikes_by_unit=relative_spikes_by_unit,
        window=(-0.5, 0.5),
        bin_size=0.5,
    )

    np.testing.assert_allclose(bin_centers, np.array([-0.25, 0.25]))
    np.testing.assert_allclose(population_rate_hz, np.array([1.0, 2.0]))


def test_plot_trial_behavior_and_spike_raster_draws_behavior_rows_above_spikes():
    """Combined trial rasters should place left/right behavior rows above unit rows."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup(
        {
            10: nap.Ts(t=np.array([19.8, 20.1], dtype=float)),
            11: nap.Ts(t=np.array([19.9, 20.2], dtype=float)),
        }
    )

    figure, axis = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10, 11], dtype=int),
        alignment_event="choice_time",
        window=(-1.0, 1.0),
        psth_bin_size=None,
    )

    tick_labels = [tick.get_text() for tick in axis.get_yticklabels()]
    assert tick_labels[-2:] == ["Right licks", "Left licks"]
    assert tick_labels[:2] == ["Unit 10", "Unit 11"]
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_returns_raster_and_psth_axes():
    """Combined trial plots should support a population PSTH beneath the raster."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup(
        {
            10: nap.Ts(t=np.array([19.8, 20.1], dtype=float)),
            11: nap.Ts(t=np.array([19.9, 20.2], dtype=float)),
        }
    )

    figure, axes = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10, 11], dtype=int),
        alignment_event="choice_time",
        window=(-0.5, 0.5),
        psth_bin_size=0.5,
    )

    assert len(axes) == 2
    assert axes[1].get_ylabel() == "Population rate (Hz/unit)"
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_uses_distinct_psth_unit_ids():
    """The PSTH can summarize all selected units while the raster shows only the visible page."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup(
        {
            10: nap.Ts(t=np.array([19.8, 21.0], dtype=float)),
            11: nap.Ts(t=np.array([20.2, 21.0], dtype=float)),
        }
    )

    figure, axes = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10], dtype=int),
        psth_unit_ids=np.array([10, 11], dtype=int),
        alignment_event="choice_time",
        window=(-0.5, 0.5),
        psth_bin_size=0.5,
    )

    raster_tick_labels = [tick.get_text() for tick in axes[0].get_yticklabels()]
    assert "Unit 10" in raster_tick_labels
    assert "Unit 11" not in raster_tick_labels
    psth_heights = np.array([patch.get_height() for patch in axes[1].patches], dtype=float)
    np.testing.assert_allclose(psth_heights, np.array([1.0, 1.0]))
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_marks_choice_on_action_row():
    """Choice markers should be drawn on the lick row matching the trial action."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup({10: nap.Ts(t=np.array([19.8, 20.1], dtype=float))})

    figure, axis = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10], dtype=int),
        alignment_event="start_time",
        window=(-0.5, 1.5),
        psth_bin_size=None,
    )

    left_choice_segments = []
    for collection in axis.collections:
        if hasattr(collection, "get_segments"):
            for segment in collection.get_segments():
                x_center = np.mean(segment[:, 0])
                y_center = np.mean(segment[:, 1])
                if np.isclose(x_center, 1.0) and np.isclose(y_center, 3.0):
                    left_choice_segments.append(segment)
    assert left_choice_segments
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_supports_choice_alignment():
    """Choice-aligned combined rasters should place the selected choice at time zero."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup({10: nap.Ts(t=np.array([19.8, 20.1], dtype=float))})

    figure, axis = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10], dtype=int),
        alignment_event="choice_time",
        window=(-1.5, 0.5),
        psth_bin_size=None,
    )

    choice_segments = []
    for collection in axis.collections:
        if hasattr(collection, "get_segments"):
            for segment in collection.get_segments():
                if np.isclose(np.mean(segment[:, 0]), 0.0):
                    choice_segments.append(segment)
    assert choice_segments
    assert "aligned to choice_time" in axis.get_title()
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_can_disable_psth():
    """Raster-only combined trial plots should remain available."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup({10: nap.Ts(t=np.array([19.8, 20.1], dtype=float))})

    figure, axis = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10], dtype=int),
        alignment_event="choice_time",
        window=(-1.5, 0.5),
        psth_bin_size=None,
    )

    assert axis.get_ylabel() == "Event / unit"
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_with_lfp_and_psth_returns_three_axes():
    """Optional LFP traces should appear above the combined raster and PSTH."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup({10: nap.Ts(t=np.array([19.8, 20.1], dtype=float))})

    figure, axes = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10], dtype=int),
        alignment_event="choice_time",
        window=(-1.5, 0.5),
        psth_bin_size=0.5,
        lfp_time_s=np.array([-0.1, 0.0, 0.1], dtype=float),
        lfp_uv=np.array([10.0, 20.0, 15.0], dtype=float),
        lfp_label="LFP saved channel 4",
    )

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "LFP (uV)"
    assert "LFP saved channel 4" in axes[0].get_title()
    assert axes[1].get_ylabel() == "Event / unit"
    assert axes[2].get_ylabel() == "Population rate (Hz/unit)"
    plt.close(figure)


def test_plot_trial_behavior_and_spike_raster_with_lfp_without_psth_returns_two_axes():
    """LFP can be shown without the population PSTH."""
    trial_df = make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([19.2, 20.1], dtype=float)),
        "right_entry": nap.Ts(t=np.array([19.4, 20.3], dtype=float)),
    }
    spike_group = nap.TsGroup({10: nap.Ts(t=np.array([19.8, 20.1], dtype=float))})

    figure, axes = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        spike_group=spike_group,
        raster_unit_ids=np.array([10], dtype=int),
        alignment_event="choice_time",
        window=(-1.5, 0.5),
        psth_bin_size=None,
        lfp_time_s=np.array([-0.1, 0.0, 0.1], dtype=float),
        lfp_uv=np.array([10.0, 20.0, 15.0], dtype=float),
        lfp_label="LFP saved channel 4",
    )

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "LFP (uV)"
    assert axes[1].get_ylabel() == "Event / unit"
    plt.close(figure)


def test_save_unit_plot_figure_writes_unit_spike_viewer_png(tmp_path: Path):
    """Saved plots should be isolated under the unit_spike_viewer figure subfolder."""
    figure, _axis = plt.subplots()

    saved_path = unit_spike_plotting.save_unit_plot_figure(
        figure=figure,
        figure_path=tmp_path,
        session_id="CT014_2025-12-23_163505",
        unit_id=10,
        region_name="HPC",
        condition="correct_rewarded",
        action_label="all",
        alignment_event="choice_time",
        page_index=0,
    )

    assert saved_path.parent == tmp_path / "unit_spike_viewer"
    assert saved_path.name == "CT014_2025-12-23_163505_HPC_unit10_correct_rewarded_all_choice_time_page0.png"
    assert saved_path.exists()
    plt.close(figure)


def test_save_unit_plot_figure_can_include_plot_type_token(tmp_path: Path):
    """Alternative unit views should be able to save without overwriting raster/PSTH plots."""
    figure, _axis = plt.subplots()

    saved_path = unit_spike_plotting.save_unit_plot_figure(
        figure=figure,
        figure_path=tmp_path,
        session_id="CT014_2025-12-23_163505",
        unit_id=10,
        region_name="HPC",
        condition="correct_rewarded",
        action_label="all",
        alignment_event="choice_time",
        page_index=0,
        plot_type="binned-rate-mean-sd",
    )

    assert saved_path.name == (
        "CT014_2025-12-23_163505_HPC_unit10_correct_rewarded_all_"
        "choice_time_binned-rate-mean-sd_page0.png"
    )
    assert saved_path.exists()
    plt.close(figure)
