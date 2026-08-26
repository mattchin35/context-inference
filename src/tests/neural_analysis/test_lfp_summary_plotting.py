"""RED contracts for cache-array LFP summary figures without Streamlit."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pytest

from src.neural_analysis.lfp_summary_plotting import (
    PlotContext,
    build_summary_figure_filename,
    plot_band_power_summary,
    plot_condition_psd,
    plot_phase_band_summary,
    plot_phase_map,
    plot_plv_distribution,
    plot_plv_exemplar,
    plot_population_ppc_maps,
    plot_ppc_band_summary,
    plot_ppc_exemplar,
    plot_unit_ppc_map,
)


def _context() -> PlotContext:
    """Return fixed cache metadata required in every summary caption."""

    return PlotContext(
        session_id="CT026_2026-08-01_130853",
        alignment_event="choice_time",
        epoch_bounds_s={
            "whole": (-2.0, 2.0),
            "before": (-2.0, 0.0),
            "after": (0.0, 2.0),
        },
        notch_enabled=True,
        gamma_exclusion_hz=(58.0, 62.0),
        reference_description="external brain reference; no CAR or bipolar rereference",
        source_voltage_unit="uV",
    )


def _assert_figure_contract(
    figure: plt.Figure,
    axes: dict[str, plt.Axes],
    names: set[str],
) -> None:
    """Assert common opaque-light plotting output without inspecting pixels."""

    assert set(axes) == names
    assert all(axis.figure is figure for axis in axes.values())
    assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    assert all(axis.get_facecolor()[:3] == (1.0, 1.0, 1.0) for axis in axes.values())
    assert all(
        axis.get_xlabel() or axis.get_ylabel() or axis.get_title()
        for axis in axes.values()
    )
    plt.close(figure)


def test_condition_psd_shows_every_condition_median_iqr_count_reference_and_units() -> None:
    """Condition spectra retain condition/trial/frequency cache axes and provenance."""

    figure, axes = plot_condition_psd(
        frequency_hz=np.array([2.0, 4.0, 6.0, 8.0]),
        condition_trial_psd_db=np.array(
            [
                [[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]],
                [[4.0, 5.0, 6.0, 7.0], [4.5, 5.5, 6.5, 7.5]],
            ]
        ),
        condition_names=("correct_rewarded", "omission"),
        contributing_trial_counts=np.array([2, 2]),
        site_label="PFC channel 5",
        epoch_name="whole",
        normalization="session_median",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"spectrum"})
    assert len(axes["spectrum"].lines) == 2
    assert "dB" in axes["spectrum"].get_ylabel()
    caption = figure.texts[-1].get_text()
    assert "correct_rewarded (n=2)" in caption and "omission (n=2)" in caption
    assert "session" in caption and "uV" in caption
    assert "\n" in caption


def test_condition_psd_preserves_nine_cache_groups_and_states_masks_overlap() -> None:
    """PSD legends retain the required cache order and disclose overlapping groups."""
    groups = (
        "correct_rewarded",
        "omission",
        "incorrect",
        "switch",
        "stay",
        "omission_switch",
        "omission_stay",
        "incorrect_switch",
        "incorrect_stay",
    )
    figure, axes = plot_condition_psd(
        frequency_hz=np.array((6.0, 8.0)),
        condition_trial_psd_db=np.ones((len(groups), 1, 2), dtype=float),
        condition_names=groups,
        contributing_trial_counts=np.ones(len(groups), dtype=np.int64),
        site_label="PFC",
        epoch_name="whole",
        normalization="session_median",
        context=_context(),
    )

    legend = axes["spectrum"].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert [label.split(" (n=")[0] for label in labels] == list(groups)
    assert "overlapping" in figure.texts[-1].get_text().lower()


def test_band_summary_uses_trial_boxplots_in_required_measurement_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Band boxes retain trial values and use the required theta/gamma ordering.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Captures Matplotlib boxplot arguments without rendering artists.
    """
    calls: list[tuple[object, dict[str, object]]] = []

    def capture_boxplot(
        axis: Axes,
        values: object,
        **kwargs: object,
    ) -> dict[str, object]:
        """Record one condition/measurement boxplot request."""
        del axis
        calls.append((values, kwargs))
        return {}

    monkeypatch.setattr(Axes, "boxplot", capture_boxplot)

    figure, axes = plot_band_power_summary(
        condition_trial_band_power_db=np.arange(24.0).reshape(2, 2, 3, 2),
        condition_names=("correct_rewarded", "omission"),
        epoch_names=("whole", "before", "after"),
        band_names=("theta", "gamma"),
        contributing_trial_counts=np.array([[2, 2, 2], [2, 2, 2]]),
        site_label="HPC1 channel 222",
        normalization="pre_session",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"band_power"})
    labels = [tick.get_text() for tick in axes["band_power"].get_xticklabels()]
    assert labels == ["correct_rewarded", "omission"]
    legend = axes["band_power"].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "theta-before",
        "theta-after",
        "gamma-before",
        "gamma-after",
    ]
    assert len(calls) == 4
    assert all(call[1]["showfliers"] is False for call in calls)
    assert all(call[1]["whis"] == 1.5 for call in calls)
    assert [call[1]["label"] for call in calls] == [
        "theta-before",
        "theta-after",
        "gamma-before",
        "gamma-after",
    ]
    assert all(len(call[0]) == 2 for call in calls)
    assert "overlapping" in figure.texts[-1].get_text().lower()
    assert axes["band_power"].get_ylabel().startswith("Band power (dB")


def test_phase_maps_and_band_summaries_expose_counts_uncertainty_and_instability() -> None:
    """ITPC/ISPC figures expose effective count, bootstrap CI, and low-n warnings."""

    figure, axes = plot_phase_map(
        metric=np.full((2, 3), 0.4),
        effective_trial_count=np.array([[3, 4, 5], [3, 4, 5]]),
        frequency_hz=np.array([6.0, 10.0]),
        relative_time_s=np.array([-0.1, 0.0, 0.1]),
        metric_name="ISPC",
        entity_label="PFC-HPC1",
        condition_name="omission",
        context=_context(),
        total_displayed_trial_count=6,
    )
    _assert_figure_contract(figure, axes, {"metric"})
    assert "Effective range/total displayed: 3-5/6" in figure.texts[-1].get_text()

    figure, axes = plot_phase_map(
        metric=np.full((2, 3), 0.4),
        effective_trial_count=np.full((2, 3), 5),
        frequency_hz=np.array([6.0, 10.0]),
        relative_time_s=np.array([-0.1, 0.0, 0.1]),
        metric_name="ITPC",
        entity_label="PFC",
        condition_name="omission",
        context=_context(),
        total_displayed_trial_count=6,
    )
    _assert_figure_contract(figure, axes, {"metric"})
    assert "Effective/total displayed: 5/6" in figure.texts[-1].get_text()

    figure, axes = plot_phase_band_summary(
        estimates=np.array([0.2, 0.4]),
        ci_low=np.array([0.1, 0.2]),
        ci_high=np.array([0.3, 0.6]),
        contributing_trial_counts=np.array([3, 12]),
        labels=("PFC", "PFC-HPC1"),
        band_name="theta",
        epoch_name="after",
        metric_name="ITPC/ISPC",
        context=_context(),
    )
    _assert_figure_contract(figure, axes, {"summary"})
    assert "unstable" in figure.texts[-1].get_text().lower()
    assert "95%" in figure.texts[-1].get_text()


def test_phase_band_summary_accepts_percentile_ci_not_containing_estimate() -> None:
    """Percentile intervals may validly lie entirely above the point estimate."""
    figure, axes = plot_phase_band_summary(
        estimates=np.array([0.2, 0.8]),
        ci_low=np.array([0.4, 0.1]),
        ci_high=np.array([0.6, 0.7]),
        contributing_trial_counts=np.array([8, 12]),
        labels=("PFC", "PFC-HPC1"),
        band_name="gamma",
        epoch_name="before",
        metric_name="ITPC/ISPC",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"summary"})
    assert len(axes["summary"].collections) >= 1


def test_phase_band_summary_reserves_space_for_nine_condition_labels() -> None:
    """Nine long condition names must remain separated from the caption."""
    labels = (
        "correct_rewarded",
        "omission",
        "incorrect",
        "switch",
        "stay",
        "omission_switch",
        "omission_stay",
        "incorrect_switch",
        "incorrect_stay",
    )
    figure, axes = plot_phase_band_summary(
        estimates=np.linspace(0.1, 0.9, len(labels)),
        ci_low=np.linspace(0.05, 0.85, len(labels)),
        ci_high=np.linspace(0.15, 0.95, len(labels)),
        contributing_trial_counts=np.arange(10, 10 + len(labels)),
        labels=labels,
        band_name="theta",
        epoch_name="before",
        metric_name="ITPC PFC",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"summary"})
    assert figure.get_size_inches()[0] >= 12.0
    assert figure.subplotpars.bottom >= 0.35


def test_plv_distribution_and_exemplar_distinguish_trial_metric_from_illustration() -> None:
    """PLV figures show coverage/count diagnostics and explicitly label illustrative trials."""

    figure, axes = plot_plv_distribution(
        trial_band_plv=np.array([[0.2, 0.4], [0.3, np.nan], [0.5, 0.6]]),
        trial_valid_sample_counts=np.array([[900, 1000], [800, 0], [1000, 1000]]),
        trial_valid_sample_fractions=np.array([[0.9, 1.0], [0.8, 0.0], [1.0, 1.0]]),
        epoch_names=("before", "after"),
        pair_label="PFC-HPC1",
        band_name="gamma",
        condition_name="omission",
        context=_context(),
    )
    _assert_figure_contract(figure, axes, {"distribution", "coverage"})

    figure, axes = plot_plv_exemplar(
        relative_time_s=np.array([-0.1, 0.0, 0.1]),
        source_traces=np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]),
        filtered_traces=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        phase_rad=np.array([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]]),
        site_labels=("PFC", "HPC1"),
        pair_label="PFC-HPC1",
        trial_index=7,
        percentile_label="95th percentile",
        pooled_plv=0.61,
        illustrative_trial_plv=0.58,
        context=_context(),
    )
    _assert_figure_contract(figure, axes, {"source", "filtered", "phase"})
    caption = figure.texts[-1].get_text().lower()
    assert "pooled" in caption and "illustrative" in caption and "trial 7" in caption


def test_plv_distribution_summarizes_large_trial_coverage_without_layout_failure() -> None:
    """Large trial populations must not expand captions by one row per trial."""
    trial_count = 427
    figure, axes = plot_plv_distribution(
        trial_band_plv=np.full((trial_count, 3), 0.5),
        trial_valid_sample_counts=np.tile((2000, 1000, 1000), (trial_count, 1)),
        trial_valid_sample_fractions=np.ones((trial_count, 3)),
        epoch_names=("whole", "before", "after"),
        pair_label="PFC-HPC1",
        band_name="theta",
        condition_name="correct_rewarded",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"distribution", "coverage"})
    caption = figure.texts[-1].get_text()
    assert "median" in caption
    assert len(caption) < 1500


def test_ppc_maps_preserve_reference_unit_order_and_reliability_inspection() -> None:
    """PPC unit heatmaps retain supplied ordering and make unreliable cells inspectable."""

    figure, axes = plot_unit_ppc_map(
        ppc=np.array([[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]]),
        computable=np.array([[True, True, True], [True, False, True]]),
        reliable=np.array([[True, False, True], [False, False, True]]),
        spike_count=np.array([[60, 20, 70], [30, 0, 80]]),
        frequency_hz=np.array([6.0, 8.0, 10.0]),
        unit_ids=("PFC:9", "HPC:2"),
        condition_name="correct_rewarded",
        site_label="PFC",
        epoch_name="whole",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"ppc", "reliability", "spike_count"})
    assert [tick.get_text() for tick in axes["ppc"].get_yticklabels()] == ["PFC:9", "HPC:2"]
    assert "unreliable" in figure.texts[-1].get_text().lower()


def test_population_ppc_prevalence_uses_eligible_denominator_and_nan_when_none() -> None:
    """Population maps distinguish reliable-unit median PPC from eligible-unit prevalence."""

    figure, axes = plot_population_ppc_maps(
        median_ppc=np.array([[0.1, 0.2], [0.3, 0.4]]),
        significant_fraction=np.array([[0.5, np.nan], [0.25, np.nan]]),
        eligible_unit_count=np.array([[4, 0], [4, 0]]),
        total_unit_count=np.array([[6, 6], [6, 6]]),
        condition_names=("correct_rewarded", "omission"),
        frequency_hz=np.array([6.0, 8.0]),
        site_label="PFC",
        epoch_name="whole",
        context=_context(),
    )

    _assert_figure_contract(figure, axes, {"median_ppc", "prevalence"})
    prevalence_array = axes["prevalence"].collections[0].get_array()
    assert np.isnan(np.asarray(prevalence_array)).any()
    assert "eligible" in figure.texts[-1].get_text().lower()


def test_ppc_band_summary_and_exemplar_include_counts_polar_frequency_and_pooled_caption() -> None:
    """PPC band and exemplar views expose reliability counts and 8/40-Hz polar selection."""

    figure, axes = plot_ppc_band_summary(
        condition_unit_band_ppc=np.arange(8.0).reshape(1, 2, 2, 2) / 10.0,
        reliable=np.array([[[[True, True], [True, False]], [[True, True], [True, True]]]]),
        condition_names=("omission",),
        epoch_names=("before", "after"),
        band_names=("theta", "gamma"),
        unit_count=2,
        site_label="PFC",
        context=_context(),
    )
    _assert_figure_contract(figure, axes, {"band_ppc"})
    assert "reliable" in figure.texts[-1].get_text().lower()

    figure, axes = plot_ppc_exemplar(
        frequency_hz=np.array([6.0, 8.0, 10.0]),
        pooled_ppc=np.array([0.1, 0.2, 0.1]),
        preferred_phase_rad=np.array([0.0, 0.2, 0.4]),
        representative_phase_hist_count=np.array([2, 5, 3, 1]),
        phase_bin_edges_rad=np.linspace(-np.pi, np.pi, 5),
        relative_time_s=np.array([-0.1, 0.0, 0.1]),
        source_trace=np.array([1.0, 2.0, 1.0]),
        filtered_trace=np.array([0.0, 1.0, 0.0]),
        spike_times_relative_s=np.array([0.0]),
        unit_id="PFC:9",
        trial_index=3,
        band_name="theta",
        representative_frequency_hz=8.0,
        percentile_label="5th percentile",
        context=_context(),
    )
    _assert_figure_contract(figure, axes, {"ppc", "polar", "trace"})
    caption = figure.texts[-1].get_text().lower()
    assert "pooled" in caption and "illustrative" in caption and "8 hz" in caption


def test_summary_filename_is_deterministic_selection_specific_and_collision_safe() -> None:
    """PNG names encode selections and sanitize labels without cross-view overwrites."""

    first = build_summary_figure_filename(
        session_id="CT026 2026/08/01",
        component="power",
        entity_label="PFC channel 5",
        condition_name="correct/rewarded",
        choice_filter="all",
        context_filter="left",
        epoch_name="whole",
        normalization="session_median",
    )
    second = build_summary_figure_filename(
        session_id="CT026 2026/08/01",
        component="power",
        entity_label="PFC channel 5",
        condition_name="correct/rewarded",
        choice_filter="all",
        context_filter="left",
        epoch_name="whole",
        normalization="pre_session",
    )

    assert first.endswith(".png")
    assert "/" not in first and " " not in first
    assert "session_median" in first and "pre_session" in second
    assert first != second
