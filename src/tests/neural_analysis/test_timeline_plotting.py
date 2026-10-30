"""Real-time heatmap geometry and display-only behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import timeline_plotting as plotting


def _forbid_fit(*args, **kwargs):
    """Fail if a display operation attempts another PCA fit."""
    pytest.fail("Figure construction must not refit PCA")


@pytest.fixture
def timeline_result():
    """Return two nonoverlapping windows at elapsed 100 and 114 s, with a fixed basis."""
    counts = np.random.default_rng(42).poisson(2, size=(2, 8, 2))
    regional = timeline.fit_timeline_activity(
        counts, ("A:7", "A:8"), np.array([0, 1]), "PFC", "A", 0.5
    )
    choices = pd.DataFrame(
        {
            "row_position": [4, 7],
            "trial_id": [105, 108],
            "choice_time": [100.0, 114.0],
            "elapsed_s": [100.0, 114.0],
            "action": [1, 0],
        }
    )
    events = pd.DataFrame(
        {
            "Time": [99.0, 101.0, 113.0, 115.0],
            "elapsed_s": [99.0, 101.0, 113.0, 115.0],
            "Event": ["LED_on", "LED_off", "left_entry", "LED_on"],
        }
    )
    return timeline.SessionTimelineResult(
        "synthetic",
        0.0,
        choices,
        events,
        {"PFC": regional},
        {"HPC": "missing"},
        {"bin_width_s": 0.5},
    )


def test_heatmap_uses_edges_and_explicit_ten_second_gap(timeline_result):
    """No heatmap cell stretches a neural score across the 102-112 s gap."""
    data = plotting.build_timeline_heatmap_data(
        timeline_result.regions["PFC"], timeline_result.choices
    )
    expected_edges = np.r_[np.arange(98.0, 102.5, 0.5), np.arange(112.0, 116.5, 0.5)]
    np.testing.assert_array_equal(data["x"], expected_edges)
    assert data["z"].shape == (2, 17)
    assert np.isnan(data["z"][:, 8]).all()
    np.testing.assert_array_equal(
        data["z"][:, :8], timeline_result.regions["PFC"].scores[0].T
    )
    assert data["customdata"][0, 0, 0] == 105
    np.testing.assert_allclose(
        data["customdata"][0, 0, 1:5].astype(float), [98.0, 98.5, -2.0, -1.5]
    )


def test_bounded_loading_preserves_true_bin_edges_and_region_scale(timeline_result):
    """Viewport filtering includes intersecting windows without clipping or rescaling cells."""
    result = timeline_result
    fig = plotting.plot_session_timeline(
        result,
        time_range_s=(98.2, 100.1),
        pc_ranges={"PFC": (2, 2)},
        loaded_range_s=(98.2, 100.1),
    )
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    np.testing.assert_array_equal(heatmap.x, np.arange(98.0, 102.5, 0.5))
    assert len(heatmap.z) == 1
    assert heatmap.zmax == result.regions["PFC"].color_limit
    assert heatmap.zmin == -result.regions["PFC"].color_limit
    assert heatmap.zsmooth is False and heatmap.connectgaps is False


def test_toggles_pc_rows_and_navigation_never_refit_or_mutate(
    timeline_result, monkeypatch
):
    """Figure construction consumes saved arrays without touching analysis state."""
    monkeypatch.setattr(timeline, "fit_regional_pca", _forbid_fit)
    regional = timeline_result.regions["PFC"]
    before = regional.scores.copy()
    first = plotting.plot_session_timeline(timeline_result, time_range_s=(90.0, 150.0))
    second = plotting.plot_session_timeline(
        timeline_result,
        time_range_s=(110.0, 120.0),
        event_tracks=(),
        pc_ranges={"PFC": (1, 1)},
    )
    np.testing.assert_array_equal(regional.scores, before)
    assert first.layout.xaxis.range == (90.0, 150.0)
    assert second.layout.xaxis.range == (110.0, 120.0)
    assert any(axis.matches for axis in first.select_xaxes())
    assert next(first.select_yaxes()).fixedrange is True
    assert any("LED" in (trace.name or "") for trace in first.data)
    assert not any("LED" in (trace.name or "") for trace in second.data)


def test_led_intervals_do_not_invent_unpaired_or_ambiguous_boundaries():
    """Repeated on/off transitions and separate LED identities are handled conservatively."""
    events = pd.DataFrame(
        {
            "Event": [
                "LED_off",
                "LED_on",
                "LED_on",
                "LED_off",
                "left_LED_on",
                "left_LED_off",
                "LED_on",
            ],
            "elapsed_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    intervals = plotting.verified_led_intervals(events)
    assert intervals.to_dict("records") == [
        {"identity": "LED", "start_s": 2.0, "end_s": 3.0},
        {"identity": "left_LED", "start_s": 4.0, "end_s": 5.0},
    ]


def test_behavior_only_figure_is_valid_and_zero_scores_are_not_missing(timeline_result):
    """Optional regional absence and center-color zero values remain distinct from gaps."""
    regional = timeline_result.regions["PFC"]
    regional.scores[0, 0, 0] = 0.0
    data = plotting.build_timeline_heatmap_data(regional, timeline_result.choices)
    assert data["z"][0, 0] == 0.0
    assert np.isnan(data["z"][0, 8])
    result = timeline.SessionTimelineResult(
        "behavior",
        0.0,
        timeline_result.choices,
        pd.DataFrame(),
        {},
        {"PFC": "missing", "HPC": "missing"},
        {},
    )
    assert plotting.plot_session_timeline(result).data


def test_context_intervals_use_logged_switch_times_and_carry_state_to_session_end():
    """Logged patch/dark entries define half-open elapsed-second intervals, even before zero."""
    events = pd.DataFrame(
        {
            "Event": [
                "enter_right_patch",
                "left_entry",
                "enter_left_patch",
                "enter_dark_period",
            ],
            "elapsed_s": [105.0, 103.0, 95.0, 110.0],
        }
    )
    intervals = plotting.context_intervals(events, 116.0)
    assert intervals.to_dict("records") == [
        {"context": "Left active", "start_s": 95.0, "end_s": 105.0},
        {"context": "Right active", "start_s": 105.0, "end_s": 110.0},
        {"context": "Dark period", "start_s": 110.0, "end_s": 116.0},
    ]
    earlier = pd.DataFrame({"Event": ["enter_left_patch"], "elapsed_s": [-2.0]})
    assert plotting.context_intervals(earlier, 5.0).start_s.tolist() == [-2.0]


def test_context_shading_is_confined_to_choice_row_and_keeps_true_boundaries(
    timeline_result,
):
    """A transition before the viewport still determines its initial state; neural rows stay unshaded."""
    result = timeline_result
    result.events.loc[len(result.events)] = [95.0, 95.0, "enter_left_patch"]
    result.events.loc[len(result.events)] = [105.0, 105.0, "enter_right_patch"]
    result.events.loc[len(result.events)] = [110.0, 110.0, "enter_dark_period"]
    figure = plotting.plot_session_timeline(
        result, time_range_s=(98.0, 116.0), loaded_range_s=(98.0, 116.0)
    )
    shapes = figure.layout.shapes
    assert [(shape.x0, shape.x1) for shape in shapes] == [
        (95.0, 105.0),
        (105.0, 110.0),
        (110.0, 116.0),
    ]
    assert all(shape.xref == "x" and shape.yref == "y domain" for shape in shapes)
    assert all(
        shape.y0 == 0 and shape.y1 == 1 and shape.layer == "below" for shape in shapes
    )
    assert len({shape.fillcolor for shape in shapes}) == 3
    assert all(0 < shape.opacity < 0.3 for shape in shapes)


@pytest.mark.parametrize(
    "entries, expected_contexts",
    [
        (
            [(100.0, "enter_right_patch"), (114.0, "enter_left_patch")],
            ("Right active", "Left active"),
        ),
        ([(105.0, "enter_right_patch")], ("Unknown", "Right active")),
        (
            [(95.0, "enter_left_patch"), (114.0, "enter_dark_period")],
            ("Left active", "Dark period"),
        ),
    ],
)
def test_choice_hover_uses_actual_context_at_choice_including_switch_boundary(
    timeline_result,
    entries,
    expected_contexts,
):
    """Context follows logged state, not chosen side, and switches take effect at their timestamp."""
    for time_s, event in entries:
        timeline_result.events.loc[len(timeline_result.events)] = [
            time_s,
            time_s,
            event,
        ]
    figure = plotting.plot_session_timeline(timeline_result)
    choices = [
        trace
        for trace in figure.data
        if trace.name in ("Left choices", "Right choices")
    ]
    assert tuple(trace.customdata[0][2] for trace in choices) == expected_contexts
    assert all("Active context" in trace.hovertemplate for trace in choices)


def test_context_toggle_reuses_scores_and_does_not_mutate_saved_choices(
    timeline_result, monkeypatch
):
    """The overlay is a plotting setting; hiding it keeps context hover and all scientific products."""
    result = timeline_result
    result.events.loc[len(result.events)] = [95.0, 95.0, "enter_left_patch"]
    before_choices = result.choices.copy(deep=True)
    before_scores = result.regions["PFC"].scores.copy()
    monkeypatch.setattr(timeline, "fit_regional_pca", _forbid_fit)
    shown = plotting.plot_session_timeline(result, show_context=True)
    hidden = plotting.plot_session_timeline(result, show_context=False)
    assert len(shown.layout.shapes) == 1
    assert not hidden.layout.shapes
    pd.testing.assert_frame_equal(result.choices, before_choices)
    np.testing.assert_array_equal(result.regions["PFC"].scores, before_scores)
    for figure in (shown, hidden):
        heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
        assert heatmap.zmin == -result.regions["PFC"].color_limit
        assert heatmap.zmax == result.regions["PFC"].color_limit


def test_missing_context_leaves_choices_unknown_and_unshaded(timeline_result):
    """Licks/LEDs or an absent event table do not imply a task context."""
    assert plotting.context_intervals(pd.DataFrame(), 116.0).empty
    figure = plotting.plot_session_timeline(timeline_result)
    assert not figure.layout.shapes
    choices = [
        trace
        for trace in figure.data
        if trace.name in ("Left choices", "Right choices")
    ]
    assert all(trace.customdata[0][2] == "Unknown" for trace in choices)


def test_pc_zscores_use_each_components_own_session_mean_and_population_sd():
    """Pool window/bin axes independently per PC, preserving shape and saved values."""
    base = np.arange(16, dtype=float).reshape(2, 8)
    scores = np.stack([100 + 3 * base, -20 - 0.2 * base], axis=-1)
    before = scores.copy()
    expected = (base - 7.5) / np.sqrt(21.25)
    actual = plotting.zscore_pc_scores(scores)
    assert actual.shape == scores.shape
    np.testing.assert_allclose(actual[..., 0], expected)
    np.testing.assert_allclose(actual[..., 1], -expected)
    np.testing.assert_allclose(actual.mean(axis=(0, 1)), 0, atol=1e-14)
    np.testing.assert_allclose(actual.std(axis=(0, 1), ddof=0), 1)
    np.testing.assert_array_equal(scores, before)


@pytest.mark.parametrize("constant", [0.0, 0.1, 5.0])
def test_pc_zscores_keep_constant_components_at_zero(constant):
    """A constant PC, including a one-observation PC, never divides by zero."""
    scores = np.stack([np.full((2, 8), constant), np.arange(16).reshape(2, 8)], axis=-1)
    actual = plotting.zscore_pc_scores(scores)
    np.testing.assert_array_equal(actual[..., 0], 0)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(plotting.zscore_pc_scores(np.ones((1, 1, 2))), 0)


def test_zscore_display_changes_only_heatmap_values_limits_and_units(
    timeline_result, monkeypatch
):
    """Geometry, context, PC order, variance hover, and saved arrays survive the toggle."""
    import json

    result = timeline_result
    result.events.loc[len(result.events)] = [95.0, 95.0, "enter_left_patch"]
    scores = result.regions["PFC"].scores
    scores[..., 0] = 100 + 3 * np.arange(16).reshape(2, 8)
    scores[..., 1] = -20 - 0.2 * np.arange(16).reshape(2, 8)
    before = scores.copy()
    monkeypatch.setattr(timeline, "fit_regional_pca", _forbid_fit)
    raw = plotting.plot_session_timeline(result)
    standardized = plotting.plot_session_timeline(result, zscore_pcs=True)
    heatmap = next(trace for trace in standardized.data if trace.type == "heatmap")
    expected = (np.arange(16).reshape(2, 8) - 7.5) / np.sqrt(21.25)
    np.testing.assert_allclose(heatmap.z[0, :8], expected[0])
    np.testing.assert_allclose(heatmap.z[0, 9:], expected[1])
    np.testing.assert_allclose(heatmap.z[1, :8], -expected[0])
    assert np.isnan(np.asarray(heatmap.z)[:, 8]).all()
    assert heatmap.zmax == pytest.approx(np.abs(expected).max())
    assert heatmap.zmin == -heatmap.zmax
    assert "z-score" in heatmap.colorbar.title.text
    assert "SD" in heatmap.colorbar.title.text
    assert "Z-score" in heatmap.hovertemplate
    np.testing.assert_array_equal(scores, before)
    raw_spec, z_spec = json.loads(raw.to_json()), json.loads(standardized.to_json())
    for spec in (raw_spec, z_spec):
        for trace in spec["data"]:
            if trace["type"] == "heatmap":
                for field in ("z", "zmin", "zmax", "hovertemplate"):
                    trace.pop(field)
                trace["colorbar"]["title"].pop("text")
    assert raw_spec == z_spec


def test_zscore_reference_and_limits_do_not_change_with_bounded_loading(
    timeline_result,
):
    """Loaded intervals and PC selection use the same full-session reference per PC."""
    result = timeline_result
    full = plotting.plot_session_timeline(result, zscore_pcs=True)
    bounded = plotting.plot_session_timeline(
        result,
        zscore_pcs=True,
        time_range_s=(112.5, 115.0),
        loaded_range_s=(112.5, 115.0),
    )
    selected = plotting.plot_session_timeline(
        result, zscore_pcs=True, pc_ranges={"PFC": (2, 2)}
    )
    all_rows, cropped, single = (
        next(trace for trace in figure.data if trace.type == "heatmap")
        for figure in (full, bounded, selected)
    )
    np.testing.assert_allclose(cropped.z, np.asarray(all_rows.z)[:, 9:])
    np.testing.assert_allclose(single.z[0], all_rows.z[1], equal_nan=True)
    assert cropped.zmax == all_rows.zmax
    assert cropped.zmin == all_rows.zmin
    np.testing.assert_array_equal(cropped.x, np.arange(112.0, 116.5, 0.5))


def test_zscore_manual_limits_are_in_sd_units_and_never_clip_data(timeline_result):
    """Manual limits only saturate colors; raw scores remain the default display."""
    result = timeline_result
    manual = plotting.plot_session_timeline(
        result, zscore_pcs=True, color_limits={"PFC": 0.5}
    )
    heatmap = next(trace for trace in manual.data if trace.type == "heatmap")
    assert (heatmap.zmin, heatmap.zmax) == (-0.5, 0.5)
    assert np.nanmax(np.abs(heatmap.z)) > 0.5
    default = plotting.plot_session_timeline(result)
    explicit_raw = plotting.plot_session_timeline(result, zscore_pcs=False)
    assert default.to_json() == explicit_raw.to_json()
    raw = next(trace for trace in default.data if trace.type == "heatmap")
    assert raw.colorbar.title.text == "PFC score"
    assert raw.zmax == result.regions["PFC"].color_limit


def test_all_constant_zscored_rows_have_valid_limits_and_measured_zero_cells(
    timeline_result,
):
    """Flat PC data remains center-colored, with only real gaps blank and finite color limits."""
    timeline_result.regions["PFC"].scores[:] = 5.0
    figure = plotting.plot_session_timeline(timeline_result, zscore_pcs=True)
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
    np.testing.assert_array_equal(np.asarray(heatmap.z)[:, :8], 0)
    assert np.isnan(np.asarray(heatmap.z)[:, 8]).all()
    assert np.isfinite([heatmap.zmin, heatmap.zmax]).all()
    assert heatmap.zmax > 0 and heatmap.zmin == -heatmap.zmax


def test_pc_and_track_row_changes_reset_y_axes_without_resetting_session_time(
    timeline_result,
):
    """Stable time UI must not preserve stale vertical ranges after subplot roles or PC ranks change."""
    first = plotting.plot_session_timeline(
        timeline_result, pc_ranges={"PFC": (1, 2)}, event_tracks=()
    )
    selected = plotting.plot_session_timeline(
        timeline_result, pc_ranges={"PFC": (2, 2)}, event_tracks=()
    )
    with_events = plotting.plot_session_timeline(
        timeline_result, pc_ranges={"PFC": (1, 2)}, event_tracks=("LED",)
    )
    assert (
        first.layout.uirevision
        == selected.layout.uirevision
        == with_events.layout.uirevision
    )
    assert first.layout.yaxis2.uirevision is not None
    assert first.layout.yaxis2.uirevision != selected.layout.yaxis2.uirevision
    assert first.layout.yaxis2.uirevision != with_events.layout.yaxis2.uirevision
    assert (
        first.layout.yaxis2.autorange == selected.layout.yaxis2.autorange == "reversed"
    )
