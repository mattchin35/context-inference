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
