"""Saved Stage 2 display contracts: aligned points, explicit preparation, no refits."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.population import timeline_plotting as plotting
from src.neural_analysis.population.window_variance import compute_window_variance
from src.neural_analysis.population.run_session_timeline import run_session_timeline
from src.tests.neural_analysis.test_timeline_plotting import (
    timeline_result as result_fixture,
)
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as session_fixture,
)
from src.tests.neural_analysis.test_session_timeline_webapp import (
    _render_timeline_for_test,
    _timeline_figure_spec,
)

timeline_result = result_fixture
synthetic_timeline_session = session_fixture


def test_metric_points_keep_choice_times_counts_and_hover_across_display_changes(
    timeline_result,
):
    """The metric uses all original axes and never connects distant choices."""
    result = timeline_result
    summary = compute_window_variance(result.regions["PFC"])
    before = summary.counts95.copy()
    first = plotting.plot_session_timeline(result, window_variance={"PFC": summary})
    second = plotting.plot_session_timeline(
        result,
        pc_ranges={"PFC": (2, 2)},
        zscore_pcs=True,
        event_tracks=(),
        loaded_range_s=(98, 102),
        window_variance={"PFC": summary},
    )
    for figure in (first, second):
        trace = next(
            trace for trace in figure.data if trace.name == "PFC fixed-axis count (95%)"
        )
        assert trace.mode == "markers"
        np.testing.assert_array_equal(trace.x, [100, 114])
        np.testing.assert_array_equal(trace.y, before)
        assert "Total variance" in trace.hovertemplate
        assert "bins" in trace.hovertemplate and "units" in trace.hovertemplate
        assert trace.customdata[0][0] == 105
    assert first.layout.uirevision == second.layout.uirevision
    assert len([trace for trace in second.data if trace.type == "heatmap"][0].z) == 1


def test_unavailable_metric_is_nan_and_does_not_add_a_fictitious_missing_region(
    timeline_result,
):
    """Unavailable windows are absent points; missing regions do not gain metric tracks."""
    summary = compute_window_variance(timeline_result.regions["PFC"])
    summary = replace(
        summary,
        counts95=np.array([summary.counts95[0], np.nan]),
        status=np.array(["ok", "zero_variance"]),
    )
    figure = plotting.plot_session_timeline(
        timeline_result, window_variance={"PFC": summary}
    )
    trace = next(
        trace for trace in figure.data if trace.name == "PFC fixed-axis count (95%)"
    )
    assert np.isnan(trace.y[1])
    assert not any("HPC fixed-axis" in (trace.name or "") for trace in figure.data)


def test_webapp_prepares_only_on_request_then_reuses_summary_and_selected_trial(
    synthetic_timeline_session, monkeypatch
):
    """A saved sidecar is reused on display reruns, with original-PC contribution readout."""
    from src.neural_analysis.webapp import timeline_view
    from src.neural_analysis.population import run_window_variance as runner
    from src.neural_analysis.population import window_variance as metric

    session = synthetic_timeline_session
    stage1 = run_session_timeline(session.metadata_path, {"PFC": "A"}).result_path
    before = stage1.read_bytes()
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select(
        "A"
    ).run()
    assert not at.exception
    assert not list(stage1.parent.glob("window_variance_*/results.npz"))
    next(
        widget for widget in at.button if widget.label == "Prepare window summary"
    ).click().run()
    assert not at.exception
    assert len(list(stage1.parent.glob("window_variance_*/results.npz"))) == 1
    figure = _timeline_figure_spec(at)
    component_position = next(index for index, element in at.main.children.items() if element.type == "bidi_component")
    assert any(
        trace.get("name") == "PFC fixed-axis count (95%)" for trace in figure["data"]
    )
    monkeypatch.setattr(
        timeline_view,
        "run_session_timeline",
        Mock(side_effect=AssertionError("No PCA during display")),
    )
    monkeypatch.setattr(runner, "compute_session_window_variance", pytest.fail)
    monkeypatch.setattr(metric, "compute_window_variance", pytest.fail)
    next(widget for widget in at.selectbox if widget.label == "Jump to trial").select(
        102
    ).run()
    assert not at.exception
    assert any("Trial 102" in widget.value for widget in at.markdown)
    table = at.dataframe[0].value
    assert {"Session PC", "Window variance fraction", "Cumulative fraction"}.issubset(
        table.columns
    )
    assert table["Cumulative fraction"].iloc[-1] >= 0.95 - 1e-12
    next(
        widget for widget in at.checkbox if widget.label == "Z-score each PC"
    ).check().run()
    assert not at.exception
    assert at.dataframe[0].value.equals(table)
    next(
        widget
        for widget in at.checkbox
        if widget.label == "Show fixed-axis window counts"
    ).uncheck().run()
    assert not at.exception
    hidden = _timeline_figure_spec(at)
    assert next(index for index, element in at.main.children.items() if element.type == "bidi_component") == component_position
    assert hidden["layout"]["uirevision"] == figure["layout"]["uirevision"]
    assert not any(
        "fixed-axis count" in trace.get("name", "") for trace in hidden["data"]
    )
    assert len(list(stage1.parent.glob("window_variance_*/results.npz"))) == 1
    assert stage1.read_bytes() == before


def test_stage2_preparation_is_disabled_without_stage1(synthetic_timeline_session):
    """Behavior remains usable while explicit Stage 2 requires prepared PCs."""
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(synthetic_timeline_session.metadata_path),)
    ).run()
    assert not at.exception
    button = next(
        widget for widget in at.button if widget.label == "Prepare window summary"
    )
    assert button.disabled


def test_twenty_ms_view_defaults_to_bounded_neural_payload(synthetic_timeline_session):
    """Measured CT026 full-session JSON exceeds transport limits at 20 ms; default to bounded loading."""
    session = synthetic_timeline_session
    run_session_timeline(session.metadata_path, {"PFC": "A"}, 0.02)
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select(
        "A"
    ).run()
    next(
        widget for widget in at.selectbox if widget.label == "Timeline bin width (ms)"
    ).select(20).run()
    assert not at.exception
    bounded = next(
        widget
        for widget in at.checkbox
        if widget.label == "Load only visible neural interval"
    )
    assert bounded.value is True
    assert "minallowed" in _timeline_figure_spec(at)["layout"]["xaxis"]
