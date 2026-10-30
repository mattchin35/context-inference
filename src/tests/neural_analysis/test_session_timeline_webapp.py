"""Timeline routing and UI state contracts without experimental data."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src.neural_analysis.webapp import app, session_inputs, timeline_view
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as timeline_session_fixture,
)

synthetic_timeline_session = timeline_session_fixture


def test_timeline_is_available_with_behavior_and_missing_neural_probes(
    synthetic_timeline_session,
):
    """Other spike views remain strict; this view permits regional/behavior-only fallback."""
    session = synthetic_timeline_session
    (session.session_root / "B/aligned.npz").unlink()
    available = session_inputs.metadata_view_availability(session)
    assert available[session_inputs.PLOT_VIEW_SESSION_TIMELINE].available
    assert not available[session_inputs.PLOT_VIEW_TRIAL_SPIKES].available


def test_timeline_dispatch_occurs_before_active_probe_and_trial_controls(
    synthetic_timeline_session, monkeypatch
):
    """The dedicated multi-probe view handles its own inputs and returns from dispatch."""
    session = synthetic_timeline_session
    selected = session_inputs.PLOT_VIEW_SESSION_TIMELINE
    fake_st = SimpleNamespace(
        sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=selected))
    )
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_session_timeline_view", renderer)
    assert app._start_metadata_webapp(session) is None
    renderer.assert_called_once_with(fake_st, session)


def test_navigation_centers_original_trial_id_and_clamps_to_session_bounds():
    """Explicit controls can reach any choice without renumbering source trials."""
    result = timeline_view.resolve_time_range(100.0, 60.0, (0.0, 200.0))
    assert result == (100.0, 160.0)
    assert timeline_view.resolve_time_range(180.0, 60.0, (0.0, 200.0)) == (140.0, 200.0)
    assert timeline_view.resolve_time_range(0.0, 60.0, (-2.0, 10.0)) == (-2.0, 10.0)


def test_pc_ranges_are_display_only_and_do_not_exceed_available_rows():
    """Few-PC populations have a valid initial selection without special estimator paths."""
    assert timeline_view.default_pc_range(3) == (1, 3)
    assert timeline_view.default_pc_range(40) == (1, 10)


def _render_timeline_for_test(metadata_path):
    """Render only the new view for an existing synthetic metadata path."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.timeline_view import render_session_timeline_view

    render_session_timeline_view(st, load_webapp_session(metadata_path))


def test_streamlit_display_changes_reuse_saved_basis(synthetic_timeline_session):
    """Actual Streamlit reruns keep one saved analysis while PC/event controls change."""
    from src.neural_analysis.population.run_session_timeline import run_session_timeline
    from src.neural_analysis.population.session_timeline_io import load_timeline_result
    import numpy as np

    session = synthetic_timeline_session
    report = run_session_timeline(session.metadata_path, {"PFC": "A"})
    before = load_timeline_result(report.result_path).regions["PFC"].scores.copy()
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    assert not at.exception
    selector = next(widget for widget in at.selectbox if widget.label == "PFC probe")
    selector.select("A").run()
    assert not at.exception
    assert at.get("plotly_chart")
    checkbox = next(widget for widget in at.checkbox if widget.label == "Trial start")
    checkbox.uncheck().run()
    assert not at.exception
    pc_range = next(widget for widget in at.slider if widget.label == "PFC PC ranks")
    pc_range.set_value((2, 2)).run()
    assert not at.exception
    assert (
        len(list((session.session_root / "analysis_runs").glob("*/results.npz"))) == 1
    )
    np.testing.assert_array_equal(
        load_timeline_result(report.result_path).regions["PFC"].scores, before
    )


def test_context_checkbox_changes_overlay_without_another_preparation(synthetic_timeline_session):
    """A saved Stage 1 result with logged transitions gains a default-on display toggle."""
    import pandas as pd
    from src.neural_analysis.population.run_session_timeline import run_session_timeline

    session = synthetic_timeline_session
    event_path = session.behavior.event_table_file
    events = pd.read_csv(event_path)
    transitions = pd.DataFrame({"Time": [1700000000., 1700000014., 1700000028.],
                                "Event": ["enter_left_patch", "enter_right_patch", "enter_dark_period"],
                                "Note": ["", "", ""]})
    pd.concat([events, transitions], ignore_index=True).to_csv(event_path, index=False)
    report = run_session_timeline(session.metadata_path, {"PFC": "A"})
    modified_at = report.result_path.stat().st_mtime_ns
    at = AppTest.from_function(_render_timeline_for_test, args=(str(session.metadata_path),)).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    assert not at.exception
    checkbox = next(widget for widget in at.checkbox if widget.label == "Show active context")
    assert checkbox.value is True
    assert len(json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]["shapes"]) == 3
    checkbox.uncheck().run()
    assert not at.exception
    assert not json.loads(at.get("plotly_chart")[0].proto.spec)["layout"].get("shapes")
    assert report.result_path.stat().st_mtime_ns == modified_at
    assert len(list((session.session_root / "analysis_runs").glob("*/results.npz"))) == 1
