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


def _timeline_figure_spec(app_test):
    """Return the Plotly figure dictionary sent through the persistent timeline component."""
    component = app_test.get("bidi_component")[0]
    return json.loads(json.loads(component.proto.json)["figure_json"])


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
    assert at.get("bidi_component")
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


def test_context_checkbox_changes_overlay_without_another_preparation(
    synthetic_timeline_session,
):
    """A saved Stage 1 result with logged transitions gains a default-on display toggle."""
    import pandas as pd
    from src.neural_analysis.population.run_session_timeline import run_session_timeline

    session = synthetic_timeline_session
    event_path = session.behavior.event_table_file
    events = pd.read_csv(event_path)
    transitions = pd.DataFrame(
        {
            "Time": [1700000000.0, 1700000014.0, 1700000028.0],
            "Event": ["enter_left_patch", "enter_right_patch", "enter_dark_period"],
            "Note": ["", "", ""],
        }
    )
    pd.concat([events, transitions], ignore_index=True).to_csv(event_path, index=False)
    report = run_session_timeline(session.metadata_path, {"PFC": "A"})
    modified_at = report.result_path.stat().st_mtime_ns
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select(
        "A"
    ).run()
    assert not at.exception
    checkbox = next(
        widget for widget in at.checkbox if widget.label == "Show active context"
    )
    assert checkbox.value is True
    assert len(_timeline_figure_spec(at)["layout"]["shapes"]) == 3
    checkbox.uncheck().run()
    assert not at.exception
    assert not _timeline_figure_spec(at)["layout"].get("shapes")
    assert report.result_path.stat().st_mtime_ns == modified_at
    assert (
        len(list((session.session_root / "analysis_runs").glob("*/results.npz"))) == 1
    )


def test_zscore_checkbox_reuses_saved_result_and_keeps_manual_limit_units_separate(
    synthetic_timeline_session, monkeypatch
):
    """Live display normalization updates units without fitting PCA or rewriting an archive."""
    from src.neural_analysis.population.run_session_timeline import run_session_timeline

    session = synthetic_timeline_session
    report = run_session_timeline(session.metadata_path, {"PFC": "A"})
    before = report.result_path.read_bytes()
    modified_at = report.result_path.stat().st_mtime_ns
    monkeypatch.setattr(
        timeline_view,
        "run_session_timeline",
        Mock(side_effect=AssertionError("Display must not prepare PCA")),
    )
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select(
        "A"
    ).run()
    assert not at.exception
    checkbox = next(
        widget for widget in at.checkbox if widget.label == "Z-score each PC"
    )
    assert checkbox.value is False
    raw = _timeline_figure_spec(at)
    raw_heatmap = next(trace for trace in raw["data"] if trace["type"] == "heatmap")
    assert raw_heatmap["colorbar"]["title"]["text"] == "PFC score"
    raw_variance = [item.value for item in at.markdown if "80% =" in item.value]
    checkbox.check().run()
    assert not at.exception
    standardized = _timeline_figure_spec(at)
    z_heatmap = next(
        trace for trace in standardized["data"] if trace["type"] == "heatmap"
    )
    assert "z-score" in z_heatmap["colorbar"]["title"]["text"]
    assert "SD" in z_heatmap["colorbar"]["title"]["text"]
    assert z_heatmap["z"] != raw_heatmap["z"]
    assert [item.value for item in at.markdown if "80% =" in item.value] == raw_variance
    assert not any("no score rescaling" in item.value for item in at.caption)
    next(
        widget for widget in at.checkbox if widget.label == "PFC manual color range"
    ).check().run()
    z_limit = next(
        widget
        for widget in at.number_input
        if widget.label == "PFC symmetric z-score limit (SD)"
    )
    z_limit.set_value(0.5).run()
    assert not at.exception
    manual = _timeline_figure_spec(at)
    manual_heatmap = next(
        trace for trace in manual["data"] if trace["type"] == "heatmap"
    )
    assert (manual_heatmap["zmin"], manual_heatmap["zmax"]) == (-0.5, 0.5)
    next(
        widget for widget in at.checkbox if widget.label == "Z-score each PC"
    ).uncheck().run()
    assert not at.exception
    raw_limit = next(
        widget
        for widget in at.number_input
        if widget.label == "PFC symmetric score limit"
    )
    assert raw_limit.value == raw_heatmap["zmax"]
    assert report.result_path.read_bytes() == before
    assert report.result_path.stat().st_mtime_ns == modified_at
    assert (
        len(list((session.session_root / "analysis_runs").glob("*/results.npz"))) == 1
    )


def test_zscore_option_is_disabled_without_prepared_pc_scores(
    synthetic_timeline_session,
):
    """A behavior-only session cannot enable a neural normalization display."""
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(synthetic_timeline_session.metadata_path),)
    ).run()
    assert not at.exception
    checkbox = next(
        widget for widget in at.checkbox if widget.label == "Z-score each PC"
    )
    assert checkbox.disabled and checkbox.value is False


def test_rank_changes_keep_component_identity_and_navigation_revision(
    synthetic_timeline_session,
):
    """Rank and z-score changes update data in place; explicit navigation requests a new range."""
    from src.neural_analysis.population.run_session_timeline import run_session_timeline

    session = synthetic_timeline_session
    report = run_session_timeline(session.metadata_path, {"PFC": "A"})
    before = report.result_path.read_bytes()
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select(
        "A"
    ).run()
    assert not at.exception
    component_id = at.get("bidi_component")[0].proto.id
    original = _timeline_figure_spec(at)
    ranks = next(widget for widget in at.slider if widget.label == "PFC PC ranks")
    ranks.set_value((2, 2)).run()
    assert not at.exception
    changed = _timeline_figure_spec(at)
    assert changed["data"] != original["data"]
    assert at.get("bidi_component")[0].proto.id == component_id
    assert changed["layout"]["uirevision"] == original["layout"]["uirevision"]
    next(
        widget for widget in at.checkbox if widget.label == "Z-score each PC"
    ).check().run()
    assert not at.exception
    assert at.get("bidi_component")[0].proto.id == component_id
    assert (
        _timeline_figure_spec(at)["layout"]["uirevision"]
        == original["layout"]["uirevision"]
    )
    duration = next(
        widget for widget in at.number_input if widget.label == "Visible duration (s)"
    )
    duration.set_value(10.0).run()
    shorter = _timeline_figure_spec(at)
    assert shorter["layout"]["uirevision"] != original["layout"]["uirevision"]
    start = next(
        widget for widget in at.number_input if widget.label == "Window start (s)"
    )
    start.set_value(20.0).run()
    moved = _timeline_figure_spec(at)
    assert moved["layout"]["xaxis"]["range"] == [20.0, 30.0]
    assert moved["layout"]["uirevision"] != shorter["layout"]["uirevision"]
    assert report.result_path.read_bytes() == before


def test_repeated_jump_and_clamped_duration_changes_still_request_navigation(
    synthetic_timeline_session,
):
    """A deliberate navigation action overrides a browser pan even if target bounds repeat."""
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(synthetic_timeline_session.metadata_path),)
    ).run()
    first = _timeline_figure_spec(at)["layout"]
    next(
        widget for widget in at.number_input if widget.label == "Visible duration (s)"
    ).set_value(200.0).run()
    full = _timeline_figure_spec(at)["layout"]
    next(
        widget for widget in at.number_input if widget.label == "Visible duration (s)"
    ).set_value(300.0).run()
    clamped = _timeline_figure_spec(at)["layout"]
    assert full["xaxis"]["range"] == clamped["xaxis"]["range"]
    assert full["uirevision"] != clamped["uirevision"]
    assert first["uirevision"] != clamped["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Jump to trial").select(
        102
    ).run()
    jump = next(widget for widget in at.button if widget.label == "Jump")
    jump.click().run()
    jumped = _timeline_figure_spec(at)["layout"]
    next(widget for widget in at.button if widget.label == "Jump").click().run()
    repeated = _timeline_figure_spec(at)["layout"]
    assert jumped["xaxis"]["range"] == repeated["xaxis"]["range"]
    assert jumped["uirevision"] != repeated["uirevision"]
