"""Real Streamlit prediction controls and raw-source-free app dispatch."""

import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src.neural_analysis.task_decoding.session_prediction.run_session import prepare_session_predictions
from src.neural_analysis.webapp import app, session_inputs
from src.tests.neural_analysis.task_decoding.session_prediction.test_scripts import (
    runnable_session as runnable_fixture, synthetic_timeline_session as timeline_fixture,
)

synthetic_timeline_session = timeline_fixture
runnable_session = runnable_fixture


def render_prediction_test_app(metadata_path):
    """Render the actual timeline view for one metadata path; all scientific data are saved."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.timeline_view import render_session_timeline_view

    render_session_timeline_view(st, load_webapp_session(metadata_path))


def figure_spec(test_app):
    """Return the persistent timeline component's JSON figure, with numeric elapsed-second axes."""
    component = test_app.get("bidi_component")[0]
    return json.loads(json.loads(component.proto.json)["figure_json"])


def test_explicit_prepare_and_display_changes_never_refit(runnable_session, monkeypatch):
    """Buttons prepare; subsequent target/mode/event changes load only saved arrays."""
    from src.neural_analysis.webapp import session_prediction_view

    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in at.selectbox if widget.label == "HPC probe").select("B").run()
    assert not list(runnable_session.parent.glob("analysis_runs/condition_session_prediction_*"))
    next(widget for widget in at.button if widget.label == "Prepare predictions").click().run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])

    def forbidden(*args, **kwargs):
        """Any fit on a display-only rerun is a failure; return never."""
        raise AssertionError("Display cannot prepare predictions")

    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    count = len(list(runnable_session.parent.glob("analysis_runs/*")))
    next(widget for widget in at.selectbox if widget.label == "Prediction target").select("relative_doubt").run()
    next(widget for widget in at.selectbox if widget.label == "Prediction PC view").select("weighted").run()
    assert any(widget.label == "PFC PC ranks" for widget in at.slider)
    assert not any("Default color limits" in item.value for item in at.caption)
    before = figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Prediction PC order").select("contribution").run()
    assert not at.exception
    assert figure_spec(at)["layout"]["uirevision"] == before
    assert len(list(runnable_session.parent.glob("analysis_runs/*"))) == count


def test_relocated_saved_display_and_actual_app_dispatch_without_raw_sources(runnable_session, tmp_path, monkeypatch):
    """The earlier availability gate and complete renderer both accept saved-only copies."""
    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"})
    copied = tmp_path / "copied"
    shutil.copytree(runnable_session.parent / "analysis_runs", copied / "analysis_runs")
    shutil.copyfile(runnable_session, copied / "neural_session.json")
    path = copied / "neural_session.json"
    session = session_inputs.load_webapp_session(path)
    assert not session.behavior.trial_table_file.exists()
    assert session_inputs.metadata_view_availability(session)[session_inputs.PLOT_VIEW_SESSION_TIMELINE].available
    fake_st = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=session_inputs.PLOT_VIEW_SESSION_TIMELINE)))
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_session_timeline_view", renderer)
    assert app._start_metadata_webapp(session) is None
    renderer.assert_called_once_with(fake_st, session)
    at = AppTest.from_function(render_prediction_test_app, args=(str(path),)).run()
    assert not at.exception
    names = [trace.get("name") for trace in figure_spec(at)["data"]]
    assert "PFC prediction" in names and "HPC prediction" in names
    assert (copied / report.result_path.relative_to(runnable_session.parent)).is_file()
