"""Real Streamlit prediction controls and raw-source-free app dispatch."""

import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from streamlit.testing.v1 import AppTest

from src.neural_analysis.task_decoding.session_prediction.run_session import prepare_session_predictions
from src.neural_analysis.task_decoding.session_prediction.results import load_prediction_result
from src.neural_analysis.population.run_window_variance import run_window_variance
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


def test_hidden_pc_view_does_not_offer_invisible_window_count_tracks(runnable_session):
    """Controls for heatmap-attached variance rows stay hidden when PC rows are hidden."""
    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"})
    saved = load_prediction_result(report.result_path)
    run_window_variance(runnable_session.parent / saved.meta["source_pca"]["path"])
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    assert not any(widget.label == "Show fixed-axis window counts" for widget in at.checkbox)


@pytest.mark.parametrize("mode", ["correct_rewarded_generalization", "mixed_outcome_decoding"])
def test_saved_only_unprepared_switch_and_hidden_predictions_keep_anchor(runnable_session, tmp_path, monkeypatch, mode):
    """Raw-free mode switches retain saved behavior/PCA and suppress every model overlay."""
    from src.neural_analysis.webapp import session_prediction_view, timeline_view

    report = prepare_session_predictions(
        runnable_session, probe_roles={"PFC": "A", "HPC": "B"}, analysis_mode=mode,
    )
    copied = tmp_path / "portable"
    shutil.copytree(runnable_session.parent / "analysis_runs", copied / "analysis_runs")
    shutil.copyfile(runnable_session, copied / "neural_session.json")

    def forbidden(*args, **kwargs):
        """Any preparation/raw lookup violates portable saved inspection; return never."""
        raise AssertionError("Saved-only switching cannot use raw inputs or fit")

    monkeypatch.setattr(timeline_view.timeline_io, "timeline_configuration", forbidden)
    monkeypatch.setattr(timeline_view, "load_timeline_behavior_cached", forbidden)
    monkeypatch.setattr(timeline_view, "run_session_timeline", forbidden)
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    at = AppTest.from_function(render_prediction_test_app, args=(str(copied / "neural_session.json"),)).run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])
    assert next(widget for widget in at.selectbox if widget.label == "Analysis").value == mode
    next(widget for widget in at.checkbox if widget.label == "Show predictions").uncheck().run()
    assert not at.exception
    assert not any("prediction" in trace.get("name", "") or "actual" in trace.get("name", "")
                   for trace in figure_spec(at)["data"])
    assert any(trace.get("name") == "PFC" for trace in figure_spec(at)["data"])
    next(widget for widget in at.checkbox if widget.label == "Show predictions").check().run()
    next(widget for widget in at.selectbox if widget.label == "Prediction PC view").select("weighted").run()
    other = "mixed_outcome_decoding" if mode == "correct_rewarded_generalization" else "correct_rewarded_generalization"
    before = figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Analysis").select(other).run()
    assert not at.exception
    spec = figure_spec(at)
    assert spec["layout"]["uirevision"] == before
    assert any("not prepared" in caption.value for caption in at.caption)
    assert any("500 ms" in caption.value and "background" in caption.value.lower() for caption in at.caption)
    assert not any("prediction" in trace.get("name", "") or "actual" in trace.get("name", "")
                   for trace in spec["data"])
    assert any(trace.get("name") == "PFC" for trace in spec["data"])
    assert next(widget for widget in at.button if widget.label == "Recompute predictions").disabled
    next(widget for widget in at.selectbox if widget.label == "Analysis").select(mode).run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])
    assert (copied / report.result_path.relative_to(runnable_session.parent)).exists()


def test_cross_configuration_mode_switch_retains_navigation_and_clear_labels(runnable_session, monkeypatch):
    """500/100-ms mode switches keep time navigation while controls follow saved basis."""
    from src.neural_analysis.webapp import session_prediction_view

    roles = {"PFC": "A", "HPC": "B"}
    reference = prepare_session_predictions(runnable_session, probe_roles=roles)
    mixed = prepare_session_predictions(runnable_session, probe_roles=roles, bin_width_ms=100,
                                        analysis_mode="mixed_outcome_decoding")
    assert load_prediction_result(reference.result_path).meta["source_pca"]["configuration_id"] != load_prediction_result(mixed.result_path).meta["source_pca"]["configuration_id"]
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", Mock(side_effect=AssertionError("No refit")))
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    next(widget for widget in at.number_input if widget.label == "Visible duration (s)").set_value(40.0).run()
    next(widget for widget in at.number_input if widget.label == "Window start (s)").set_value(50.0).run()
    next(widget for widget in at.selectbox if widget.label == "Jump to trial").select(9).run()
    initial = figure_spec(at)
    assert any("State generalization (correct-rewarded training)" in item["text"]
               for item in initial["layout"]["annotations"])
    next(widget for widget in at.selectbox if widget.label == "Analysis").select("mixed_outcome_decoding").run()
    assert not at.exception
    updated = figure_spec(at)
    assert updated["layout"]["uirevision"] == initial["layout"]["uirevision"]
    assert updated["layout"]["xaxis"]["range"] == initial["layout"]["xaxis"]["range"]
    assert updated["layout"]["yaxis"]["uirevision"] != initial["layout"]["yaxis"]["uirevision"]
    assert next(widget for widget in at.number_input if widget.label == "Visible duration (s)").value == 40
    assert next(widget for widget in at.number_input if widget.label == "Window start (s)").value == 50
    assert next(widget for widget in at.selectbox if widget.label == "Jump to trial").value == 9
    assert next(widget for widget in at.selectbox if widget.label == "Prediction target").options == ["State decoding", "Action decoding"]
    assert "mixed_outcome_decoding" in next(widget for widget in at.selectbox if widget.label == "Saved predictions").options[0]
    assert any("State decoding (mixed-outcome training)" in item["text"] for item in updated["layout"]["annotations"])
    assert any("sample size" in item.value and "regularization" in item.value for item in at.caption)
    next(widget for widget in at.button if widget.label == "Jump").click().run()
    assert figure_spec(at)["layout"]["uirevision"] != updated["layout"]["uirevision"]


def test_gui_prepare_forwards_selected_mode_and_bounded_caption(runnable_session):
    """Explicit mixed Prepare yields two-target products and explains bounded resets."""
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in at.selectbox if widget.label == "Analysis").select("mixed_outcome_decoding").run()
    next(widget for widget in at.button if widget.label == "Prepare predictions").click().run()
    assert not at.exception
    path = next(runnable_session.parent.glob("analysis_runs/condition_session_prediction_*/results.npz"))
    assert load_prediction_result(path).inputs.target_names == ("current_state", "current_action")
    next(widget for widget in at.checkbox if widget.label == "Load only visible neural interval").check().run()
    assert any("reset" in item.value.lower() and "loaded" in item.value.lower() for item in at.caption)
