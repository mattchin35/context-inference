"""CCA routing and real Streamlit display/preparation boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src.neural_analysis.population import run_session_cca as runner
from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.webapp import app, session_inputs
from src.tests.neural_analysis.session_cca.test_synthetic_integration import (
    shared_private_session as shared_fixture,
)


shared_private_session = shared_fixture
ROLES = {"PFC": "A", "HPC": "B"}


def test_cca_dispatch_precedes_active_probe_and_trial_loaders(shared_private_session, monkeypatch):
    """A dedicated metadata route owns all CCA inputs and returns before trial controls."""
    selected = "Session timeline and CCA"
    assert selected in session_inputs.PLOT_VIEW_OPTIONS
    fake_st = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=selected)))
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_session_cca_view", renderer)
    assert app._start_metadata_webapp(shared_private_session) is None
    renderer.assert_called_once_with(fake_st, shared_private_session)


def test_saved_cca_is_available_without_behavior_or_spike_sources(shared_private_session):
    """A completed historical product is checked before generic raw prerequisites."""
    session = shared_private_session
    runner.run_session_cca(session.metadata_path, ROLES)
    _remove_raw_sources(session)
    availability = session_inputs.metadata_view_availability(session)
    assert availability["Session timeline and CCA"].available
    assert not availability[session_inputs.PLOT_VIEW_TRIAL_SPIKES].available


def test_unprepared_cca_supports_behavior_only_and_never_fits_on_display(shared_private_session, monkeypatch):
    """Incomplete regional selections still render choices and navigation without analysis."""
    monkeypatch.setattr(runner.cca, "fit_paired_cca", _forbid_call)
    at = AppTest.from_function(_render_cca, args=(str(shared_private_session.metadata_path),)).run()
    assert not at.exception
    figure = _figure(at)
    assert not [trace for trace in figure["data"] if trace["type"] == "heatmap"]
    assert any(trace["name"] == "Left choices" for trace in figure["data"])
    assert any(widget.label == "Jump to trial" for widget in at.selectbox)
    assert not (shared_private_session.session_root / "analysis_runs").exists()


def test_saved_display_and_preparation_widgets_do_not_refit_or_relabel(shared_private_session, monkeypatch):
    """All display controls consume saved 500-ms scores even when prep settings change."""
    session = shared_private_session
    report = runner.run_session_cca(session.metadata_path, ROLES)
    before = report.result_path.read_bytes()
    monkeypatch.setattr(cca_io, "cca_configuration", _forbid_call)
    monkeypatch.setattr(runner.cca, "fit_paired_cca", _forbid_call)
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    figure = _figure(at)
    original_revision = figure["layout"]["uirevision"]
    heatmaps = [trace for trace in figure["data"] if trace["type"] == "heatmap"]
    assert len(heatmaps) == 2 and len(heatmaps[0]["y"]) == 3
    assert next(widget for widget in at.selectbox if widget.label == "Trace pair").value == 1
    next(widget for widget in at.slider if widget.label == "Canonical pair ranks").set_value((2, 3)).run()
    assert not at.exception
    next(widget for widget in at.selectbox if widget.label == "Trace pair").select(3).run()
    next(widget for widget in at.checkbox if widget.label == "Show canonical traces").uncheck().run()
    assert not at.exception
    assert _figure(at)["layout"]["uirevision"] == original_revision
    next(widget for widget in at.selectbox if widget.label == "CCA bin width (ms)").select(20).run()
    assert not at.exception
    # Prep settings cannot change the saved bin geometry or its descriptive caption.
    assert any("500 ms" in item.value for item in at.caption)
    assert _figure(at)["layout"]["uirevision"] == original_revision
    assert report.result_path.read_bytes() == before
    assert len(list((session.session_root / "analysis_runs").glob("*/results.npz"))) == 1


def test_copied_saved_view_loads_without_any_raw_arrays(shared_private_session, monkeypatch):
    """Raw-source absence prevents fresh reuse but not archived behavior/CCA viewing."""
    session = shared_private_session
    runner.run_session_cca(session.metadata_path, ROLES)
    _remove_raw_sources(session)
    monkeypatch.setattr(cca_io, "cca_configuration", _forbid_call)
    monkeypatch.setattr(cca_io, "load_cca_inputs", _forbid_call)
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    assert len([trace for trace in _figure(at)["data"] if trace["type"] == "heatmap"]) == 2


def test_prepare_and_recompute_are_explicit_and_additive(shared_private_session):
    """Two explicit buttons prepare once then create a distinct immutable rerun."""
    session = shared_private_session
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in at.selectbox if widget.label == "HPC probe").select("B").run()
    assert not (session.session_root / "analysis_runs").exists()
    next(widget for widget in at.button if widget.label == "Prepare").click().run()
    assert not at.exception
    paths = list((session.session_root / "analysis_runs").glob("*/results.npz"))
    assert len(paths) == 1
    before = paths[0].read_bytes()
    next(widget for widget in at.button if widget.label == "Recompute").click().run()
    assert not at.exception
    assert len(list((session.session_root / "analysis_runs").glob("*/results.npz"))) == 2
    assert paths[0].read_bytes() == before


def test_preparation_error_keeps_selected_valid_figure(shared_private_session, monkeypatch):
    """A failed explicit action reports its error while the prior valid product stays visible."""
    from src.neural_analysis.webapp import cca_view

    session = shared_private_session
    report = runner.run_session_cca(session.metadata_path, ROLES)
    monkeypatch.setattr(cca_view, "run_session_cca", Mock(side_effect=ValueError("numerical validation failed")))
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in at.selectbox if widget.label == "HPC probe").select("B").run()
    next(widget for widget in at.button if widget.label == "Recompute").click().run()
    assert not at.exception
    assert any("numerical validation failed" in error.value for error in at.error)
    assert len([trace for trace in _figure(at)["data"] if trace["type"] == "heatmap"]) == 2
    assert report.result_path.is_file()


def _render_cca(metadata_path: str) -> None:
    """Render the actual isolated metadata CCA route for an existing JSON path."""
    import streamlit as st
    from src.neural_analysis.webapp.cca_view import render_session_cca_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_session_cca_view(st, load_webapp_session(metadata_path))


def _figure(app_test: AppTest) -> dict:
    """Return the actual JSON figure passed to the persistent chart component."""
    component = app_test.get("bidi_component")[0]
    return json.loads(json.loads(component.proto.json)["figure_json"])


def _remove_raw_sources(session) -> None:
    """Remove only this fixture's raw trial/event/NPY/NPZ sources, preserving saved products."""
    session.behavior.trial_table_file.unlink()
    session.behavior.event_table_file.unlink()
    for probe in ("A", "B"):
        for name in ("aligned.npz", "spike_clusters.npy"):
            (session.session_root / probe / name).unlink()


def _forbid_call(*args, **kwargs):
    """Fail when a saved-only display attempts current configuration/input fitting work."""
    raise AssertionError("Saved display must use archived scores and behavior only")
