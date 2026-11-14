"""Separate dropdown routing and a source-free population decoder inspector."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.population_decoding import pipeline, preparation
from src.neural_analysis.population_decoding.run_session import run_session
from src.neural_analysis.webapp import app, session_inputs
from src.neural_analysis.webapp.population_decoding_view import render_population_decoding_view


def render_for_test(metadata_path):
    """Input metadata filename; render only saved population results in Streamlit."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.population_decoding_view import render_population_decoding_view
    render_population_decoding_view(st, load_webapp_session(metadata_path))


def forbid_analysis(*args, **kwargs):
    """Input arbitrary arguments; forbid any computation/source loading in saved view."""
    pytest.fail("Saved view attempted analysis/source loading")


def test_separate_route_dispatches_before_live_probe_controls(population_config_path, monkeypatch):
    """Input metadata/patch; both dropdown entries remain distinct and new route returns early."""
    session = session_inputs.load_webapp_session(population_config_path.parent / "neural_session.json")
    assert session_inputs.PLOT_VIEW_CONTEXT_ENCODING == "Single unit selectivity"
    assert session_inputs.PLOT_VIEW_POPULATION_DECODING == "Population task decoding"
    assert session_inputs.PLOT_VIEW_POPULATION_DECODING in session_inputs.PLOT_VIEW_OPTIONS
    fake = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(
        return_value=session_inputs.PLOT_VIEW_POPULATION_DECODING)))
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake)
    monkeypatch.setattr(app, "render_population_decoding_view", renderer)
    assert app._start_metadata_webapp(session) is None
    renderer.assert_called_once_with(fake, session)


def test_view_and_selector_changes_work_without_raw_sources(population_config_path, monkeypatch):
    """Input source session; remove raw data then inspect units/PCs without fitting."""
    report = run_session(population_config_path)
    metadata = population_config_path.parent / "neural_session.json"
    for path in (population_config_path.parent / "trials.csv", population_config_path.parent / "events.csv",
                 population_config_path.parent / "A/aligned.npz", population_config_path.parent / "B/aligned.npz"):
        path.unlink()
    monkeypatch.setattr(pipeline, "fit_target", forbid_analysis)
    monkeypatch.setattr(preparation, "prepare_session", forbid_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_analysis)
    session = session_inputs.load_webapp_session(metadata)
    assert session_inputs.metadata_view_availability(session)[session_inputs.PLOT_VIEW_POPULATION_DECODING].available
    view = AppTest.from_function(render_for_test, args=(str(metadata),)).run()
    assert not view.exception
    assert view.get("bidi_component")
    assert any(item.value == "Population task decoding" for item in view.subheader)
    next(item for item in view.selectbox if item.label == "Population features").select("pcs").run()
    assert not view.exception and view.get("bidi_component")
    next(item for item in view.selectbox if item.label == "Target").select("outcome").run()
    assert not view.exception


def test_empty_view_explains_how_to_compute(population_config_path):
    """Input metadata without results; view gives CLI guidance instead of computing."""
    view = AppTest.from_function(render_for_test, args=(str(population_config_path.parent / "neural_session.json"),)).run()
    assert not view.exception
    assert any("No saved" in item.value for item in view.info)
    assert view.code
