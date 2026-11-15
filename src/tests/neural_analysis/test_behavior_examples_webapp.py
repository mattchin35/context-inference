"""Behavior-example routing, navigation and export without neural IO."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.tests.neural_analysis.event_sequences.example_fixtures import example_session, example_trials


def _render_examples(metadata_path: str) -> None:
    """Input: synthetic metadata Path string; render the single-session Streamlit view."""
    import streamlit as st
    from src.neural_analysis.webapp.behavior_examples_view import render_behavior_examples_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_behavior_examples_view(st, load_webapp_session(metadata_path))


def _widget(app_test, kind: str, label: str):
    """Input: AppTest/widget-kind/label strings. Return the uniquely labeled widget."""
    return next(item for item in getattr(app_test, kind) if item.label == label)


def _forbid(*args, **kwargs) -> None:
    """Input: any attempted IO/preparation arguments. Fail if browsing calls neural operations."""
    raise AssertionError("Unexpected IO while browsing behavior examples")


def test_early_metadata_dispatch_skips_neural_inventory(example_session, monkeypatch):
    """Input: resolved session and fake sidebar. Dispatch before shared neural availability."""
    from src.neural_analysis.webapp import app, session_inputs

    assert "Behavior examples" in session_inputs.PLOT_VIEW_OPTIONS
    renderer = Mock()
    monkeypatch.setattr(app, "st", SimpleNamespace(sidebar=SimpleNamespace(
        caption=Mock(), selectbox=Mock(return_value="Behavior examples"))))
    monkeypatch.setattr(app, "metadata_view_availability", _forbid)
    monkeypatch.setattr(app, "render_behavior_examples_view", renderer)
    assert app._start_metadata_webapp(example_session) is None
    renderer.assert_called_once()
    assert session_inputs.behavior_examples_availability(example_session).available


def test_real_widgets_browse_and_export_without_source_reloads(example_session, monkeypatch):
    """Input: session without probes. Navigation/context changes must reuse saved behavioral results."""
    view_module = importlib.import_module("src.neural_analysis.webapp.behavior_examples_view")
    from src.neural_analysis.event_sequences import io as neural_io

    monkeypatch.setattr(neural_io, "discover_event_sequences", _forbid)
    monkeypatch.setattr(neural_io, "load_probe_spike_arrays", _forbid)
    app_test = AppTest.from_function(_render_examples, args=(str(example_session.metadata_path),), default_timeout=30).run()
    assert not app_test.exception
    assert _widget(app_test, "selectbox", "Pattern").value == "probe"
    assert _widget(app_test, "number_input", "Trials before").value == 5
    assert _widget(app_test, "number_input", "Trials after").value == 5
    assert app_test.get("plotly_chart")
    monkeypatch.setattr(pd, "read_csv", _forbid)
    _widget(app_test, "button", "Next").click().run()
    assert not app_test.exception
    assert _widget(app_test, "selectbox", "Example").value == 1
    _widget(app_test, "number_input", "Trials before").set_value(0).run()
    _widget(app_test, "button", "Export PNG").click().run()
    assert not app_test.exception
    assert app_test.success
    _widget(app_test, "selectbox", "Pattern").select("indecisive_transition").run()
    _widget(app_test, "selectbox", "Initial side").select("right").run()
    assert not app_test.exception
    assert _widget(app_test, "selectbox", "Example").value == 0
    _widget(app_test, "selectbox", "Pattern").select("probe").run()
    _widget(app_test, "slider", "Incorrect repeats").set_value(4).run()
    assert not app_test.exception
    assert any("No matching" in message.value for message in app_test.info)


def test_malformed_trial_fields_report_error(example_session):
    """Input: CSV without correctness. Show an actionable error rather than a Streamlit exception."""
    table = pd.read_csv(example_session.behavior.trial_table_file).drop(columns="correct")
    table.to_csv(example_session.behavior.trial_table_file, index=False)
    app_test = AppTest.from_function(_render_examples, args=(str(example_session.metadata_path),), default_timeout=30).run()
    assert not app_test.exception
    assert any("correct" in error.value for error in app_test.error)
