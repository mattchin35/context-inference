"""Saved-load cache and explicit navigation contracts for the CCA viewer."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from streamlit.testing.v1 import AppTest

from src.neural_analysis.population import run_session_cca as runner
from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.webapp import app
from src.tests.neural_analysis.session_cca.test_synthetic_integration import (
    shared_private_session as shared_fixture,
)
from src.tests.neural_analysis.session_cca.test_webapp import _figure, _render_cca


shared_private_session = shared_fixture
ROLES = {"PFC": "A", "HPC": "B"}


def test_saved_result_cache_uses_archive_path_size_and_modification_time(shared_private_session, monkeypatch):
    """Repeated display loads reuse one decode; a changed file stat invalidates that cache."""
    from src.neural_analysis.webapp import cca_view

    report = runner.run_session_cca(shared_private_session.metadata_path, ROLES)
    load = Mock(wraps=cca_io.load_cca_result)
    monkeypatch.setattr(cca_io, "load_cca_result", load)
    cca_view.load_cca_result_cached.clear()
    path = report.result_path
    before = path.stat()
    first = cca_view.load_cca_result_cached(str(path), before.st_size, before.st_mtime_ns)
    cached = cca_view.load_cca_result_cached(str(path), before.st_size, before.st_mtime_ns)
    assert load.call_count == 1
    np.testing.assert_array_equal(first.fit.scores["PFC"], cached.fit.scores["PFC"])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    after = path.stat()
    cca_view.load_cca_result_cached(str(path), after.st_size, after.st_mtime_ns)
    assert load.call_count == 2
    cca_view.load_cca_result_cached.clear()


def test_saved_fine_bins_default_to_bounded_plotting_and_explicit_controls_load_elsewhere(shared_private_session):
    """Saved 20-ms geometry sets the default; start/Jump change bounds and time revision."""
    session = shared_private_session
    report = runner.run_session_cca(session.metadata_path, ROLES, .02)
    before = report.result_path.read_bytes()
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    assert next(widget for widget in at.checkbox if widget.label == "Load only visible neural interval").value is True
    initial = _figure(at)["layout"]
    assert initial["xaxis"]["maxallowed"] < 150
    next(widget for widget in at.number_input if widget.label == "Window start (s)").set_value(100.0).run()
    assert not at.exception
    moved = _figure(at)["layout"]
    assert moved["uirevision"] != initial["uirevision"]
    assert moved["xaxis"]["range"] == [100.0, 160.0]
    assert moved["xaxis"]["minallowed"] >= 40
    assert moved["xaxis"]["maxallowed"] >= 220
    next(widget for widget in at.selectbox if widget.label == "Jump to trial").select(118).run()
    assert _figure(at)["layout"]["uirevision"] == moved["uirevision"]
    next(widget for widget in at.button if widget.label == "Jump").click().run()
    assert not at.exception
    jumped = _figure(at)["layout"]
    assert jumped["uirevision"] != moved["uirevision"]
    assert jumped["xaxis"]["range"][0] > 180
    assert jumped["xaxis"]["range"][1] > 240
    assert report.result_path.read_bytes() == before


def test_manual_route_explains_metadata_requirement_before_any_path_or_signal_work(monkeypatch):
    """Selecting CCA without metadata exits cleanly before manual source initialization."""
    fake_st = SimpleNamespace(
        set_page_config=Mock(), title=Mock(), info=Mock(),
        sidebar=SimpleNamespace(button=Mock(return_value=False),
                                selectbox=Mock(return_value="Session timeline and CCA")),
    )
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "_initialize_path_input_state", Mock(side_effect=AssertionError("manual paths must not initialize")))
    app.main([])
    assert fake_st.info.call_count == 1
    assert "--session-metadata" in fake_st.info.call_args.args[0]
