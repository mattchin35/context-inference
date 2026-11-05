"""Saved-only route, independent products and manual Refresh contracts (WP6b)."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.phase_synchrony import computation, preparation, results, run_session as runner
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.webapp import app, session_inputs, phase_synchrony_view as view
from src.tests.neural_analysis.phase_synchrony.test_preparation import binary_session
from src.tests.neural_analysis.phase_synchrony.test_runner import runnable_session, config


def _resolve(path):
    """Return resolved metadata from one synthetic session JSON path; no scientific arrays."""
    return resolve_session_metadata(load_session_metadata(path), path)


def _forbid_work(*args, **kwargs):
    """Reject analysis or raw loading during any saved-view operation."""
    pytest.fail("Saved viewer attempted analysis/raw loading")


def _forbid_analysis(monkeypatch):
    """Forbid all numerical preparation/estimators and the session runner; return None."""
    monkeypatch.setattr(runner, "run_session", _forbid_work)
    for name in ("load_site_lfp", "prepare_site_lfp", "prepare_site_phase", "load_selected_unit_spikes"):
        monkeypatch.setattr(preparation, name, _forbid_work)
    for name in ("compute_site_psd", "compute_band_power", "compute_welch_coherence", "compute_lfp_phase_summary", "compute_selected_unit_ppc"):
        monkeypatch.setattr(computation, name, _forbid_work)


def test_route_requires_behavior_only_and_dispatches_before_probe_controls(runnable_session, monkeypatch):
    """A missing unrelated probe/spike source cannot disable the dedicated saved-result timeline."""
    session = _resolve(runnable_session)
    route = session_inputs.PLOT_VIEW_PHASE_SYNCHRONY
    assert route in session_inputs.PLOT_VIEW_OPTIONS
    assert session_inputs.metadata_view_availability(session)[route].available
    fake_st = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=route)))
    render = Mock()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_phase_synchrony_view", render)
    assert app._start_metadata_webapp(session) is None
    render.assert_called_once_with(fake_st, session)
    session.behavior.trial_table_file.unlink()
    assert not session_inputs.metadata_view_availability(session)[route].available


def test_saved_power_is_discovered_independently_of_unit_selection(runnable_session, config, monkeypatch):
    """Changing selected saved units does not invalidate unrelated power or open numerical sources."""
    selected = replace(config, sites=("NamedSite",), pairs=())
    report = runner.run_session(runnable_session, selected)
    _forbid_analysis(monkeypatch)
    session = _resolve(runnable_session)
    first = view.discover_view_products(session, selected)
    changed = view.discover_view_products(session, replace(selected, unit_ids=("ProbeZ:29",), lfp_site="NamedSite"))
    assert [product.path for product in first] == [report.products[0]]
    assert [product.path for product in changed] == [report.products[0]]
    assert first[0].meta["run_id"] and first[0].meta["completed_utc"]
    assert view.discover_view_products(session, replace(selected, intervals=("secondary",))) == ()
    session.behavior.trial_table_file.write_text(session.behavior.trial_table_file.read_text() + "\n", encoding="ascii")
    assert view.discover_view_products(session, selected) == ()


@pytest.mark.parametrize("target_site,change_phase", [("NamedSite", False), ("NamedSite", True), ("OtherSite", False)])
def test_newer_failed_attempt_is_separate_from_selected_completed_source(runnable_session, config, monkeypatch, target_site, change_phase):
    """Recorded UTC attempt ordering reports failure while preserving the older compatible result."""
    selected = replace(config, sites=("NamedSite",), pairs=())
    report = runner.run_session(runnable_session, selected)
    if target_site == "OtherSite":
        selected = replace(selected, sites=("NamedSite", "OtherSite"))
    session = _resolve(runnable_session)
    attempt = session.session_root / "analysis_runs" / "session_timeline_phase_synchrony_20300102_030405_000000_new"
    attempt.mkdir()
    attempt_config = replace(selected, phase_frequencies_hz=(6., 8., 10.)) if change_phase else selected
    (attempt / "config.json").write_text(json.dumps({"session_id": session.session_id, **asdict(attempt_config)}), encoding="ascii")
    (attempt / "measurements.json").write_text(json.dumps({
        "run_id": "20300102_030405_000000_new", "stages": [{"stage": "power", "selection": [target_site],
        "interval": "primary", "status": "failed", "error": "Synthetic interrupted input"}],
    }), encoding="ascii")
    _forbid_analysis(monkeypatch)
    products = view.discover_view_products(session, selected)
    assert [product.path for product in products] == list(report.products)
    statuses = view.newer_attempt_statuses(session.session_root, selected, products)
    failure = next(status for status in statuses if status["stage"] == "power" and tuple(status["selection"]) == (target_site,))
    assert failure["status"] == "failed" and failure["run_id"].endswith("_new")
    assert failure["displayed_run_id"] == (products[0].meta["run_id"] if target_site == "NamedSite" else None)
    assert failure["error"] == "Synthetic interrupted input"


def _render_for_test(metadata_path):
    """Render only the saved synchrony route for one synthetic metadata path."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.phase_synchrony_view import render_phase_synchrony_view
    render_phase_synchrony_view(st, load_webapp_session(metadata_path))


def _figure_spec(app_test):
    """Return the actual persistent component ID and Plotly figure specification."""
    component = app_test.get("bidi_component")[0]
    return component.proto.id, json.loads(json.loads(component.proto.json)["figure_json"])


def _metric_specs(spec):
    """Return saved scalar trace dictionaries from the component payload."""
    return [trace for trace in spec["data"] if isinstance(trace.get("meta"), dict)
            and trace["meta"].get("kind") == "metric"]


def test_manual_refresh_discovers_new_power_and_keeps_chart_navigation(runnable_session, config):
    """Real Streamlit reruns transition behavior-only to partial saved power with stable chart identity."""
    at = AppTest.from_function(_render_for_test, args=(str(runnable_session),)).run()
    assert not at.exception
    before_id, before = _figure_spec(at)
    assert not _metric_specs(before)
    report = runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    next(button for button in at.button if button.label == "Refresh results").click().run()
    assert not at.exception
    after_id, after = _figure_spec(at)
    assert after_id == before_id and after["layout"]["uirevision"] == before["layout"]["uirevision"]
    assert _metric_specs(after) and {trace["meta"]["family"] for trace in _metric_specs(after)} == {"power"}
    assert any("coherence" in element.value.lower() for element in at.info)
    assert len(list((runnable_session.parent / "analysis_runs").glob("session_timeline_phase_synchrony_*"))) == 1
    assert any(report.products[0].name in str(value) for frame in at.dataframe
               for value in frame.value.to_numpy().ravel())


def test_display_changes_and_refresh_never_prepare_signals(runnable_session, config, monkeypatch):
    """Interval/metric/event choices use saved products; manual Refresh does not invoke the runner."""
    runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    _forbid_analysis(monkeypatch)
    at = AppTest.from_function(_render_for_test, args=(str(runnable_session),)).run()
    assert not at.exception
    first_id, first = _figure_spec(at)
    interval = next(widget for widget in at.selectbox if widget.label == "Centered interval")
    interval.select("secondary").run()
    assert not at.exception
    second_id, second = _figure_spec(at)
    assert first_id == second_id and first["layout"]["uirevision"] == second["layout"]["uirevision"]
    assert not _metric_specs(second)
    next(button for button in at.button if button.label == "Refresh results").click().run()
    assert not at.exception
    assert len(list((runnable_session.parent / "analysis_runs").glob("session_timeline_phase_synchrony_*"))) == 1


def test_inspection_requires_explicit_request_and_loads_saved_prerequisites_only(runnable_session, config, monkeypatch):
    """Inspection resolves exact prepared phase IDs and waveform rows without preparing absent prerequisites."""
    selected = replace(config, components=("spike-ppc",), sites=("NamedSite",), pairs=(),
                       unit_ids=("ProbeZ:17",), lfp_site="NamedSite")
    runner.run_session(runnable_session, selected)
    _forbid_analysis(monkeypatch)
    session = _resolve(runnable_session)
    products = view.discover_view_products(session, selected)
    inspection = view.load_inspection_products(session, selected, products, site_id="NamedSite", unit_id="ProbeZ:17")
    families = {product.meta["identity"]["family"] for product in inspection}
    assert "phase" in families and "spike-ppc" in families
    assert "waveforms" not in families  # Spike PPC did not prepare Welch waveform windows.
    phase = next(product for product in inspection if product.meta["identity"]["family"] == "phase")
    assert results.preparation_identity(phase.record) in products[0].meta["identity"]["preparation_ids"]


def test_empty_site_display_selection_hides_site_metrics(runnable_session, config):
    """Clearing a display multiselect means hide those rows, independent of offline empty-means-default config."""
    runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    at = AppTest.from_function(_render_for_test, args=(str(runnable_session),)).run()
    assert not at.exception and _metric_specs(_figure_spec(at)[1])
    next(widget for widget in at.multiselect if widget.label == "LFP sites").set_value([]).run()
    assert not at.exception
    assert not _metric_specs(_figure_spec(at)[1])


def test_partial_power_labels_each_unavailable_selected_site(runnable_session, config):
    """An available power family cannot hide another selected site's missing product."""
    runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    at = AppTest.from_function(_render_for_test, args=(str(runnable_session),)).run()
    assert not at.exception
    assert any("power" in element.value and "OtherSite" in element.value for element in at.info)


def test_initial_saved_unit_uses_its_actual_lfp_site(runnable_session, config):
    """A saved unit on the second metadata site appears without silently selecting the first site."""
    selected = replace(config, components=("spike-ppc",), unit_ids=("ProbeZ:17",), lfp_site="OtherSite")
    runner.run_session(runnable_session, selected)
    at = AppTest.from_function(_render_for_test, args=(str(runnable_session),)).run()
    assert not at.exception
    assert next(widget for widget in at.selectbox if widget.label == "Spike-PPC LFP site").value == "OtherSite"
    assert any(trace["meta"]["family"] == "spike-ppc" for trace in _metric_specs(_figure_spec(at)[1]))


def test_inspection_without_explicit_unit_does_not_substitute_a_saved_unit(runnable_session, config):
    """No-unit LFP inspection excludes spike and polar observations even if a unit result is available."""
    selected = replace(config, components=("spike-ppc",), sites=("NamedSite",), pairs=(),
                       unit_ids=("ProbeZ:17",), lfp_site="NamedSite")
    runner.run_session(runnable_session, selected)
    session = _resolve(runnable_session)
    products = view.discover_view_products(session, selected)
    inspection = view.load_inspection_products(session, selected, products, site_id="NamedSite", unit_id=None)
    assert all(product.meta["identity"]["family"] != "spike-ppc" for product in inspection)
