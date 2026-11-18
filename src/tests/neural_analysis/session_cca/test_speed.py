"""Optional CCA speed rows reuse the existing independent treadmill pipeline."""

import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.mouse_behavior_preprocessing.treadmill_speed import TreadmillSpeed
from src.neural_analysis import session_speed
from src.neural_analysis.population import cca_plotting
from src.neural_analysis.population.run_session_cca import run_session_cca
from src.tests.neural_analysis.session_cca.test_plotting import cca_display_result as display_fixture
from src.tests.neural_analysis.session_cca.test_webapp import _figure, _render_cca
from src.tests.neural_analysis.test_session_speed import (
    synthetic_speed_session as speed_session_fixture,
    synthetic_speed_messages as messages_fixture,
    synthetic_timeline_session as session_fixture,
)


cca_display_result = display_fixture
synthetic_speed_session = speed_session_fixture
synthetic_speed_messages = messages_fixture
synthetic_timeline_session = session_fixture
ROLES = {"PFC": "A", "HPC": "B"}


def test_speed_keeps_signed_samples_real_gaps_and_fixed_scale_without_changing_cca(cca_display_result):
    """Display-only (S,) UTC/mm/s steps remain intact across neural slices and row changes."""
    result = cca_display_result
    speed = TreadmillSpeed(
        result.time_origin_s + np.array([97., 99., 103., 107., 113., 117.]),
        np.array([-3., 0., 7., -2., 1., 4.]), np.arange(6),
        np.array([0., 180.]), result.time_origin_s + np.array([0., 180.]), {},
    )
    before = {region: scores.copy() for region, scores in result.fit.scores.items()}
    original = cca_plotting.plot_session_cca(result)
    figures = [
        cca_plotting.plot_session_cca(result, speed=speed, loaded_range_s=(98., 102.), event_tracks=("LED",)),
        cca_plotting.plot_session_cca(result, speed=speed, pair_range=(2, 3), show_traces=False),
    ]
    limits = []
    for figure in figures:
        trace = next(trace for trace in figure.data if trace.name == "Signed speed (mm/s)")
        np.testing.assert_array_equal(trace.x, speed.utc_s - result.time_origin_s)
        np.testing.assert_array_equal(trace.y, speed.speed_mm_s)
        np.testing.assert_array_equal(trace.customdata[:, 0], speed.utc_s)
        np.testing.assert_array_equal(trace.customdata[:, 1], speed.source_rows)
        assert trace.type == "scattergl" and trace.line.shape == "hv"
        assert trace.connectgaps is False and "mm/s" in trace.hovertemplate and "UTC" in trace.hovertemplate
        heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
        assert int(trace.yaxis[1:]) < int(heatmap.yaxis[1:])
        axis = figure.layout["yaxis" + trace.yaxis[1:]]
        assert axis.zeroline and axis.range[0] < -3 < 0 < 7 < axis.range[1]
        limits.append(axis.range)
        assert figure.layout.uirevision == original.layout.uirevision
        assert axis.uirevision != original.layout.yaxis.uirevision
    assert limits[0] == limits[1]
    for region, scores in before.items():
        np.testing.assert_array_equal(result.fit.scores[region], scores)


def test_explicit_speed_prepare_and_toggle_reuse_cca_and_speed_archives(synthetic_speed_session, monkeypatch):
    """Actual Streamlit controls prepare once, then use caches without fitting or raw decoding."""
    from src.neural_analysis.webapp import cca_view

    session = synthetic_speed_session
    cca_path = run_session_cca(session.metadata_path, ROLES).result_path
    original = cca_path.read_bytes()
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    assert next(widget for widget in at.checkbox if widget.label == "Show speed").disabled
    assert not list((session.session_root / "analysis_runs").glob("treadmill_speed_*/results.npz"))
    next(widget for widget in at.button if widget.label == "Prepare speed").click().run()
    assert not at.exception
    assert next(widget for widget in at.checkbox if widget.label == "Show speed").value
    visible = _figure(at)
    assert any(trace.get("name") == "Signed speed (mm/s)" for trace in visible["data"])
    chart_position = _chart_position(at)
    speed_path, = (session.session_root / "analysis_runs").glob("treadmill_speed_*/results.npz")
    speed_before = speed_path.read_bytes()
    monkeypatch.setattr(session_speed.irig_serial_io, "load_coolterm_serial_txt", _forbid_work)
    monkeypatch.setattr(session_speed, "align_treadmill_speed", _forbid_work)
    monkeypatch.setattr(cca_view, "run_session_cca", _forbid_work)
    next(widget for widget in at.checkbox if widget.label == "Show speed").uncheck().run()
    assert not at.exception
    hidden = _figure(at)
    assert not any(trace.get("name") == "Signed speed (mm/s)" for trace in hidden["data"])
    assert hidden["layout"]["uirevision"] == visible["layout"]["uirevision"]
    assert _chart_position(at) == chart_position
    next(widget for widget in at.checkbox if widget.label == "Show speed").check().run()
    next(widget for widget in at.selectbox if widget.label == "Trace pair").select(2).run()
    next(widget for widget in at.selectbox if widget.label == "CCA bin width (ms)").select(20).run()
    assert not at.exception
    assert any(trace.get("name") == "Signed speed (mm/s)" for trace in _figure(at)["data"])
    assert len(list((session.session_root / "analysis_runs").glob("treadmill_speed_*/results.npz"))) == 1
    assert cca_path.read_bytes() == original and speed_path.read_bytes() == speed_before


@pytest.mark.parametrize("source_state", ["undeclared", "missing", "malformed"])
def test_unavailable_speed_affects_only_its_optional_row(synthetic_speed_session, source_state):
    """Absent or unusable treadmill files leave a saved paired CCA figure available."""
    session = synthetic_speed_session
    cca_path = run_session_cca(session.metadata_path, ROLES).result_path
    if source_state == "undeclared":
        metadata = json.loads(session.metadata_path.read_text())
        metadata["behavior"].pop("treadmill")
        session.metadata_path.write_text(json.dumps(metadata))
    elif source_state == "missing":
        session_speed.run_session_speed(session.metadata_path)
        session.behavior.treadmill_file.unlink()
    else:
        session.behavior.treadmill_file.write_text("not a treadmill log\n")
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    if source_state == "malformed":
        next(widget for widget in at.button if widget.label == "Prepare speed").click().run()
    else:
        assert next(widget for widget in at.button if widget.label == "Prepare speed").disabled
    assert not at.exception
    assert next(widget for widget in at.checkbox if widget.label == "Show speed").disabled
    assert any("Speed unavailable" in caption.value for caption in at.caption)
    assert len([trace for trace in _figure(at)["data"] if trace["type"] == "heatmap"]) == 2
    assert cca_path.is_file()


def test_speed_can_display_without_a_prepared_cca(synthetic_speed_session):
    """Prepared speed and archived behavioral time do not require population fitting."""
    session = synthetic_speed_session
    session_speed.run_session_speed(session.metadata_path)
    at = AppTest.from_function(_render_cca, args=(str(session.metadata_path),)).run()
    assert not at.exception
    figure = _figure(at)
    assert any(trace.get("name") == "Signed speed (mm/s)" for trace in figure["data"])
    assert not any(trace["type"] == "heatmap" for trace in figure["data"])


def _chart_position(app_test: AppTest) -> int:
    """Return the persistent component's main-block position for an actual AppTest."""
    return next(index for index, element in app_test.main.children.items() if element.type == "bidi_component")


def _forbid_work(*args, **kwargs):
    """Reject raw decoding or CCA preparation during display-only interactions."""
    pytest.fail("Display controls must reuse prepared speed and CCA")
