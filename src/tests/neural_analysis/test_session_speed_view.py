"""Optional signed-speed plotting and explicit preparation in the existing viewer."""

from dataclasses import replace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.irig_tools import irig_serial_io
from src.mouse_behavior_preprocessing.treadmill_speed import align_treadmill_speed
from src.neural_analysis.population import timeline_plotting as plotting
from src.neural_analysis.population.run_session_timeline import run_session_timeline
from src.neural_analysis.population.run_window_variance import run_window_variance
from src.tests.neural_analysis.test_treadmill_speed import (
    synthetic_speed_messages as messages_fixture,
)
from src.tests.neural_analysis.test_session_speed import (
    synthetic_speed_session as speed_session_fixture,
)
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as session_fixture,
)
from src.tests.neural_analysis.test_timeline_plotting import (
    timeline_result as result_fixture,
)
from src.tests.neural_analysis.test_session_timeline_webapp import (
    _render_timeline_for_test,
    _timeline_figure_spec,
)

synthetic_speed_messages = messages_fixture
synthetic_speed_session = speed_session_fixture
synthetic_timeline_session = session_fixture
timeline_result = result_fixture


def test_speed_is_a_signed_step_row_after_events_and_spans_choice_gaps(
    timeline_result, synthetic_speed_messages
):
    """UTC-to-elapsed conversion is display-only; full speed survives neural payload filtering."""
    speed = align_treadmill_speed(
        irig_serial_io.parse_coolterm_serial_text(
            synthetic_speed_messages["text"], pc_timestamp_format="ignore"
        )
    )
    result = replace(timeline_result, time_origin_s=1_699_999_980.0)
    before = speed.speed_mm_s.copy()
    original = plotting.plot_session_timeline(result)
    first = plotting.plot_session_timeline(
        result, speed=speed, loaded_range_s=(98, 102)
    )
    second = plotting.plot_session_timeline(
        result, speed=speed, event_tracks=(), zscore_pcs=True, pc_ranges={"PFC": (2, 2)}
    )
    for figure in (first, second):
        trace = next(
            trace for trace in figure.data if trace.name == "Signed speed (mm/s)"
        )
        assert (
            trace.type == "scattergl"
            and trace.mode == "lines"
            and trace.line.shape == "hv"
        )
        assert trace.connectgaps is False
        np.testing.assert_allclose(
            trace.x, synthetic_speed_messages["utc_s"] - result.time_origin_s
        )
        np.testing.assert_array_equal(trace.y, before)
        np.testing.assert_array_equal(trace.customdata[:, 0], speed.utc_s)
        assert "mm/s" in trace.hovertemplate and "UTC" in trace.hovertemplate
        # This actual speed lies in the neural 102-112 second gap's broader session,
        # and every sample remains transmitted even for the 98-102 neural slice.
        assert len(trace.x) == len(speed.utc_s)
        heatmap = next(item for item in figure.data if item.type == "heatmap")
        speed_row = int(trace.yaxis[1:])
        assert speed_row < int(heatmap.yaxis[1:])
        yaxis = figure.layout[f"yaxis{speed_row}"]
        assert yaxis.range[0] < 0 < yaxis.range[1] and yaxis.zeroline
        assert figure.layout.uirevision == original.layout.uirevision
    assert (
        first.layout[
            f"yaxis{int(next(t for t in first.data if t.name == 'Signed speed (mm/s)').yaxis[1:])}"
        ].range
        == second.layout.yaxis2.range
    )
    np.testing.assert_array_equal(speed.speed_mm_s, before)


def test_speed_controls_are_explicit_and_display_changes_reuse_every_archive(
    synthetic_speed_session, monkeypatch
):
    """Preparation is independent; hiding speed keeps component position and navigation revision."""
    from src.neural_analysis import session_speed as storage
    from src.neural_analysis.webapp import timeline_view

    session = synthetic_speed_session
    stage1 = run_session_timeline(session.metadata_path, {"PFC": "A"}).result_path
    stage2 = run_window_variance(stage1).result_path
    stage1_before, stage2_before = stage1.read_bytes(), stage2.read_bytes()
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    next(item for item in at.selectbox if item.label == "PFC probe").select("A").run()
    assert not at.exception
    assert not list(
        (session.session_root / "analysis_runs").glob("treadmill_speed_*/results.npz")
    )
    assert next(item for item in at.checkbox if item.label == "Show speed").disabled
    next(item for item in at.button if item.label == "Prepare speed").click().run()
    assert not at.exception
    assert next(item for item in at.checkbox if item.label == "Show speed").value
    figure = _timeline_figure_spec(at)
    assert any(trace.get("name") == "Signed speed (mm/s)" for trace in figure["data"])
    position = next(
        index
        for index, element in at.main.children.items()
        if element.type == "bidi_component"
    )
    monkeypatch.setattr(storage.irig_serial_io, "load_coolterm_serial_txt", pytest.fail)
    monkeypatch.setattr(storage, "align_treadmill_speed", pytest.fail)
    monkeypatch.setattr(timeline_view, "run_session_timeline", pytest.fail)
    next(item for item in at.checkbox if item.label == "Show speed").uncheck().run()
    assert not at.exception
    hidden = _timeline_figure_spec(at)
    assert hidden["layout"]["uirevision"] == figure["layout"]["uirevision"]
    assert (
        next(
            index
            for index, element in at.main.children.items()
            if element.type == "bidi_component"
        )
        == position
    )
    assert not any(
        trace.get("name") == "Signed speed (mm/s)" for trace in hidden["data"]
    )
    next(item for item in at.checkbox if item.label == "Show speed").check().run()
    next(item for item in at.checkbox if item.label == "Z-score each PC").check().run()
    next(
        item for item in at.selectbox if item.label == "Timeline bin width (ms)"
    ).select(20).run()
    assert not at.exception
    assert any(
        trace.get("name") == "Signed speed (mm/s)"
        for trace in _timeline_figure_spec(at)["data"]
    )
    assert (
        len(
            list(
                (session.session_root / "analysis_runs").glob(
                    "treadmill_speed_*/results.npz"
                )
            )
        )
        == 1
    )
    assert stage1.read_bytes() == stage1_before and stage2.read_bytes() == stage2_before


def test_behavior_only_speed_and_missing_source_do_not_require_neural_preparation(
    synthetic_speed_session,
):
    """Speed can prepare with no PCs, and a missing log affects only its optional row."""
    from src.neural_analysis import session_speed

    session = synthetic_speed_session
    session_speed.run_session_speed(session.metadata_path)
    at = AppTest.from_function(
        _render_timeline_for_test, args=(str(session.metadata_path),)
    ).run()
    assert not at.exception
    assert any(
        trace.get("name") == "Signed speed (mm/s)"
        for trace in _timeline_figure_spec(at)["data"]
    )
    assert not any(
        trace["type"] == "heatmap" for trace in _timeline_figure_spec(at)["data"]
    )
    session.behavior.treadmill_file.unlink()
    at.run()
    assert not at.exception
    assert next(item for item in at.button if item.label == "Prepare speed").disabled
    assert next(item for item in at.checkbox if item.label == "Show speed").disabled
