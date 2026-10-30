"""Timeline routing and UI state contracts without experimental data."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from src.neural_analysis.webapp import app, session_inputs, timeline_view
from src.tests.neural_analysis.test_session_timeline_io import synthetic_timeline_session


def test_timeline_is_available_with_behavior_and_missing_neural_probes(synthetic_timeline_session):
    """Other spike views remain strict; this view permits regional/behavior-only fallback."""
    session = synthetic_timeline_session
    (session.session_root / "B/aligned.npz").unlink()
    available = session_inputs.metadata_view_availability(session)
    assert available[session_inputs.PLOT_VIEW_SESSION_TIMELINE].available
    assert not available[session_inputs.PLOT_VIEW_TRIAL_SPIKES].available


def test_timeline_dispatch_occurs_before_active_probe_and_trial_controls(synthetic_timeline_session, monkeypatch):
    """The dedicated multi-probe view handles its own inputs and returns from dispatch."""
    session = synthetic_timeline_session
    selected = session_inputs.PLOT_VIEW_SESSION_TIMELINE
    fake_st = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=selected)))
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_session_timeline_view", renderer)
    assert app._start_metadata_webapp(session) is None
    renderer.assert_called_once_with(fake_st, session)


def test_navigation_centers_original_trial_id_and_clamps_to_session_bounds():
    """Explicit controls can reach any choice without renumbering source trials."""
    result = timeline_view.resolve_time_range(100., 60., (0., 200.))
    assert result == (100., 160.)
    assert timeline_view.resolve_time_range(180., 60., (0., 200.)) == (140., 200.)
    assert timeline_view.resolve_time_range(0., 60., (-2., 10.)) == (-2., 10.)


def test_pc_ranges_are_display_only_and_do_not_exceed_available_rows():
    """Few-PC populations have a valid initial selection without special estimator paths."""
    assert timeline_view.default_pc_range(3) == (1, 3)
    assert timeline_view.default_pc_range(40) == (1, 10)
