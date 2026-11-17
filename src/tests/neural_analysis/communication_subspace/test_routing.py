"""Communication inspection routes must precede unrelated raw-data gates."""

from types import SimpleNamespace
from unittest.mock import Mock

from src.neural_analysis.webapp import app, session_inputs


def test_communication_route_opens_controls_without_generic_source_checks(monkeypatch):
    """Metadata identity is sufficient to open controls; no signal axes are loaded."""
    label = "Communication subspace"
    assert label in session_inputs.PLOT_VIEW_OPTIONS
    session = SimpleNamespace(subject_id="synthetic", session_id="session")
    interface = SimpleNamespace(sidebar=SimpleNamespace(
        caption=Mock(), selectbox=Mock(return_value=label)))
    render = Mock()
    monkeypatch.setattr(app, "st", interface)
    monkeypatch.setattr(app, "render_communication_subspace_view", render)
    monkeypatch.setattr(app, "metadata_view_availability", Mock(
        side_effect=AssertionError("Generic prerequisites must not gate controls")))
    assert app._start_metadata_webapp(session) is None
    render.assert_called_once_with(interface, session)


def test_manual_route_explains_metadata_before_loading_paths(monkeypatch):
    """Manual launch returns a metadata instruction without loading arrays or paths."""
    interface = SimpleNamespace(
        set_page_config=Mock(), title=Mock(), info=Mock(),
        sidebar=SimpleNamespace(button=Mock(return_value=False),
                                selectbox=Mock(return_value="Communication subspace")),
    )
    monkeypatch.setattr(app, "st", interface)
    monkeypatch.setattr(app, "_initialize_path_input_state", Mock(
        side_effect=AssertionError("Manual source loading must not start")))
    app.main([])
    assert interface.info.call_count == 1
    assert "--session-metadata" in interface.info.call_args.args[0]
