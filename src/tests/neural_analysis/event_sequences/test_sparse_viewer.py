"""Real Streamlit reruns preserve color preferences across tools and scaling."""

import json

from streamlit.testing.v1 import AppTest

from src.neural_analysis.event_sequences.run_session import EventSequenceRequest, prepare_event_sequences
from src.tests.neural_analysis.test_event_sequence_webapp import _render_event_tool


def _widget(app, kind, label):
    """Input: AppTest/type/label strings. Return the matching live widget."""
    return next(item for item in getattr(app, kind) if item.label == label)


def _limits(app):
    """Input: AppTest. Return the shared Plotly color-axis bounds in displayed units."""
    assert not app.exception
    axis = json.loads(app.get("plotly_chart")[0].proto.spec)["layout"]["coloraxis"]
    return axis["cmin"], axis["cmax"]


def test_sparse_color_preferences_survive_widget_cleanup_and_tool_changes(event_session, monkeypatch):
    """Input: session. Return None; raw/zscore/source preferences survive actual Streamlit reruns."""
    prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "pc", region="PFC"))
    prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "sparse_pc", region="PFC"))
    from src.neural_analysis.webapp import event_sequence_views
    from src.tests.neural_analysis.event_sequences.test_sparse_sources import _forbidden
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", _forbidden)
    app = AppTest.from_function(_render_event_tool, args=(str(event_session.metadata_path),)).run()
    assert not app.exception
    assert _widget(app, "selectbox", "Neural source").value == "sparse_pc"
    _widget(app, "checkbox", "Manual color limit").check().run()
    _widget(app, "number_input", "Color limit L").set_value(.15).run()
    assert _limits(app) == (-.15, .15)
    _widget(app, "slider", "Sparse component").set_value(2).run()
    assert not _widget(app, "checkbox", "Manual color limit").value
    _widget(app, "slider", "Sparse component").set_value(1).run()
    assert _limits(app) == (-.15, .15)
    _widget(app, "selectbox", "Test tool").select("transition").run()
    assert _limits(app) == (-.15, .15)
    _widget(app, "selectbox", "Sequence family").select("probe").run()
    assert _limits(app) == (-.15, .15)
    _widget(app, "checkbox", "Z-score").check().run()
    assert not _widget(app, "checkbox", "Manual color limit").value
    _widget(app, "checkbox", "Manual color limit").check().run()
    _widget(app, "number_input", "Color limit L").set_value(2.).run()
    assert _limits(app) == (-2., 2.)
    _widget(app, "checkbox", "Z-score").uncheck().run()
    assert _limits(app) == (-.15, .15)
    _widget(app, "selectbox", "Neural source").select("pc").run()
    assert not _widget(app, "checkbox", "Manual color limit").value
    _widget(app, "selectbox", "Neural source").select("sparse_pc").run()
    assert _limits(app) == (-.15, .15)
    _widget(app, "selectbox", "Test tool").select("central").run()
    assert _limits(app) == (-.15, .15)
