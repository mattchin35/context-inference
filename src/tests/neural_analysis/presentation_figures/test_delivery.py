"""The offline exporter and webapp read prepared results without accessing spikes."""

from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from src.neural_analysis.presentation_figures import results
from src.neural_analysis.webapp import app, session_inputs


def test_regional_manifest_reader_preserves_training_block_support(tmp_path, monkeypatch):
    """The exported and webapp Figures 1-2 receive the saved per-fold class counts."""
    from src.neural_analysis.presentation_figures import export_figures, regional_plotting

    folds = pd.DataFrame({"fold_id": [0], "blocks_0": [4], "blocks_1": [5]})
    results.save_result(tmp_path / "regional.npz", {"folds": folds}, {})
    manifest = results.complete_run(tmp_path, kind="session", identity="saved",
                                    session_ids=["s"], archives={"regional": "regional.npz"})
    render = Mock(return_value={})
    monkeypatch.setattr(regional_plotting, "render_regional", render)
    export_figures.read_panels(manifest, "regional")
    pd.testing.assert_frame_equal(render.call_args.args[0]["folds"], folds)


def test_export_writes_png_and_self_contained_replay(tmp_path, monkeypatch):
    """A completed session manifest yields static PNGs and offline HTML in a chosen path."""
    from src.neural_analysis.presentation_figures import export_figures as delivery

    results.save_result(tmp_path / "a.npz", {}, {})
    manifest = results.complete_run(tmp_path, kind="session", identity="saved",
                                    session_ids=["synthetic"], archives={"regional": "a.npz"})
    def read_panels(path, group, theme):
        """Record one saved panel group, returning static and interactive figure dictionaries."""
        assert path == manifest and theme == "light"
        if group != "regional":
            return {}, {}
        fig, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])
        return {"figure1": fig}, {"switch": go.Figure(go.Scatter(x=[0, 1], y=[.25, .75]))}

    monkeypatch.setattr(delivery, "read_panels", read_panels)
    paths = delivery.export_figures(manifest, tmp_path / "export", theme="light")
    assert {p.name for p in paths} == {"figure1.png", "switch.html"}
    assert (tmp_path / "export/figure1.png").read_bytes().startswith(b"\x89PNG")
    document = (tmp_path / "export/switch.html").read_text()
    assert "Plotly.newPlot" in document and '<script src="https://' not in document


def test_archive_view_routes_before_raw_data_gates(monkeypatch):
    """A metadata-defined session can open saved figures with no raw inputs."""
    label = "PFC and HPC presentation figures"
    assert label in session_inputs.PLOT_VIEW_OPTIONS
    interface = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=label)))
    session = SimpleNamespace(session_id="synthetic", subject_id="mouse")
    render = Mock()
    monkeypatch.setattr(app, "st", interface)
    monkeypatch.setattr(app, "render_presentation_view", render)
    monkeypatch.setattr(app, "metadata_view_availability", Mock(side_effect=AssertionError("raw data gate")))
    app._start_metadata_webapp(session)
    render.assert_called_once_with(interface, session)
