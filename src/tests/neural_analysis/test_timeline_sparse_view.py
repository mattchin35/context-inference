"""Sparse heatmaps share time navigation while excluding ordinary-PCA summaries/models."""

from dataclasses import replace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.population import timeline_plotting as plotting
from src.neural_analysis.population.run_session_timeline import run_session_timeline
from src.neural_analysis.population.session_timeline import SPARSE_ANALYSIS_VERSION
from src.neural_analysis.population.timeline_sparse_pca import fit_sparse_timeline_activity
from src.neural_analysis.population.window_variance import compute_window_variance
from src.neural_analysis.webapp import timeline_view
from src.tests.neural_analysis.test_session_timeline_io import synthetic_timeline_session as session_fixture
from src.tests.neural_analysis.test_session_timeline_webapp import _render_timeline_for_test, _timeline_figure_spec
from src.tests.neural_analysis.test_timeline_plotting import timeline_result as result_fixture

synthetic_timeline_session = session_fixture
timeline_result = result_fixture


@pytest.fixture
def sparse_result(timeline_result):
    """Return the existing two-choice elapsed-time fixture with a sparse (2,8,2) tensor."""
    regional = fit_sparse_timeline_activity(
        np.random.default_rng(42).poisson(2, (2, 8, 2)), ("A:7", "A:8"),
        np.array([0, 1]), "PFC", "A", .5,
    )
    return replace(timeline_result, regions={"PFC": regional}, meta={
        "bin_width_s": .5, "algorithm": "sparse_pca", "analysis_version": SPARSE_ANALYSIS_VERSION,
    })


def test_sparse_heatmap_reuses_time_geometry_and_omits_variance(sparse_result, timeline_result):
    """Same temporal edges/gaps and display normalization, with stable SC IDs and honest hover."""
    original = plotting.plot_session_timeline(timeline_result)
    raw = plotting.plot_session_timeline(sparse_result)
    standardized = plotting.plot_session_timeline(sparse_result, zscore_pcs=True)
    for figure in (raw, standardized):
        trace = next(item for item in figure.data if item.type == "heatmap")
        reference = next(item for item in original.data if item.type == "heatmap")
        np.testing.assert_array_equal(trace.x, reference.x)
        assert np.isnan(trace.z[:, 8]).all()
        assert trace.customdata.shape == (2, 17, 8)
        assert all(value is None for value in trace.customdata[:, :, 7].ravel())
        assert "session variance" not in trace.hovertemplate
        assert "Trial" in trace.hovertemplate and "UTC" in trace.hovertemplate
        assert list(figure.layout["yaxis" + trace.yaxis[1:]].ticktext) == ["SC1", "SC2"]
        assert figure.layout.uirevision == original.layout.uirevision
        assert any("sparse components" in item.text for item in figure.layout.annotations)
    trace = next(item for item in standardized.data if item.type == "heatmap")
    assert "component SD" in trace.colorbar.title.text
    np.testing.assert_array_equal(trace.z[:, :8], plotting.zscore_window_activity(sparse_result.regions["PFC"].scores)[0].T)
    bounded = plotting.plot_session_timeline(sparse_result, loaded_range_s=(98, 102), pc_ranges={"PFC": (2, 2)})
    assert next(item for item in bounded.data if item.type == "heatmap").z.shape == (1, 8)


def test_sparse_plot_refuses_attached_ordinary_summary(sparse_result, timeline_result):
    """Direct plotting cannot silently attach percentages/counts from an unrelated PCA basis."""
    ordinary = compute_window_variance(timeline_result.regions["PFC"])
    with pytest.raises(ValueError, match="ordinary PCA"):
        plotting.plot_session_timeline(sparse_result, window_variance={"PFC": ordinary})


def test_sparse_controls_skip_prediction_anchor_and_preserve_display_identity(synthetic_timeline_session, monkeypatch):
    """Actual Streamlit reruns use only the selected sparse cache; no fitting on display changes."""
    session = synthetic_timeline_session
    ordinary = run_session_timeline(session.metadata_path, {"PFC": "A"}).result_path
    sparse = run_session_timeline(session.metadata_path, {"PFC": "A"}, algorithm="sparse_pca").result_path
    before = [(path, path.stat().st_mtime_ns, path.read_bytes()) for path in (ordinary, sparse)]
    at = AppTest.from_function(_render_timeline_for_test, args=(str(session.metadata_path),)).run()
    next(item for item in at.selectbox if item.label == "PFC probe").select("A").run()
    assert not at.exception
    position = next(i for i, element in at.main.children.items() if element.type == "bidi_component")
    initial = _timeline_figure_spec(at)
    with monkeypatch.context() as guard:
        guard.setattr(timeline_view, "render_prediction_controls", pytest.fail)
        guard.setattr(timeline_view, "_window_summary_controls", pytest.fail)
        guard.setattr(timeline_view, "run_session_timeline", pytest.fail)
        next(item for item in at.selectbox if item.label == "Decomposition").select("sparse_pca").run()
        assert not at.exception
        assert not any(item.label == "Prepare window summary" for item in at.button)
        assert not any(item.label == "Show predictions" for item in at.checkbox)
        assert not any("Variance coverage" in item.value for item in at.markdown)
        assert any("ridge" in item.value for item in at.caption)
        next(item for item in at.slider if item.label == "PFC sparse components").set_value((2, 2)).run()
        next(item for item in at.checkbox if item.label == "Z-score each component").check().run()
        assert not at.exception
        figure = _timeline_figure_spec(at)
        assert figure["layout"]["uirevision"] == initial["layout"]["uirevision"]
        assert next(i for i, element in at.main.children.items() if element.type == "bidi_component") == position
        next(item for item in at.number_input if item.label == "SparsePCA alpha").set_value(.3).run()
        assert not at.exception
        assert not any(item["type"] == "heatmap" for item in _timeline_figure_spec(at)["data"])
        assert not next(item for item in at.button if item.label == "Prepare").disabled
    next(item for item in at.selectbox if item.label == "Decomposition").select("pca").run()
    assert not at.exception
    assert any(item.label == "Show predictions" for item in at.checkbox)
    assert any(item.label == "PFC PC ranks" for item in at.slider)
    for path, mtime, data in before:
        assert path.stat().st_mtime_ns == mtime and path.read_bytes() == data


def test_sparse_prepare_is_explicit_and_alpha_is_forwarded(synthetic_timeline_session):
    """Prepare makes a sparse product with selected alpha; changing alpha only selects another cache."""
    from src.neural_analysis.population.session_timeline_io import load_timeline_result

    session = synthetic_timeline_session
    at = AppTest.from_function(_render_timeline_for_test, args=(str(session.metadata_path),)).run()
    next(item for item in at.selectbox if item.label == "Decomposition").select("sparse_pca").run()
    next(item for item in at.selectbox if item.label == "PFC probe").select("A").run()
    next(item for item in at.number_input if item.label == "SparsePCA alpha").set_value(.3).run()
    assert not (session.session_root / "analysis_runs").exists()
    next(item for item in at.button if item.label == "Prepare").click().run()
    assert not at.exception
    paths = list((session.session_root / "analysis_runs").glob("*/results.npz"))
    assert len(paths) == 1
    assert load_timeline_result(paths[0]).regions["PFC"].transform.parameters["alpha"] == .3
    assert any(item["type"] == "heatmap" for item in _timeline_figure_spec(at)["data"])
