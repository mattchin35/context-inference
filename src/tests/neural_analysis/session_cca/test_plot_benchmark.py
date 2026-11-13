"""Real saved-product figure and serialization measurements, excluding the browser."""

import numpy as np

from src.neural_analysis.population import run_session_cca as runner
from src.neural_analysis.population import session_cca_io as cca_io
from src.tests.neural_analysis.session_cca.test_results import (
    cca_result as result_fixture,
    synthetic_timeline_session as session_fixture,
)


cca_result = result_fixture
synthetic_timeline_session = session_fixture


def test_benchmark_measures_real_full_figure_json_and_preserves_numeric_fields(cca_result, tmp_path):
    """Measured seconds/bytes reflect default saved-score plotting, without a refit."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    measured = runner.benchmark_cca_result(path)

    for field in ("npz_load_s", "figure_construction_s", "serialization_s"):
        assert np.isfinite(measured[field]) and measured[field] >= 0
    assert measured["npz_bytes"] == path.stat().st_size
    assert measured["score_bytes"] == sum(values.nbytes for values in cca_result.fit.scores.values())
    assert measured["loaded_range_s"] is None
    assert measured["browser_rendering"] == "not measured"
    assert isinstance(measured["preparation_timings_s"], dict)
    from src.neural_analysis.population.cca_plotting import plot_session_cca

    expected = plot_session_cca(cca_io.load_cca_result(path)).to_json().encode("utf-8")
    assert measured["json_bytes"] == len(expected)


def test_bounded_benchmark_measures_smaller_real_payload_without_changing_fit(cca_result, tmp_path):
    """One intersecting window reduces serialized data while file/full score bytes stay fixed."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    original = path.read_bytes()
    full = runner.benchmark_cca_result(path)
    bounded = runner.benchmark_cca_result(path, loaded_range_s=(16., 21.))

    assert bounded["loaded_range_s"] == [16., 21.]
    assert 0 < bounded["json_bytes"] < full["json_bytes"]
    assert bounded["score_bytes"] == full["score_bytes"]
    assert bounded["npz_bytes"] == full["npz_bytes"]
    assert path.read_bytes() == original
    from src.neural_analysis.population.cca_plotting import plot_session_cca

    expected = plot_session_cca(cca_io.load_cca_result(path), loaded_range_s=(16., 21.)).to_json().encode("utf-8")
    assert bounded["json_bytes"] == len(expected)
