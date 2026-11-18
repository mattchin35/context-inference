"""Fixed-session-axis participation, separate from plotting and PCA fitting."""

from dataclasses import replace
import importlib

import numpy as np
import pytest

from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.interregional.records import RegionalPCATransform


def _regional_scores(scores, totals, unit_count=None):
    """Return an identity-axis regional fixture for (window,bin,PC) float scores.

    totals is direct unit-space sample variance (window,); unit_count can
    exceed retained PCs to exercise a deliberately restricted basis.
    """
    windows, bins, pcs = scores.shape
    units = pcs if unit_count is None else unit_count
    ids = tuple(f"A:{unit}" for unit in range(units))
    spectrum = np.arange(pcs, 0, -1, dtype=float)
    fitted = RegionalPCATransform(
        retained_unit_ids=ids,
        omitted_unit_ids=(),
        input_unit_ids=ids,
        training_mean=np.zeros(units),
        training_scale=np.ones(units),
        components=np.eye(units)[:pcs],
        explained_variance=spectrum,
        explained_variance_ratio=spectrum / spectrum.sum(),
        n_training_observations=windows * bins,
    )
    return timeline.RegionalTimelinePCA(
        "PFC",
        "A",
        np.arange(windows),
        np.linspace(-2, 2, bins + 1),
        np.asarray(scores, dtype=float),
        fitted,
        float(units),
        np.asarray(totals, dtype=float),
        float(np.abs(scores).max()),
    )


def test_window_specific_pc2_pc7_pc12_contributions_return_three_fixed_axes():
    """Rank by local sample variance, not global rank, mean energy, or displayed-PC count."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    signal = np.tile([-1.0, 1.0], 4) * np.sqrt(7 / 8)
    scores = np.zeros((2, 8, 12))
    for pc, variance in ((1, 0.6), (6, 0.25), (11, 0.1), (0, 0.05)):
        scores[0, :, pc] = 100 + signal * np.sqrt(variance)
    scores[1] = 0.1
    regional = _regional_scores(scores, [1.0, 0.0])
    before = scores.copy()
    summary = metric.compute_window_variance(regional)
    assert summary.basis_complete
    assert summary.status.tolist() == ["ok", "zero_variance"]
    assert summary.counts95[0] == 3 and np.isnan(summary.counts95[1])
    np.testing.assert_array_equal(summary.pc_ids[0, :3], [2, 7, 12])
    np.testing.assert_allclose(summary.fractions[0, :3], [0.6, 0.25, 0.1])
    assert summary.cumulative[0, 2] == pytest.approx(0.95)
    np.testing.assert_array_equal(regional.scores, before)


def test_fixed_axis_count_can_exceed_bin_count_minus_one_and_ties_keep_session_rank():
    """Orthogonal global directions can each have a rank-one local diagonal trajectory."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    directions = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float
    )
    scores = np.stack([-directions, directions], axis=1)
    summary = metric.compute_window_variance(_regional_scores(scores, np.full(4, 8.0)))
    np.testing.assert_array_equal(summary.counts95, [4, 4, 4, 4])
    np.testing.assert_array_equal(summary.pc_ids, np.tile([1, 2, 3, 4], (4, 1)))
    np.testing.assert_allclose(summary.fractions, 0.25)


def test_complete_fitted_basis_conserves_saved_unit_variance_without_refitting(
    monkeypatch,
):
    """Numerators and denominator use matching sample variance and original saved scaling."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    counts = np.random.default_rng(73).poisson(3, size=(5, 8, 6))
    regional = timeline.fit_timeline_activity(
        counts, tuple(f"A:{i}" for i in range(6)), np.arange(5), "PFC", "A", 0.5
    )
    monkeypatch.setattr(timeline, "fit_regional_pca", pytest.fail)
    summary = metric.compute_window_variance(regional)
    np.testing.assert_allclose(
        summary.pc_variance.sum(axis=1), regional.window_total_variance
    )
    np.testing.assert_allclose(summary.fractions.sum(axis=1), 1)
    assert summary.status.tolist() == ["ok"] * 5
    for row, count in enumerate(summary.counts95.astype(int)):
        assert summary.cumulative[row, count - 1] >= 0.95 - 1e-12
        if count > 1:
            assert summary.cumulative[row, count - 2] < 0.95 - 1e-12


@pytest.mark.parametrize(
    "variances, expected_status, expected_count",
    [([0.9, 0.06], "restricted", 2), ([0.8], "not_reached", np.nan)],
)
def test_restricted_bases_use_original_denominator_and_explicit_status(
    variances, expected_status, expected_count
):
    """A retained subset is never renormalized or presented as an exact complete-basis count."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    signal = np.tile([-1.0, 1.0], 4) * np.sqrt(7 / 8)
    scores = np.tile(signal[:, None] * np.sqrt(variances), (2, 1, 1))
    summary = metric.compute_window_variance(
        _regional_scores(scores, [1, 1], unit_count=3)
    )
    assert not summary.basis_complete
    assert summary.status.tolist() == [expected_status] * 2
    np.testing.assert_allclose(summary.counts95, expected_count, equal_nan=True)
    np.testing.assert_allclose(summary.fractions[0], variances)


def test_invalid_and_inconsistent_windows_are_unavailable_without_repair():
    """Zero, missing, invalid totals, and a purported full-basis mismatch never become zero dimensions."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    scores = np.tile(np.arange(8, dtype=float)[None, :, None], (5, 1, 2))
    scores[1, 0, 0] = np.nan
    summary = metric.compute_window_variance(
        _regional_scores(scores, [0, 12, np.nan, -1, 100])
    )
    assert summary.status.tolist() == [
        "zero_variance",
        "nonfinite",
        "nonfinite",
        "invalid_total",
        "variance_mismatch",
    ]
    assert np.isnan(summary.counts95).all()
    one_bin = _regional_scores(np.ones((2, 1, 2)), [0, 0])
    insufficient = metric.compute_window_variance(one_bin)
    assert insufficient.status.tolist() == ["insufficient_bins"] * 2
    assert np.isnan(insufficient.counts95).all()


def test_constant_decimal_scores_do_not_gain_variance_from_mean_roundoff():
    """A sustained elevation is not within-window participation or mean-square energy."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    summary = metric.compute_window_variance(
        _regional_scores(np.full((2, 8, 2), 0.1), [0, 0])
    )
    np.testing.assert_array_equal(summary.pc_variance, 0)
    assert np.isnan(summary.counts95).all()


def test_threshold_tolerance_and_region_specific_window_identity_are_retained():
    """The stated fraction tolerance handles a threshold roundoff without losing source positions."""
    metric = importlib.import_module("src.neural_analysis.population.window_variance")
    signal = np.tile([-1.0, 1.0], 4) * np.sqrt(7 / 8)
    scores = np.tile(signal[:, None] * np.sqrt([0.95 - 5e-13, 0.05 + 5e-13]), (2, 1, 1))
    regional = replace(
        _regional_scores(scores, [1, 1]), choice_positions=np.array([4, 9])
    )
    summary = metric.compute_window_variance(regional)
    np.testing.assert_array_equal(summary.counts95, [1, 1])
    np.testing.assert_array_equal(summary.choice_positions, [4, 9])
