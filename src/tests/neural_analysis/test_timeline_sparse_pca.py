"""Sparse timeline coefficients retain count scaling, axes, and sklearn semantics."""

from dataclasses import replace

import numpy as np
import pytest
from sklearn.decomposition import PCA, SparsePCA

from src.neural_analysis.interregional import pca
from src.neural_analysis.population import timeline_sparse_pca as sparse


@pytest.fixture
def sparse_counts():
    """Return float64 (4 windows,8 bins,5 units) counts; final unit is constant."""
    rng = np.random.default_rng(84)
    first = rng.poisson(3, (4, 8, 1))
    second = rng.poisson(2, (4, 8, 1))
    return np.concatenate(
        [first, second, first + second, first + 2 * second, np.ones_like(first)],
        axis=2,
    ).astype(float)


def test_shared_scaling_preserves_existing_full_pca_exactly(sparse_counts):
    """The extracted arithmetic agrees with the former explicit scaling/fit path."""
    observations = sparse_counts.reshape(-1, 5)
    means = observations.mean(0, dtype=np.float64)
    scales = observations.std(0, ddof=0, dtype=np.float64)
    mask = scales > 0
    expected = (observations[:, mask] - means[mask]) / scales[mask]
    standardized, retained, actual_mean, actual_scale = pca.standardize_unit_activity(
        observations
    )
    for actual, reference in zip(
        (standardized, retained, actual_mean, actual_scale),
        (expected, mask, means[mask], scales[mask]),
    ):
        np.testing.assert_array_equal(actual, reference)
    reference = PCA(n_components=4, svd_solver="full", whiten=False).fit(expected)
    fitted = pca.fit_regional_pca(sparse_counts, tuple("abcde"), 5)
    np.testing.assert_array_equal(fitted.components, reference.components_)
    np.testing.assert_array_equal(
        fitted.explained_variance, reference.explained_variance_
    )
    np.testing.assert_array_equal(
        fitted.explained_variance_ratio, reference.explained_variance_ratio_
    )
    np.testing.assert_array_equal(
        pca.transform_regional_activity(sparse_counts, tuple("abcde"), fitted).reshape(
            -1, 4
        ),
        expected @ reference.components_.T,
    )


def test_sparse_scores_match_explicit_sklearn_fit_and_preserve_order(sparse_counts):
    """Use full-session population SDs and sklearn's ridge scores, without rotation sorting."""
    before = sparse_counts.copy()
    observations = sparse_counts.reshape(-1, 5)[:, :4]
    expected = (observations - observations.mean(0)) / observations.std(0, ddof=0)
    estimator = SparsePCA(
        n_components=4,
        alpha=0.3,
        ridge_alpha=0.01,
        method="lars",
        max_iter=1000,
        tol=1e-8,
        random_state=0,
        n_jobs=1,
    )
    coefficients = estimator.fit_transform(expected)
    result = sparse.fit_sparse_timeline_activity(
        sparse_counts,
        tuple("abcde"),
        np.array([0, 2, 3, 6]),
        "PFC",
        "A",
        0.5,
        alpha=0.3,
    )
    assert result.scores.shape == (4, 8, 4)
    # Independent array layouts can change rounding at machine precision.
    np.testing.assert_allclose(
        result.scores.reshape(-1, 4), coefficients, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        result.transform.components, estimator.components_, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        result.transform.center, estimator.mean_, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        result.transform.objective_history, estimator.error_, atol=1e-12, rtol=1e-12
    )
    assert result.transform.n_iter == estimator.n_iter_
    assert result.transform.input_unit_ids == tuple("abcde")
    assert result.transform.retained_unit_ids == tuple("abcd")
    assert result.transform.omitted_unit_ids == ("e",)
    assert result.transform.parameters["n_components"] == 4
    assert result.transform.parameters["random_state"] == 0
    assert not hasattr(result.transform, "explained_variance_ratio")
    np.testing.assert_array_equal(result.choice_positions, [0, 2, 3, 6])
    np.testing.assert_array_equal(sparse_counts, before)
    repeated = sparse.fit_sparse_timeline_activity(
        sparse_counts,
        tuple("abcde"),
        np.array([0, 2, 3, 6]),
        "PFC",
        "A",
        0.5,
        alpha=0.3,
    )
    np.testing.assert_array_equal(repeated.scores, result.scores)


def test_nonorthogonal_sparse_scores_use_ridge_not_loading_dot_product():
    """More units than observations yields a genuinely nonorthogonal reference fixture."""
    counts = np.random.default_rng(84).poisson(3, (1, 8, 12)).astype(float)
    result = sparse.fit_sparse_timeline_activity(
        counts,
        tuple(f"A:{i}" for i in range(12)),
        np.array([0]),
        "PFC",
        "A",
        0.5,
        alpha=0.3,
    )
    fitted = result.transform
    standardized = (
        counts.reshape(8, 12) - fitted.training_mean
    ) / fitted.training_scale
    reference = SparsePCA(**fitted.parameters).fit(standardized)
    np.testing.assert_allclose(
        result.scores[0], reference.transform(standardized), atol=1e-12
    )
    assert not np.allclose(fitted.components @ fitted.components.T, np.eye(8))
    assert not np.allclose(result.scores[0], standardized @ fitted.components.T)


def test_sparse_fit_is_capped_at_thirty_components():
    """The user-selected cap fits 30 joint atoms rather than a full unit basis."""
    counts = np.random.default_rng(105).poisson(3, (6, 8, 35)).astype(float)
    result = sparse.fit_sparse_timeline_activity(
        counts,
        tuple(f"A:{i}" for i in range(35)),
        np.arange(6),
        "PFC",
        "A",
        0.5,
    )
    assert result.scores.shape == (6, 8, 30)
    assert result.transform.components.shape == (30, 35)
    assert result.transform.parameters["n_components"] == 30
    standardized = (
        counts.reshape(-1, 35) - result.transform.training_mean
    ) / result.transform.training_scale
    reference = SparsePCA(**result.transform.parameters).fit_transform(standardized)
    np.testing.assert_allclose(
        result.scores.reshape(-1, 30), reference, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("alpha", [-1, np.nan, np.inf, True, "1"])
def test_invalid_alpha_is_rejected(alpha):
    """Scientific settings require a finite nonnegative number, not coercible text/flags."""
    with pytest.raises(ValueError, match="alpha"):
        sparse.sparse_pca_parameters(alpha)


def test_zero_alpha_and_zero_loading_components_are_explicit(sparse_counts):
    """Zero penalty is accepted; an excessive penalty preserves zero component identities."""
    assert sparse.sparse_pca_parameters(0)["alpha"] == 0
    result = sparse.fit_sparse_timeline_activity(
        sparse_counts,
        tuple("abcde"),
        np.arange(4),
        "PFC",
        "A",
        0.5,
        alpha=1e6,
    )
    assert result.scores.shape == (4, 8, 4)
    assert result.transform.zero_component_ids == (1, 2, 3, 4)
    assert not result.scores.any()
    assert result.color_limit > 0


def test_iteration_limit_is_diagnostic_not_automatic_refit(sparse_counts):
    """Keep finite coefficients and mark uncertainty when the recorded limit was reached."""
    result = sparse.fit_sparse_timeline_activity(
        sparse_counts,
        tuple("abcde"),
        np.arange(4),
        "PFC",
        "A",
        0.5,
    )
    at_limit = replace(result.transform, n_iter=result.transform.parameters["max_iter"])
    assert at_limit.iteration_limit_reached
    assert not result.transform.iteration_limit_reached


@pytest.mark.parametrize(
    "invalid", ["nonfinite", "constant", "unit_order", "window_shape"]
)
def test_invalid_count_contract_fails_before_fit(sparse_counts, invalid):
    """No silent shape changes, nonfinite omission, duplicate IDs, or all-constant fits."""
    counts = sparse_counts.copy()
    ids, positions = tuple("abcde"), np.arange(4)
    if invalid == "nonfinite":
        counts[0, 0, 0] = np.nan
    elif invalid == "constant":
        counts[:] = 1
    elif invalid == "unit_order":
        ids = tuple("aabcd")
    else:
        positions = np.arange(3)
    with pytest.raises(ValueError):
        sparse.fit_sparse_timeline_activity(counts, ids, positions, "PFC", "A", 0.5)
