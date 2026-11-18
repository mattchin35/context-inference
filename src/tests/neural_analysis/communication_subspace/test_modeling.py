"""Analytic checks for training transforms and penalized reduced-rank regression."""

import numpy as np
import pytest

from src.neural_analysis.communication_subspace.modeling import (
    fit_ridge_decomposition,
    fit_training_transform,
    inverse_transform_predictions,
    predict_rank,
    select_alpha,
    transform_counts,
)


def test_training_transform_uses_population_sd_and_training_rows_only():
    """No inputs; variable (row, neuron) counts use training ddof=0 and remain unchanged."""
    train = np.array([[1., 4., 7.], [3., 4., 11.], [5., 4., 15.]])
    held_out = np.array([[7., 1000., 19.]])
    train_copy = train.copy()
    held_out_copy = held_out.copy()

    transform = fit_training_transform(train)
    np.testing.assert_array_equal(transform.variable_mask, [True, False, True])
    np.testing.assert_allclose(transform.means, [3., 4., 11.])
    np.testing.assert_allclose(transform.scales[[0, 2]], [np.sqrt(8 / 3), np.sqrt(32 / 3)])
    np.testing.assert_allclose(transform_counts(train, transform).mean(axis=0), [0., 0.], atol=1e-15)
    np.testing.assert_allclose(transform_counts(train, transform).std(axis=0), [1., 1.])
    np.testing.assert_allclose(
        transform_counts(held_out, transform),
        [[4 / np.sqrt(8 / 3), 8 / np.sqrt(32 / 3)]],
    )
    np.testing.assert_array_equal(train, train_copy)
    np.testing.assert_array_equal(held_out, held_out_copy)


def test_inverse_transform_restores_counts_and_constant_target_identity():
    """No inputs; variable z predictions (row, variable target) restore all counts/bin."""
    train = np.array([[2., 7., 10.], [4., 7., 14.], [6., 7., 18.]])
    transform = fit_training_transform(train)
    predicted = inverse_transform_predictions(np.array([[0., 1.], [-1., 0.]]), transform)
    assert predicted.shape == (2, 3)
    np.testing.assert_allclose(predicted, [[4., 7., 14. + np.sqrt(32 / 3)],
                                                [4. - np.sqrt(8 / 3), 7., 14.]])


def test_small_nonzero_variance_is_not_treated_as_constant():
    """No inputs; finite low-variance (row, neuron) counts retain variable identity."""
    counts = np.array([[0., 1.], [1e-10, 1.], [2e-10, 1.]])
    transform = fit_training_transform(counts)
    np.testing.assert_array_equal(transform.variable_mask, [True, False])
    np.testing.assert_allclose(transform_counts(counts, transform)[:, 0],
                               [-np.sqrt(3 / 2), 0., np.sqrt(3 / 2)])


def test_penalized_rrr_matches_independent_augmented_design_oracle():
    """No inputs; rank-r coefficients match augmented least-squares reduced-rank fit."""
    x = np.array([[-2., 1.], [-1., -1.], [0., 1.], [1., -1.], [2., 0.]])
    y = np.array([[2., 0., 1.], [0., 2., -1.], [1., 1., 2.],
                  [2., -1., 0.], [4., 0., 3.]])
    alpha = 0.7
    sample_count, source_count = x.shape
    augmented_x = np.vstack((x, np.sqrt(sample_count * alpha) * np.eye(source_count)))
    augmented_y = np.vstack((y, np.zeros((source_count, y.shape[1]))))
    q_matrix, r_matrix = np.linalg.qr(augmented_x, mode="reduced")
    u_matrix, singular_values, v_transpose = np.linalg.svd(q_matrix.T @ augmented_y,
                                                            full_matrices=False)
    rank_one_oracle = np.linalg.solve(
        r_matrix, (u_matrix[:, :1] * singular_values[:1]) @ v_transpose[:1, :]
    )
    ridge_oracle = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]

    decomposition = fit_ridge_decomposition(x, y, alpha)
    np.testing.assert_allclose(decomposition.ridge_coefficients, ridge_oracle, atol=1e-12)
    np.testing.assert_allclose(predict_rank(x, decomposition, 1), x @ rank_one_oracle,
                               atol=1e-12)
    assert np.linalg.matrix_rank(rank_one_oracle) == 1
    assert decomposition.max_rank == 2
    np.testing.assert_allclose(predict_rank(x, decomposition, 2), x @ ridge_oracle,
                               atol=1e-12)
    np.testing.assert_allclose(predict_rank(x, decomposition, None), x @ ridge_oracle,
                               atol=1e-12)


def test_collinear_sources_fit_ridge_and_limit_identifiable_rank():
    """No inputs; collinear (row, source) design permits ridge but only one RRR rank."""
    x = np.array([[-2., -4.], [-1., -2.], [0., 0.], [1., 2.], [2., 4.]])
    y = np.column_stack((x[:, 0], -x[:, 0]))
    decomposition = fit_ridge_decomposition(x, y, 0.1)
    assert decomposition.max_rank == 1
    np.testing.assert_allclose(predict_rank(x, decomposition, 1),
                               predict_rank(x, decomposition, None), atol=1e-12)
    with pytest.raises(ValueError, match="rank"):
        predict_rank(x, decomposition, 2)
    with pytest.raises(ValueError, match="rank"):
        predict_rank(x, decomposition, 0)


def test_zero_cross_covariance_does_not_remove_supported_ranks():
    """No inputs; zero X.T Y leaves ranks 1..2 supported with zero predictions."""
    x = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                  [1., 1.], [-1., -1.]])
    y = np.array([[1., -1.], [1., -1.], [-1., -1.], [-1., -1.],
                  [0., 2.], [0., 2.]])
    np.testing.assert_allclose(x.T @ y, 0., atol=1e-15)
    decomposition = fit_ridge_decomposition(x, y, 0.3)
    assert decomposition.max_rank == 2
    for rank in (1, 2):
        np.testing.assert_allclose(predict_rank(x, decomposition, rank), 0., atol=1e-14)


def test_rank_tolerance_uses_training_design_not_crosscovariance():
    """No inputs; float64 SVD tolerance excludes a near-null source direction."""
    x = np.array([[1., 0.], [-1., 0.], [0., 1e-16], [0., -1e-16]])
    y = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    decomposition = fit_ridge_decomposition(x, y, 0.1)
    assert decomposition.max_rank == 1


def test_nonfinite_model_inputs_fail_explicitly():
    """No inputs; nonfinite dimensionless arrays fail before fitting or prediction."""
    x = np.array([[-1., 0.], [0., 1.], [1., -1.]])
    y = np.array([[0.], [1.], [2.]])
    with pytest.raises(ValueError, match="finite"):
        fit_ridge_decomposition(x, np.array([[0.], [np.nan], [2.]]), 0.1)
    decomposition = fit_ridge_decomposition(x, y, 0.1)
    bad_x = x.copy()
    bad_x[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        predict_rank(bad_x, decomposition, None)


def test_select_alpha_prefers_larger_penalty_at_numerical_tie():
    """No inputs; finite 1D candidate R2 scores choose largest alpha within 1e-12."""
    alphas = np.array([1e-2, 1e-1, 1.])
    assert select_alpha(alphas, np.array([0.4, 0.5, 0.5 - 5e-13])) == 1.
    assert select_alpha(alphas, np.array([0.4, 0.5, 0.5 - 2e-12])) == 0.1
    assert select_alpha(alphas, np.array([np.nan, np.nan, np.nan])) is None
