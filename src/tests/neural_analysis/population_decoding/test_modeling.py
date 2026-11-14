"""Weighted binary/multinomial classifiers and rotation invariance."""

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from src.neural_analysis.population_decoding.modeling import fit_classifier, predict_probabilities
from src.neural_analysis.population_decoding.features import fit_fold_pca, transform_pc_activity
from src.neural_analysis.task_decoding.pooled.decoding import stack_training_window


@pytest.mark.parametrize("classes", [2, 4])
def test_weighted_logistic_stationarity_and_probability_order(classes):
    """Input class count; verify fitted objective gradient and ordered (M,C) probabilities."""
    rng = np.random.default_rng(10 + classes)
    x = rng.normal(size=(120, 5))
    y = np.tile(np.arange(classes), 120 // classes)
    weights = rng.uniform(.1, 1, 120)
    fitted = fit_classifier(x, y, weights, tuple(range(classes)))
    assert fitted.available, fitted.reason
    probability, logp = predict_probabilities(x, fitted)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(np.exp(logp), probability)
    assert fitted.coefficients.shape == (1 if classes == 2 else 4, 5)
    if classes == 2:
        residual = probability[:, 1] - y
        gradient = x.T @ (weights * residual) + fitted.coefficients[0]
        intercept_gradient = np.sum(weights * residual)
    else:
        residual = probability - np.eye(classes)[y]
        gradient = (weights[:, None] * residual).T @ x + fitted.coefficients
        intercept_gradient = np.sum(weights[:, None] * residual, axis=0)
    assert np.max(np.abs(gradient)) < .02
    assert np.max(np.abs(intercept_gradient)) < .02


def test_all_pc_predictions_match_unit_rotation():
    """No inputs; an orthogonal full-PC basis preserves L2 readout predictions."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 5))
    y = (x[:, 0] + rng.normal(size=100) > 0).astype(int)
    pca = fit_fold_pca(x)
    pc = transform_pc_activity(x, pca)
    raw = fit_classifier(x, y, np.ones(100), (0, 1))
    rotated = fit_classifier(pc, y, np.ones(100), (0, 1))
    np.testing.assert_allclose(predict_probabilities(x, raw)[0],
                               predict_probabilities(pc, rotated)[0], atol=2e-4)


@pytest.mark.parametrize("bins", [2, 4, 10, 20, 40, 100])
def test_equal_trial_weight(bins):
    """Input supervised-bin count; flattened (T*B,F) rows sum to weight one/trial."""
    _, _, weight = stack_training_window(np.ones((3, bins, 2)), np.array([0, 1, 0]))
    np.testing.assert_allclose(weight.reshape(3, bins).sum(axis=1), 1)


def test_missing_training_class_is_unavailable():
    """No inputs; missing outcome class never becomes an invented probability column."""
    fitted = fit_classifier(np.ones((8, 2)), np.arange(8) % 3, np.ones(8), (0, 1, 2, 3))
    assert not fitted.available and "class" in fitted.reason


def test_stable_log_probabilities_at_extreme_logits():
    """No inputs; log probabilities remain finite when exp(probability) underflows."""
    from dataclasses import replace
    fitted = fit_classifier(np.array([[-1.], [1.]]), np.array([0, 1]), np.ones(2), (0, 1))
    fitted = replace(fitted, coefficients=np.array([[1000.]]), intercepts=np.array([0.]))
    p, lp = predict_probabilities(np.array([[-1.], [1.]]), fitted)
    assert np.isfinite(lp).all()
    np.testing.assert_allclose(lp[:, 1], [-1000, 0], atol=1e-12)
