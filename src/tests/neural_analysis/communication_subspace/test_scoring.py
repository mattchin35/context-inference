"""Held-out per-neuron R2 and complete-support summary contracts."""

import numpy as np
import pytest

from src.neural_analysis.communication_subspace.scoring import score_target_predictions


def test_per_target_scores_keep_negative_and_undefined_r2():
    """No inputs; (sample, neuron) counts/bin yield per-neuron SSE, SST, and R2."""
    actual = np.array([[0., 3., 5.], [1., 3., 5.], [2., 3., 5.]])
    predicted = np.array([[2., 3., 5.], [2., 3., 5.], [2., 3., 5.]])
    scores = score_target_predictions(actual, predicted)
    np.testing.assert_allclose(scores.sse, [5., 0., 0.])
    np.testing.assert_allclose(scores.sst, [2., 0., 0.])
    np.testing.assert_array_equal(scores.valid_mask, [True, False, False])
    assert scores.r2[0] == -1.5
    assert np.isnan(scores.r2[1:]).all()
    assert scores.mean_r2 == -1.5


def test_constant_training_target_can_be_scored_on_variable_test():
    """No inputs; a constant train target's mean predicts variable test counts/bin."""
    actual = np.array([[3.], [5.], [7.]])
    prediction_from_training_mean = np.full((3, 1), 4.)
    scores = score_target_predictions(actual, prediction_from_training_mean)
    np.testing.assert_allclose(scores.sse, [11.])
    np.testing.assert_allclose(scores.sst, [8.])
    assert scores.valid_mask.tolist() == [True]
    assert scores.r2[0] == -0.375


def test_nonfinite_prediction_fails_whole_score_instead_of_dropping_target():
    """No inputs; nonfinite (sample, neuron) predictions raise a fit/scoring error."""
    actual = np.array([[0., 1.], [1., 2.], [2., 3.]])
    prediction = actual.copy()
    prediction[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_target_predictions(actual, prediction)


def test_all_constant_test_targets_have_no_mean_score():
    """No inputs; zero test SST for every neuron yields undefined fold mean R2."""
    actual = np.full((3, 2), 5.)
    predicted = np.zeros((3, 2))
    scores = score_target_predictions(actual, predicted)
    np.testing.assert_array_equal(scores.valid_mask, [False, False])
    assert np.isnan(scores.r2).all()
    assert np.isnan(scores.mean_r2)
