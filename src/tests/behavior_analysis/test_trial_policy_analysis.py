"""Hand calculations and independent HMM references for trial-policy inference."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp


@pytest.fixture
def history():
    """Return four retained rows, including an interior no-choice observation."""
    from src.behavior_analysis.trial_policy_models import prepare_trial_policy_history

    table = pd.DataFrame({"cur_trial": [10, 12, 15, 19], "cur_block": [2, 2, 3, 3],
                          "action": [0, "no_choice", 1, 0], "reward": [1, 0, 0, 1]})
    return prepare_trial_policy_history(table, session_id="synthetic")


def enumerate_posterior(log_likelihoods, transition):
    """Return exact marginals (T,K) by enumerating paths; inputs are log masses/A."""
    count, models = log_likelihoods.shape
    paths = np.asarray(list(product(range(models), repeat=count)))
    log_weights = np.full(len(paths), -np.log(models))
    for row in range(count):
        log_weights += log_likelihoods[row, paths[:, row]]
        if row:
            log_weights += np.log(transition[paths[:, row - 1], paths[:, row]])
    weights = np.exp(log_weights - logsumexp(log_weights))
    return np.array([[weights[paths[:, row] == model].sum() for model in range(models)]
                     for row in range(count)])


def stable_posterior(log_likelihoods, transition):
    """Return normalized log-domain forward/backward reference marginals (T,K)."""
    count, models = log_likelihoods.shape
    log_transition = np.log(transition)
    forward = np.empty_like(log_likelihoods)
    forward[0] = -np.log(models) + log_likelihoods[0]
    forward[0] -= logsumexp(forward[0])
    for row in range(1, count):
        forward[row] = log_likelihoods[row] + logsumexp(
            forward[row - 1, :, None] + log_transition, axis=0)
        forward[row] -= logsumexp(forward[row])
    backward = np.zeros_like(log_likelihoods)
    for row in range(count - 2, -1, -1):
        backward[row] = logsumexp(log_transition + log_likelihoods[row + 1][None, :]
                                  + backward[row + 1][None, :], axis=1)
        backward[row] -= logsumexp(backward[row])
    joint = forward + backward
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def test_bernoulli_direction_indifference_common_mask_and_nullable_ties(history):
    """Left-positive scores produce hand-checked likelihoods and shared exclusions."""
    from src.behavior_analysis import trial_policy_analysis as analysis

    scores = np.array([[-1, 1, 0], [-1, 1, 0], [1, 1, -1], [1, -1, -1.]])
    available = np.ones_like(scores, dtype=bool)
    available[3, 0] = False
    result = analysis.compute_trial_policy_comparison(history, scores, available, ("a", "b", "c"))
    np.testing.assert_allclose(result.p_right[0], [.9, .1, .5])
    np.testing.assert_allclose(result.log_likelihoods[0], np.log([.9, .1, .5]))
    np.testing.assert_allclose(result.independent_probs[0], np.array([.9, .1, .5]) / 1.5)
    assert result.independent_state.tolist() == [0, -1, -1, -1]
    assert result.independent_tie.tolist() == [0, -1, 1, -1]
    assert result.greedy_choice[0].tolist() == [0, 1, -1]
    assert result.model_indifferent[0].tolist() == [False, False, True]
    assert not result.model_indifferent[3, 0]
    assert result.comparison_valid.tolist() == [True, False, True, False]
    assert result.history_update_valid[3]
    assert np.isnan(result.log_likelihoods[[1, 3]]).all()
    assert not result.independent_best_model_mask[[1, 3]].any()
    assert result.independent_best_model_mask[2].tolist() == [True, True, False]
    assert result.hmm_smoothed_probs is None
    np.testing.assert_allclose(result.choice_residuals[0], [0.1, .9, .5])


def test_interior_neutral_row_matches_enumerated_paths(history, monkeypatch):
    """Smoothing retains the interior time step and never passes NaNs to Dynamax."""
    from src.behavior_analysis import trial_policy_analysis as analysis

    scores = np.array([[-1, 1], [1, -1], [1, -1], [-1, 1.]])
    result = analysis.compute_trial_policy_comparison(history, scores, np.ones_like(scores, bool), ("a", "b"))
    smoothed = analysis.smooth_trial_policy_comparison(result, stay_prob=.8)
    emissions = result.log_likelihoods.copy()
    emissions[~result.comparison_valid] = 0
    expected = enumerate_posterior(emissions, smoothed.transition_matrix)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs, expected, atol=2e-6, rtol=0)
    assert smoothed.hmm_smoothed_probs.shape == (4, 2)
    assert np.isfinite(smoothed.hmm_smoothed_probs).all()
    assert result.hmm_smoothed_probs is None
    assert np.isnan(smoothed.log_likelihoods[1]).all()
    compressed = enumerate_posterior(emissions[[0, 2, 3]], smoothed.transition_matrix)
    assert not np.allclose(expected[[0, 2, 3]], compressed)


@pytest.mark.parametrize("models", [1, 2, 3, 4])
def test_identical_policies_uniform_transition_and_model_permutation(history, models):
    """Generic K matrices preserve uniform ties, independent inference and ordering."""
    from src.behavior_analysis import trial_policy_analysis as analysis

    ids = tuple(f"candidate_{index}" for index in range(models))
    scores = np.ones((4, models))
    result = analysis.compute_trial_policy_comparison(history, scores, np.ones_like(scores, bool), ids)
    uniform = analysis.smooth_trial_policy_comparison(result)
    np.testing.assert_allclose(uniform.hmm_smoothed_probs, 1 / models, atol=2e-6, rtol=0)
    assert uniform.hmm_smoothed_tie.tolist() == [models > 1] * 4
    scores[:, ::2] *= -1
    distinct = analysis.compute_trial_policy_comparison(history, scores, np.ones_like(scores, bool), ids)
    smoothed = analysis.smooth_trial_policy_comparison(distinct, stay_prob=1 / models)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs[distinct.comparison_valid],
                               distinct.independent_probs[distinct.comparison_valid], atol=2e-6, rtol=0)
    permuted = analysis.compute_trial_policy_comparison(history, scores[:, ::-1],
                                                       np.ones_like(scores, bool), ids[::-1])
    np.testing.assert_allclose(permuted.independent_probs[:, ::-1], distinct.independent_probs, equal_nan=True)
    np.testing.assert_array_equal(permuted.independent_best_model_mask[:, ::-1], distinct.independent_best_model_mask)


@pytest.mark.parametrize("rows,valid,status", [(0, False, "empty"), (1, False, "no_valid_comparisons"),
                                              (3, False, "no_valid_comparisons"), (1, True, "ok")])
@pytest.mark.parametrize("models", [1, 4])
def test_empty_single_row_and_prior_only_status(rows, valid, status, models):
    """Edge shapes have defined evidence status and optional aligned HMM outputs."""
    from src.behavior_analysis import trial_policy_analysis as analysis
    from src.behavior_analysis.trial_policy_models import prepare_trial_policy_history

    table = pd.DataFrame({"cur_trial": range(rows), "cur_block": [0] * rows,
                          "action": [0 if valid else None] * rows, "reward": [0] * rows})
    prepared = prepare_trial_policy_history(table, session_id="edge")
    scores = np.zeros((rows, models))
    result = analysis.compute_trial_policy_comparison(prepared, scores, np.ones_like(scores, bool),
                                                       tuple(f"m{index}" for index in range(models)))
    assert result.status == status
    assert result.independent_probs.shape == (rows, models)
    result = analysis.smooth_trial_policy_comparison(result)
    assert result.hmm_smoothed_probs.shape == (rows, models)
    np.testing.assert_allclose(result.hmm_smoothed_probs, 1 / models, atol=2e-6, rtol=0)


@pytest.mark.parametrize("error", [0, .5, 1, -1, np.nan, np.inf, 1e-20])
def test_invalid_error_probabilities_raise(history, error):
    """Invalid or unrepresentable Bernoulli settings fail before numerical scoring."""
    from src.behavior_analysis import trial_policy_analysis as analysis

    with pytest.raises(ValueError, match="error"):
        analysis.compute_trial_policy_comparison(history, np.ones((4, 1)), np.ones((4, 1), bool),
                                                  ("a",), opposite_action_error_prob=error)


@pytest.mark.parametrize("models", [2, 3, 4])
def test_transition_boundaries_and_long_reversals(history, models):
    """Inclusive bounds work in real Dynamax and adjacent outside requests fail."""
    from src.behavior_analysis import trial_policy_analysis as analysis
    from src.behavior_analysis.trial_policy_models import prepare_trial_policy_history

    table = pd.DataFrame({"cur_trial": range(121), "cur_block": [0] * 121,
                          "action": [0] * 121, "reward": [0] * 121})
    prepared = prepare_trial_policy_history(table, session_id="reversal")
    scores = np.ones((121, models))
    scores[:60, 0] = -1
    scores[60:, 1] = -1
    independent = analysis.compute_trial_policy_comparison(prepared, scores, np.ones_like(scores, bool),
                                                           tuple(f"m{i}" for i in range(models)))
    lower, upper = 1e-6, 1 - (models - 1) * 1e-6
    for stay in (lower, .9, upper):
        result = analysis.smooth_trial_policy_comparison(independent, stay_prob=stay)
        expected = stable_posterior(result.log_likelihoods, result.transition_matrix)
        np.testing.assert_allclose(result.hmm_smoothed_probs, expected, atol=2e-6, rtol=0)
        np.testing.assert_allclose(result.hmm_smoothed_probs.sum(axis=1), 1, atol=2e-6, rtol=0)
    for stay in (0, 1, np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf), np.nan, np.inf):
        with pytest.raises(ValueError, match="stay"):
            analysis.smooth_trial_policy_comparison(independent, stay_prob=stay)


def test_probability_maxima_tolerance_and_tiny_score_direction(history):
    """Posterior ties use an absolute tolerance; policy scores retain exact signs."""
    from src.behavior_analysis import trial_policy_analysis as analysis

    masks, states, ties = analysis.assign_unique_probability_maxima(
        np.array([[.5, .5 + 1e-7], [.5, .5 + 1e-4], [np.nan, np.nan]]),
        np.array([True, True, False]))
    assert masks.tolist() == [[True, True], [False, True], [False, False]]
    assert states.tolist() == [-1, 1, -1]
    assert ties.tolist() == [1, 0, -1]
    scores = np.tile([-1e-100, 1e-100], (4, 1))
    result = analysis.compute_trial_policy_comparison(history, scores, np.ones_like(scores, bool), ("a", "b"))
    np.testing.assert_allclose(result.p_right, np.tile([.9, .1], (4, 1)))


def test_unexpected_nonfinite_scores_are_errors_not_missing_predictions(history):
    """A claimed available numerical failure cannot become ordinary missingness."""
    from src.behavior_analysis import trial_policy_analysis as analysis

    with pytest.raises(ValueError, match="finite"):
        analysis.compute_trial_policy_comparison(history, np.full((4, 1), np.inf),
                                                  np.ones((4, 1), bool), ("a",))
