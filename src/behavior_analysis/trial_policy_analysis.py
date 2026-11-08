"""Bernoulli comparison and retrospective smoothing of fixed causal policies."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
from scipy.special import softmax
from scipy.stats import bernoulli

from .exemplar_hmm_modeling import (
    build_sticky_transition_matrix,
    smooth_fixed_exemplar_likelihoods_with_dynamax,
)
from .trial_policy_models import PreparedTrialHistory


PROBABILITY_TIE_ATOL = 1e-6
MIN_TRANSITION_PROBABILITY = 1e-6
TRANSITION_ROUNDOFF_ATOL = 1e-12


@dataclass(frozen=True)
class TrialPolicyResult:
    """Data-only trial result: T retained rows by K ordered candidate columns.

    Probabilities/residuals are dimensionless; log likelihoods use natural logs.
    observed_action uses raw 0=Right/1=Left and NaN for no choice. Greedy sides
    use the same codes with -1 for unavailable/indifferent. State indices are
    zero-based model_ids indices; -1 is unavailable/ambiguous. Independent ties
    use int8 -1=unavailable, 0=unique, 1=tied. HMM arrays are None when disabled.
    Identifiers are Unicode; row_position is int64[T], original retained order.
    """

    model_ids: np.ndarray
    session_id: str
    trial_id: np.ndarray
    block_id: np.ndarray
    row_position: np.ndarray
    observed_action: np.ndarray
    comparison_valid: np.ndarray
    history_update_valid: np.ndarray
    comparison_exclusion_reason: np.ndarray
    history_update_exclusion_reason: np.ndarray
    prediction_valid: np.ndarray
    model_indifferent: np.ndarray
    greedy_choice: np.ndarray
    p_right: np.ndarray
    log_likelihoods: np.ndarray
    choice_residuals: np.ndarray
    independent_probs: np.ndarray
    independent_best_model_mask: np.ndarray
    independent_state: np.ndarray
    independent_tie: np.ndarray
    status: str
    initial_probs: np.ndarray | None = None
    transition_matrix: np.ndarray | None = None
    hmm_smoothed_probs: np.ndarray | None = None
    hmm_smoothed_best_model_mask: np.ndarray | None = None
    hmm_smoothed_state: np.ndarray | None = None
    hmm_smoothed_tie: np.ndarray | None = None


def assign_unique_probability_maxima(
    probabilities: np.ndarray, valid_rows: np.ndarray, *, atol: float = PROBABILITY_TIE_ATOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tied-maximum bool[T,K], state int64[T], and tie int8[T] arrays.

    probabilities is a dimensionless float[T,K] matrix (K>=1); valid_rows is
    bool[T]. Valid rows must be finite. Invalid rows receive all-false masks
    and -1 state/tie values. Absolute probability tolerance atol has rtol=0.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    valid_rows = np.asarray(valid_rows, dtype=bool)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must have shape (T,K), K >= 1")
    if valid_rows.shape != (len(probabilities),):
        raise ValueError("valid_rows must have shape (T,)")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("Probability tie atol must be finite and nonnegative")
    selected = probabilities[valid_rows]
    if not np.isfinite(selected).all():
        raise ValueError("Valid probabilities must be finite")
    best = np.zeros(probabilities.shape, dtype=bool)
    states = np.full(len(probabilities), -1, dtype=np.int64)
    ties = np.full(len(probabilities), -1, dtype=np.int8)
    if len(selected):
        winners = np.isclose(selected, selected.max(axis=1, keepdims=True), atol=atol, rtol=0)
        unique = winners.sum(axis=1) == 1
        best[valid_rows] = winners
        states[valid_rows] = np.where(unique, winners.argmax(axis=1), -1)
        ties[valid_rows] = (~unique).astype(np.int8)
    return best, states, ties


def compute_trial_policy_comparison(
    history: PreparedTrialHistory, scores: np.ndarray, prediction_valid: np.ndarray,
    model_ids: Sequence[str], *, opposite_action_error_prob: float = .10,
    probability_tie_atol: float = PROBABILITY_TIE_ATOL,
) -> TrialPolicyResult:
    """Compare pre-choice policies on a common animal-choice eligibility mask.

    history contains normalized (T,) observations, raw 0=Right/1=Left actions.
    scores is dimensionless left-positive float[T,K]; prediction_valid is
    bool[T,K]. model_ids supplies K unique ordered labels. Shared dimensionless
    error probability is strictly between zero and .5, with a representable
    complement below one. Returns full-row TrialPolicyResult, without HMM fields
    or I/O. Unavailable comparisons have NaN likelihoods/probabilities/residuals.
    """
    error = float(opposite_action_error_prob)
    if not np.isfinite(error) or not 0 < error < .5 or not 0 < 1 - error < 1:
        raise ValueError("error probability and its complement must be representable strictly inside (0,1), error < .5")
    ids = np.asarray(tuple(model_ids), dtype=str)
    if ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("model_ids must be a nonempty unique ordered sequence")
    scores = np.asarray(scores, dtype=float)
    available = np.asarray(prediction_valid, dtype=bool)
    shape = (len(history.actions), len(ids))
    if scores.shape != shape or available.shape != shape:
        raise ValueError("scores and prediction_valid must have shape (T,K)")
    if not np.isfinite(scores[available]).all():
        raise ValueError("Available policy scores must be finite")

    indifferent = available & (scores == 0)
    greedy = np.full(shape, -1, dtype=np.int8)
    greedy[available & (scores < 0)] = 0
    greedy[available & (scores > 0)] = 1
    probabilities = np.full(shape, np.nan)
    probabilities[greedy == 0] = 1 - error
    probabilities[greedy == 1] = error
    probabilities[indifferent] = .5
    comparison_valid = history.comparison_valid & available.all(axis=1)
    likelihoods = np.full(shape, np.nan)
    residuals = np.full(shape, np.nan)
    independent = np.full(shape, np.nan)
    y_right = 1 - history.actions[comparison_valid]
    likelihoods[comparison_valid] = bernoulli.logpmf(y_right[:, None], p=probabilities[comparison_valid])
    residuals[comparison_valid] = y_right[:, None] - probabilities[comparison_valid]
    if comparison_valid.any():
        independent[comparison_valid] = softmax(likelihoods[comparison_valid], axis=1)
    best, states, ties = assign_unique_probability_maxima(independent, comparison_valid, atol=probability_tie_atol)
    reasons = history.comparison_exclusion_reason.tolist()
    for row in np.flatnonzero(~available.all(axis=1)):
        reasons[row] = ";".join(filter(None, (reasons[row], "unavailable_prediction")))
    status = "empty" if not shape[0] else ("ok" if comparison_valid.any() else "no_valid_comparisons")
    return TrialPolicyResult(
        ids, history.session_id, history.trial_id.copy(), history.block_id.copy(), history.row_position.copy(),
        history.actions.copy(), comparison_valid, history.history_update_valid.copy(), np.asarray(reasons, dtype=str),
        history.history_update_exclusion_reason.copy(), available.copy(), indifferent, greedy, probabilities,
        likelihoods, residuals, independent, best, states, ties, status,
    )


def trial_policy_transition_matrix(num_models: int, stay_prob: float) -> np.ndarray:
    """Build fixed float64[K,K] transitions within the supported trialwise range.

    num_models is integer K>=1; stay_prob is a dimensionless finite scalar.
    K>1 requires inclusive [1e-6,1-(K-1)*1e-6] requested bounds. K=1 uses [[1]].
    Derived float64/JAX entries allow only 1e-12 absolute roundoff at the floor.
    This wrapper validates trial settings without changing the block helper.
    """
    stay = float(stay_prob)
    if not np.isfinite(stay) or not 0 <= stay <= 1:
        raise ValueError("stay probability must be finite and in [0,1]")
    if num_models > 1 and not MIN_TRANSITION_PROBABILITY <= stay <= 1 - (num_models - 1) * MIN_TRANSITION_PROBABILITY:
        raise ValueError("stay probability is outside the supported trial transition bounds")
    matrix = build_sticky_transition_matrix(num_models, stay)
    if num_models > 1:
        for values in (matrix, np.asarray(jnp.asarray(matrix))):
            if np.any(values.astype(float) < MIN_TRANSITION_PROBABILITY - TRANSITION_ROUNDOFF_ATOL):
                raise ValueError("stay probability produces unsupported transition entries after dtype conversion")
    return matrix


def smooth_trial_policy_comparison(
    result: TrialPolicyResult, *, stay_prob: float = .90,
    probability_tie_atol: float = PROBABILITY_TIE_ATOL,
) -> TrialPolicyResult:
    """Return a new result with retrospective full-row Dynamax marginal outputs.

    result contains (T,K) independent arrays; stay_prob is a dimensionless fixed
    transition setting, and tie tolerance is absolute probability units. Entire
    excluded rows use zero HMM log evidence, preserving one transition step per
    retained row. Empty input skips Dynamax; all-invalid input remains prior-only.
    Independent fields and their missing values remain unchanged.
    """
    trials, models = result.log_likelihoods.shape
    transitions = trial_policy_transition_matrix(models, stay_prob)
    initial = np.full(models, 1 / models)
    if trials and result.comparison_valid.any():
        likelihoods = np.where(result.comparison_valid[:, None], result.log_likelihoods, 0.)
        smoothing = smooth_fixed_exemplar_likelihoods_with_dynamax(likelihoods, stay_prob=stay_prob)
        probabilities = smoothing.smoothed_probs
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=2e-6, rtol=0):
            raise ValueError(f"Session {result.session_id}: invalid Dynamax posterior probabilities")
    else:
        probabilities = np.full((trials, models), 1 / models)
    best, states, ties = assign_unique_probability_maxima(probabilities, np.ones(trials, bool), atol=probability_tie_atol)
    return replace(result, initial_probs=initial, transition_matrix=transitions,
                   hmm_smoothed_probs=probabilities, hmm_smoothed_best_model_mask=best,
                   hmm_smoothed_state=states, hmm_smoothed_tie=ties.astype(bool))
