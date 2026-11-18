"""Constant-choice evidence and nine-state integration for the chance baseline."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import TaskParams
from src.behavior_modeling.benchmark_models import ObservableHistory, build_observable_models


PREVIOUS_IDS = ("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch", "q_learning",
                "ideal_observer", "perseveration", "win_switch_lose_stay",
                "home_referenced_expectancy_doubt")


@pytest.fixture
def trials():
    """Return binary choice/outcome table [T=6,C], with task coding 1=Left/0=Right."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 20, 21, 25], "cur_block": [1, 1, 1, 2, 2, 3],
                         "action": [1, 0, 1, 1, 0, 0], "reward": [1, 0, 0, 1, 0, 1]})


def compare(table, ids=("chance",), error=.1):
    """Table[T,C]/IDs[K]/error scalar yield prepared history and comparison[T,K]."""
    candidates = models.resolve_trial_policy_candidates(ids)
    history = models.prepare_trial_policy_history(table, session_id="chance_test")
    scores, available = models.replay_trial_policy_candidates(history, candidates, TaskParams())
    result = analysis.compute_trial_policy_comparison(
        history, scores, available, ids, opposite_action_error_prob=error,
        choice_parameters=models.effective_trial_policy_choice_parameters(candidates, error_prob=error))
    return history, result


def test_chance_appends_to_defaults_and_has_no_history_or_free_parameters():
    """No inputs; the ninth default has a constant readout and empty feature parameters."""
    assert models.DEFAULT_MODEL_IDS == PREVIOUS_IDS + ("chance",)
    candidate, = models.resolve_trial_policy_candidates(("chance",))
    assert candidate.parameter_names == () and candidate.choice_temperature is None
    assert candidate.history_update_rule == "none"
    assert models.effective_trial_policy_choice_parameters((candidate,)) == {"chance": {"rule": "constant_half"}}
    assert models.effective_trial_policy_parameters(
        (candidate,), TaskParams(n_actions=3, perseveration_decay=-1)) == {"chance": {}}
    with pytest.raises(ValueError, match="chance"):
        models.effective_trial_policy_choice_parameters((candidate,), temperatures={"chance": .4})


@pytest.mark.parametrize("error", [.01, .1, .3])
def test_constant_probabilities_and_likelihoods_match_benchmark(trials, error):
    """Table/error inputs give 0.5 probabilities and natural-log evidence -ln(2)."""
    _, result = compare(trials, error=error)
    np.testing.assert_array_equal(result.policy_scores, np.zeros((len(trials), 1)))
    np.testing.assert_array_equal(result.p_right, np.full((len(trials), 1), .5))
    np.testing.assert_array_equal(result.log_likelihoods, np.full((len(trials), 1), -np.log(2)))
    assert result.model_indifferent.all() and np.all(result.greedy_choice == -1)
    benchmark = next(model for model in build_observable_models() if model.model_id == "chance")
    reference = benchmark.replay(ObservableHistory(
        trials.action.to_numpy(), trials.reward.to_numpy(), np.ones(len(trials), bool)))
    np.testing.assert_array_equal(result.p_right[:, 0], 1 - reference.p_left)


def test_observed_history_and_missing_rewards_do_not_change_predictions(trials):
    """Table[T,C] preserves predictions when actions change and every reward is absent."""
    changed = trials.assign(action=1-trials.action, reward=np.nan)
    original = changed.copy(deep=True)
    history, result = compare(changed)
    assert history.comparison_valid.all() and not history.history_update_valid.any()
    assert np.all(result.p_right == .5) and np.all(result.log_likelihoods == -np.log(2))
    pd.testing.assert_frame_equal(changed, original)


@pytest.mark.parametrize("excluded", ["manual", "unknown_flag", "no_choice", "invalid_switch"])
def test_excluded_attempts_have_no_predictions_or_likelihoods(trials, excluded):
    """Table/exclusion inputs preserve all T rows while excluding one comparison."""
    table = trials.astype({"action": object}).assign(experimenter_reward_given=0.)
    if excluded == "manual":
        table.loc[2, "experimenter_reward_given"] = 1
    elif excluded == "unknown_flag":
        table.loc[2, "experimenter_reward_given"] = np.nan
    else:
        table.loc[2, "action"] = "no_choice"
        if excluded == "invalid_switch":
            table.loc[2, "reward"] = 0
            table["trial_outcome"] = "choice"
            table.loc[2, "trial_outcome"] = "invalid_switch"
    _, result = compare(table)
    assert not result.prediction_valid[2, 0] and not result.comparison_valid[2]
    assert np.isnan(result.policy_scores[2, 0]) and np.isnan(result.log_likelihoods[2, 0])
    assert np.all(result.p_right[[0, 1, 3, 4, 5]] == .5)


def test_unavailable_other_prediction_excludes_common_evidence(trials, monkeypatch):
    """Table/spy fixtures ensure excluded rows pass zero log evidence for every HMM state."""
    history = models.prepare_trial_policy_history(trials, session_id="chance_test")
    scores = np.zeros((len(trials), 2))
    available = np.ones_like(scores, bool)
    available[2, 1] = False
    settings = {"chance": {"rule": "constant_half"},
                "other": {"rule": "fixed_error", "opposite_action_error_prob": .1}}
    result = analysis.compute_trial_policy_comparison(
        history, scores, available, ("chance", "other"), choice_parameters=settings)
    assert result.p_right[2, 0] == .5 and not result.comparison_valid[2]
    assert np.isnan(result.log_likelihoods[2]).all()
    smoother = Mock(wraps=analysis.smooth_fixed_exemplar_likelihoods_with_dynamax)
    monkeypatch.setattr(analysis, "smooth_fixed_exemplar_likelihoods_with_dynamax", smoother)
    analysis.smooth_trial_policy_comparison(result)
    np.testing.assert_array_equal(smoother.call_args.args[0][2], [0., 0.])


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_fresh_crops_and_empty_inputs(trials, rows):
    """Table/integer T gives aligned [T,1] arrays and the existing empty status."""
    _, result = compare(trials.iloc[2:2+rows])
    smoothed = analysis.smooth_trial_policy_comparison(result)
    assert smoothed.p_right.shape == smoothed.hmm_smoothed_probs.shape == (rows, 1)
    if rows:
        assert np.all(result.p_right == .5) and np.all(smoothed.hmm_smoothed_probs == 1)
    else:
        assert result.status == "empty"


def test_all_invalid_nine_state_input_is_prior_only(trials):
    """Table[T,C] with no choices retains uniform [T,9] posterior and no comparison evidence."""
    _, result = compare(trials.assign(action="no_choice", reward=0), PREVIOUS_IDS + ("chance",))
    smoothed = analysis.smooth_trial_policy_comparison(result)
    assert smoothed.status == "no_valid_comparisons"
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs, np.full((len(trials), 9), 1/9))
    assert np.isnan(result.log_likelihoods).all() and smoothed.hmm_smoothed_tie.all()


def test_neutral_nine_state_evidence_preserves_ties(trials):
    """Table[T,C] and neutral scores[T,9] preserve equal priors and nine-state transitions."""
    candidates = models.resolve_trial_policy_candidates()
    history = models.prepare_trial_policy_history(trials, session_id="chance_test")
    shape = (len(trials), len(candidates))
    result = analysis.compute_trial_policy_comparison(
        history, np.zeros(shape), np.ones(shape, bool), models.DEFAULT_MODEL_IDS,
        choice_parameters=models.effective_trial_policy_choice_parameters(candidates))
    smoothed = analysis.smooth_trial_policy_comparison(result)
    np.testing.assert_allclose(smoothed.initial_probs, np.full(9, 1/9))
    np.testing.assert_allclose(smoothed.transition_matrix[0], [.9] + [.0125]*8)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs, np.full(shape, 1/9), atol=2e-6)
    assert smoothed.hmm_smoothed_best_model_mask.all() and smoothed.hmm_smoothed_tie.all()


def test_alternating_choices_favor_chance_over_confident_constant_policies():
    """A deterministic T=80 choice sequence favors chance over fixed Left/Right policies."""
    table = pd.DataFrame({"cur_trial": np.arange(80), "cur_block": 1,
                          "action": np.tile([0, 1], 40), "reward": 0})
    history = models.prepare_trial_policy_history(table, session_id="chance_test")
    scores = np.tile([1., -1., 0.], (len(table), 1))
    settings = {name: {"rule": "fixed_error", "opposite_action_error_prob": .1}
                for name in ("left", "right")}
    candidate, = models.resolve_trial_policy_candidates(("chance",))
    settings.update(models.effective_trial_policy_choice_parameters((candidate,)))
    comparison = analysis.compute_trial_policy_comparison(
        history, scores, np.ones_like(scores, bool), ("left", "right", "chance"), choice_parameters=settings)
    smoothed = analysis.smooth_trial_policy_comparison(comparison)
    assert smoothed.hmm_smoothed_probs[10:-10, 2].mean() > .5


def test_eight_original_predictions_and_saved_arrays_remain_compatible(trials, tmp_path):
    """Table/Path fixtures preserve original scores/evidence and round-trip the ninth column."""
    history, current = compare(trials, PREVIOUS_IDS + ("chance",))
    _, previous = compare(trials, PREVIOUS_IDS)
    for field in ("policy_scores", "p_right", "log_likelihoods", "prediction_valid"):
        np.testing.assert_array_equal(getattr(current, field)[:, :8], getattr(previous, field))
    candidates = models.resolve_trial_policy_candidates()
    meta = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test")
    assert meta["analysis_version"] == "trial_policy_hmm_v9"
    assert meta["candidate_parameters"]["chance"] == {}
    assert meta["candidate_history_update_rules"]["chance"] == "none"
    assert meta["candidate_choice_parameters"]["chance"] == {"rule": "constant_half"}
    assert "0.5" in meta["choice_probability_definition"]
    old_meta = results.build_trial_policy_metadata(
        history, models.resolve_trial_policy_candidates(PREVIOUS_IDS), TaskParams(), parameter_source="test")
    assert old_meta["signature"] != meta["signature"]
    path = results.save_trial_policy_results(current, processed_data_path=tmp_path, metadata=meta)
    loaded, saved = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(loaded.log_likelihoods, current.log_likelihoods)
    assert saved["candidate_choice_parameters"]["chance"] == {"rule": "constant_half"}
    old_path = results.save_trial_policy_results(previous, processed_data_path=tmp_path, metadata=old_meta)
    historical, _ = results.load_trial_policy_results(old_path)
    assert historical.p_right.shape == (len(trials), 8)
