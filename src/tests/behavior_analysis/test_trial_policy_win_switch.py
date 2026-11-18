"""Win-switch/lose-stay uses the existing rule with trial-HMM discrete errors."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import trial_features
from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import TaskParams
from src.behavior_modeling.benchmark_models import ObservableHistory, build_observable_models


DEFAULT_IDS = ("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch", "q_learning",
               "ideal_observer", "perseveration", "win_switch_lose_stay",
               "home_referenced_expectancy_doubt")


@pytest.fixture
def trials():
    """Return chronological binary action/outcome table [T=6,C], 1=Left/0=Right."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 20, 21, 25], "cur_block": [1, 1, 1, 2, 2, 2],
                         "action": [1, 0, 0, 1, 1, 0], "reward": [1, 0, 1, 0, 1, 0]})


def compare(table, ids=("win_switch_lose_stay",)):
    """Table[T,C]/IDs[K] yield prepared history and comparison with arrays[T,K]."""
    candidates = models.resolve_trial_policy_candidates(ids)
    history = models.prepare_trial_policy_history(table, session_id="win_switch_test")
    scores, available = models.replay_trial_policy_candidates(history, candidates, TaskParams())
    result = analysis.compute_trial_policy_comparison(
        history, scores, available, ids,
        choice_parameters=models.effective_trial_policy_choice_parameters(candidates))
    return history, result


def test_eight_defaults_and_retained_optional_candidates():
    """No inputs; the selected defaults change while both former IDs remain usable."""
    assert models.DEFAULT_MODEL_IDS == DEFAULT_IDS + ("chance",)
    optional = ("reward_decay_hmm", "probe_persistence_doubt")
    assert tuple(candidate.model_id for candidate in models.resolve_trial_policy_candidates(optional)) == optional
    candidate, = models.resolve_trial_policy_candidates(("win_switch_lose_stay",))
    assert candidate.parameter_names == () and candidate.history_update_rule == "observed_choice_and_reward"
    assert models.effective_trial_policy_choice_parameters((candidate,)) == {
        "win_switch_lose_stay": {"rule": "fixed_error", "opposite_action_error_prob": .1}}


def test_default_subset_accepts_temperature_settings_for_optional_graded_models():
    """No inputs; shared temperature configs may retain known unselected optional IDs."""
    settings = models.effective_trial_policy_choice_parameters(
        models.resolve_trial_policy_candidates(),
        temperatures={"reward_decay_hmm": .4, "probe_persistence_doubt": .5})
    assert set(settings) == set(DEFAULT_IDS + ("chance",))


def test_reward_switch_omission_stay_matches_existing_benchmark(trials, monkeypatch):
    """Table fixture gives signed rules[T] matching benchmark, with 90/10 likelihoods."""
    benchmark = next(model for model in build_observable_models() if model.model_id == "win_switch_lose_stay")
    reference = benchmark.replay(ObservableHistory(
        trials.action.to_numpy(), trials.reward.to_numpy(), np.ones(len(trials), bool)))
    shared = Mock(wraps=trial_features.win_stay_lose_switch_regressor)
    monkeypatch.setattr(trial_features, "win_stay_lose_switch_regressor", shared)
    _, actual = compare(trials)
    assert shared.call_count == 1
    np.testing.assert_array_equal(actual.policy_scores[:, 0], [0, -1, -1, 1, 1, -1])
    np.testing.assert_array_equal(actual.policy_scores[:, 0], reference.model_value)
    np.testing.assert_allclose(actual.p_right[:, 0], [.5, .9, .9, .1, .1, .9])
    np.testing.assert_allclose(actual.log_likelihoods[:, 0], np.log([.5, .9, .9, .9, .9, .9]))


@pytest.mark.parametrize("excluded", ["manual", "no_choice", "unknown_flag", "missing_reward"])
def test_exclusions_and_missing_rewards_keep_last_completed_pair(trials, excluded):
    """Table/condition inputs freeze history, retaining predictions for missing rewards."""
    table = trials.astype({"action": object, "reward": float}).assign(experimenter_reward_given=0.)
    if excluded == "manual":
        table.loc[2, "experimenter_reward_given"] = 1
    elif excluded == "unknown_flag":
        table.loc[2, "experimenter_reward_given"] = np.nan
    elif excluded == "no_choice":
        table.loc[2, "action"] = "no_choice"
    else:
        table.loc[2, "reward"] = np.nan
    history, actual = compare(table)
    _, compressed = compare(trials.drop(index=2))
    np.testing.assert_array_equal(actual.policy_scores[[0, 1, 3, 4, 5]], compressed.policy_scores)
    assert not history.history_update_valid[2]
    assert actual.comparison_valid[2] == (excluded == "missing_reward")
    if excluded == "missing_reward":
        assert actual.p_right[2, 0] == actual.p_right[3, 0] == .9


def test_causality_and_fresh_crop_start(trials):
    """Table[T,C] preserves prefixes and begins each retained crop at 50/50."""
    _, baseline = compare(trials)
    changed = trials.copy()
    changed.loc[2:, ["action", "reward"]] = 1 - changed.loc[2:, ["action", "reward"]]
    _, altered = compare(changed)
    np.testing.assert_array_equal(baseline.policy_scores[:3], altered.policy_scores[:3])
    _, crop = compare(trials.iloc[2:])
    assert crop.p_right[0, 0] == .5


def test_eight_model_roundtrip_and_metadata(trials, tmp_path):
    """Table/Path fixtures save [T,8] posterior arrays and parameter-free rule metadata."""
    history, comparison = compare(trials, DEFAULT_IDS)
    smoothed = analysis.smooth_trial_policy_comparison(comparison)
    assert smoothed.hmm_smoothed_probs.shape == (len(trials), 8)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs.sum(axis=1), 1, atol=2e-6)
    meta = results.build_trial_policy_metadata(
        history, models.resolve_trial_policy_candidates(), TaskParams(), parameter_source="test")
    assert meta["analysis_version"] == results.ANALYSIS_VERSION
    assert meta["candidate_parameters"]["win_switch_lose_stay"] == {}
    path = results.save_trial_policy_results(smoothed, processed_data_path=tmp_path, metadata=meta)
    restored, saved = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(restored.hmm_smoothed_probs, smoothed.hmm_smoothed_probs)
    assert saved["candidate_choice_parameters"]["win_switch_lose_stay"]["rule"] == "fixed_error"
