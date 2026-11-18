"""Weighted choice history and the approved graded perseveration readout."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.behavior_analysis import trial_features
from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import TaskParams, collect_trial_index_features


TEMPERATURE = 1 / np.log(9.)


@pytest.fixture
def trials():
    """Return chronological binary choices/rewards [T=6,C], Left=1 and Right=0."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 20, 21, 25], "cur_block": [1, 1, 1, 2, 2, 3],
                         "action": [1, 1, 0, 1, 0, 0], "reward": [1, 0, 1, 0, 0, 1]})


def replay(table, params=None):
    """Table[T,C]/optional TaskParams yield history and perseveration comparison[T,1]."""
    params = TaskParams() if params is None else params
    candidates = models.resolve_trial_policy_candidates(("perseveration",))
    history = models.prepare_trial_policy_history(table, session_id="perseveration_test")
    scores, available = models.replay_trial_policy_candidates(history, candidates, params)
    comparison = analysis.compute_trial_policy_comparison(
        history, scores, available, ("perseveration",),
        choice_parameters=models.effective_trial_policy_choice_parameters(candidates))
    return history, comparison


def test_replaces_default_persistence_and_records_approved_parameters():
    """No inputs; select eight defaults and retain the former immediate-choice model."""
    assert len(models.DEFAULT_MODEL_IDS) == 8
    assert models.DEFAULT_MODEL_IDS[5] == "perseveration"
    assert "immediate_persistence" not in models.DEFAULT_MODEL_IDS
    candidate, = models.resolve_trial_policy_candidates(("perseveration",))
    old, = models.resolve_trial_policy_candidates(("immediate_persistence",))
    assert old.choice_rule == "fixed_error"
    assert candidate.parameter_names == ("perseveration_decay",)
    assert candidate.history_update_rule == "observed_choice"
    assert models.effective_trial_policy_parameters((candidate,), TaskParams()) == {"perseveration": {"decay": .25}}
    assert models.effective_trial_policy_choice_parameters((candidate,)) == {
        "perseveration": {"rule": "logistic", "temperature": TEMPERATURE}}


def test_hand_values_and_shared_window_feature(trials, monkeypatch):
    """Table[T,C] gives the same weighted signed score[T] as window feature collection."""
    helper = Mock(wraps=trial_features.perseveration_regressor)
    monkeypatch.setattr(trial_features, "perseveration_regressor", helper)
    _, actual = replay(trials)
    assert helper.call_count == 1
    alpha = np.exp(-.25)
    expected = np.array([0., 1., 1., (-1 + alpha + alpha**2)/(1 + alpha + alpha**2),
                         (1 - alpha + alpha**2 + alpha**3)/(1 + alpha + alpha**2 + alpha**3),
                         (-1 + alpha - alpha**2 + alpha**3 + alpha**4)/(1 + alpha + alpha**2 + alpha**3 + alpha**4)])
    np.testing.assert_allclose(actual.policy_scores[:, 0], expected, atol=1e-15)
    assert actual.policy_scores[3, 0] > 0  # A one-choice excursion does not erase the earlier preference.
    features = trials.assign(right_omissions=0, left_omissions=0, right_cf_omissions=0,
                             left_cf_omissions=0, consecutive_omissions=0, relative_monotonic_cf_value=0)
    window_features = collect_trial_index_features(features)
    np.testing.assert_array_equal(actual.policy_scores[:, 0], np.asarray(window_features.perseveration_regressor, float))
    np.testing.assert_allclose(actual.p_right[:, 0], expit(-expected / TEMPERATURE))
    assert actual.p_right[0, 0] == .5 and actual.p_right[1, 0] == pytest.approx(.1)
    assert .1 < actual.p_right[3, 0] < .5


@pytest.mark.parametrize("excluded", ["manual", "unknown_flag", "no_choice", "invalid_switch"])
def test_excluded_attempts_freeze_choice_history(trials, excluded):
    """Table/condition inputs exclude one attempt without advancing weighted history."""
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
    original = table.copy(deep=True)
    _, actual = replay(table)
    _, compressed = replay(trials.drop(index=2))
    assert not actual.prediction_valid[2, 0] and np.isnan(actual.policy_scores[2, 0])
    np.testing.assert_array_equal(actual.policy_scores[[0, 1, 3, 4, 5]], compressed.policy_scores)
    pd.testing.assert_frame_equal(table, original)


def test_missing_rewards_still_update_choice_history_and_all_rewards_are_irrelevant(trials):
    """Table[T,C] preserves scores/probabilities even with every reward missing."""
    _, expected = replay(trials)
    history, missing = replay(trials.assign(reward=np.nan))
    assert history.comparison_valid.all() and not history.history_update_valid.any()
    np.testing.assert_array_equal(missing.policy_scores, expected.policy_scores)
    np.testing.assert_array_equal(missing.p_right, expected.p_right)
    assert "perseveration" in " ".join(results.trial_policy_warnings(history)).lower()


@pytest.mark.parametrize("count", [0, 1, 4])
def test_causal_crops_start_indifferent(trials, count):
    """Table/integer T give [T,1] scores initialized at the first retained row."""
    _, crop = replay(trials.iloc[2:2+count])
    assert crop.p_right.shape == (count, 1)
    if count:
        assert crop.p_right[0, 0] == .5
    _, baseline = replay(trials)
    later = trials.copy()
    later.loc[2:, "action"] = 1 - later.loc[2:, "action"]
    _, changed = replay(later)
    np.testing.assert_array_equal(changed.policy_scores[:3], baseline.policy_scores[:3])


@pytest.mark.parametrize("decay", [-.1, np.nan, np.inf, "invalid"])
def test_invalid_selected_decay_rejected_without_affecting_other_candidates(trials, decay):
    """Bad scalar decay fails selected replay; the unrelated immediate model remains usable."""
    params = replace(TaskParams(), perseveration_decay=decay)
    with pytest.raises(ValueError, match="perseveration_decay"):
        replay(trials, params)
    old = models.resolve_trial_policy_candidates(("immediate_persistence",))
    assert models.effective_trial_policy_parameters(old, params) == {"immediate_persistence": {}}


def test_zero_decay_and_temperature_override(trials):
    """Table[T,C] with zero decay averages all preceding choices, preserving exact neutrality."""
    _, uniform = replay(trials, replace(TaskParams(), perseveration_decay=0))
    np.testing.assert_allclose(uniform.policy_scores[:, 0], [0, 1, 1, 1/3, .5, .2])
    candidates = models.resolve_trial_policy_candidates(("perseveration",))
    assert models.effective_trial_policy_choice_parameters(candidates, temperatures={"perseveration": .7}) == {
        "perseveration": {"rule": "logistic", "temperature": .7}}
    old = models.resolve_trial_policy_candidates(("immediate_persistence",))
    assert set(models.effective_trial_policy_choice_parameters(old, temperatures={"perseveration": .7})) == {"immediate_persistence"}


def test_cache_identity_roundtrip_and_choice_only_provenance(trials, tmp_path):
    """Table/Path fixtures persist [T,1] arrays and changes in relevant decay/temperature."""
    history, comparison = replay(trials)
    candidates = models.resolve_trial_policy_candidates(("perseveration",))
    meta = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test")
    assert meta["analysis_version"] == "trial_policy_hmm_v8"
    assert meta["candidate_history_update_rules"] == {"perseveration": "observed_choice"}
    for kwargs in ({"feature_params": replace(TaskParams(), perseveration_decay=.5)},
                   {"choice_temperatures": {"perseveration": .7}}):
        options = {"feature_params": TaskParams(), **kwargs}
        altered = results.build_trial_policy_metadata(history, candidates, parameter_source="test", **options)
        assert altered["signature"] != meta["signature"]
    path = results.save_trial_policy_results(comparison, processed_data_path=tmp_path, metadata=meta)
    restored, saved = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(restored.policy_scores, comparison.policy_scores)
    assert saved["candidate_choice_parameters"] == meta["candidate_choice_parameters"]
