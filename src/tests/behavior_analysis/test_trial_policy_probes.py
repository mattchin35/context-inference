"""Native probe-policy probabilities, histories, and trial-HMM provenance."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import (
    TaskParams, make_candidate_expectant_switching_config,
)
from src.behavior_modeling import expectant_switching as expectant


PROBE_IDS = ("probe_persistence_doubt", "home_referenced_expectancy_doubt")
PREVIOUS_IDS = ("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch",
                "q_learning", "reward_decay_hmm", "ideal_observer", "immediate_persistence")


@pytest.fixture
def trials():
    """Return binary choice/outcome history [T=9,C], with Left=1/Right=0."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 17, 20, 21, 25, 30, 31],
                         "cur_block": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                         "action": [1, 1, 1, 0, 1, 0, 0, 1, 0],
                         "reward": [1, 1, 1, 0, 0, 1, 0, 0, 1]})


def compare(table, ids=PROBE_IDS, params=None):
    """Table[T,C], IDs[K], optional TaskParams yield history and comparison[T,K]."""
    params = TaskParams() if params is None else params
    candidates = models.resolve_trial_policy_candidates(ids)
    history = models.prepare_trial_policy_history(table, session_id="probe_test")
    scores, available = models.replay_trial_policy_candidates(history, candidates, params)
    choices = models.effective_trial_policy_choice_parameters(candidates)
    comparison = analysis.compute_trial_policy_comparison(
        history, scores, available, ids, choice_parameters=choices)
    return history, comparison


def test_probe_candidates_append_to_defaults_and_retain_native_choice_rule():
    """No inputs; nine ordered defaults include native logistic drive scale 0.5."""
    assert models.DEFAULT_MODEL_IDS == PREVIOUS_IDS + PROBE_IDS
    candidates = models.resolve_trial_policy_candidates(PROBE_IDS[::-1])
    assert tuple(candidate.model_id for candidate in candidates) == PROBE_IDS[::-1]
    assert models.effective_trial_policy_choice_parameters(candidates) == {
        model_id: {"rule": "logistic", "temperature": .5} for model_id in PROBE_IDS}
    assert all(candidate.history_update_rule == "observed_choice_and_reward" for candidate in candidates)


@pytest.mark.parametrize("custom", [False, True])
def test_adapters_reuse_existing_replays_and_native_probabilities(trials, custom, monkeypatch):
    """Table/boolean fixtures verify shared drive[T] and native probabilities[T,2]."""
    params = TaskParams()
    if custom:
        params = replace(params, probe_doubt_persistence_weight=.7, probe_doubt_probe_weight=1.6,
                         probe_doubt_doubt_weight=.9, probe_doubt_lambda=.2,
                         home_expectancy_threshold=1.5, home_expectancy_scale=.8,
                         home_expectancy_weight=.8, home_doubt_weight=1.3, home_doubt_lambda=.3)
    config = make_candidate_expectant_switching_config(params)
    expected = expectant.replay_candidate_expectant_switching(
        trials.action.to_numpy(), trials.reward.to_numpy(), np.ones(len(trials), bool), config)
    spies = []
    for function_name in ("replay_probe_persistence_doubt", "replay_home_referenced_expectancy_doubt"):
        original = getattr(expectant, function_name)
        spy = Mock(wraps=original)
        spies.append(spy)
        monkeypatch.setattr(expectant, function_name, spy)
    _, actual = compare(trials, params=params)
    assert all(spy.call_count == 1 for spy in spies)
    for column, model_id in enumerate(PROBE_IDS):
        np.testing.assert_array_equal(actual.policy_scores[:, column], expected[f"{model_id}_drive"])
        np.testing.assert_allclose(actual.p_right[:, column], 1 - expected[f"{model_id}_prob_left"], atol=2e-16)


def test_probe_drives_after_reward_and_omission_are_hand_calculated(trials):
    """Table[T,C] yields reward probes followed by a return toward rewarded home."""
    _, actual = compare(trials)
    expected = [0, -1, -1, -1, -1 + 1.4 * (1 - np.exp(-.5))]
    np.testing.assert_allclose(actual.policy_scores[:5, 0], expected, atol=1e-15)
    assert actual.p_right[1, 0] == pytest.approx(expit(2))
    assert actual.policy_scores[4, 0] < 0  # Omitted Right probe still favors Right initially.
    assert actual.policy_scores[5, 0] == pytest.approx(1.)  # Opposite omissions cancel doubt.


def test_home_reward_count_survives_omissions_and_resets_on_opposite_reward(trials):
    """Table[T,C] keeps rewarded home across omissions and changes it after a reward."""
    _, actual = compare(trials)
    strengths = (1 + np.tanh((np.array([1, 2, 3]) - 2.5) / .5)) / 2
    np.testing.assert_allclose(actual.policy_scores[1:4, 1], 1 - 2 * strengths)
    third_reward_drive = 1 - 2 * strengths[-1]
    assert actual.policy_scores[4, 1] == pytest.approx(third_reward_drive + 1 - np.exp(-.5))
    assert actual.policy_scores[5, 1] == pytest.approx(third_reward_drive)
    assert actual.policy_scores[6, 1] == pytest.approx(-(1 - 2 * strengths[0]))


@pytest.mark.parametrize("excluded", ["manual", "no_choice", "unknown_flag", "invalid_switch", "missing_reward"])
def test_exclusions_and_missing_rewards_freeze_all_probe_state(trials, excluded):
    """Table/exclusion inputs preserve [T,2] predictions and freeze both histories."""
    table = trials.astype({"action": object, "reward": float}).assign(experimenter_reward_given=0.)
    if excluded == "manual":
        table.loc[3, "experimenter_reward_given"] = 1
    elif excluded == "unknown_flag":
        table.loc[3, "experimenter_reward_given"] = np.nan
    elif excluded == "missing_reward":
        table.loc[3, "reward"] = np.nan
    else:
        table.loc[3, "action"] = "no_choice"
        if excluded == "invalid_switch":
            table["trial_outcome"] = "choice"
            table.loc[3, "trial_outcome"] = "invalid_switch"
    original = table.copy(deep=True)
    history, actual = compare(table)
    _, compressed = compare(trials.drop(index=3))
    np.testing.assert_array_equal(actual.policy_scores[np.arange(len(trials)) != 3], compressed.policy_scores)
    assert not history.history_update_valid[3]
    if excluded == "missing_reward":
        assert actual.comparison_valid[3] and actual.prediction_valid[3].all()
        np.testing.assert_array_equal(actual.policy_scores[3], actual.policy_scores[4])
    else:
        assert not actual.comparison_valid[3] and np.isnan(actual.policy_scores[3]).all()
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("count", [0, 1, 5])
def test_session_and_crop_initialization_and_causal_prefix(trials, count):
    """Table/integer T yield [T,2] arrays with fresh neutral starts and no lookahead."""
    _, crop = compare(trials.iloc[2:2 + count])
    assert crop.p_right.shape == (count, 2)
    if count:
        np.testing.assert_array_equal(crop.p_right[0], [.5, .5])
    _, baseline = compare(trials)
    changed = trials.copy()
    changed.loc[3:, ["action", "reward"]] = 1 - changed.loc[3:, ["action", "reward"]]
    _, altered = compare(changed)
    np.testing.assert_array_equal(altered.policy_scores[:4], baseline.policy_scores[:4])


@pytest.mark.parametrize("model_id,field,value", [
    (PROBE_IDS[0], "probe_doubt_lambda", 0),
    (PROBE_IDS[0], "probe_doubt_probe_weight", -1),
    (PROBE_IDS[1], "home_expectancy_scale", 0),
    (PROBE_IDS[1], "home_doubt_weight", -1),
])
def test_only_selected_probe_parameters_are_validated(trials, model_id, field, value):
    """Selected ID/bad scalar rejects invalid parameters but unrelated subsets work."""
    params = replace(TaskParams(), **{field: value})
    with pytest.raises(ValueError):
        compare(trials, (model_id,), params)
    other = PROBE_IDS[1] if model_id == PROBE_IDS[0] else PROBE_IDS[0]
    _, valid = compare(trials, (other,), params)
    assert np.isfinite(valid.p_right).all()


def test_nine_state_roundtrip_and_probe_parameters_in_cache_identity(trials, tmp_path):
    """Table/Path fixtures save nine-state [T,9] arrays and effective probe settings."""
    history, comparison = compare(trials, models.DEFAULT_MODEL_IDS)
    smoothed = analysis.smooth_trial_policy_comparison(comparison)
    assert smoothed.hmm_smoothed_probs.shape == (len(trials), 9)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs.sum(axis=1), 1, atol=2e-6)
    candidates = models.resolve_trial_policy_candidates()
    meta = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test")
    assert meta["analysis_version"] == "trial_policy_hmm_v6"
    assert "../behavior_modeling/expectant_switching.py" in results.ANALYSIS_SOURCE_FILES
    assert meta["candidate_parameters"][PROBE_IDS[0]] == {
        "persistence_weight": 1., "probe_weight": 2., "doubt_weight": 1.4, "doubt_lambda": .5}
    assert meta["candidate_parameters"][PROBE_IDS[1]]["curve"] == {"threshold": 2.5, "scale": .5}
    altered = results.build_trial_policy_metadata(
        history, candidates, replace(TaskParams(), home_expectancy_threshold=3.), parameter_source="test")
    assert altered["signature"] != meta["signature"]
    old_candidates = models.resolve_trial_policy_candidates(PREVIOUS_IDS)
    old = results.build_trial_policy_metadata(history, old_candidates, TaskParams(), parameter_source="test")
    old_altered = results.build_trial_policy_metadata(
        history, old_candidates, replace(TaskParams(), home_expectancy_threshold=3.), parameter_source="test")
    assert old["signature"] == old_altered["signature"]
    path = results.save_trial_policy_results(smoothed, processed_data_path=tmp_path, metadata=meta)
    loaded, saved = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(loaded.hmm_smoothed_probs, smoothed.hmm_smoothed_probs)
    np.testing.assert_array_equal(loaded.policy_scores, smoothed.policy_scores)
    assert saved["candidate_choice_parameters"][PROBE_IDS[1]] == {"rule": "logistic", "temperature": .5}
    exported = results.attach_trial_policy_outputs(trials, loaded, metadata=saved)
    for column, model_id in enumerate(PROBE_IDS, start=7):
        np.testing.assert_array_equal(exported[f"trial_policy_p_right_{model_id}"], loaded.p_right[:, column])
