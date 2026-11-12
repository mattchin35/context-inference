"""Immediate-persistence reuse and its choice-only trial-HMM history contract."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import trial_features
from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import TaskParams


@pytest.fixture
def choices():
    """Return binary mouse history (T=6), with original nonconsecutive trial IDs."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 20, 21, 25], "cur_block": [1, 1, 2, 2, 2, 3],
                         "action": [1, 1, 0, 1, 0, 0], "reward": [1, 0, 1, 0, 0, 1]},
                        index=[8, 4, 5, 3, 9, 2])


def replay(table, ids=("immediate_persistence",)):
    """DataFrame[T,C] and IDs[K] yield prepared history and float/bool[T,K]."""
    history = models.prepare_trial_policy_history(table, session_id="persistence_test")
    scores, available = models.replay_trial_policy_candidates(
        history, models.resolve_trial_policy_candidates(ids), TaskParams())
    return history, scores, available


def test_seventh_default_is_parameter_free_and_left_positive(choices):
    """Table[T,C] yields actual previous-choice scores in {-1,0,+1}, unitless."""
    assert len(models.DEFAULT_MODEL_IDS) == 7
    assert models.DEFAULT_MODEL_IDS[-1] == "immediate_persistence"
    _, scores, available = replay(choices)
    np.testing.assert_array_equal(scores[:, 0], [0, 1, 1, -1, 1, -1])
    assert available.all()
    candidate, = models.resolve_trial_policy_candidates(("immediate_persistence",))
    assert candidate.parameter_names == ()
    assert candidate.history_update_rule == "observed_choice"
    assert models.effective_trial_policy_parameters((candidate,), TaskParams()) == {"immediate_persistence": {}}


def test_persistence_is_not_weighted_perseveration(choices):
    """Table[T,C] distinguishes repeat-last-choice from averaging older choices."""
    _, scores, _ = replay(choices)
    assert scores[3, 0] == -1
    assert trial_features.perseveration_regressor(choices.action)[3] > 0


def test_current_and_future_actions_or_any_rewards_do_not_change_past_scores(choices):
    """Table[T,C] preserves causal prefixes and is independent of all rewards."""
    _, expected, _ = replay(choices)
    _, reward_changed, _ = replay(choices.assign(reward=1 - choices.reward))
    np.testing.assert_array_equal(reward_changed, expected)
    altered = choices.copy()
    altered.iloc[2:, altered.columns.get_loc("action")] = 1 - altered.iloc[2:].action
    _, actual, _ = replay(altered)
    np.testing.assert_array_equal(actual[:3], expected[:3])
    assert actual[3, 0] == -expected[3, 0]


@pytest.mark.parametrize("excluded", ["manual", "no_choice", "invalid_switch", "unknown_flag"])
def test_excluded_attempts_do_not_replace_previous_choice(choices, excluded):
    """Table/exclusion inputs skip prediction and memory updates on invalid attempts."""
    table = choices.astype({"action": object}).assign(experimenter_reward_given=0.)
    if excluded == "manual":
        table.loc[5, "experimenter_reward_given"] = 1
    elif excluded == "unknown_flag":
        table.loc[5, "experimenter_reward_given"] = np.nan
    else:
        table.loc[5, "action"] = "no_choice"
        if excluded == "invalid_switch":
            table.loc[5, "reward"] = 0
            table["trial_outcome"] = "choice"
            table.loc[5, "trial_outcome"] = "invalid_switch"
    _, actual, available = replay(table)
    _, compressed, _ = replay(choices.iloc[[0, 1, 3, 4, 5]])
    assert not available[2, 0] and np.isnan(actual[2, 0])
    np.testing.assert_array_equal(actual[[0, 1, 3, 4, 5]], compressed)


def test_missing_reward_updates_choice_memory_but_freezes_outcome_models(choices):
    """Table[T,C] with a missing outcome still records the actual observed action."""
    table = choices.astype({"reward": float})
    table.iloc[2, table.columns.get_loc("reward")] = np.nan
    ids = ("immediate_persistence", "win_stay_lose_switch", "q_learning")
    history, actual, available = replay(table, ids)
    assert history.comparison_valid[2] and not history.history_update_valid[2]
    assert available[2].all()
    assert actual[2, 0] == 1 and actual[3, 0] == -1
    _, compressed, _ = replay(choices.iloc[[0, 1, 3, 4, 5]], ids)
    np.testing.assert_array_equal(actual[[0, 1, 3, 4, 5], 1:], compressed[:, 1:])


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_each_retained_session_or_crop_starts_indifferent(choices, rows):
    """Table/integer T yield aligned (T,1) arrays initialized at the retained start."""
    _, scores, available = replay(choices.iloc[2:2 + rows])
    assert scores.shape == available.shape == (rows, 1)
    if rows:
        assert scores[0, 0] == 0 and available.all()


def test_trial_and_benchmark_share_the_existing_previous_choice_computation(choices, monkeypatch):
    """Table/monkeypatch fixtures verify both adapters use the same shared helper."""
    from src.behavior_modeling.benchmark_models import ObservableHistory, build_observable_models

    helper = trial_features.immediate_persistence_regressor
    calls = []

    def record_helper(*args, **kwargs):
        """Observation arrays and helper options yield float[T], recording reuse."""
        calls.append(True)
        return helper(*args, **kwargs)

    monkeypatch.setattr(trial_features, "immediate_persistence_regressor", record_helper)
    _, trial_scores, _ = replay(choices)
    benchmark = next(candidate for candidate in build_observable_models()
                     if candidate.model_id == "immediate_persistence")
    valid = np.array([True, True, False, True, True, True])
    history = ObservableHistory(choices.action.to_numpy(), choices.reward.to_numpy(), valid)
    benchmark_scores = benchmark.replay(history).model_value
    assert len(calls) == 2
    expected = trial_features.immediate_persistence_regressor(choices.action, experimenter_reward_given=~valid)
    np.testing.assert_array_equal(benchmark_scores, expected)
    np.testing.assert_array_equal(trial_scores[:, 0], [0, 1, 1, -1, 1, -1])


def test_seven_state_roundtrip_and_missing_reward_provenance(choices, tmp_path):
    """Table/Path fixtures preserve [T,7] posteriors and explicit choice-only rules."""
    table = choices.astype({"reward": float})
    table.iloc[2, table.columns.get_loc("reward")] = np.nan
    history, scores, available = replay(table, models.DEFAULT_MODEL_IDS)
    comparison = analysis.compute_trial_policy_comparison(history, scores, available, models.DEFAULT_MODEL_IDS)
    assert comparison.p_right[:, -1].tolist() == [.5, .1, .1, .9, .1, .9]
    smoothed = analysis.smooth_trial_policy_comparison(comparison)
    assert smoothed.hmm_smoothed_probs.shape == (6, 7)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs.sum(axis=1), 1, atol=2e-6)
    np.testing.assert_allclose(smoothed.transition_matrix[0], [.9] + [.1 / 6] * 6)
    candidates = models.resolve_trial_policy_candidates()
    meta = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test")
    assert meta["candidate_history_update_rules"]["immediate_persistence"] == "observed_choice"
    assert meta["candidate_history_update_rules"]["win_stay_lose_switch"] == "observed_choice_and_reward"
    assert "choice/outcome" in meta["history_update_valid_definition"]
    assert "persistence" in " ".join(results.trial_policy_warnings(history)).lower()
    assert "skipped all candidate history updates" not in " ".join(results.trial_policy_warnings(history))
    assert meta["analysis_version"] == "trial_policy_hmm_v4"
    path = results.save_trial_policy_results(smoothed, processed_data_path=tmp_path, metadata=meta)
    restored, saved_meta = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(restored.p_right, smoothed.p_right)
    assert saved_meta["candidate_history_update_rules"] == meta["candidate_history_update_rules"]
    attached = results.attach_trial_policy_outputs(table, restored, metadata=saved_meta)
    assert not attached.iloc[2]["trial_policy_history_update_valid"]
    assert attached.iloc[3]["trial_policy_p_right_immediate_persistence"] == .9
    legacy = results.build_trial_policy_metadata(history, candidates[:-1], TaskParams(), parameter_source="test")
    assert legacy["signature"] != meta["signature"]
