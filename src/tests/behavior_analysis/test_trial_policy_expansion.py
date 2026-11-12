"""Causal replay and provenance contracts for the six-candidate comparison."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from src.behavior_analysis import trial_features
from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import TaskParams


ORIGINAL_IDS = ("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch")
NEW_IDS = ("q_learning", "reward_decay_hmm", "ideal_observer")
ALL_IDS = ORIGINAL_IDS + NEW_IDS


@pytest.fixture
def history_table():
    """Return chronological binary observations (T=7), with nontrivial row IDs."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 18, 20, 21, 25],
                         "cur_block": [1, 1, 1, 2, 2, 2, 3],
                         "action": [0, 1, 0, 0, 1, 1, 0],
                         "reward": [1, 0, 0, 0, 0, 1, 0]}, index=[7, 4, 8, 2, 9, 3, 0])


def replay(table, ids=ALL_IDS, params=None):
    """Table[T,C], IDs[K], optional TaskParams yield history and float/bool[T,K]."""
    history = models.prepare_trial_policy_history(table, session_id="test_session")
    scores, valid = models.replay_trial_policy_candidates(
        history, models.resolve_trial_policy_candidates(ids), TaskParams() if params is None else params)
    return history, scores, valid


def test_default_order_preserves_six_candidates_before_persistence():
    """No inputs; verify stable default identities and selectable subsets."""
    assert models.DEFAULT_MODEL_IDS[:7] == ALL_IDS + ("immediate_persistence",)
    assert tuple(candidate.model_id for candidate in models.resolve_trial_policy_candidates()) == models.DEFAULT_MODEL_IDS
    assert tuple(candidate.model_id for candidate in models.resolve_trial_policy_candidates(NEW_IDS[::-1])) == NEW_IDS[::-1]


def test_q_learning_hand_values_and_taskparams_rate(history_table):
    """Table[T,C] yields pre-update Q-left minus Q-right, in reward-indicator units."""
    _, scores, _ = replay(history_table, ("q_learning",))
    np.testing.assert_allclose(scores[:, 0], [0, -.3, -.3, -.21, -.147, -.147, .153], atol=1e-15)
    _, slower, _ = replay(history_table, ("q_learning",), TaskParams(QL_learning_rate=.1))
    assert slower[1, 0] == pytest.approx(-.1)
    differing = history_table.iloc[:5].assign(action=[0, 1, 1, 1, 0], reward=[1, 1, 0, 0, 0])
    _, pair, _ = replay(differing, ("q_learning", "forgetting_q_learning"))
    assert pair[2, 0] == 0 and pair[2, 1] > 0


def test_decay_hmm_reward_omission_and_hazard_hand_values(history_table):
    """Table[T,C] verifies reward Bayes, omission log-odds decay, then hazard."""
    _, scores, _ = replay(history_table.iloc[:3], ("reward_decay_hmm",))
    first_belief = .2
    after_omission = .2 + .6 * expit(.9 * logit(first_belief))
    expected = [0, np.tanh(logit(first_belief)), np.tanh(logit(after_omission))]
    np.testing.assert_allclose(scores[:, 0], expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize("kwargs,actions,rewards,neutral_from", [
    ({}, [0, 1] * 8, [0] * 16, 0),
    ({"lambda_decay": 1.}, [1, 1, 0, 1], [1, 0, 0, 0], 2),
    ({"state_transition_prob": .5}, [0, 1, 1, 0], [1, 0, 1, 0], 0),
])
def test_decay_feature_symmetric_belief_is_exactly_neutral(kwargs, actions, rewards, neutral_from):
    """Parameter dict and observations[T] preserve exact zero for symmetric beliefs."""
    values = trial_features.hmm_relative_value_reward_decay(actions, rewards, **kwargs)
    np.testing.assert_array_equal(values[neutral_from:], np.zeros(len(values) - neutral_from))


def test_observer_hand_counts_resets_and_policy_reversal(history_table):
    """Table[T,C] yields raw decay score minus pre-choice left-minus-right doubt."""
    _, scores, _ = replay(history_table, ("reward_decay_hmm", "ideal_observer"))
    right = np.array([0, 0, 0, 1, 2, 2, 0])
    left = np.array([0, 0, 1, 1, 1, 2, 0])
    doubt = (1 - np.exp(-.5 * left)) - (1 - np.exp(-.5 * right))
    np.testing.assert_allclose(scores[:, 1], scores[:, 0] - doubt, atol=1e-15)
    assert scores[4, 0] < 0 < scores[4, 1]
    assert scores[6, 1] == scores[6, 0]
    composed = trial_features.hmm_relative_value_reward_decay(
        history_table.action, history_table.reward, lambda_decay=.1)
    expected = composed - trial_features.relative_doubt_index(right, left, lam=.5)
    np.testing.assert_allclose(scores[:, 1], expected, atol=1e-15)


@pytest.mark.parametrize("model_id", NEW_IDS)
def test_new_candidates_are_causal_crop_local_and_ignore_stale_context(history_table, model_id):
    """Table[T,C] and ID yield causal scores independent of stale/task columns."""
    original = history_table.copy(deep=True)
    _, expected, _ = replay(history_table, (model_id,))
    changed = history_table.assign(context=99, cur_block=999, Qlearning_rel_value=99,
                                  HMM_rel_value_logodds_decay=99, observer_value=99,
                                  relative_doubt_index=99, p_switch=.99, p_active_rew=.01)
    changed.iloc[3:, changed.columns.get_loc("action")] = 1 - changed.iloc[3:].action
    changed.iloc[3:, changed.columns.get_loc("reward")] = 1 - changed.iloc[3:].reward
    _, actual, _ = replay(changed, (model_id,))
    np.testing.assert_array_equal(actual[:4], expected[:4])
    _, cropped, _ = replay(history_table.iloc[3:], (model_id,))
    assert cropped[0, 0] == 0
    pd.testing.assert_frame_equal(history_table, original)


@pytest.mark.parametrize("model_id", NEW_IDS)
@pytest.mark.parametrize("excluded", ["manual", "no_choice", "unknown", "missing_reward"])
def test_new_candidates_freeze_all_state_on_excluded_updates(history_table, model_id, excluded):
    """Table/ID/exclusion inputs keep replay[T,1] equivalent to removing the update."""
    changed = history_table.astype({"action": object, "reward": object}).assign(experimenter_reward_given=0.)
    if excluded == "manual":
        changed.iloc[1, changed.columns.get_loc("experimenter_reward_given")] = 1
    elif excluded == "unknown":
        changed.iloc[1, changed.columns.get_loc("experimenter_reward_given")] = np.nan
    elif excluded == "no_choice":
        changed.iloc[1, changed.columns.get_loc("action")] = "no_choice"
    else:
        changed.iloc[1, changed.columns.get_loc("reward")] = np.nan
    _, actual, valid = replay(changed, (model_id,))
    _, compressed, _ = replay(history_table.iloc[[0, 2, 3, 4, 5, 6]], (model_id,))
    np.testing.assert_array_equal(actual[[0, 2, 3, 4, 5, 6]], compressed)
    assert valid[1, 0] == (excluded == "missing_reward")
    if excluded == "missing_reward":
        assert actual[1, 0] == actual[2, 0]


def test_decay_and_observer_mirror_symmetry(history_table):
    """Table[T,C] with reversed actions negates dimensionless pre-choice values."""
    _, original, _ = replay(history_table, NEW_IDS)
    _, mirrored, _ = replay(history_table.assign(action=1 - history_table.action), NEW_IDS)
    np.testing.assert_allclose(mirrored, -original, atol=3e-15, rtol=0)


@pytest.mark.parametrize("model_id", ["reward_decay_hmm", "ideal_observer"])
def test_impossible_decay_reward_raises_with_original_trial_and_model(history_table, model_id):
    """Table[T,C]/ID with zero reward support raises with session/trial/model context."""
    with pytest.raises(ValueError, match=rf"test_session.*10.*{model_id}.*normalization.*row 0"):
        replay(history_table, (model_id,), TaskParams(active_reward_probability=0))


def test_decay_validation_opt_in_and_valid_non_neutral_compatibility():
    """No inputs; strict reward checks preserve finite supported legacy arithmetic."""
    legacy = trial_features.hmm_relative_value_reward_decay([0, 1, 1], [1, 0, 1])
    checked = trial_features.hmm_relative_value_reward_decay(
        [0, 1, 1], [1, 0, 1], validate_outcome_evidence=True)
    np.testing.assert_array_equal(checked, legacy)
    np.testing.assert_allclose(legacy, [0, -.8823529411764706, -.5540345040855856], atol=1e-12)
    trial_features.hmm_relative_value_reward_decay([0], [1], active_reward_probability=0)
    with pytest.raises(ValueError, match="normalization.*row 0"):
        trial_features.hmm_relative_value_reward_decay(
            [0], [1], active_reward_probability=0, validate_outcome_evidence=True)


@pytest.mark.parametrize("model_id,field,value", [
    ("q_learning", "QL_learning_rate", -.1), ("q_learning", "QL_learning_rate", np.nan),
    ("q_learning", "n_actions", 3), ("reward_decay_hmm", "hmm_reward_decay_lambda", 1.1),
    ("reward_decay_hmm", "hmm_reward_decay_lambda", np.inf),
    ("ideal_observer", "omission_lam", 0), ("ideal_observer", "omission_lam", np.nan),
])
def test_new_parameter_validation_is_selected_model_local(history_table, model_id, field, value):
    """Table/ID/parameter scalar rejects invalid selected dynamics only."""
    params = replace(TaskParams(), **{field: value})
    with pytest.raises(ValueError, match=field):
        replay(history_table, (model_id,), params)
    replay(history_table, ("win_stay_lose_switch",), params)


def test_six_state_probabilities_permutation_and_npz_provenance(history_table, tmp_path):
    """Table/Path fixtures preserve [T,6] probabilities, order and complete provenance."""
    history, scores, valid = replay(history_table)
    independent = analysis.compute_trial_policy_comparison(history, scores, valid, ALL_IDS)
    smoothed = analysis.smooth_trial_policy_comparison(independent)
    np.testing.assert_allclose(smoothed.hmm_smoothed_probs.sum(axis=1), 1, atol=2e-6)
    assert smoothed.transition_matrix.shape == (6, 6)
    np.testing.assert_allclose(smoothed.transition_matrix[0], [.9, .02, .02, .02, .02, .02])
    permuted = analysis.smooth_trial_policy_comparison(analysis.compute_trial_policy_comparison(
        history, scores[:, ::-1], valid[:, ::-1], ALL_IDS[::-1]))
    np.testing.assert_allclose(permuted.hmm_smoothed_probs[:, ::-1], smoothed.hmm_smoothed_probs, atol=2e-6)
    candidates = models.resolve_trial_policy_candidates(ALL_IDS)
    meta = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test")
    assert meta["analysis_version"] == results.ANALYSIS_VERSION
    assert meta["candidate_parameters"]["q_learning"] == {"learning_rate": .3, "n_actions": 2}
    assert meta["candidate_parameters"]["reward_decay_hmm"]["lambda_decay"] == .1
    assert meta["candidate_parameters"]["ideal_observer"]["omission_lam"] == .5
    assert set(meta["candidate_indifference_rules"]) == set(ALL_IDS)
    assert "belief" in meta["candidate_indifference_rules"]["reward_decay_hmm"]
    path = results.save_trial_policy_results(smoothed, processed_data_path=tmp_path, metadata=meta)
    restored, restored_meta = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(restored.p_right, smoothed.p_right)
    assert restored_meta["candidate_parameters"] == meta["candidate_parameters"]
    assert tuple(restored.model_ids) == ALL_IDS
    for field in ("QL_learning_rate", "hmm_reward_decay_lambda", "omission_lam"):
        changed = results.build_trial_policy_metadata(history, candidates, replace(TaskParams(), **{field: .4}),
                                                       parameter_source="test")
        assert changed["signature"] != meta["signature"]


def test_counterfactual_source_is_hashed(history_table, monkeypatch):
    """Table/monkeypatch fixtures bind cache identity to the shared doubt source."""
    from pathlib import Path

    original = Path.read_bytes
    before = results._analysis_source_digest()

    def changed_counter_source(path):
        """Path input returns source bytes with a synthetic change only for doubt."""
        content = original(path)
        return content + b"\n# changed doubt\n" if path.name == "counterfactual_doubt.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed_counter_source)
    assert results._analysis_source_digest() != before


def test_original_three_v1_artifact_still_loads(history_table, tmp_path):
    """Table/Path fixtures load a legacy-shaped archive without new metadata keys."""
    from dataclasses import fields

    history, scores, valid = replay(history_table, ORIGINAL_IDS)
    comparison = analysis.compute_trial_policy_comparison(history, scores, valid, ORIGINAL_IDS)
    legacy_meta = {"analysis_version": "trial_policy_hmm_v1", "status": "ok", "score_indifference_rule": "exact_zero"}
    arrays = {field.name: getattr(comparison, field.name) for field in fields(comparison)
              if getattr(comparison, field.name) is not None and field.name != "status"}
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, **arrays, meta=legacy_meta)
    loaded, metadata = results.load_trial_policy_results(path)
    assert tuple(loaded.model_ids) == ORIGINAL_IDS and loaded.p_right.shape == (7, 3)
    assert metadata == legacy_meta
