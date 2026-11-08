"""Contracts for source-local, causal replay of the approved trial policies."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import trial_features
from src.behavior_analysis.gather_trial_features import TaskParams


MODEL_IDS = ("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch")


@pytest.fixture
def trial_table():
    """Return a fresh DataFrame (T=5 rows) of dimensionless binary observations."""
    return pd.DataFrame(
        {"action": [0, 1, 1, 0, 1], "reward": [1, 0, 1, 0, 1],
         "cur_trial": [10, 12, 15, 16, 20], "cur_block": [1, 1, 2, 2, 2]},
        index=[8, 3, 7, 9, 4],
    )


def _replay(table, params=None, model_ids=MODEL_IDS, session_id="mouse_session"):
    """Map DataFrame[T,C], TaskParams and ordered IDs to history and float/bool[T,K]."""
    from src.behavior_analysis import trial_policy_models as models

    history = models.prepare_trial_policy_history(table, session_id=session_id)
    candidates = models.resolve_trial_policy_candidates(model_ids)
    scores, valid = models.replay_trial_policy_candidates(
        history, candidates, TaskParams() if params is None else params
    )
    return history, scores, valid


def test_hand_calculated_prechoice_values_use_actual_mouse_history(trial_table):
    """No inputs; assert dimensionless [trial,model] scores before each outcome."""
    history, scores, valid = _replay(trial_table)
    # Beliefs are [right,left]: .5/.5 -> .8/.2 -> 27/35,8/35 -> .2/.8.
    bayesian = np.tanh(np.log([1, 1 / 4, 8 / 27, 4, 27 / 8]))
    expected = np.column_stack(
        ([0, -.3, -.21, .153, .1071], bayesian, [0, -1, -1, 1, 1])
    )
    np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=0)
    assert valid.dtype == bool and valid.all()
    assert history.comparison_valid.all() and history.history_update_valid.all()
    np.testing.assert_array_equal(history.row_position, np.arange(5))
    np.testing.assert_array_equal(history.trial_id, ["10", "12", "15", "16", "20"])


def test_context_blocks_stale_feature_columns_and_dataframe_index_do_not_drive_replay(trial_table):
    """No inputs; changing non-history columns leaves [T,K] scores unchanged."""
    original = trial_table.copy(deep=True)
    _, expected, _ = _replay(trial_table)
    changed = trial_table.assign(cur_block=99, context=[1, 0, 1, 0, 1],
                                 FQlearning_rel_value=99, HMM_rel_value_logodds=-99,
                                 wsls_regressor=42).reset_index(drop=True)
    _, actual, _ = _replay(changed)
    np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(trial_table, original)


@pytest.mark.parametrize("changed_position", [0, 2, 4])
def test_current_and_future_changes_cannot_change_prechoice_scores(trial_table, changed_position):
    """DataFrame[T,C] and integer row input; assert causal prefix equality [t+1,K]."""
    _, baseline, _ = _replay(trial_table)
    changed = trial_table.copy()
    changed.iloc[changed_position:, changed.columns.get_loc("action")] = 1 - changed.iloc[changed_position:]["action"]
    changed.iloc[changed_position:, changed.columns.get_loc("reward")] = 1 - changed.iloc[changed_position:]["reward"]
    _, actual, _ = _replay(changed)
    np.testing.assert_array_equal(actual[:changed_position + 1], baseline[:changed_position + 1])


def test_cropped_and_separate_sessions_initialize_at_first_retained_row(trial_table):
    """No inputs; fresh sessions/crops retain original IDs and start at zero scores."""
    full, _, _ = _replay(trial_table)
    cropped, scores, _ = _replay(trial_table.iloc[2:], session_id="second")
    np.testing.assert_array_equal(scores[0], [0, 0, 0])
    np.testing.assert_array_equal(cropped.trial_id, ["15", "16", "20"])
    np.testing.assert_array_equal(cropped.row_position, [0, 1, 2])
    assert cropped.session_id != full.session_id
    assert cropped.metadata["initialization"] == "first_retained_row"


@pytest.mark.parametrize("action", [.5, 1.2, -1, 2, np.inf, "left", "right", "true", "garbage"])
def test_actions_are_exact_binary_not_truncated_or_textual(trial_table, action):
    """Scalar action input must raise ValueError with session/trial context."""
    table = trial_table.astype({"action": object})
    table.iloc[2, table.columns.get_loc("action")] = action
    with pytest.raises(ValueError, match="mouse_session.*15.*action"):
        _replay(table)


@pytest.mark.parametrize("reward", [.5, 2, -1, np.inf, "true", "water"])
def test_supplied_rewards_are_finite_binary_indicators(trial_table, reward):
    """Scalar reward input must raise rather than silently binarize observations."""
    table = trial_table.astype({"reward": object})
    table.iloc[2, table.columns.get_loc("reward")] = reward
    with pytest.raises(ValueError, match="mouse_session.*15.*reward"):
        _replay(table)


def test_numeric_csv_and_pandas_representations_normalize_source_locally(trial_table):
    """No inputs; numeric strings/bools/missing aliases yield float64[T] history."""
    table = trial_table.assign(action=["0.0", "1", 1., "0", True],
                               reward=["1.0", "0", True, 0., "1"],
                               give_reward=["0", "0.0", False, "None", np.nan],
                               experimenter_reward_given=[0., False, "0", pd.NA, ""])
    history, scores, valid = _replay(table)
    assert history.actions.dtype == np.float64
    assert history.rewards.dtype == np.float64
    np.testing.assert_array_equal(history.comparison_valid, [1, 1, 1, 0, 0])
    assert np.isnan(scores[3:]).all() and not valid[3:].any()
    assert history.metadata["source_flag_columns"] == ["give_reward", "experimenter_reward_given"]


@pytest.mark.parametrize("flag", [None, np.nan, pd.NA, "None", "", "nan", "NaN", "true", "false", "yes", .5, 2])
def test_unknown_flag_excludes_choice_and_update(trial_table, flag):
    """Scalar flag input freezes all policy histories and marks unknown flags."""
    table = trial_table.assign(experimenter_reward_given=[0, flag, 0, 0, 0])
    history, scores, valid = _replay(table)
    _, compressed, _ = _replay(trial_table.iloc[[0, 2, 3, 4]])
    assert not history.comparison_valid[1] and not history.history_update_valid[1]
    assert "unknown_manual_flag" in history.comparison_exclusion_reason[1]
    assert np.isnan(scores[1]).all() and not valid[1].any()
    np.testing.assert_array_equal(scores[[0, 2, 3, 4]], compressed)


@pytest.mark.parametrize("action", [None, np.nan, pd.NA, "None", "no_choice", "", "nan", "NaN"])
def test_nochoice_rows_are_unavailable_and_freeze_all_histories(trial_table, action):
    """Missing action scalar yields NaN scores, unlike the valid initial zero row."""
    table = trial_table.astype({"action": object})
    table.iloc[1, table.columns.get_loc("action")] = action
    history, scores, valid = _replay(table)
    _, compressed, _ = _replay(trial_table.iloc[[0, 2, 3, 4]])
    assert np.isnan(history.actions[1])
    assert "no_choice" in history.comparison_exclusion_reason[1]
    assert not history.history_update_valid[1] and not valid[1].any()
    assert valid[0].all() and (scores[0] == 0).all()
    np.testing.assert_array_equal(scores[[0, 2, 3, 4]], compressed)


def test_manual_rows_freeze_histories_without_advancing_forgetting_or_hazard(trial_table):
    """No inputs; known manual rows have no prediction and no outcome update."""
    history, scores, valid = _replay(trial_table.assign(give_reward=[0, 1, 0, 0, 0]))
    _, compressed, _ = _replay(trial_table.iloc[[0, 2, 3, 4]])
    assert "manual_reward" in history.comparison_exclusion_reason[1]
    assert history.adapter_skip_flags[1] and not history.history_update_valid[1]
    assert not valid[1].any() and np.isnan(scores[1]).all()
    np.testing.assert_array_equal(scores[[0, 2, 3, 4]], compressed)


@pytest.mark.parametrize("reward", [None, np.nan, pd.NA, "None", "", "NaN"])
def test_missing_reward_preserves_current_score_and_freezes_entire_update(trial_table, reward):
    """Missing reward scalar permits current scoring but skips forgetting/hazard/WSLS."""
    table = trial_table.astype({"reward": object})
    table.iloc[1, table.columns.get_loc("reward")] = reward
    history, scores, valid = _replay(table)
    _, compressed, _ = _replay(trial_table.iloc[[0, 2, 3, 4]])
    assert history.comparison_valid[1] and valid[1].all()
    assert not history.history_update_valid[1] and not history.adapter_skip_flags[1]
    assert "missing_reward" in history.history_update_exclusion_reason[1]
    assert history.comparison_exclusion_reason[1] == ""
    np.testing.assert_array_equal(scores[1], scores[2])
    np.testing.assert_array_equal(scores[[0, 2, 3, 4]], compressed)


def test_absent_flag_assumption_and_schema_identity_are_explicit(trial_table):
    """No inputs; equivalent histories with different source schemas have distinct digests."""
    tables = [trial_table, trial_table.assign(give_reward=0),
              trial_table.assign(experimenter_reward_given=0),
              trial_table.assign(give_reward=0, experimenter_reward_given=0)]
    histories = [_replay(table)[0] for table in tables]
    assert histories[0].metadata["absent_flag_assumed_animal_controlled"] is True
    assert len({history.metadata["input_digest"] for history in histories}) == 4
    reordered = _replay(tables[-1][list(reversed(tables[-1].columns))])[0]
    assert reordered.metadata["input_digest"] == histories[-1].metadata["input_digest"]
    changed_reward = _replay(trial_table.assign(reward=[0, 0, 1, 0, 1]))[0]
    assert changed_reward.metadata["input_digest"] != histories[0].metadata["input_digest"]


def test_conflicting_flag_aliases_raise_after_normalization(trial_table):
    """No inputs; disagreeing source aliases cannot silently select one flag column."""
    with pytest.raises(ValueError, match="conflict"):
        _replay(trial_table.assign(give_reward=0, experimenter_reward_given=[0, 1, 0, 0, 0]))


@pytest.mark.parametrize("column", ["cur_trial", "cur_block", "action", "reward"])
def test_nonempty_history_requires_identifiers_and_observations(trial_table, column):
    """Column name input identifies a required nonempty-table field."""
    with pytest.raises(ValueError, match=column):
        _replay(trial_table.drop(columns=column))


@pytest.mark.parametrize("column", ["session_ID", "source_session_id"])
def test_session_membership_is_validated_and_source_block_ids_win(trial_table, column):
    """Session-column name input validates full IDs, preserving source block display IDs."""
    with pytest.raises(ValueError, match="session"):
        _replay(trial_table.assign(**{column: ["mouse_session"] * 4 + ["other"]}))
    history, _, _ = _replay(trial_table.assign(source_cur_block=[4, 4, 7, 7, 7]))
    np.testing.assert_array_equal(history.block_id, ["4", "4", "7", "7", "7"])


def test_empty_history_and_one_trial_shapes():
    """No inputs; empty and singleton DataFrames return aligned [T,K] arrays."""
    history, scores, valid = _replay(pd.DataFrame())
    assert scores.shape == valid.shape == (0, 3)
    assert history.trial_id.shape == history.block_id.shape == (0,)
    _, scores, valid = _replay(pd.DataFrame({"action": [1], "reward": [0], "cur_trial": [4], "cur_block": [9]}))
    np.testing.assert_array_equal(scores, [[0, 0, 0]])
    assert valid.all()


@pytest.mark.parametrize("ids", [(), (MODEL_IDS[0], MODEL_IDS[0]), ("observer",), ("HMM_rel_value_logodds_decay",)])
def test_candidate_selection_rejects_empty_duplicate_unknown_and_future_ids(ids):
    """Tuple[str] candidate input must be nonempty, unique and currently implemented."""
    from src.behavior_analysis import trial_policy_models as models

    with pytest.raises(ValueError):
        models.resolve_trial_policy_candidates(ids)


def test_candidate_permutation_and_one_candidate_reuse_existing_functions(trial_table, monkeypatch):
    """No inputs; ordered adapter selection avoids executing unselected policies."""
    from src.behavior_analysis import trial_policy_models as models

    _, expected, _ = _replay(trial_table)
    _, permuted, _ = _replay(trial_table, model_ids=MODEL_IDS[::-1])
    np.testing.assert_array_equal(permuted, expected[:, ::-1])
    monkeypatch.setattr(trial_features, "hmm_relative_value", _unexpected_policy_call)
    _, single, _ = _replay(trial_table, model_ids=(MODEL_IDS[0],))
    np.testing.assert_array_equal(single, expected[:, :1])
    definition = models.resolve_trial_policy_candidates((MODEL_IDS[0],))[0]
    assert definition.indifference_rule == "exact_zero"
    assert callable(definition.adapter) and definition.model_id == MODEL_IDS[0]


def _unexpected_policy_call(*args, **kwargs):
    """Arbitrary policy-call arguments are forbidden; raises AssertionError, no result."""
    raise AssertionError("unselected Bayesian adapter executed")


@pytest.mark.parametrize("params", [TaskParams(FQL_decay=np.nan), TaskParams(FQL_decay=-.1),
                                        TaskParams(n_actions=3), TaskParams(state_transition_prob=np.nan),
                                        TaskParams(active_reward_probability=1.1),
                                        TaskParams(inactive_reward_probability=-.1),
                                        TaskParams(tanh_scale=0), TaskParams(tanh_scale=np.inf),
                                        TaskParams(correct_reward_size=np.nan)])
def test_invalid_effective_policy_parameters_raise(trial_table, params):
    """TaskParams input must describe finite supported two-action policy dynamics."""
    with pytest.raises(ValueError):
        _replay(trial_table, params=params)


def test_irrelevant_parameters_do_not_change_selected_policy(trial_table):
    """No inputs; fast-learn and Bayesian-only fields do not control FQL replay."""
    _, baseline, _ = _replay(trial_table, model_ids=(MODEL_IDS[0],))
    _, actual, _ = _replay(trial_table, params=TaskParams(FQL_reward_update_rate_fast_learn=.99,
                                                       active_reward_probability=np.nan),
                            model_ids=(MODEL_IDS[0],))
    np.testing.assert_array_equal(actual, baseline)


def test_effective_parameters_record_only_selected_policy_inputs():
    """No inputs; explicit TaskParams map to plain per-model effective parameter dictionaries."""
    from src.behavior_analysis import trial_policy_models as models

    candidates = models.resolve_trial_policy_candidates(MODEL_IDS)
    effective = models.effective_trial_policy_parameters(candidates, TaskParams(FQL_decay=.6))
    assert tuple(effective) == MODEL_IDS
    assert effective[MODEL_IDS[0]] == {"decay": .6, "reward_update_rate": .4, "n_actions": 2}
    assert effective[MODEL_IDS[1]] == {
        "state_transition_prob": .2, "active_reward_probability": .8,
        "inactive_reward_probability": 0., "correct_reward_size": 1.,
        "incorrect_reward_size": 0., "tanh_scale": 1., "value_mode": "bayesian_log_odds",
    }
    assert effective[MODEL_IDS[2]] == {}


def test_tiny_fql_scores_and_positive_adapter_scaling_preserve_directions():
    """No inputs; positive scores below 1e-20 remain available, nonzero and left-positive."""
    from src.behavior_analysis import trial_policy_models as models

    table = pd.DataFrame({"action": [1] * 90, "reward": [1] + [0] * 89,
                          "cur_trial": np.arange(90), "cur_block": 1})
    history, scores, valid = _replay(table, params=TaskParams(FQL_decay=.5), model_ids=(MODEL_IDS[0],))
    assert 0 < scores[-1, 0] < 1e-20 and valid.all()
    definition = models.resolve_trial_policy_candidates((MODEL_IDS[0],))[0]
    scaled = replace(definition, model_id="scaled_fql", adapter=_positively_scaled_fql)
    scaled_scores, scaled_valid = models.replay_trial_policy_candidates(history, (scaled,), TaskParams(FQL_decay=.5))
    np.testing.assert_array_equal(np.sign(scaled_scores), np.sign(scores))
    np.testing.assert_array_equal(scaled_valid, valid)


def _positively_scaled_fql(history, params):
    """Prepared history and TaskParams return dimensionless float/bool[T] scaled FQL."""
    from src.behavior_analysis import trial_policy_models as models

    definition = models.resolve_trial_policy_candidates((MODEL_IDS[0],))[0]
    scores, valid = definition.adapter(history, params)
    return scores * 1e-9, valid


def test_impossible_bayesian_evidence_identifies_session_trial_and_model(trial_table):
    """No inputs; an impossible supplied reward raises contextual calculation failure."""
    with pytest.raises(ValueError, match="mouse_session.*10.*bayesian_hmm"):
        _replay(trial_table, params=TaskParams(active_reward_probability=0, inactive_reward_probability=0))
