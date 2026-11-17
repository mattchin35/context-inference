"""Window-specific inverse WSLS values retain full recorded behavioral history."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_window_analysis as analysis
from src.behavior_analysis import choice_window_results as results
from src.behavior_analysis import trial_features
from src.behavior_analysis.session_analysis import DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS
from src.tests.behavior_analysis.test_trial_policy_win_switch import compare


AGENT = "win_switch_lose_stay_mouse_agreement"
VALUE = "win_switch_lose_stay_regressor"


@pytest.fixture
def trials():
    """Return six chronological trials, shape (6,4); 1=left, reward unitless."""
    return pd.DataFrame({"action": [1, 0, 0, 1, 1, 0], "reward": [1, 0, 1, 0, 1, 0],
                         "cur_trial": [10, 12, 15, 20, 21, 25], "cur_block": [1, 1, 1, 2, 2, 2]},
                        index=[5, 9, 15, 20, 30, 50])


def test_preparation_matches_canonical_rule_without_mutating_input(trials):
    """Table[T,C] gains one signed column on a copy, preserving index and metadata."""
    trials.attrs["units"] = "unitless choices"
    original = trials.copy(deep=True)
    prepared, provenance = analysis.prepare_choice_window_agent_values(trials)
    np.testing.assert_equal(prepared[VALUE].to_numpy(), [0, -1, -1, 1, 1, -1])
    _, reference = compare(trials.reset_index(drop=True))
    np.testing.assert_equal(prepared[VALUE].to_numpy(), reference.policy_scores[:, 0])
    pd.testing.assert_frame_equal(trials, original)
    pd.testing.assert_frame_equal(prepared.drop(columns=VALUE), original)
    assert prepared.attrs == trials.attrs
    assert provenance[VALUE]["source"] == "trial_features.win_stay_lose_switch_regressor"


@pytest.mark.parametrize("excluded", ["manual", "no_choice", "unknown_flag", "missing_reward"])
def test_fallback_reuses_history_and_exclusion_rules(trials, excluded):
    """Special row[T] freezes history; missing rewards keep a usable prediction."""
    table = trials.reset_index(drop=True).astype({"action": object, "reward": float}).assign(give_reward=0.)
    if excluded == "manual":
        table.loc[2, "give_reward"] = 1
    elif excluded == "unknown_flag":
        table.loc[2, "give_reward"] = np.nan
    elif excluded == "no_choice":
        table.loc[2, "action"] = "no_choice"
    else:
        table.loc[2, "reward"] = np.nan
    prepared, _ = analysis.prepare_choice_window_agent_values(table)
    _, reference = compare(table)
    np.testing.assert_equal(prepared[VALUE].to_numpy(dtype=float), reference.policy_scores[:, 0])
    assert "give_reward" in prepared and "experimenter_reward_given" not in prepared


def test_existing_wsls_values_take_priority_and_preserve_missing(trials, monkeypatch):
    """Saved unitless values[T] are negated directly, including zero and NaN."""
    trials["wsls_regressor"] = [0., 1., None, -1., "None", 0.]
    def fail_if_replayed(*args, **kwargs):
        """Arbitrary inputs; fail if already available full-history features are replayed."""
        pytest.fail("Saved WSLS values must be reused")
    monkeypatch.setattr(trial_features, "win_stay_lose_switch_regressor", fail_if_replayed)
    prepared, provenance = analysis.prepare_choice_window_agent_values(trials)
    np.testing.assert_equal(prepared[VALUE].to_numpy(), [0., -1., np.nan, 1., np.nan, 0.])
    assert provenance[VALUE]["source"] == "wsls_regressor"


def test_insufficient_inputs_remain_usable_without_new_agent():
    """Legacy action/value tables[T,C] remain valid when rewards/WSLS are absent."""
    table = pd.DataFrame({"action": [0, 1], "Qlearning_rel_value": [-1., 1.]})
    prepared, provenance = analysis.prepare_choice_window_agent_values(table)
    pd.testing.assert_frame_equal(prepared, table)
    assert provenance == {} and AGENT not in analysis.resolve_agent_value_columns(prepared)
    assert AGENT not in DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS


def test_window_scoring_preserves_initial_tie_and_cross_window_history(trials, tmp_path):
    """Two-choice windows[T=6] use signed scores, unchanged bounds and separate denominators."""
    trials["wsls_regressor"] = trial_features.win_stay_lose_switch_regressor(trials.action, trials.reward)
    table, path = results.save_choice_window_results(trials, tmp_path, "session", (2,))
    inverse = table.loc[table.agent.eq(AGENT)]
    wsls = table.loc[table.agent.eq("wsls_mouse_agreement")]
    assert inverse.n_agreements.tolist() == [1, 2, 2]
    assert wsls.n_agreements.tolist() == [0, 0, 0]
    assert inverse.n_comparisons.tolist() == [2, 2, 2]
    assert inverse.trial_start.tolist() == [0, 2, 4]
    assert inverse.trial_stop.tolist() == [2, 4, 6]
    _, metadata = results.load_choice_window_results(path)
    assert metadata["analysis_version"] == "1.1.0"
    assert metadata["input_shape"] == list(trials.shape)
    assert metadata["prepared_input_shape"] == [6, trials.shape[1] + 1]
    assert metadata["derived_agent_values"][VALUE]["source"] == "wsls_regressor"
    assert (path.parent / "trial_features.py").exists()
    assert VALUE not in trials


def test_missing_inverse_predictions_reduce_only_its_denominator(trials, tmp_path):
    """Missing WSLS values[T] stay missing rather than being imputed from rewards."""
    trials["wsls_regressor"] = [0, 1, np.nan, -1, -1, 1]
    table, _ = results.save_choice_window_results(trials, tmp_path, "session", (2,))
    assert table.loc[table.agent.eq(AGENT), "n_comparisons"].tolist() == [2, 1, 2]
