"""Contracts for consecutive groups of valid mouse choices."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_window_analysis as windows
from src.behavior_analysis.session_analysis import add_block_agent_mouse_agreement_columns


@pytest.fixture
def trial_df():
    """Return a 47-choice session, shape (47, 5), with left-positive values."""
    return pd.DataFrame({
        "action": np.zeros(47, dtype=int),
        "reward": np.zeros(47),
        "cur_block": np.repeat(np.arange(7), 7)[:47],
        "agent_value": np.full(47, -1.0),
        "session_ID": ["mouse_session"] * 47,
    })


def test_windows_partition_choices_and_discard_tail(trial_df):
    """Ten/twenty-choice partitions start together and omit incomplete tails."""
    scores = windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})
    for size, count in [(10, 4), (20, 2)]:
        result = windows.summarize_choice_window_agreement(trial_df, scores, size, "mouse_session")
        assert result["choice_start"].tolist() == list(range(0, 40, size))
        assert result["choice_stop"].tolist() == list(range(size, 41, size))
        assert result["trial_start"].tolist() == result["choice_start"].tolist()
        assert result["trial_stop"].tolist() == result["choice_stop"].tolist()
        assert result["n_valid_choices"].tolist() == [size] * count
        assert result["agreement"].tolist() == [1.0] * count
        assert result["n_context_transitions"].gt(0).all()


@pytest.mark.parametrize("reward_column", ["give_reward", "experimenter_reward_given"])
def test_membership_excludes_non_choices_and_manual_rewards(trial_df, reward_column):
    """Membership is positional and skips both legacy and current exclusions."""
    trial_df["action"] = trial_df["action"].astype(object)
    trial_df.loc[[1, 3], "action"] = ["no_choice", "None"]
    trial_df[reward_column] = 0
    trial_df.loc[5, reward_column] = 1
    trial_df.index = [100] * len(trial_df)
    before = trial_df.copy(deep=True)
    scores = windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})
    result = windows.summarize_choice_window_agreement(trial_df, scores, 10, "mouse_session")
    assert result.iloc[0]["trial_stop"] == 13
    assert result.iloc[1]["trial_start"] == 13
    assert result["n_comparisons"].tolist() == [10] * 4
    pd.testing.assert_frame_equal(before, trial_df)


def test_missing_predictions_do_not_change_membership_or_reset_ties(trial_df):
    """Agents share windows but have separate denominators and continuous ties."""
    trial_df["action"] = 1
    trial_df["agent_value"] = 0.0
    trial_df.loc[0, "agent_value"] = 1.0
    trial_df.loc[9, "agent_value"] = np.nan
    trial_df["other_value"] = pd.Series([-1.0] * len(trial_df), dtype=object)
    trial_df.loc[10:19, "other_value"] = "None"
    scores = windows.compute_trial_agent_agreements(
        trial_df, {"agent": "agent_value", "other": "other_value"}
    )
    result = windows.summarize_choice_window_agreement(trial_df, scores, 10, "mouse_session")
    agent = result[result.agent.eq("agent")]
    other = result[result.agent.eq("other")]
    assert agent.n_comparisons.tolist() == [9, 10, 10, 10]
    assert agent.agreement.tolist() == [1.0] * 4
    assert other.iloc[1].n_comparisons == 0
    assert np.isnan(other.iloc[1].agreement)
    assert agent.trial_start.tolist() == other.trial_start.tolist()


@pytest.mark.parametrize("length", [0, 9, 10, 20])
def test_short_empty_and_exact_sessions(trial_df, length):
    """Empty results preserve the schema; complete windows are never discarded."""
    trial_df = trial_df.iloc[:length]
    scores = windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})
    result = windows.summarize_choice_window_agreement(trial_df, scores, 10, "mouse_session")
    assert len(result) == length // 10
    assert list(result.columns) == list(windows.RESULT_COLUMNS)


def test_no_valid_choices_has_no_windows(trial_df):
    """An entirely manual-reward session has no eligible windows."""
    trial_df["experimenter_reward_given"] = 1
    scores = windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})
    assert windows.summarize_choice_window_agreement(trial_df, scores, 10, "mouse_session").empty


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_window_sizes_fail(trial_df, size):
    """Window sizes must be positive integers, excluding booleans."""
    scores = windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})
    with pytest.raises(ValueError, match="positive integer"):
        windows.summarize_choice_window_agreement(trial_df, scores, size, "mouse_session")


@pytest.mark.parametrize("bad_action", [2, -1, "unexpected"])
def test_invalid_retained_actions_fail(trial_df, bad_action):
    """Malformed actions cannot silently become valid mouse choices."""
    trial_df["action"] = trial_df["action"].astype(object)
    trial_df.loc[0, "action"] = bad_action
    with pytest.raises(ValueError, match="action"):
        windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})


def test_agent_selection_and_mixed_session_validation(trial_df):
    """Missing explicit agents and mixed sessions fail; defaults use present values."""
    with pytest.raises(ValueError, match="missing"):
        windows.compute_trial_agent_agreements(trial_df, {"agent": "missing"})
    with pytest.raises(ValueError, match="agent"):
        windows.compute_trial_agent_agreements(trial_df)
    trial_df["Qlearning_rel_value"] = -1.0
    scores = windows.compute_trial_agent_agreements(trial_df)
    assert scores.columns.tolist() == ["qlearning_mouse_agreement"]
    trial_df.loc[1, "session_ID"] = "another_session"
    with pytest.raises(ValueError, match="session"):
        windows.compute_trial_agent_agreements(trial_df)


def test_infinite_predictions_fail(trial_df):
    """Infinite values are malformed rather than confident predictions."""
    trial_df.loc[0, "agent_value"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})


def test_twenty_choice_counts_equal_adjacent_ten_choice_counts(trial_df):
    """Nested partitions combine raw counts, not unweighted means."""
    trial_df.loc[[2, 4, 8], "agent_value"] = np.nan
    trial_df.loc[10:19, "agent_value"] = 1.0
    scores = windows.compute_trial_agent_agreements(trial_df, {"agent": "agent_value"})
    ten = windows.summarize_choice_window_agreement(trial_df, scores, 10, "mouse_session")
    twenty = windows.summarize_choice_window_agreement(trial_df, scores, 20, "mouse_session")
    for index in range(2):
        pair = ten.iloc[index * 2:index * 2 + 2]
        assert twenty.iloc[index].n_comparisons == pair.n_comparisons.sum()
        assert twenty.iloc[index].n_agreements == pair.n_agreements.sum()
        assert twenty.iloc[index].agreement == pair.n_agreements.sum() / pair.n_comparisons.sum()


def test_window_computation_preserves_context_block_analysis(trial_df):
    """Existing block outputs remain identical before and after window analysis."""
    blocks = pd.DataFrame({"block_ix": pd.unique(trial_df.cur_block)})
    mapping = {"agent": "agent_value"}
    before = add_block_agent_mouse_agreement_columns(blocks, trial_df, mapping)
    scores = windows.compute_trial_agent_agreements(trial_df, mapping)
    windows.summarize_choice_window_agreement(trial_df, scores, 10, "mouse_session")
    after = add_block_agent_mouse_agreement_columns(blocks, trial_df, mapping)
    pd.testing.assert_frame_equal(before, after)


@pytest.fixture
def multisession_windows():
    """Return seven window-agent rows; agreement is unitless, counts are choices."""
    return pd.DataFrame({
        "session_id": ["s1", "s1", "s1", "s2", "s2", "s1", "s2"],
        "date": ["2026-11-01"] * 7,
        "window_size": [10, 10, 10, 10, 10, 20, 20],
        "window_index": [0, 1, 2, 0, 1, 0, 0],
        "agent": ["qlearning_mouse_agreement"] * 7,
        "agreement": [0.0, 1.0, np.nan, 0.25, 0.75, 0.8, np.nan],
        "n_comparisons": [10, 5, 0, 4, 4, 20, 0],
        "n_agreements": [0, 5, 0, 1, 3, 16, 0],
    })


def test_session_window_summary_separates_sizes_sessions_and_missing_scores(multisession_windows):
    """Plot summaries use window quartiles while retaining counts and pooled means."""
    original = multisession_windows.copy(deep=True)
    summary = windows.summarize_choice_window_sessions(multisession_windows)
    assert len(summary) == 4
    first = summary.loc[summary.session_id.eq("s1") & summary.window_size.eq(10)].iloc[0]
    assert first.n_windows == 3
    assert first.n_scored_windows == 2
    assert first.agreement_q1 == 0.25
    assert first.agreement_median == 0.5
    assert first.agreement_q3 == 0.75
    assert first.agreement_pooled == pytest.approx(1 / 3)
    other = summary.loc[summary.session_id.eq("s2") & summary.window_size.eq(10)].iloc[0]
    assert other.agreement_q1 == 0.375
    assert other.agreement_q3 == 0.625
    missing = summary.loc[summary.session_id.eq("s2") & summary.window_size.eq(20)].iloc[0]
    assert missing.n_scored_windows == 0
    assert np.isnan(missing.agreement_median)
    assert np.isnan(missing.agreement_pooled)
    pd.testing.assert_frame_equal(multisession_windows, original)


def test_empty_session_window_summary_has_quartile_schema(multisession_windows):
    """No complete windows still produces a table suitable for placeholder figures."""
    summary = windows.summarize_choice_window_sessions(multisession_windows.iloc[:0])
    assert summary.empty
    assert {"session_id", "window_size", "agent", "agreement_q1", "agreement_q3"}.issubset(summary.columns)
