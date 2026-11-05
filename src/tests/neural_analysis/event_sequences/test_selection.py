"""Scientific selection contracts on full chronological processed trial tables.

Condition keys use ASCII identifiers. Every selected integer is a zero-based
source row position, never a DataFrame label or original trial identity.
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences.selection import (
    prepare_sequence_trials,
    select_central_sequences,
    select_ideal_transitions,
)


CENTRAL_KEYS = list(product(
    ("correct_rewarded", "omission"), ("left", "right"),
    ("correct_rewarded", "omission", "incorrect"), ("repeat", "switch"),
))
TRANSITION_KEYS = list(product(("left_to_right", "right_to_left"), range(1, 5)))


def _trials(actions: list[int], correct: list[int], reward: list[int]) -> pd.DataFrame:
    """Return (N, 8) trials from unitless (N,) labels; times are UTC seconds."""
    size = len(actions)
    return pd.DataFrame({
        "cur_trial": np.arange(101, 101 + size),
        "choice_time": 1_700_000_000.0 + 5.0 * np.arange(size),
        "action": actions, "correct": correct, "reward": reward,
        "experimenter_reward_given": np.zeros(size, dtype=int),
        "cur_block": np.zeros(size, dtype=int),
        "state_int": np.ones(size, dtype=int),
    }, index=[f"source-{size - position}" for position in range(size)])


def _central_trials(key: tuple[str, str, str, str]) -> pd.DataFrame:
    """Return (3, 8) trials matching one unitless condition key at position 1."""
    central_outcome, side, previous_outcome, next_action = key
    action = int(side == "left")
    previous_action = 1 - action if previous_outcome == "incorrect" else action
    return _trials(
        [previous_action, action, action if next_action == "repeat" else 1 - action],
        [int(previous_outcome != "incorrect"), 1, 0],
        [int(previous_outcome == "correct_rewarded"), int(central_outcome == "correct_rewarded"), 0],
    )


def _transition_trials(direction: str, repeats: int) -> pd.DataFrame:
    """Return (x+2, 8) ideal episode; direction labels and x are unitless."""
    old_action = int(direction == "left_to_right")
    trials = _trials(
        [old_action] * (repeats + 1) + [1 - old_action],
        [1] + [0] * repeats + [1], [1] + [0] * repeats + [1],
    )
    trials["cur_block"] = [8] + [9] * (repeats + 1)
    trials["state_int"] = [old_action] + [1 - old_action] * (repeats + 1)
    return trials


def _assert_central_mapping(matches: dict, expected: dict) -> None:
    """Assert all 24 keys and unitless integer arrays (M,) equal expectations."""
    assert list(matches) == CENTRAL_KEYS
    for key, positions in matches.items():
        assert isinstance(positions, np.ndarray)
        assert positions.ndim == 1
        assert np.issubdtype(positions.dtype, np.integer)
        np.testing.assert_array_equal(positions, expected.get(key, []))


def _assert_transition_mapping(matches: dict, expected: dict) -> None:
    """Assert eight keys and unitless integer arrays (E,x+2) equal expectations."""
    assert list(matches) == TRANSITION_KEYS
    for key, positions in matches.items():
        assert isinstance(positions, np.ndarray)
        assert positions.ndim == 2
        assert positions.shape[1] == key[1] + 2
        assert np.issubdtype(positions.dtype, np.integer)
        np.testing.assert_array_equal(
            positions, np.asarray(expected.get(key, []), dtype=int).reshape(-1, key[1] + 2),
        )


@pytest.mark.parametrize("key", CENTRAL_KEYS)
def test_all_24_central_conditions_return_only_central_source_position(key):
    """Input: one unitless key. Output: None; verify a (1,) match and 23 empties."""
    trials = _central_trials(key)
    original = trials.copy(deep=True)
    _assert_central_mapping(select_central_sequences(trials), {key: [1]})
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("key", TRANSITION_KEYS)
def test_all_eight_transition_conditions_return_entire_episode(key):
    """Input: direction/x key. Output: None; verify (1,x+2) source positions."""
    trials = _transition_trials(*key)
    original = trials.copy(deep=True)
    _assert_transition_mapping(select_ideal_transitions(trials), {key: [list(range(key[1] + 2))]})
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("row", range(3))
@pytest.mark.parametrize("column,value", [
    ("experimenter_reward_given", 1), ("experimenter_reward_given", -2),
    ("experimenter_reward_given", "None"), ("experimenter_reward_given", np.nan),
    ("action", "no_choice"), ("action", "None"), ("action", 2),
    ("choice_time", "None"), ("choice_time", np.inf), ("choice_time", -np.inf),
])
def test_invalid_participating_rows_break_central_motif(row, column, value):
    """Input: row index and scalar invalid value. Output: None; all arrays empty."""
    trials = _central_trials(CENTRAL_KEYS[0]).astype(object)
    trials.iloc[row, trials.columns.get_loc(column)] = value
    _assert_central_mapping(select_central_sequences(trials), {})


@pytest.mark.parametrize("ids", [[101, 103, 104], [101, 102, 104], [101, "None", 103]])
def test_original_trial_id_gaps_break_adjacency(ids):
    """Input: unitless (3,) IDs. Output: None; adjacent rows cannot bridge ID gaps."""
    trials = _central_trials(CENTRAL_KEYS[0])
    trials["cur_trial"] = ids
    _assert_central_mapping(select_central_sequences(trials), {})


def test_invalid_intervening_row_is_not_filtered_before_neighbor_search():
    """No inputs; return None after checking invalid original rows break motifs."""
    key = ("correct_rewarded", "left", "correct_rewarded", "switch")
    trials = _central_trials(key)
    blocker = trials.iloc[[1]].copy()
    blocker["action"] = "no_choice"
    trials = pd.concat([trials.iloc[:1], blocker, trials.iloc[1:]])
    trials["cur_trial"] = [101, 102, 103, 104]
    _assert_central_mapping(select_central_sequences(trials), {})


@pytest.mark.parametrize("correct,reward", [(0, 0), (0, 1), (1, 0), (1, 1), ("None", "nan"), (np.nan, np.nan)])
def test_next_trial_outcomes_are_unconstrained(correct, reward):
    """Input: next scalar outcome labels. Output: None; next action alone is used."""
    key = ("correct_rewarded", "right", "incorrect", "switch")
    trials = _central_trials(key).astype(object)
    trials.iloc[2, trials.columns.get_loc("correct")] = correct
    trials.iloc[2, trials.columns.get_loc("reward")] = reward
    _assert_central_mapping(select_central_sequences(trials), {key: [1]})


@pytest.mark.parametrize("column,row", [("correct", 0), ("reward", 0), ("correct", 1), ("reward", 1)])
@pytest.mark.parametrize("sentinel", ["None", "", "nan", "NaN", None, np.nan])
def test_missing_constrained_outcomes_exclude_central_motif(column, row, sentinel):
    """Input: outcome location and missing scalar. Output: None; no match remains."""
    trials = _central_trials(CENTRAL_KEYS[0]).astype(object)
    trials.iloc[row, trials.columns.get_loc(column)] = sentinel
    _assert_central_mapping(select_central_sequences(trials), {})


def test_incorrect_previous_trial_reward_is_unconstrained():
    """No inputs; return None after checking incorrect means correctness zero."""
    key = ("omission", "left", "incorrect", "repeat")
    trials = _central_trials(key).astype(object)
    trials.iloc[0, trials.columns.get_loc("reward")] = "None"
    _assert_central_mapping(select_central_sequences(trials), {key: [1]})


@pytest.mark.parametrize("correct,reward", [(0, 0), (0, 1), (1, 1)])
def test_omission_requires_correct_and_unrewarded(correct, reward):
    """Input: scalar central labels. Output: None; generic no-reward is no omission."""
    key = ("omission", "left", "correct_rewarded", "repeat")
    trials = _central_trials(key)
    trials.iloc[1, trials.columns.get_loc("correct")] = correct
    trials.iloc[1, trials.columns.get_loc("reward")] = reward
    assert select_central_sequences(trials)[key].size == 0


def test_central_sequences_allow_crossing_block_and_context_boundaries():
    """No inputs; return None after checking Part I does not require same block."""
    key = CENTRAL_KEYS[0]
    trials = _central_trials(key)
    trials["cur_block"] = [5, 6, 7]
    trials["state_int"] = [0, 1, 0]
    _assert_central_mapping(select_central_sequences(trials), {key: [1]})


@pytest.mark.parametrize("column", ["cur_trial", "choice_time", "action", "correct", "reward", "experimenter_reward_given"])
def test_missing_central_input_column_is_unavailable_not_empty(column):
    """Input: required column name. Output: None; missing schema raises ValueError."""
    trials = _central_trials(CENTRAL_KEYS[0]).drop(columns=[column])
    with pytest.raises(ValueError, match=column if column != "cur_trial" else "cur_trial|trial_id"):
        select_central_sequences(trials)


@pytest.mark.parametrize("column", ["cur_block", "state_int"])
def test_missing_context_disables_only_transition_selection(column):
    """Input: context field name. Output: None; Part I remains available."""
    key = CENTRAL_KEYS[0]
    trials = _central_trials(key).drop(columns=[column])
    _assert_central_mapping(select_central_sequences(trials), {key: [1]})
    with pytest.raises(ValueError, match=column):
        select_ideal_transitions(trials)


def test_numeric_strings_legacy_manual_alias_and_trial_id_alias():
    """No inputs; return None after numeric strings/legacy names preserve selection."""
    key = ("right_to_left", 2)
    trials = _transition_trials(*key).astype(str).rename(columns={
        "cur_trial": "trial_id", "experimenter_reward_given": "give_reward",
    })
    _assert_transition_mapping(select_ideal_transitions(trials), {key: [[0, 1, 2, 3]]})
    central = _central_trials(CENTRAL_KEYS[0]).astype(str).rename(columns={
        "cur_trial": "trial_id", "experimenter_reward_given": "give_reward",
    })
    _assert_central_mapping(select_central_sequences(central), {CENTRAL_KEYS[0]: [1]})


@pytest.mark.parametrize("selector", [select_central_sequences, select_ideal_transitions])
def test_conflicting_manual_reward_aliases_are_rejected(selector):
    """Input: selector callable (DataFrame -> mapping). Output: None; reject conflict."""
    trials = _transition_trials("left_to_right", 2)
    trials["give_reward"] = [0, 1, 0, 0]
    with pytest.raises(ValueError, match="conflicting"):
        selector(trials)


@pytest.mark.parametrize("selector", [select_central_sequences, select_ideal_transitions])
def test_nonchronological_source_choices_are_rejected_without_sorting(selector):
    """Input: selector callable (DataFrame -> mapping). Output: None; reject time reversal."""
    trials = _transition_trials("left_to_right", 2)
    trials["choice_time"] = [100.0, 110.0, 105.0, 120.0]
    original = trials.copy(deep=True)
    with pytest.raises(ValueError, match="chronological"):
        selector(trials)
    pd.testing.assert_frame_equal(trials, original)


def test_normalized_full_table_preserves_three_distinct_identities():
    """No inputs; return None after verifying (N,) positions, IDs, and source labels."""
    trials = _central_trials(CENTRAL_KEYS[0])
    trials.index = pd.Index(["duplicate", "duplicate", "last"], name="source_label")
    original = trials.copy(deep=True)
    normalized = prepare_sequence_trials(trials)
    assert len(normalized) == 3
    np.testing.assert_array_equal(normalized["row_position"], [0, 1, 2])
    np.testing.assert_array_equal(normalized["trial_id"], [101, 102, 103])
    np.testing.assert_array_equal(normalized["index_label"], ["duplicate", "duplicate", "last"])
    _assert_central_mapping(select_central_sequences(trials), {CENTRAL_KEYS[0]: [1]})
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("size", [0, 1, 2])
def test_short_tables_keep_all_conditions_with_correct_empty_shapes(size):
    """Input: number of source rows. Output: None; retain empty (0,)/(0,x+2) arrays."""
    trials = _central_trials(CENTRAL_KEYS[0]).iloc[:size]
    _assert_central_mapping(select_central_sequences(trials), {})
    _assert_transition_mapping(select_ideal_transitions(trials), {})


def test_multiple_matches_are_chronological_original_positions():
    """No inputs; return None after checking two (M,) and (E,4) chronological matches."""
    key = ("omission", "left", "incorrect", "switch")
    trials = pd.concat([_central_trials(key), _central_trials(key)])
    trials["cur_trial"] = np.arange(71, 77)
    trials["choice_time"] = 1_700_000_000.0 + np.arange(6) * 5
    _assert_central_mapping(select_central_sequences(trials), {key: [1, 4]})
    episode = _transition_trials("left_to_right", 2)
    transitions = pd.concat([episode, episode])
    transitions["cur_trial"] = np.arange(71, 79)
    transitions["choice_time"] = 1_700_000_000.0 + np.arange(8) * 5
    _assert_transition_mapping(select_ideal_transitions(transitions), {
        ("left_to_right", 2): [[0, 1, 2, 3], [4, 5, 6, 7]],
    })


@pytest.mark.parametrize("row,column,value", [
    (1, "cur_block", 8), (0, "state_int", 0), (0, "state_int", 2),
    (1, "state_int", 1), (1, "state_int", 2), (1, "state_int", "None"),
    (0, "action", 0), (0, "correct", 0), (0, "reward", 0),
    (1, "action", 0), (2, "correct", 1), (2, "reward", "None"),
    (3, "action", 1), (3, "correct", 0), (3, "reward", 0),
    (2, "cur_block", 10), (3, "cur_block", 10), (2, "state_int", 1),
    (3, "state_int", 1), (2, "cur_trial", 105), (2, "action", "no_choice"),
    (1, "choice_time", np.inf), (3, "choice_time", "None"),
])
def test_transition_requires_true_boundary_and_exact_unbroken_episode(row, column, value):
    """Input: one mutated scalar. Output: None; reject broken left-to-right x2 episode."""
    trials = _transition_trials("left_to_right", 2).astype(object)
    trials.iloc[row, trials.columns.get_loc(column)] = value
    # Reward is unconstrained on an incorrect repeat, including a missing value.
    expected = {("left_to_right", 2): [[0, 1, 2, 3]]} if (row, column) == (2, "reward") else {}
    _assert_transition_mapping(select_ideal_transitions(trials), expected)


@pytest.mark.parametrize("row", range(4))
@pytest.mark.parametrize("manual", [1, "None", np.nan])
def test_manual_or_unknown_flag_anywhere_breaks_transition(row, manual):
    """Input: episode position/flag scalar. Output: None; every participant is ordinary."""
    trials = _transition_trials("left_to_right", 2).astype(object)
    trials.iloc[row, trials.columns.get_loc("experimenter_reward_given")] = manual
    _assert_transition_mapping(select_ideal_transitions(trials), {})


def test_context_change_without_block_change_does_not_establish_trigger():
    """No inputs; return None after excluding rewarded/error sequence without boundary."""
    trials = _transition_trials("right_to_left", 1)
    trials["cur_block"] = 8
    _assert_transition_mapping(select_ideal_transitions(trials), {})


def test_first_opposite_choice_omission_cannot_be_skipped_for_later_reward():
    """No inputs; return None after checking immediate first switch must be rewarded."""
    trials = _transition_trials("left_to_right", 2)
    trials.loc[trials.index[-1], "reward"] = 0
    final_reward = trials.iloc[[-1]].copy()
    final_reward["reward"] = 1
    trials = pd.concat([trials, final_reward])
    trials["cur_trial"] = np.arange(101, 106)
    _assert_transition_mapping(select_ideal_transitions(trials), {})


def test_five_incorrect_repeats_do_not_match_shorter_motif():
    """No inputs; return None after excluding x5 without truncation to x1..x4."""
    _assert_transition_mapping(select_ideal_transitions(_transition_trials("left_to_right", 5)), {})
