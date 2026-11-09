"""Part III contracts, with source episodes specified independently of selectors.

All positions and labels are unitless; choice timestamps are UTC Unix seconds.
Synthetic motif construction describes trials, never searches for matches.
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import selection


ADDITIONAL_KEYS = (
    list(product(("probe",), ("left", "right"), range(1, 5)))
    + list(product(("omission_probe",), ("left", "right"), range(1, 5)))
    + [("lucky_omission_switch", "left", None), ("lucky_omission_switch", "right", None)]
    + list(product(("indecisive_transition",), ("left", "right"), range(1, 4)))
)
LONG_KEYS = [
    ("probe", "left", 4), ("omission_probe", "left", 4),
    ("lucky_omission_switch", "left", None), ("indecisive_transition", "left", 3),
]


def _episode(key: tuple[str, str, int | None]) -> pd.DataFrame:
    """Return synthetic (S,8) raw trials for a unitless family/side/x key.

    action/state use 0=right, 1=left; trial IDs are consecutive integers,
    choices are UTC seconds, and deliberately duplicated index labels are
    unrelated to original trial IDs or zero-based source row positions.
    """
    family, side, repeats = key
    initial = int(side == "left")
    opposite = 1 - initial
    if family == "probe":
        actions = [initial] + [opposite] * repeats + [initial]
        correct = [1] + [0] * repeats + [1]
        rewards = [1] + [0] * repeats + [1]
        blocks = [31] * len(actions)
        states = [initial] * len(actions)
    elif family == "omission_probe":
        actions = [initial, initial] + [opposite] * repeats + [initial]
        correct = [1, 1] + [0] * repeats + [1]
        rewards = [1, 0] + [0] * repeats + [1]
        blocks = [31] * len(actions)
        states = [initial] * len(actions)
    elif family == "lucky_omission_switch":
        actions, correct, rewards = [initial, initial, opposite], [1, 1, 1], [1, 0, 1]
        blocks, states = [31, 31, 47], [initial, initial, opposite]
    elif family == "indecisive_transition":
        actions = [initial] + [initial] * repeats + [opposite, initial, opposite]
        correct = [1] + [0] * repeats + [1, 0, 1]
        rewards = [1] + [0] * repeats + [1, 0, 1]
        blocks = [31] + [47] * (len(actions) - 1)
        states = [initial] + [opposite] * (len(actions) - 1)
    else:
        raise ValueError(f"Unknown fixture family: {family}")
    size = len(actions)
    return pd.DataFrame({
        "cur_trial": np.arange(701, 701 + size),
        "choice_time": 1_700_000_000.0 + 5 * np.arange(size),
        "action": actions, "correct": correct, "reward": rewards,
        "experimenter_reward_given": np.zeros(size, dtype=int),
        "cur_block": blocks, "state_int": states,
    }, index=pd.Index(["duplicate"] * size, name="unrelated_label"))


def _join_trials(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Join (N_i,C) trial tables into (sum N_i,C) with fresh IDs/UTC seconds.

    Row order and duplicate source labels are preserved; only fixture IDs and
    timestamps are reassigned to establish continuous original chronology.
    """
    trials = pd.concat(parts)
    trials["cur_trial"] = np.arange(701, 701 + len(trials))
    trials["choice_time"] = 1_700_000_000.0 + 5 * np.arange(len(trials))
    return trials


def _select(trials: pd.DataFrame) -> dict:
    """Map full chronological (N,C) trials to integer (E,S) original positions.

    UTC times and all unitless behavior labels are passed through unchanged.
    Attribute lookup makes the missing public selector a focused RED failure.
    """
    return getattr(selection, "select_additional_sequences")(trials)


def _assert_matches(matches: dict, expected: dict) -> None:
    """Check all 24 keys and unitless integer (E,S) original-position arrays.

    expected maps only populated family/side/x keys to explicit episode rows.
    Return None; every unspecified condition must keep its correct (0,S) shape.
    """
    assert len(matches) == 24
    assert set(matches) == set(ADDITIONAL_KEYS)
    for key in ADDITIONAL_KEYS:
        family, _, repeats = key
        length = 3 if repeats is None else repeats + {
            "probe": 2, "omission_probe": 3, "indecisive_transition": 4,
        }[family]
        positions = matches[key]
        assert isinstance(positions, np.ndarray), key
        assert positions.ndim == 2, key
        assert positions.shape[1] == length, key
        assert np.issubdtype(positions.dtype, np.integer), key
        np.testing.assert_array_equal(
            positions, np.asarray(expected.get(key, []), dtype=int).reshape(-1, length),
            err_msg=str(key),
        )


def _expected_episode(key: tuple[str, str, int | None], length: int, start: int = 0) -> dict:
    """Return explicit (E,S) position lists for one synthetic episode of S rows.

    key is unitless and start is its zero-based source position. An indecisive
    episode also contains the independently valid opposite-side x1 probe tail.
    No behavior is searched and no matcher logic is used.
    """
    expected = {key: [list(range(start, start + length))]}
    if key[0] == "indecisive_transition":
        opposite_side = "right" if key[1] == "left" else "left"
        expected[("probe", opposite_side, 1)] = [list(range(start + length - 3, start + length))]
    return expected


def _assert_target_absent(trials: pd.DataFrame, key: tuple[str, str, int | None]) -> None:
    """Check no (E,S) episode remains for a unitless target in full (N,C) trials.

    Return None. Other families can retain valid subepisodes after a mutation;
    in particular, breaking an indecisive start need not break its probe tail.
    """
    assert _select(trials)[key].shape[0] == 0, key


@pytest.mark.parametrize("key", ADDITIONAL_KEYS)
def test_all_conditions_return_full_original_episode_positions(key):
    """Input: one unitless key. Return None after checking one (1,S) episode."""
    trials = _episode(key)
    original = trials.copy(deep=True)
    _assert_matches(_select(trials), _expected_episode(key, len(trials)))
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("key", LONG_KEYS)
@pytest.mark.parametrize("column,value", [
    ("experimenter_reward_given", 1), ("experimenter_reward_given", -2),
    ("experimenter_reward_given", "None"), ("experimenter_reward_given", np.nan),
    ("action", "no_choice"), ("action", 2),
    ("choice_time", "None"), ("choice_time", np.inf),
])
def test_every_participating_row_must_be_an_ordinary_choice(key, column, value):
    """Input: key/invalid scalar. Return None; reject each invalid (S,C) episode."""
    for row in range(len(_episode(key))):
        trials = _episode(key).astype(object)
        trials.iloc[row, trials.columns.get_loc(column)] = value
        _assert_target_absent(trials, key)


@pytest.mark.parametrize("key", LONG_KEYS)
@pytest.mark.parametrize("identity", [699, 701, 704.5, "None", np.nan])
def test_trial_id_gaps_repeats_reversals_and_nonintegers_break_adjacency(key, identity):
    """Input: key/invalid ID scalar. Return None; every episode edge needs +1 IDs."""
    for row in range(len(_episode(key))):
        trials = _episode(key).astype(object)
        # At row zero, 699 is a gap; at later rows, these values are never +1.
        trials.iloc[row, trials.columns.get_loc("cur_trial")] = identity
        if identity == 701 and row == 0:
            continue
        _assert_target_absent(trials, key)


@pytest.mark.parametrize("key", LONG_KEYS)
def test_invalid_intervening_rows_cannot_be_filtered_out_before_matching(key):
    """Input: unitless key. Return None; inserting a non-choice prevents bridging."""
    episode = _episode(key)
    blocker = episode.iloc[[1]].astype(object)
    blocker["action"] = "no_choice"
    trials = _join_trials([episode.iloc[:1], blocker, episode.iloc[1:]])
    _assert_target_absent(trials, key)


@pytest.mark.parametrize("key", LONG_KEYS)
def test_action_and_constrained_outcome_required_on_every_segment(key):
    """Input: unitless key. Return None; reject wrong side/correct/reward labels."""
    episode = _episode(key)
    for row in range(len(episode)):
        for column in ("action", "correct", "reward"):
            if column == "reward" and episode.iloc[row]["correct"] == 0:
                continue
            trials = episode.copy(deep=True)
            trials.iloc[row, trials.columns.get_loc(column)] = 1 - episode.iloc[row][column]
            _assert_target_absent(trials, key)


@pytest.mark.parametrize("key", LONG_KEYS)
@pytest.mark.parametrize("sentinel", ["None", np.nan])
def test_missing_constrained_outcomes_are_not_inferred_from_context(key, sentinel):
    """Input: key/missing scalar. Return None; missing required outcomes reject."""
    episode = _episode(key)
    for row in range(len(episode)):
        for column in ("correct", "reward"):
            if column == "reward" and episode.iloc[row]["correct"] == 0:
                continue
            trials = episode.astype(object)
            trials.iloc[row, trials.columns.get_loc(column)] = sentinel
            _assert_target_absent(trials, key)


@pytest.mark.parametrize("key", [LONG_KEYS[0], LONG_KEYS[1], LONG_KEYS[3]])
@pytest.mark.parametrize("reward", [1, "None", np.nan])
def test_incorrect_excursion_and_later_return_rewards_are_unconstrained(key, reward):
    """Input: key/reward scalar. Return None; correct=0 alone defines incorrect."""
    trials = _episode(key).astype(object)
    trials.loc[trials["correct"] == 0, "reward"] = reward
    _assert_matches(_select(trials), _expected_episode(key, len(trials)))


@pytest.mark.parametrize("family", ["probe", "omission_probe"])
@pytest.mark.parametrize("column,value", [
    ("cur_block", 47), ("cur_block", "None"),
    ("state_int", 0), ("state_int", 2), ("state_int", "None"),
])
def test_probes_require_one_block_and_the_initial_active_context(family, column, value):
    """Input: family/context scalar. Return None; reject change on any episode row."""
    key = (family, "left", 4)
    for row in range(len(_episode(key))):
        trials = _episode(key).astype(object)
        trials.iloc[row, trials.columns.get_loc(column)] = value
        _assert_matches(_select(trials), {})
    if column == "state_int":
        trials[column] = value
        _assert_matches(_select(trials), {})


@pytest.mark.parametrize("family", ["probe", "omission_probe"])
def test_omission_return_cannot_be_skipped_for_a_later_reward(family):
    """Input: probe family. Return None; an immediate return omission is ineligible."""
    key = (family, "left", 2)
    episode = _episode(key)
    episode.iloc[-1, episode.columns.get_loc("reward")] = 0
    later_reward = episode.iloc[[-1]].copy()
    later_reward["reward"] = 1
    _assert_matches(_select(_join_trials([episode, later_reward])), {})


@pytest.mark.parametrize("family", ["probe", "omission_probe", "indecisive_transition"])
def test_initial_run_above_supported_range_does_not_match_a_shorter_condition(family):
    """Input: family. Return None; x5 probe/x4 indecisive cannot be truncated."""
    repeats = 4 if family == "indecisive_transition" else 5
    matches = _select(_episode((family, "left", repeats)))
    assert all(positions.shape[0] == 0 for key, positions in matches.items() if key[0] == family)


@pytest.mark.parametrize("family", ["probe", "omission_probe", "indecisive_transition"])
def test_variable_run_needs_at_least_one_initial_incorrect_trial(family):
    """Input: family string. Return None; zero-error run matches none of its keys."""
    matches = _select(_episode((family, "left", 0)))
    assert all(positions.shape[0] == 0 for key, positions in matches.items() if key[0] == family)


def test_omission_probe_needs_exactly_one_omission_before_excursion():
    """No inputs; return None after rejecting (8,C) motif with two omissions."""
    episode = _episode(("omission_probe", "left", 4))
    trials = _join_trials([episode.iloc[:2], episode.iloc[[1]], episode.iloc[2:]])
    _assert_matches(_select(trials), {})


@pytest.mark.parametrize("blocks,states", [
    ([31, 47, 47], [1, 0, 0]),  # Boundary is before the omission.
    ([31, 31, 31], [1, 1, 0]),  # Context alone does not establish a boundary.
    ([31, 31, 47], [1, 1, 1]),  # Block alone does not establish a boundary.
    ([31, 31, 47], [1, 1, 2]),
    ([31, 47, 48], [1, 1, 0]),
    ([31, 31, 47], [0, 0, 1]),  # Active context must agree with the choices.
])
def test_lucky_boundary_is_exactly_after_the_old_context_omission(blocks, states):
    """Input: unitless (3,) block/state lists. Return None; reject false boundary."""
    trials = _episode(("lucky_omission_switch", "left", None))
    trials["cur_block"], trials["state_int"] = blocks, states
    _assert_matches(_select(trials), {})


def test_lucky_omission_does_not_relax_the_existing_ideal_rewarded_trigger():
    """No inputs; return None after asserting lucky (1,3), every ideal (0,S)."""
    key = ("lucky_omission_switch", "left", None)
    trials = _episode(key)
    _assert_matches(_select(trials), {key: [[0, 1, 2]]})
    assert all(positions.shape[0] == 0 for positions in selection.select_ideal_transitions(trials).values())


@pytest.mark.parametrize("side,repeats", list(product(("left", "right"), range(1, 4))))
def test_indecisive_episode_also_keeps_its_original_ideal_prefix(side, repeats):
    """Input: unitless side/x. Return None; keep full (1,x+4) and ideal (1,x+2)."""
    key = ("indecisive_transition", side, repeats)
    trials = _episode(key)
    _assert_matches(_select(trials), _expected_episode(key, repeats + 4))
    direction = "left_to_right" if side == "left" else "right_to_left"
    ideal = selection.select_ideal_transitions(trials)
    for ideal_key, positions in ideal.items():
        expected = [list(range(repeats + 2))] if ideal_key == (direction, repeats) else []
        np.testing.assert_array_equal(positions, np.asarray(expected, dtype=int).reshape(-1, ideal_key[1] + 2))


@pytest.mark.parametrize("column,value", [
    ("cur_block", 31), ("cur_block", 99),
    ("state_int", 1), ("state_int", 2),
])
def test_all_indecisive_posttrigger_rows_stay_in_one_new_context(column, value):
    """Input: scalar context mutation. Return None; reject every broken post row."""
    key = ("indecisive_transition", "left", 3)
    for row in range(1, 7):
        trials = _episode(key)
        trials.iloc[row, trials.columns.get_loc(column)] = value
        _assert_target_absent(trials, key)


@pytest.mark.parametrize("blocks,states", [
    ([31] * 5, [1, 0, 0, 0, 0]),
    ([31, 47, 47, 47, 47], [1] * 5),
    ([31, 47, 47, 47, 47], [0] * 5),
    ([31, 47, 47, 47, 47], [2, 0, 0, 0, 0]),
])
def test_indecisive_trigger_needs_a_real_opposite_active_context_boundary(blocks, states):
    """Input: unitless (5,) block/state lists. Return None; reject false trigger."""
    key = ("indecisive_transition", "left", 1)
    trials = _episode(key)
    trials["cur_block"], trials["state_int"] = blocks, states
    _assert_target_absent(trials, key)


def test_indecisive_later_incorrect_return_is_exactly_one_trial():
    """No inputs; return None; later two-error return cannot be a full episode."""
    episode = _episode(("indecisive_transition", "left", 2))
    trials = _join_trials([episode.iloc[:-1], episode.iloc[[-2]], episode.iloc[[-1]]])
    assert _select(trials)[("indecisive_transition", "left", 2)].shape[0] == 0
    np.testing.assert_array_equal(selection.select_ideal_transitions(trials)[("left_to_right", 2)], [[0, 1, 2, 3]])


@pytest.mark.parametrize("column", [
    "cur_trial", "choice_time", "action", "correct", "reward",
    "experimenter_reward_given", "cur_block", "state_int",
])
def test_missing_required_columns_raise_named_value_error(column):
    """Input: absent field name. Return None; unavailable schema is never n=0."""
    trials = _episode(LONG_KEYS[0]).drop(columns=[column])
    with pytest.raises(ValueError, match=column if column != "cur_trial" else "cur_trial|trial_id"):
        _select(trials)
    if column in ("cur_block", "state_int"):
        assert len(selection.select_central_sequences(trials)) == 24


@pytest.mark.parametrize("key", LONG_KEYS)
@pytest.mark.parametrize("column", ["cur_block", "state_int"])
def test_missing_row_context_invalidates_only_episodes_containing_it(key, column):
    """Input: key/context field. Return None; keep unaffected second (1,S) match."""
    episode = _episode(key)
    length = len(episode)
    for row in range(length):
        trials = _join_trials([episode, episode]).astype(object)
        trials.iloc[row, trials.columns.get_loc(column)] = "None"
        np.testing.assert_array_equal(_select(trials)[key], [list(range(length, 2 * length))])


@pytest.mark.parametrize("key", LONG_KEYS)
def test_numeric_strings_and_legacy_aliases_keep_full_episode_positions(key):
    """Input: unitless key. Return None; string labels/IDs and flags normalize."""
    trials = _episode(key).astype(str).rename(columns={
        "cur_trial": "trial_id", "experimenter_reward_given": "give_reward",
    })
    _assert_matches(_select(trials), _expected_episode(key, len(trials)))


def test_conflicting_manual_reward_aliases_raise_value_error():
    """No inputs; return None after rejecting conflicting flags in a (6,C) table."""
    trials = _episode(LONG_KEYS[0])
    trials["give_reward"] = 0
    trials.iloc[2, trials.columns.get_loc("give_reward")] = 1
    with pytest.raises(ValueError, match="conflicting"):
        _select(trials)


def test_reversed_choice_chronology_is_rejected_without_sorting():
    """No inputs; return None after rejecting reversed UTC times and preserving rows."""
    trials = _episode(LONG_KEYS[0])
    trials.iloc[2, trials.columns.get_loc("choice_time")] = trials.iloc[0]["choice_time"]
    original = trials.copy(deep=True)
    with pytest.raises(ValueError, match="chronological"):
        _select(trials)
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("size", [0, 1, 2])
def test_short_tables_return_all_correctly_shaped_empty_conditions(size):
    """Input: row count 0..2. Return None; each of 24 arrays retains shape (0,S)."""
    _assert_matches(_select(_episode(LONG_KEYS[0]).iloc[:size]), {})


@pytest.mark.parametrize("key", LONG_KEYS)
def test_multiple_episodes_are_ordered_by_start_and_use_source_positions(key):
    """Input: key. Return None; two (2,S) rows follow original source chronology."""
    episode = _episode(key)
    length = len(episode)
    trials = _join_trials([episode, episode])
    expected = _expected_episode(key, length)
    for condition, rows in _expected_episode(key, length, length).items():
        expected[condition].extend(rows)
    _assert_matches(_select(trials), expected)
