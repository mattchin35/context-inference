"""Select behavioral motifs before filtering trials for neural coverage.

All selection arrays refer to the full chronological input table's zero-based
positions. Original trial IDs and DataFrame labels remain separate identities.
No behavioral rows are discarded, sorted, or reconstructed from raw logs.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    is_present_value,
    is_zero_flag,
    normalize_experimenter_reward_column,
)


CentralCondition = tuple[str, str, str, str]
TransitionCondition = tuple[str, int]

CENTRAL_CONDITIONS = tuple(product(
    ("correct_rewarded", "omission"), ("left", "right"),
    ("correct_rewarded", "omission", "incorrect"), ("repeat", "switch"),
))
TRANSITION_CONDITIONS = tuple(product(
    ("left_to_right", "right_to_left"), range(1, 5),
))


# Public table preparation and motif selection.
def prepare_sequence_trials(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a full processed trial table without dropping or sorting rows.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Shape (N, C), one row per original chronological trial. Required fields
        are cur_trial (or trial_id), choice_time (UTC Unix seconds), action
        (0=right, 1=left), correct/reward (1/0), and a known manual-reward flag
        column experimenter_reward_given (or legacy give_reward). Missing row
        values are allowed and invalidate only motifs that require them.
        Optional state_int (0=right, 1=left, 2=dark) and cur_block are unitless.

    Returns
    -------
    pd.DataFrame
        A new (N, C') full table with a RangeIndex. row_position is integer
        zero-based source position; trial_id retains the original identity
        values; index_label retains the original DataFrame labels, including
        duplicates. Numeric behavior/context columns use float values with
        missing/unparseable entries as NaN. No units or axis order change.
        Unknown/non-choice rows remain present. Raises ValueError for missing
        required columns, conflicting manual aliases, or ordinary choices
        whose finite timestamps decrease in source order.
    """
    required = {"choice_time", "action", "correct", "reward"}
    missing = sorted(required - set(trial_df.columns))
    if missing:
        raise ValueError(f"Trial table is missing required columns: {missing}")
    id_column = "cur_trial" if "cur_trial" in trial_df else "trial_id"
    if id_column not in trial_df:
        raise ValueError("Trial table needs cur_trial or trial_id identities.")
    if not {EXPERIMENTER_REWARD_GIVEN_COLUMN, "give_reward"}.intersection(trial_df.columns):
        raise ValueError("Trial table needs experimenter_reward_given or legacy give_reward.")

    # Reset only the working copy: shared flag helpers use label-based indexing,
    # while source labels may legitimately repeat.
    prepared = normalize_experimenter_reward_column(trial_df.reset_index(drop=True))
    prepared["row_position"] = np.arange(len(trial_df), dtype=int)
    prepared["trial_id"] = trial_df[id_column].to_numpy(copy=True)
    prepared["index_label"] = trial_df.index.to_numpy(copy=True)
    for column in ("choice_time", "action", "correct", "reward",
                   EXPERIMENTER_REWARD_GIVEN_COLUMN, "state_int", "cur_block"):
        if column in prepared:
            prepared[column] = _numeric_values(prepared[column])

    times = prepared["choice_time"].to_numpy()
    ordinary = _ordinary_choices(prepared)
    if np.any(np.diff(times[ordinary]) < 0):
        raise ValueError("Ordinary choices must be in chronological source order.")
    return prepared


def select_central_sequences(trial_df: pd.DataFrame) -> dict[CentralCondition, np.ndarray]:
    """Select central trials by their immediate original neighbors' behavior.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Full chronological (N, C) table with the required fields/units described
        by prepare_sequence_trials. Context/block columns are not required.
        Every participating row needs a finite UTC choice time, action 0/1,
        known zero manual flag, and consecutive integer original trial IDs.

    Returns
    -------
    dict[tuple[str, str, str, str], np.ndarray]
        All 24 ordered (central_outcome, side, previous_outcome, next_action)
        keys. Outcomes are correct_rewarded/omission (previous also incorrect),
        sides left/right, and next actions repeat/switch. Each integer (M,)
        array contains chronological zero-based central source positions;
        absent conditions retain shape (0,). Correctness/reward are required
        only where constrained; next outcomes and incorrect-previous rewards
        are unconstrained. Block/context crossings are allowed. No neural
        coverage checks or unit transformations occur.
    """
    trials = prepare_sequence_trials(trial_df)
    matches = {key: np.empty(0, dtype=int) for key in CENTRAL_CONDITIONS}
    if len(trials) < 3:
        return matches

    action = trials["action"].to_numpy()
    outcomes = _outcome_masks(trials)
    ordinary = _ordinary_choices(trials)
    adjacent = _consecutive_trial_pairs(trials["trial_id"])
    complete_triples = ordinary[:-2] & ordinary[1:-1] & ordinary[2:]
    complete_triples &= adjacent[:-1] & adjacent[1:]
    central_positions = np.arange(1, len(trials) - 1, dtype=int)

    for key in CENTRAL_CONDITIONS:
        central_outcome, side, previous_outcome, next_action = key
        central_action = int(side == "left")
        previous_action = 1 - central_action if previous_outcome == "incorrect" else central_action
        following_action = central_action if next_action == "repeat" else 1 - central_action
        selected = complete_triples & outcomes[central_outcome][1:-1]
        selected &= action[1:-1] == central_action
        selected &= outcomes[previous_outcome][:-2] & (action[:-2] == previous_action)
        selected &= action[2:] == following_action
        matches[key] = central_positions[selected]
    return matches


def select_ideal_transitions(trial_df: pd.DataFrame) -> dict[TransitionCondition, np.ndarray]:
    """Select complete ideal transition episodes at true processed block boundaries.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Full chronological (N, C) trial table, as in prepare_sequence_trials,
        additionally requiring unitless cur_block and state_int fields. A
        trigger is a rewarded/correct old-active-side choice immediately before
        a different block with opposite active context (0/1). Dark/unknown
        contexts do not establish a trigger.

    Returns
    -------
    dict[tuple[str, int], np.ndarray]
        Eight ordered (direction, x) keys, directions left_to_right and
        right_to_left, x=1..4. Integer arrays have shape (E, x+2), including
        (0, x+2) for empty conditions. Columns contain source positions of the
        trigger, exactly x incorrect old-side repeats, and the immediate
        correct/rewarded opposite choice. Every row is an ordinary choice with
        consecutive trial IDs. All post-trigger rows share the new block and
        context. Repeat rewards are unconstrained. Episodes remain in trigger
        chronology; neural coverage is checked separately downstream.
    """
    missing = sorted({"cur_block", "state_int"} - set(trial_df.columns))
    if missing:
        raise ValueError(f"Trial table is missing transition columns: {missing}")
    trials = prepare_sequence_trials(trial_df)
    matches = {
        key: np.empty((0, key[1] + 2), dtype=int) for key in TRANSITION_CONDITIONS
    }
    if len(trials) < 3:
        return matches

    action = trials["action"].to_numpy()
    state = trials["state_int"].to_numpy()
    block = trials["cur_block"].to_numpy()
    outcomes = _outcome_masks(trials)
    ordinary = _ordinary_choices(trials)
    adjacent = _consecutive_trial_pairs(trials["trial_id"])
    boundary = np.isfinite(block[:-1]) & np.isfinite(block[1:])
    boundary &= block[:-1] != block[1:]
    boundary &= np.isin(state[:-1], [0, 1]) & (state[1:] == 1 - state[:-1])
    boundary &= outcomes["correct_rewarded"][:-1] & (action[:-1] == state[:-1])

    # At most six rows participate in an episode. Iterating actual boundaries
    # keeps exact-episode checks readable without an N-by-condition tensor.
    for key in TRANSITION_CONDITIONS:
        direction, repeats = key
        old_action = int(direction == "left_to_right")
        episodes = []
        triggers = np.flatnonzero(boundary & (action[:-1] == old_action))
        for trigger in triggers:
            final = trigger + repeats + 1
            if final >= len(trials):
                continue
            if not ordinary[trigger:final + 1].all() or not adjacent[trigger:final].all():
                continue
            following = slice(trigger + 1, final + 1)
            if not np.all(block[following] == block[trigger + 1]):
                continue
            if not np.all(state[following] == 1 - old_action):
                continue
            incorrect_run = slice(trigger + 1, final)
            if not outcomes["incorrect"][incorrect_run].all():
                continue
            if not np.all(action[incorrect_run] == old_action):
                continue
            if action[final] == 1 - old_action and outcomes["correct_rewarded"][final]:
                episodes.append(np.arange(trigger, final + 1, dtype=int))
        matches[key] = np.asarray(episodes, dtype=int).reshape(-1, repeats + 2)
    return matches


# Positional numeric and validity helpers.
def _numeric_values(values: pd.Series) -> np.ndarray:
    """Return float (N,) values from a unit-preserving (N,) Series.

    Project missing sentinels, real missing values, and nonnumeric strings
    become NaN. All physical units remain those of the input column.
    """
    present_values = values.where(is_present_value(values), np.nan)
    return pd.to_numeric(present_values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _ordinary_choices(trials: pd.DataFrame) -> np.ndarray:
    """Return bool (N,) ordinary choices from normalized (N,C) trial rows.

    choice_time is UTC seconds; action and manual flags are unitless. Only
    finite times, action 0/1, and explicitly zero manual flags are accepted.
    """
    return (
        np.isfinite(trials["choice_time"].to_numpy())
        & np.isin(trials["action"].to_numpy(), [0, 1])
        & is_zero_flag(trials[EXPERIMENTER_REWARD_GIVEN_COLUMN]).to_numpy(dtype=bool)
    )


def _consecutive_trial_pairs(trial_ids: pd.Series) -> np.ndarray:
    """Return bool (max(N-1,0),) original adjacency from unitless (N,) IDs.

    Both IDs must be finite integers, and the following ID must be exactly one
    larger. Missing IDs, gaps, repeats, and reversals break adjacency.
    """
    identities = _numeric_values(trial_ids)
    finite = np.isfinite(identities)
    integer = finite & (identities == np.floor(identities))
    return integer[:-1] & integer[1:] & (np.diff(identities) == 1)


def _outcome_masks(trials: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return named bool (N,) masks from normalized (N,C) unitless outcomes.

    Correct-rewarded/omission constrain both correct and reward. Incorrect
    constrains only correct=0, so missing reward remains allowed there.
    """
    correct = trials["correct"].to_numpy()
    reward = trials["reward"].to_numpy()
    return {
        "correct_rewarded": (correct == 1) & (reward == 1),
        "omission": (correct == 1) & (reward == 0),
        "incorrect": correct == 0,
    }
