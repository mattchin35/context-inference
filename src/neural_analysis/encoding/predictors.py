"""Pure context coding on the complete, unfiltered chronological trial sequence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    is_present_value,
    is_zero_flag,
    make_no_choice_action_mask,
    normalize_experimenter_reward_column,
)
from src.neural_analysis.encoding.configuration import ENCODINGS, TIMING_MODES


# Public computations retain source row positions independently of index labels.
def select_baseline_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Identify the descriptive PCA pool before predictor-specific missingness.

    Parameters
    ----------
    trials : pd.DataFrame
        Complete chronological (W,C) table with cur_trial, cur_block, start_time,
        choice_time, state_int, action and experimenter_reward_given (or legacy
        give_reward). Times are UTC seconds; state/action use 0=right, 1=left,
        state 2=dark. Missing states/choices/flags are excluded conservatively.

    Returns
    -------
    pd.DataFrame
        W rows with RangeIndex, row_position, original trial_id and block_id,
        Boolean baseline_valid, and semicolon-separated baseline_invalid_reason.
        Manual reward, no animal choice, missing choice time, and invalid/dark
        entering context exclude rows. Input and original row order are unchanged.
    """
    _validate_trial_sequence(trials)
    working = normalize_experimenter_reward_column(trials).reset_index(drop=True)
    _require_columns(working, {"cur_block", "action", EXPERIMENTER_REWARD_GIVEN_COLUMN})
    states = _numeric(working["state_int"])
    actions = _numeric(working["action"])
    no_choice = make_no_choice_action_mask(working["action"]).to_numpy() | ~np.isin(actions, [0, 1])
    flags = _numeric(working[EXPERIMENTER_REWARD_GIVEN_COLUMN])
    flag_present = is_present_value(working[EXPERIMENTER_REWARD_GIVEN_COLUMN]).to_numpy()
    if np.any(flag_present & ~np.isfinite(flags)):
        raise ValueError("experimenter_reward_given must contain finite numeric flags")
    manual = ~is_zero_flag(working[EXPERIMENTER_REWARD_GIVEN_COLUMN]).to_numpy()
    missing_choice = ~np.isfinite(_numeric(working["choice_time"]))
    dark = states == 2
    invalid_context = ~np.isin(states, [0, 1, 2])
    reasons = []
    for row in range(len(working)):
        row_reasons = []
        for mask, reason in ((manual, "manual_reward"), (no_choice, "no_animal_choice"),
                             (missing_choice, "missing_choice_time"), (dark, "dark_context"),
                             (invalid_context, "invalid_context")):
            if mask[row]:
                row_reasons.append(reason)
        reasons.append(";".join(row_reasons))
    return pd.DataFrame({
        "row_position": np.arange(len(working)),
        "trial_id": working["cur_trial"].to_numpy(copy=True),
        "block_id": working["cur_block"].to_numpy(copy=True),
        "baseline_valid": ~(manual | no_choice | missing_choice | dark | invalid_context),
        "baseline_invalid_reason": reasons,
    })


def build_context_values(
    trials: pd.DataFrame, relative_bin_edges_s: np.ndarray, timing_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign context before any filtering using entering and immediate next states.

    Parameters
    ----------
    trials : pd.DataFrame
        Complete (W,C) chronological trial table, including excluded rows. Required
        columns are cur_trial, start_time, choice_time (UTC seconds), state_int.
    relative_bin_edges_s : np.ndarray
        Finite strictly increasing (B+1,) edges in choice-relative seconds, with
        zero as an edge. Each bin is half-open; bins starting at zero are post.
    timing_mode : str
        choice_updated uses state_int[i] before zero and state_int[i+1] after;
        trial_window repeats state_int[i] for all bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 context values and Boolean validity, each (W,B), axes trial/bin.
        Only 0=right and 1=left are valid. Unavailable values are NaN. Terminal
        choice_updated post bins stay unavailable; no events or carry-forward
        inference are used. This function does not apply baseline exclusions.
    """
    _validate_trial_sequence(trials)
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"timing_mode must be one of {TIMING_MODES}")
    edges = np.asarray(relative_bin_edges_s, dtype=float)
    if (edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0) or not np.any(edges == 0)
            or edges[0] >= 0 or edges[-1] <= 0):
        raise ValueError("bin edges must increase, span zero, and include zero as an edge")
    states = _numeric(trials["state_int"])
    entering = np.where(np.isin(states, [0, 1]), states, np.nan)
    values = np.repeat(entering[:, None], len(edges) - 1, axis=1)
    if timing_mode == "choice_updated":
        following = np.full(len(entering), np.nan)
        following[:-1] = entering[1:]
        values[:, edges[:-1] >= 0] = following[:, None]
    return values, np.isfinite(values)


def encode_context(values: np.ndarray, encoding: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Encode native context values without an intercept column or standardization.

    Parameters
    ----------
    values : np.ndarray
        (N,) numeric context codes: 0=right, 1=left, other values unavailable.
        Callers explicitly select/flatten trial-bin axes before this boundary.
    encoding : str
        signed_context or separate_context.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[str, ...]]
        Float64 X (N,P), Boolean valid mask (N,), and P ordered column names.
        Signed is right=-1/left=+1; separate columns left_context/right_context
        contain native 0/1 indicators. Invalid rows contain NaN, never zero.
    """
    states = np.asarray(values, dtype=float)
    if states.ndim != 1:
        raise ValueError("context values must be one-dimensional")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding supports only context encodings {ENCODINGS}")
    valid = np.isin(states, [0, 1])
    if encoding == "signed_context":
        matrix = (2 * states - 1)[:, None]
        columns = ("signed_context",)
    else:
        matrix = np.column_stack((states == 1, states == 0)).astype(float)
        columns = ("left_context", "right_context")
    matrix[~valid] = np.nan
    return matrix, valid, columns


def validate_trial_tables(augmented: pd.DataFrame, source: pd.DataFrame) -> None:
    """Require exact complete chronological source correspondence before filtering.

    Parameters
    ----------
    augmented, source : pd.DataFrame
        Complete (W,C) tables containing cur_trial, start_time, choice_time and
        state_int. IDs need not start at zero; times are UTC seconds and state
        codes are dimensionless. DataFrame index labels and extra columns are
        ignored, but table order and original trial identity are preserved.

    Returns
    -------
    None
        Raises ValueError for missing/duplicate IDs, chronology, or differing
        IDs/timestamps/context. Paired missing values compare equal. No inferred
        join, sorting, event repair, or input mutation occurs.
    """
    _validate_trial_sequence(augmented)
    _validate_trial_sequence(source)
    if len(augmented) != len(source):
        raise ValueError("Complete trial tables have different row counts")
    for column in ("cur_trial", "start_time", "choice_time", "state_int"):
        first = _numeric(augmented[column])
        second = _numeric(source[column])
        if not np.array_equal(first, second, equal_nan=True):
            raise ValueError(f"Trial tables disagree in {column}")


def diagnose_context_events(trials: pd.DataFrame, events: pd.DataFrame) -> list[str]:
    """Compare compressed context order; event times never override trial states.

    Parameters
    ----------
    trials : pd.DataFrame
        Complete (W,C) table with state_int (0=right,1=left,2=dark), trial IDs
        and start/choice UTC seconds.
    events : pd.DataFrame
        (E,C) event log containing Time (UTC seconds) and Event strings. Only
        enter_right_patch, enter_left_patch, enter_dark_period are compared.
        Logs may lag choice-time updates. Their original timestamps stay intact.

    Returns
    -------
    list[str]
        Advisory messages, empty for matching state order. Absent/invalid logs
        and differing state order are reported, never repaired or made fatal.
        A fully matching trial sequence may have surplus logged segments strictly
        after the latest trial start/finite choice; these unused terminal events
        are silent. Delayed matching entries are retained. Genuine mismatches
        report only the first differing segment and counts, not full sequences.
        This diagnostic does not supply terminal or invalid predictor values.
    """
    _validate_trial_sequence(trials)
    if not {"Time", "Event"}.issubset(events.columns):
        return ["Context event log lacks Time/Event columns"]
    event_codes = {"enter_right_patch": 0, "enter_left_patch": 1, "enter_dark_period": 2}
    entries = events.loc[events["Event"].isin(event_codes)].copy()
    if entries.empty:
        return ["No recorded context-entry events are available for state-order diagnostics"]
    times = _numeric(entries["Time"])
    if not np.isfinite(times).all():
        return ["Context event timestamps contain invalid UTC seconds"]
    entries = entries.assign(_time=times).sort_values("_time", kind="stable")
    event_states = entries["Event"].map(event_codes).to_numpy(dtype=int)
    new_segment = np.r_[True, event_states[1:] != event_states[:-1]]
    logged = event_states[new_segment].tolist()
    segment_times = entries["_time"].to_numpy()[new_segment]
    states = _numeric(trials["state_int"])
    messages = []
    if not np.isin(states, [0, 1, 2]).all():
        messages.append("Trial context contains unavailable states; event-order comparison is incomplete")
    expected = _state_order(states[np.isin(states, [0, 1, 2])])
    if logged == expected:
        return messages
    first_difference = 0
    while (first_difference < min(len(expected), len(logged))
           and expected[first_difference] == logged[first_difference]):
        first_difference += 1
    # Only an exact trial-context prefix permits ignoring unused terminal events.
    # The last start also protects a represented final trial without a choice.
    choices = _numeric(trials["choice_time"])
    boundary = max(_numeric(trials["start_time"])[-1], choices[np.isfinite(choices)].max(initial=-np.inf))
    if (expected and first_difference == len(expected)
            and np.all(segment_times[first_difference:] > boundary)):
        return messages
    labels = ("right", "left", "dark")
    trial_state = labels[expected[first_difference]] if first_difference < len(expected) else "missing"
    logged_state = labels[logged[first_difference]] if first_difference < len(logged) else "missing"
    messages.append(f"Context state order differs at segment {first_difference + 1}: "
                    f"trial={trial_state}, logged={logged_state} "
                    f"({len(expected)} trial segments, {len(logged)} logged segments)")
    return messages


# Shared schema and numeric helpers.
def _require_columns(table: pd.DataFrame, required: set[str]) -> None:
    """Check (W,C) DataFrame/required string set; raise for missing columns, return None."""
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Trial table is missing columns: {sorted(missing)}")


def _numeric(values: pd.Series) -> np.ndarray:
    """Convert (W,) Series to float64 (W,), preserving units; missing/invalid -> NaN."""
    return pd.to_numeric(values.where(is_present_value(values)), errors="coerce").to_numpy(dtype=float)


def _validate_trial_sequence(table: pd.DataFrame) -> None:
    """Check full (W,C) trial identity/UTC chronology; return None or raise ValueError."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Complete trial table must be a nonempty DataFrame")
    _require_columns(table, {"cur_trial", "start_time", "choice_time", "state_int"})
    ids = _numeric(table["cur_trial"])
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("cur_trial must contain unique finite integer trial IDs")
    starts = _numeric(table["start_time"])
    if not np.isfinite(starts).all() or not np.all(np.diff(starts) > 0):
        raise ValueError("start_time must contain finite strictly chronological UTC seconds")
    choices = _numeric(table["choice_time"])
    present = is_present_value(table["choice_time"]).to_numpy()
    if np.any(present & ~np.isfinite(choices)) or np.any(choices < starts):
        raise ValueError("choice_time must be missing or finite UTC seconds at/after start_time")


def _state_order(states: np.ndarray) -> list[int]:
    """Compress repeated (N,) valid dimensionless state codes to ordered int list."""
    if len(states) == 0:
        return []
    return states[np.r_[True, states[1:] != states[:-1]]].astype(int).tolist()
