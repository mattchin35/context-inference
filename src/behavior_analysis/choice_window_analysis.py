"""Greedy mouse-agent agreement in non-overlapping valid-choice windows.

Agent predictions retain full-session history. Only aggregation uses windows;
context block membership and the input trial table are never modified.
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd

from .project_utils import normalize_experimenter_reward_column
from .session_analysis import (
    DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS,
    compute_agent_mouse_agreement_by_trial,
    make_behavioral_choice_mask,
)


ANALYSIS_VERSION = "1.0.0"
RESULT_DTYPES = {
    "session_id": "str", "window_size": "int64", "window_index": "int64",
    "choice_start": "int64", "choice_stop": "int64",
    "trial_start": "int64", "trial_stop": "int64", "n_valid_choices": "int64",
    "first_context_block": "str", "last_context_block": "str",
    "n_context_transitions": "int64", "agent": "str",
    "n_comparisons": "int64", "n_agreements": "int64", "agreement": "float64",
}
RESULT_COLUMNS = tuple(RESULT_DTYPES)


def valid_choice_positions(trial_df: pd.DataFrame) -> np.ndarray:
    """Validate one session and return eligible original row positions.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Chronological recorded trials. ``action`` uses 0=right, 1=left, or
        existing no-choice sentinels. Optional reward flags mark manual
        rewards. Optional ``session_ID``/``session_id`` must identify at most
        one session. DataFrame index labels are not trial coordinates.

    Returns
    -------
    numpy.ndarray, shape (n_valid_choices,), dtype int64
        Increasing zero-based positional indices, in recorded-trial units.
        No-choice and manual-reward rows are excluded.
    """
    if "action" not in trial_df:
        raise ValueError("trial_df is missing required action column.")
    for column in ("session_ID", "session_id", "source_session_id"):
        if column in trial_df and trial_df[column].dropna().nunique() > 1:
            raise ValueError("Choice windows require one session per call.")
    normalized = normalize_experimenter_reward_column(trial_df).reset_index(drop=True)
    valid = make_behavioral_choice_mask(normalized)
    actions = pd.to_numeric(normalized["action"], errors="coerce")
    if (valid & ~actions.isin([0, 1]).to_numpy()).any():
        raise ValueError("Retained action values must be numeric 0=right or 1=left.")
    return np.flatnonzero(valid)


def resolve_agent_value_columns(
    trial_df: pd.DataFrame,
    agent_value_columns: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve available default agents or validate an explicit mapping.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Trial table containing signed left-positive agent decision values.
    agent_value_columns : Mapping[str, str] or None
        Output agent identifiers mapped to input value columns. None selects
        available members of the existing block-agreement registry.

    Returns
    -------
    dict[str, str]
        Nonempty agent-to-value-column mapping preserving registry/input order.
        Explicit missing columns raise ValueError rather than disappearing.
    """
    if agent_value_columns is None:
        selected = {
            agent: column for agent, column in DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS.items()
            if column in trial_df
        }
    else:
        selected = dict(agent_value_columns)
    if not selected:
        raise ValueError("No agent value columns are available for choice-window analysis.")
    missing = [column for column in selected.values() if column not in trial_df]
    if missing:
        raise ValueError(f"trial_df is missing requested agent value columns: {missing}")
    return selected


def compute_trial_agent_agreements(
    trial_df: pd.DataFrame,
    agent_value_columns: Mapping[str, str] | None = None,
    initial_tie_choice: int = 0,
) -> pd.DataFrame:
    """Score every agent over the full session before any window aggregation.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Chronological single-session data satisfying ``valid_choice_positions``.
        Agent values are unitless, signed and left-positive. Nonnumeric or
        missing predictions have no comparison; infinite predictions are errors.
    agent_value_columns : Mapping[str, str] or None
        Agent identifiers mapped to value columns; None uses available defaults.
    initial_tie_choice : int, default=0
        First retained tie choice, 0=right or 1=left. Later ties repeat the
        previous greedy choice, including across context/window boundaries.

    Returns
    -------
    pandas.DataFrame, shape (n_trials, n_agents)
        Range-indexed numeric scores aligned by position to original trials.
        Values are 0.0, 1.0, or NaN; fractions are unitless. Input is unchanged.
    """
    valid_choice_positions(trial_df)
    if initial_tie_choice not in (0, 1):
        raise ValueError("initial_tie_choice must be 0=right or 1=left.")
    mapping = resolve_agent_value_columns(trial_df, agent_value_columns)
    normalized = normalize_experimenter_reward_column(trial_df).reset_index(drop=True)
    scores = {}
    for agent, column in mapping.items():
        values = pd.to_numeric(normalized[column], errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Agent column {column} must contain finite values or missing predictions.")
        agreement = compute_agent_mouse_agreement_by_trial(normalized, column, initial_tie_choice)
        scores[agent] = pd.to_numeric(pd.Series(agreement), errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(scores, index=pd.RangeIndex(len(trial_df)))


def validate_window_size(window_size: int) -> int:
    """Return a positive integer size in choices; reject bool/noninteger inputs.

    Parameters
    ----------
    window_size : int
        Requested valid mouse-choice count per window.

    Returns
    -------
    int
        Validated size, in valid mouse choices.
    """
    if isinstance(window_size, (bool, np.bool_)) or not isinstance(window_size, Integral) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    return int(window_size)


def summarize_choice_window_agreement(
    trial_df: pd.DataFrame,
    trial_agreements: pd.DataFrame,
    window_size: int,
    session_id: str,
) -> pd.DataFrame:
    """Aggregate aligned scores over full non-overlapping valid-choice windows.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Single-session chronological trials. Optional ``cur_block`` is used
        only for context diagnostics over the original recorded-trial span.
    trial_agreements : pandas.DataFrame, shape (n_trials, n_agents)
        Positional scores from ``compute_trial_agent_agreements``: 0/1/NaN.
    window_size : int
        Valid choices per window. The last incomplete window is discarded.
    session_id : str
        Source session identifier copied into every result row.

    Returns
    -------
    pandas.DataFrame, shape (n_full_windows * n_agents, len(RESULT_COLUMNS))
        Long-form result with fixed dtypes. Choice and original trial bounds
        are zero-based and half-open. Counts are choices; agreement is a
        unitless fraction or NaN when n_comparisons=0. Context IDs are strings;
        unavailable context diagnostics use empty IDs and transition count -1.
        No input shapes, ordering, or values are changed.
    """
    size = validate_window_size(window_size)
    positions = valid_choice_positions(trial_df)
    if len(trial_agreements) != len(trial_df) or trial_agreements.shape[1] == 0:
        raise ValueError("trial_agreements must have shape (n_trials, n_agents) with at least one agent.")
    scores = trial_agreements.to_numpy(dtype=float)
    if not (np.isnan(scores) | (scores == 0) | (scores == 1)).all():
        raise ValueError("trial_agreements must contain only 0, 1, or NaN.")
    full_choice_count = len(positions) // size * size
    members = positions[:full_choice_count].reshape(-1, size)
    rows = []
    for window_index, trial_positions in enumerate(members):
        start, stop = int(trial_positions[0]), int(trial_positions[-1]) + 1
        first_context, last_context, transitions = "", "", -1
        if "cur_block" in trial_df:
            contexts = trial_df["cur_block"].iloc[start:stop]
            if contexts.notna().all() and not contexts.astype(str).isin(["None", ""]).any():
                context_values = contexts.astype(str).to_numpy()
                first_context, last_context = context_values[0], context_values[-1]
                transitions = int(np.count_nonzero(context_values[1:] != context_values[:-1]))
        for agent_index, agent in enumerate(trial_agreements.columns):
            values = scores[trial_positions, agent_index]
            count = int(np.count_nonzero(~np.isnan(values)))
            agreements = int(np.nansum(values))
            rows.append({
                "session_id": str(session_id), "window_size": size, "window_index": window_index,
                "choice_start": window_index * size, "choice_stop": (window_index + 1) * size,
                "trial_start": start, "trial_stop": stop, "n_valid_choices": size,
                "first_context_block": first_context, "last_context_block": last_context,
                "n_context_transitions": transitions, "agent": str(agent),
                "n_comparisons": count, "n_agreements": agreements,
                "agreement": agreements / count if count else np.nan,
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).astype(RESULT_DTYPES)


def summarize_choice_window_sessions(window_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize window agreement separately by session, size, and agent.

    Parameters
    ----------
    window_table : pandas.DataFrame, shape (n_window_agent_rows, n_columns)
        Collected window outputs with session_id, date, window_size, agent,
        window_index, agreement, n_comparisons, and n_agreements. Agreement is
        a unitless fraction or NaN; counts are valid mouse choices. Multiple
        sessions may share a date and remain distinct groups.

    Returns
    -------
    pandas.DataFrame, shape (n_session_size_agent_groups, 12)
        Group identifiers, total/scored window counts, summed comparison and
        agreement counts, window agreement median/Q1/Q3, and count-weighted
        pooled agreement. Missing scores do not enter quartiles; a group with
        no comparisons has NaN agreement summaries. Input is not modified.
    """
    keys = ["session_id", "date", "window_size", "agent"]
    required = {*keys, "window_index", "agreement", "n_comparisons", "n_agreements"}
    missing = sorted(required.difference(window_table.columns))
    if missing:
        raise ValueError(f"window_table is missing session-summary columns: {missing}")
    summary_input = window_table.loc[:, [
        *keys, "window_index", "agreement", "n_comparisons", "n_agreements",
    ]].copy()
    # Empty collections may have object-typed columns; quartiles still require
    # a numeric agreement axis even when there are no observations to reduce.
    summary_input["agreement"] = summary_input["agreement"].astype(float)
    groups = summary_input.groupby(keys, sort=False)
    summary = groups.agg(
        n_windows=("window_index", "size"), n_scored_windows=("agreement", "count"),
        n_comparisons=("n_comparisons", "sum"), n_agreements=("n_agreements", "sum"),
        agreement_median=("agreement", "median"),
    )
    summary["agreement_q1"] = groups["agreement"].quantile(0.25)
    summary["agreement_q3"] = groups["agreement"].quantile(0.75)
    summary["agreement_pooled"] = summary["n_agreements"] / summary["n_comparisons"].replace(0, np.nan)
    return summary.reset_index()
