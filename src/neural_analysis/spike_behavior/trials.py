"""Trial validity, outcome, choice-side, switch, and stay classification."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    get_experimenter_reward_flags,
    normalize_experimenter_reward_column,
)

def _numeric_present(values: pd.Series) -> pd.Series:
    """Return rows with numeric, non-missing values."""
    return pd.to_numeric(values, errors="coerce").notna()

def select_valid_lick_peth_trials(
    trial_df: pd.DataFrame,
    require_led_time: bool = True,
) -> pd.Series:
    """
    Select trials suitable for lick PETH plots.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Required columns are
        ``start_time`` and ``choice_time`` in seconds. ``led_on_time`` in seconds is required
        when ``require_led_time=True``. Legacy ``give_reward`` and current
        ``experimenter_reward_given`` reward flags are accepted.
    require_led_time : bool, default=True
        Whether selected trials must have numeric ``led_on_time`` values in seconds.

    Returns
    -------
    pd.Series
        Boolean mask with shape ``(n_trials,)`` indexed like ``trial_df``. True rows have no
        experimenter reward and numeric alignment/event times.
    """

    required_columns = {"start_time", "choice_time"}
    if require_led_time:
        required_columns.add("led_on_time")
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    experimenter_reward_flags = get_experimenter_reward_flags(trial_df, default_zero=True)
    valid_mask = pd.Series(experimenter_reward_flags == 0, index=trial_df.index)
    valid_mask &= _numeric_present(trial_df["start_time"])
    valid_mask &= _numeric_present(trial_df["choice_time"])
    if require_led_time:
        valid_mask &= _numeric_present(trial_df["led_on_time"])
    return valid_mask

def make_lick_peth_trial_type_masks(
    trial_df: pd.DataFrame,
    require_led_time: bool = True,
) -> dict[str, pd.Series]:
    """
    Build trial-type masks for lick PETH plots.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Required columns are ``start_time``,
        ``choice_time``, ``correct``, ``reward``, and ``action``. ``led_on_time`` is required
        when ``require_led_time=True``. Times are in seconds. Choices use the project convention
        ``0=right`` and ``1=left``.
    require_led_time : bool, default=True
        Whether selected trials must have numeric ``led_on_time`` values in seconds.

    Returns
    -------
    dict[str, pd.Series]
        Boolean masks indexed like ``trial_df``. Current keys are ``valid``, side-specific
        correct, incorrect, omission, switch, and stay masks. Correct masks include only
        correct rewarded trials. Switch/stay masks label the current unrewarded trial based
        on the next valid row's action, matching ``make_trial_type_masks``.
    """

    required_columns = {"correct", "reward", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    valid = select_valid_lick_peth_trials(trial_df, require_led_time=require_led_time)
    correct = pd.to_numeric(trial_df["correct"], errors="coerce").eq(1)
    rewarded = pd.to_numeric(trial_df["reward"], errors="coerce").eq(1)
    action = pd.to_numeric(trial_df["action"], errors="coerce")

    left_action = action.eq(1)
    right_action = action.eq(0)
    correct_rewarded = valid & correct & rewarded
    incorrect = valid & pd.to_numeric(trial_df["correct"], errors="coerce").eq(0) & action.notna()
    omission = valid & correct & pd.to_numeric(trial_df["reward"], errors="coerce").eq(0)

    current_unrewarded = valid & pd.to_numeric(trial_df["reward"], errors="coerce").eq(0)
    current_action_valid = action.notna()
    next_valid = valid.shift(-1).fillna(False)
    next_action = action.shift(-1)
    next_action_valid = next_action.notna()
    comparable_next_trial = current_unrewarded & current_action_valid & next_valid & next_action_valid
    switch = comparable_next_trial & next_action.ne(action)
    stay = comparable_next_trial & next_action.eq(action)

    return {
        "valid": valid,
        "left_correct": correct_rewarded & left_action,
        "right_correct": correct_rewarded & right_action,
        "left_incorrect": incorrect & left_action,
        "right_incorrect": incorrect & right_action,
        "left_omission": omission & left_action,
        "right_omission": omission & right_action,
        "left_switch": switch & left_action,
        "right_switch": switch & right_action,
        "left_stay": stay & left_action,
        "right_stay": stay & right_action,
    }

def resolve_trial_end(trial_row: pd.Series) -> float:
    """
    Resolve a trial end time using the legacy reward-choice-start priority order.

    Parameters
    ----------
    trial_row : pd.Series
        Trial metadata row containing ``start_time``, ``choice_time``, and ``reward_time``.
        Time values are scalar floats in seconds.

    Returns
    -------
    float
        Trial end time in seconds.
    """

    reward_time = float(trial_row["reward_time"]) if pd.notna(trial_row["reward_time"]) else np.nan
    choice_time = float(trial_row["choice_time"]) if pd.notna(trial_row["choice_time"]) else np.nan
    start_time = float(trial_row["start_time"])

    if not np.isnan(reward_time):
        return reward_time
    if not np.isnan(choice_time):
        return choice_time
    return start_time

def make_trial_type_masks(trial_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Build boolean trial masks for simple neural-behavior analyses.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``experimenter_reward_given``, ``correct``, ``reward``, and ``action``.
        Legacy ``give_reward`` columns are accepted and normalized at this
        boundary. Nonzero experimenter-reward values mark trials invalid for
        neural analysis. ``correct`` and ``reward`` are scalar trial outcomes
        with no unit conversion. ``action`` is a scalar choice label.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary of boolean masks indexed like ``trial_df``. Keys are:
        ``valid``,
        ``correct_rewarded``,
        ``rewarded``,
        ``incorrect``,
        ``omission``,
        ``switch``,
        ``stay``,
        ``omission_switch``,
        ``omission_stay``,
        ``incorrect_switch``,
        ``incorrect_stay``.
    """

    trial_df = normalize_experimenter_reward_column(trial_df)
    required_columns = {EXPERIMENTER_REWARD_GIVEN_COLUMN, "correct", "reward", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    valid = trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN].eq(0)
    correct_rewarded = valid & trial_df["correct"].eq(1) & trial_df["reward"].eq(1)
    incorrect = valid & trial_df["correct"].eq(0) & trial_df["action"].notna()
    omission = valid & trial_df["correct"].eq(1) & trial_df["reward"].eq(0)

    current_unrewarded = valid & trial_df["reward"].eq(0)
    current_action_valid = trial_df["action"].notna()
    next_valid = trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN].shift(-1).eq(0).fillna(False)
    next_action = trial_df["action"].shift(-1)
    next_action_valid = next_action.notna()
    comparable_next_trial = current_unrewarded & current_action_valid & next_valid & next_action_valid
    switch = comparable_next_trial & next_action.ne(trial_df["action"])
    stay = comparable_next_trial & next_action.eq(trial_df["action"])

    masks = {
        "valid": valid,
        "correct_rewarded": correct_rewarded,
        "rewarded": correct_rewarded,
        "incorrect": incorrect,
        "omission": omission,
        "switch": switch,
        "stay": stay,
    }
    masks["omission_switch"] = masks["omission"] & masks["switch"]
    masks["omission_stay"] = masks["omission"] & masks["stay"]
    masks["incorrect_switch"] = masks["incorrect"] & masks["switch"]
    masks["incorrect_stay"] = masks["incorrect"] & masks["stay"]
    return masks

def summarize_trial_masks(
    trial_masks: Mapping[str, pd.Series],
    condition_names: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Summarize selected trial masks in a user-specified condition order.

    Parameters
    ----------
    trial_masks : Mapping[str, pd.Series]
        Mapping from condition name to boolean trial mask. Each mask must be one-dimensional and
        indexed like the trial table.
    condition_names : list[str]
        Ordered list of condition names to summarize.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        ``(summary_df, condition_trial_indices)`` where ``summary_df`` has one row per requested
        condition with columns ``condition`` and ``n_trials``, and ``condition_trial_indices``
        maps each condition name to a one-dimensional integer array of selected trial indices.
    """

    summary_rows: list[dict[str, Any]] = []
    condition_trial_indices: dict[str, np.ndarray] = {}
    for condition_name in condition_names:
        if condition_name not in trial_masks:
            raise ValueError(f"Requested condition {condition_name!r} is not present in trial_masks.")
        trial_indices = np.flatnonzero(np.asarray(trial_masks[condition_name], dtype=bool))
        condition_trial_indices[condition_name] = trial_indices
        summary_rows.append(
            {
                "condition": condition_name,
                "n_trials": int(trial_indices.size),
            }
        )

    return pd.DataFrame(summary_rows), condition_trial_indices
