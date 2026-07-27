"""General session-level behavioral performance metrics."""

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import (
    get_experimenter_reward_flags,
    is_zero_flag,
    make_no_choice_action_mask,
    normalize_experimenter_reward_column,
)

MISSING_SUMMARY_VALUE = "None"


def make_valid_behavior_mask(trial_df: pd.DataFrame) -> np.ndarray:
    """Return rows that contain animal choices used for behavior summaries.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required column
        is `action`. Optional columns `experimenter_reward_given` or legacy
        `give_reward` mark manual rewards; missing reward flags are treated as
        all-zero.

    Returns
    -------
    np.ndarray
        Boolean array with shape `(n_trials,)`. True marks trials with an
        animal left/right choice and no experimenter-delivered reward.
    """
    if "action" not in trial_df.columns:
        raise ValueError("trial_df must contain an 'action' column.")

    normalized = normalize_experimenter_reward_column(trial_df)
    experimenter_reward_given = get_experimenter_reward_flags(normalized, default_zero=True)
    no_choice = make_no_choice_action_mask(normalized["action"]).to_numpy()
    return is_zero_flag(experimenter_reward_given).to_numpy() & ~no_choice


def numeric_valid_values(
    trial_df: pd.DataFrame,
    column_name: str,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return one numeric column restricted to valid trials.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.
    column_name : str
        Name of the column to extract. Values are unitless trial outcomes or
        probabilities depending on the column.
    valid_mask : np.ndarray
        Boolean mask with shape `(n_trials,)`; True rows are retained.

    Returns
    -------
    np.ndarray
        Float array with shape `(n_valid_trials,)`. Invalid numeric values in
        retained rows raise a ValueError instead of silently becoming zeros.
    """
    if column_name not in trial_df.columns:
        raise ValueError(f"trial_df must contain a '{column_name}' column.")
    if valid_mask.shape[0] != trial_df.shape[0]:
        raise ValueError(
            "valid_mask must have one entry per trial: "
            f"got {valid_mask.shape[0]} for {trial_df.shape[0]} trials."
        )

    retained_values = trial_df.loc[valid_mask, column_name]
    numeric_values = pd.to_numeric(retained_values, errors="coerce")
    invalid_values = retained_values.loc[numeric_values.isna()]
    if not invalid_values.empty:
        invalid_summary = sorted(invalid_values.astype(str).unique())
        raise ValueError(f"{column_name} values must be numeric, got: {invalid_summary}")
    return numeric_values.to_numpy(dtype=float)


def compute_oracle_choice_accuracy(trial_df: pd.DataFrame) -> float | str:
    """Return the fraction of valid trials where the animal chose the oracle side.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action` and `correct`; optional experimenter-reward columns
        are used to exclude manual rewards. `correct` is unitless and encoded
        as 0/1 or numeric strings.

    Returns
    -------
    float or str
        Mean correct value over valid trials, unitless. Returns `"None"` when
        there are no valid animal-choice trials.
    """
    valid_mask = make_valid_behavior_mask(trial_df)
    if not valid_mask.any():
        return MISSING_SUMMARY_VALUE
    correct = numeric_valid_values(trial_df, "correct", valid_mask)
    return float(np.mean(correct))


def compute_actual_reward_collected(trial_df: pd.DataFrame) -> float | str:
    """Return total animal-earned reward over valid trials.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action` and `reward`; optional experimenter-reward columns
        are used to exclude manual rewards. `reward` is in task reward units.

    Returns
    -------
    float or str
        Sum of `reward` over valid trials, in task reward units. Returns
        `"None"` when there are no valid animal-choice trials.
    """
    valid_mask = make_valid_behavior_mask(trial_df)
    if not valid_mask.any():
        return MISSING_SUMMARY_VALUE
    reward = numeric_valid_values(trial_df, "reward", valid_mask)
    return float(np.sum(reward))


def compute_oracle_expected_reward(trial_df: pd.DataFrame) -> float | str:
    """Return expected reward if the oracle side were chosen on valid trials.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action` and `p_active_rew`; optional experimenter-reward
        columns are used to exclude manual rewards. `p_active_rew` is the active
        side reward probability for each trial, unitless in `[0, 1]`.

    Returns
    -------
    float or str
        Sum of active reward probabilities over valid trials, in expected
        reward units. Returns `"None"` when there are no valid trials.
    """
    valid_mask = make_valid_behavior_mask(trial_df)
    if not valid_mask.any():
        return MISSING_SUMMARY_VALUE
    p_active_reward = numeric_valid_values(trial_df, "p_active_rew", valid_mask)
    return float(np.sum(p_active_reward))


def summarize_oracle_behavior(trial_df: pd.DataFrame) -> dict:
    """Summarize oracle choice and reward-collection performance for one session.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required columns
        are `action`, `correct`, and `reward`. Optional columns
        `experimenter_reward_given` or legacy `give_reward` exclude manual
        rewards. Optional `p_active_rew` supplies the active-side reward
        probability needed for oracle expected reward.

    Returns
    -------
    dict
        Session-level scalar summary with keys:
        - `oracle_choice_accuracy` : float or `"None"`, unitless fraction
        - `actual_reward_collected` : float or `"None"`, task reward units
        - `oracle_expected_reward` : float or `"None"`, expected reward units
        - `oracle_reward_fraction` : float or `"None"`, actual / expected
        - `oracle_reward_difference` : float or `"None"`, actual - expected
    """
    valid_mask = make_valid_behavior_mask(trial_df)
    if not valid_mask.any():
        return {
            "oracle_choice_accuracy": MISSING_SUMMARY_VALUE,
            "actual_reward_collected": MISSING_SUMMARY_VALUE,
            "oracle_expected_reward": MISSING_SUMMARY_VALUE,
            "oracle_reward_fraction": MISSING_SUMMARY_VALUE,
            "oracle_reward_difference": MISSING_SUMMARY_VALUE,
        }

    correct = numeric_valid_values(trial_df, "correct", valid_mask)
    reward = numeric_valid_values(trial_df, "reward", valid_mask)
    oracle_choice_accuracy = float(np.mean(correct))
    actual_reward_collected = float(np.sum(reward))

    if "p_active_rew" not in trial_df.columns:
        return {
            "oracle_choice_accuracy": oracle_choice_accuracy,
            "actual_reward_collected": actual_reward_collected,
            "oracle_expected_reward": MISSING_SUMMARY_VALUE,
            "oracle_reward_fraction": MISSING_SUMMARY_VALUE,
            "oracle_reward_difference": MISSING_SUMMARY_VALUE,
        }

    p_active_reward = numeric_valid_values(trial_df, "p_active_rew", valid_mask)
    oracle_expected_reward = float(np.sum(p_active_reward))
    oracle_reward_difference = actual_reward_collected - oracle_expected_reward
    if oracle_expected_reward > 0:
        oracle_reward_fraction = actual_reward_collected / oracle_expected_reward
    else:
        oracle_reward_fraction = MISSING_SUMMARY_VALUE

    return {
        "oracle_choice_accuracy": oracle_choice_accuracy,
        "actual_reward_collected": actual_reward_collected,
        "oracle_expected_reward": oracle_expected_reward,
        "oracle_reward_fraction": oracle_reward_fraction,
        "oracle_reward_difference": oracle_reward_difference,
    }


def summarize_rewarded_choice_switching(trial_df: pd.DataFrame) -> dict:
    """Summarize next-trial switching after rewarded mouse choices.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action` and `reward`; optional `experimenter_reward_given`
        or legacy `give_reward` columns exclude manual rewards. Choice
        convention is `0` right and `1` left. Reward values are in task reward
        units, and `reward > 0` marks a rewarded trial.

    Returns
    -------
    dict
        Session-level scalar summary with keys:
        - `rewarded_choice_switch_probability` : float or `"None"`, fraction
          of eligible rewarded adjacent trial pairs where `action[t+1]`
          differs from `action[t]`
        - `rewarded_choice_switch_n_trials` : int, number of eligible rewarded
          adjacent trial pairs
    """
    valid_mask = make_valid_behavior_mask(trial_df)
    if valid_mask.shape[0] < 2:
        return {
            "rewarded_choice_switch_probability": MISSING_SUMMARY_VALUE,
            "rewarded_choice_switch_n_trials": 0,
        }

    reward = numeric_valid_values(trial_df, "reward", valid_mask)
    full_reward = np.full(trial_df.shape[0], np.nan, dtype=float)
    full_reward[valid_mask] = reward

    actions = pd.to_numeric(trial_df["action"], errors="coerce").to_numpy(dtype=float)
    eligible_pairs = valid_mask[:-1] & valid_mask[1:] & (full_reward[:-1] > 0)
    n_eligible = int(np.sum(eligible_pairs))
    if n_eligible == 0:
        return {
            "rewarded_choice_switch_probability": MISSING_SUMMARY_VALUE,
            "rewarded_choice_switch_n_trials": 0,
        }

    switched = actions[1:][eligible_pairs] != actions[:-1][eligible_pairs]
    return {
        "rewarded_choice_switch_probability": float(np.mean(switched)),
        "rewarded_choice_switch_n_trials": n_eligible,
    }
