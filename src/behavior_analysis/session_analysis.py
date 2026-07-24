from pathlib import Path
import pandas as pd
import numpy as np
import re
import warnings
from collections import defaultdict
from src.behavior_analysis import block_exemplar_models
from src.behavior_analysis import block_residual_models
from src.behavior_analysis import general_behavior_assessment
from src.behavior_analysis import ideal_observer
from src.behavior_analysis import switch_persistence
import src.behavior_analysis.decision_variable_counters as counters
from src.behavior_analysis.project_utils import (
    get_experimenter_reward_flags,
    is_present_value,
    is_zero_flag,
    make_no_choice_action_mask,
    normalize_experimenter_reward_column,
)
from formulaic import model_matrix
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Protocol

eps = np.finfo(float).eps

DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS = {
    "qlearning_mouse_agreement": "Qlearning_rel_value",
    "fql_mouse_agreement": "FQlearning_rel_value",
    "hmm_logodds_mouse_agreement": "HMM_rel_value_logodds",
    "hmm_logodds_decay_mouse_agreement": "HMM_rel_value_logodds_decay",
    "perseveration_mouse_agreement": "perseveration_regressor",
    "doubt_perseveration_mouse_agreement": "doubt_perseveration_value",
    "wsls_mouse_agreement": "wsls_regressor",
    "observer_mouse_agreement": "observer_value",
}


class Session(Protocol):
    multi_session_save_path: Path
    session_data_home: Path
    sess_id_full: str
    raw_behavior_folder: Path
    processed_data_path: Path
    figure_path: Path
    mouse: str
    date: str
    timestamp: str
    session_info_fname: str
    session_info: dict


@dataclass
class DecisionVariableState:
    """Mutable trial-history counters used to build decision variables."""
    consecutive_rewards: int = 0
    consecutive_omissions: int = 0
    consecutive_rewards_memory: int = 0
    consecutive_omissions_memory: int = 0
    negative_value: int = 0
    previous_trial_rewarded: bool = False
    left_value: int = 0
    right_value: int = 0
    left_omissions: int = 0
    right_omissions: int = 0
    left_value_cf: int = 0
    right_value_cf: int = 0
    left_omissions_cf: int = 0
    right_omissions_cf: int = 0
    left_monotonic_cf_value: int = 0
    right_monotonic_cf_value: int = 0


def _get_normalized_state_labels(trial_df: pd.DataFrame) -> np.ndarray:
    """Normalize real-session and simulated state labels to `left`/`right`.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table containing a `state` column. Real sessions use
        `left_patch`/`right_patch`; simulated runs use `left`/`right`.

    Returns
    -------
    np.ndarray
        Normalized state labels with shape `(n_trials,)` and values drawn from
        `{"left", "right"}` where possible. Unknown labels are preserved so
        downstream boolean masks simply evaluate to False.
    """
    normalized_states = trial_df["state"].astype(str).to_numpy(dtype=object)
    normalized_states[normalized_states == "left_patch"] = "left"
    normalized_states[normalized_states == "right_patch"] = "right"
    return normalized_states


def _get_uncued_block_mask(trial_df: pd.DataFrame) -> np.ndarray:
    """Return the uncued-block mask for real or simulated trial schemas.

    Adapter logic:
    - real sessions: uncued blocks have null-like `block_stimulus`
    - simulated runs: uncued blocks have `model_stimulus == -1`

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table containing either `block_stimulus` or `model_stimulus`.

    Returns
    -------
    np.ndarray
        Boolean mask with shape `(n_trials,)`, where True marks uncued trials.
    """
    if "block_stimulus" in trial_df.columns:
        stimulus = trial_df["block_stimulus"]
        return stimulus.isnull().to_numpy() | stimulus.astype(str).eq("None").to_numpy()

    if "model_stimulus" in trial_df.columns:
        stimulus = trial_df["model_stimulus"]
        return stimulus.isnull().to_numpy() | stimulus.astype(str).eq("-1").to_numpy()

    raise ValueError(
        "trial_df must contain either 'block_stimulus' (real sessions) or "
        "'model_stimulus' (simulated runs)."
    )


def _get_experimenter_reward_given_array(trial_df: pd.DataFrame) -> np.ndarray:
    """Return experimenter-reward flags, defaulting missing values to zero.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table that may or may not contain an experimenter-reward column.

    Returns
    -------
    np.ndarray
        Array of shape `(n_trials,)`. Missing columns are treated as zeros,
        which is the correct behavior for simulated runs.
    """
    return get_experimenter_reward_flags(trial_df, default_zero=True)


def _get_choice_latency(trial_df: pd.DataFrame) -> np.ndarray:
    """Return trialwise choice latencies or NaN when timing columns are absent."""
    if {"choice_time", "start_time"}.issubset(trial_df.columns):
        choice_latency = np.full(trial_df.shape[0], np.nan, dtype=float)
        present_choice_time = is_present_value(trial_df["choice_time"])
        if present_choice_time.any():
            choice_latency[present_choice_time.to_numpy()] = (
                trial_df.loc[present_choice_time, "choice_time"].astype(float).to_numpy()
                - trial_df.loc[present_choice_time, "start_time"].astype(float).to_numpy()
            )
        return choice_latency
    return np.full(trial_df.shape[0], np.nan, dtype=float)


def get_block_types(trial_df: pd.DataFrame) -> np.array:
    """Assign block types for real or simulated runs.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table containing a `state` column and either:
        - `block_stimulus` for real sessions
        - `model_stimulus` for simulated runs

    Returns
    -------
    np.ndarray
        Block-type labels with shape `(n_trials,)`, values in
        `{"left_cued", "left_uncued", "right_cued", "right_uncued"}`.
    """
    normalized_states = _get_normalized_state_labels(trial_df)
    uncued_block = _get_uncued_block_mask(trial_df)
    cued_block = ~uncued_block
    left_cued_ix = (normalized_states == 'left') & cued_block
    left_uncued_ix = (normalized_states == 'left') & uncued_block
    right_cued_ix = (normalized_states == 'right') & cued_block
    right_uncued_ix = (normalized_states == 'right') & uncued_block

    block_types = np.zeros(trial_df.shape[0], dtype=object)
    block_types[left_cued_ix] = 'left_cued'
    block_types[right_cued_ix] = 'right_cued'
    block_types[left_uncued_ix] = 'left_uncued'
    block_types[right_uncued_ix] = 'right_uncued'
    return block_types


def get_numeric_correct_values(
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Return numeric correctness values, validating rows used for summaries.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required column is
        `correct`, with values encoded as numeric 0/1, booleans, or string
        equivalents such as `"0"`, `"1"`, `"True"`, and `"False"`.
    valid_mask : np.ndarray, pd.Series, or None
        Boolean mask with shape `(n_trials,)`. Rows marked True must contain a
        valid correctness value. If None, all rows are validated.

    Returns
    -------
    np.ndarray
        Float array with shape `(n_trials,)`; correct choices are 1.0 and
        incorrect choices are 0.0. Rows outside `valid_mask` may contain NaN if
        their source value was missing.
    """
    if "correct" not in trial_df.columns:
        raise ValueError("trial_df must contain a 'correct' column.")

    if valid_mask is None:
        valid_mask_array = np.ones(trial_df.shape[0], dtype=bool)
    else:
        valid_mask_array = np.asarray(valid_mask, dtype=bool)
        if valid_mask_array.shape[0] != trial_df.shape[0]:
            raise ValueError(
                "valid_mask must have one entry per trial: "
                f"got {valid_mask_array.shape[0]} for {trial_df.shape[0]} trials."
            )

    correct_values = trial_df["correct"].copy()
    text_values = correct_values.astype(str).str.strip().str.lower()
    normalized_values = correct_values.mask(text_values == "true", 1)
    normalized_values = normalized_values.mask(text_values == "false", 0)
    numeric_values = pd.to_numeric(normalized_values, errors="coerce")

    invalid_mask = valid_mask_array & numeric_values.isna().to_numpy()
    if invalid_mask.any():
        invalid_values = sorted(correct_values.loc[invalid_mask].astype(str).unique())
        raise ValueError(f"correct values must be numeric 0/1 for behavioral choices, got: {invalid_values}")

    return numeric_values.to_numpy(dtype=float)


def get_numeric_reward_values(
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Return numeric reward values, validating rows used for summaries.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required column is
        `reward`, with values encoded as numeric 0/1, booleans, or string
        equivalents such as `"0"`, `"1"`, `"True"`, and `"False"`.
    valid_mask : np.ndarray, pd.Series, or None
        Boolean mask with shape `(n_trials,)`. Rows marked True must contain a
        valid reward value. If None, all rows are validated.

    Returns
    -------
    np.ndarray
        Float array with shape `(n_trials,)`; rewarded trials are 1.0 and
        unrewarded trials are 0.0. Rows outside `valid_mask` may contain NaN if
        their source value was missing.
    """
    if "reward" not in trial_df.columns:
        raise ValueError("trial_df must contain a 'reward' column.")

    if valid_mask is None:
        valid_mask_array = np.ones(trial_df.shape[0], dtype=bool)
    else:
        valid_mask_array = np.asarray(valid_mask, dtype=bool)
        if valid_mask_array.shape[0] != trial_df.shape[0]:
            raise ValueError(
                "valid_mask must have one entry per trial: "
                f"got {valid_mask_array.shape[0]} for {trial_df.shape[0]} trials."
            )

    reward_values = trial_df["reward"].copy()
    text_values = reward_values.astype(str).str.strip().str.lower()
    normalized_values = reward_values.mask(text_values == "true", 1)
    normalized_values = normalized_values.mask(text_values == "false", 0)
    numeric_values = pd.to_numeric(normalized_values, errors="coerce")

    invalid_mask = valid_mask_array & numeric_values.isna().to_numpy()
    if invalid_mask.any():
        invalid_values = sorted(reward_values.loc[invalid_mask].astype(str).unique())
        raise ValueError(f"reward values must be numeric 0/1 for behavioral choices, got: {invalid_values}")

    return numeric_values.to_numpy(dtype=float)


def percent_correct(augmented_trial_df: pd.DataFrame) -> dict:
    """Calculate the percentage of correct choices made by the agent. Calculate for
    left uncued, right uncued, left cued, and right cued."""
    assert 'block_type' in augmented_trial_df.keys(), "'block_type' key was not found in dataframe."
    assert 'correct' in augmented_trial_df.keys(), "'correct' key was not found in dataframe."

    behavioral_choice_ix = make_behavioral_choice_mask(augmented_trial_df)
    correct_values = get_numeric_correct_values(augmented_trial_df, behavioral_choice_ix)
    left_cued_ix = (augmented_trial_df['block_type'] == 'left_cued') & behavioral_choice_ix
    left_uncued_ix = (augmented_trial_df['block_type'] == 'left_uncued') & behavioral_choice_ix
    right_cued_ix = (augmented_trial_df['block_type'] == 'right_cued') & behavioral_choice_ix
    right_uncued_ix = (augmented_trial_df['block_type'] == 'right_uncued') & behavioral_choice_ix

    if left_cued_ix.sum():
        left_cued_correct = np.sum(correct_values[np.asarray(left_cued_ix)]) / np.sum(left_cued_ix)
    else:
        left_cued_correct = 'None'

    if left_uncued_ix.sum():
        left_uncued_correct = np.sum(correct_values[np.asarray(left_uncued_ix)]) / np.sum(left_uncued_ix)
    else:
        left_uncued_correct = 'None'

    if right_cued_ix.sum():
        right_cued_correct = np.sum(correct_values[np.asarray(right_cued_ix)]) / np.sum(right_cued_ix)
    else:
        right_cued_correct = 'None'

    if right_uncued_ix.sum():
        right_uncued_correct = np.sum(correct_values[np.asarray(right_uncued_ix)]) / np.sum(right_uncued_ix)
    else:
        right_uncued_correct = 'None'

    if behavioral_choice_ix.sum():
        overall = np.sum(correct_values[np.asarray(behavioral_choice_ix)]) / np.sum(behavioral_choice_ix)
    else:
        overall = 'None'
    return dict(left_cued_correct=left_cued_correct, left_uncued_correct=left_uncued_correct,
                right_cued_correct=right_cued_correct, right_uncued_correct=right_uncued_correct,
                overall_correct=overall)


def compute_history_ideal_mouse_agreement_by_trial(
    augmented_trial_df: pd.DataFrame,
    params: ideal_observer.IdealObserverParams | None = None,
) -> np.ndarray:
    """Return trialwise agreement with the greedy mouse-history ideal policy.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `action` and `reward`, plus either precomputed ideal-observer feature
        columns (`HMM_rel_value_logodds_decay`, `relative_doubt_index`) or the
        columns needed by `ideal_observer.compute_history_ideal_values`.
        Experimenter rewards and no-choice trials are excluded.
    params : ideal_observer.IdealObserverParams or None, default=None
        Observer parameters. None uses the same defaults as the session-level
        ideal-observer summaries.

    Returns
    -------
    np.ndarray
        Object array with shape `(n_trials,)`. Valid ideal-comparison rows
        contain `1.0` for mouse/ideal agreement and `0.0` for disagreement.
        Invalid rows contain the string sentinel `"None"`.
    """
    if params is None:
        params = ideal_observer.IdealObserverParams()

    agreement = np.full(augmented_trial_df.shape[0], "None", dtype=object)
    behavioral_choice_ix = make_behavioral_choice_mask(augmented_trial_df)
    ideal_values = ideal_observer.compute_history_ideal_values(
        augmented_trial_df,
        params=params,
    )
    numeric_ideal_values = pd.to_numeric(pd.Series(ideal_values), errors="coerce")
    mouse_actions = pd.to_numeric(augmented_trial_df["action"], errors="coerce")
    valid_rows = (
        behavioral_choice_ix
        & numeric_ideal_values.notna().to_numpy()
        & mouse_actions.notna().to_numpy()
    )
    if not valid_rows.any():
        return agreement

    retained_values = numeric_ideal_values.loc[valid_rows].to_numpy(dtype=float)
    ideal_actions = ideal_observer.choose_greedy_left_positive_actions(
        retained_values,
        initial_choice=params.initial_tie_choice,
    )
    retained_mouse_actions = mouse_actions.loc[valid_rows].to_numpy(dtype=int)
    agreement[valid_rows] = (ideal_actions == retained_mouse_actions).astype(float)
    return agreement


def compute_agent_mouse_agreement_by_trial(
    augmented_trial_df: pd.DataFrame,
    value_column: str,
    initial_tie_choice: int = ideal_observer.RIGHT_CHOICE,
) -> np.ndarray:
    """Return trialwise mouse agreement with one agent's greedy policy.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `action`, an optional experimenter-reward column accepted by
        `make_behavioral_choice_mask`, and `value_column`. Actions use
        `0=right`, `1=left`; no-choice and experimenter-reward rows are
        excluded.
    value_column : str
        Name of a signed left-positive agent value column. Positive values
        greedily choose left, negative values choose right, and exact zeroes
        repeat the previous greedy agent choice.
    initial_tie_choice : int, default=ideal_observer.RIGHT_CHOICE
        Greedy choice used when the first valid comparison row is an exact tie.

    Returns
    -------
    np.ndarray
        Object array with shape `(n_trials,)`. Valid comparison rows contain
        `1.0` for mouse/agent agreement and `0.0` for disagreement. Invalid
        rows contain the string sentinel `"None"`.

    Raises
    ------
    ValueError
        If `value_column` is absent from `augmented_trial_df`.
    """
    if value_column not in augmented_trial_df.columns:
        raise ValueError(f"augmented_trial_df is missing requested agent value column: {value_column}")

    agreement = np.full(augmented_trial_df.shape[0], "None", dtype=object)
    behavioral_choice_ix = make_behavioral_choice_mask(augmented_trial_df)
    agent_values = pd.to_numeric(augmented_trial_df[value_column], errors="coerce")
    mouse_actions = pd.to_numeric(augmented_trial_df["action"], errors="coerce")
    valid_rows = (
        behavioral_choice_ix
        & agent_values.notna().to_numpy()
        & mouse_actions.notna().to_numpy()
    )
    if not valid_rows.any():
        return agreement

    retained_values = agent_values.loc[valid_rows].to_numpy(dtype=float)
    agent_actions = ideal_observer.choose_greedy_left_positive_actions(
        retained_values,
        initial_choice=initial_tie_choice,
    )
    retained_mouse_actions = mouse_actions.loc[valid_rows].to_numpy(dtype=int)
    agreement[valid_rows] = (agent_actions == retained_mouse_actions).astype(float)
    return agreement


def safe_side_bias_ratio(numerator: int, denominator: int) -> float | str:
    """Return a side-bias ratio or the missing-value sentinel for zero denominators.

    Parameters
    ----------
    numerator : int
        Count of mouse choices for one side, in trials.
    denominator : int
        Count of reference trials for one side, in trials.

    Returns
    -------
    float or str
        `numerator / denominator` when `denominator > 0`; otherwise `"None"`.
        This ratio is unitless and is not bounded by 1.
    """
    if denominator == 0:
        return "None"
    return float(numerator / denominator)


def safe_signed_bias(numerator: int, denominator: int) -> float | str:
    """Return a signed bias value or the missing-value sentinel.

    Parameters
    ----------
    numerator : int
        Difference between mouse left-choice count and the reference left count,
        in trials.
    denominator : int
        Number of valid comparison trials, in trials.

    Returns
    -------
    float or str
        `numerator / denominator` when `denominator > 0`; otherwise `"None"`.
        Positive values indicate excess left choices relative to the reference.
    """
    if denominator == 0:
        return "None"
    return float(numerator / denominator)


def add_prev_rewards_centered(
    block_performance: pd.DataFrame,
    output_column: str,
) -> pd.DataFrame:
    """Add a centered previous-reward count column.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Block table with shape `(n_blocks, n_columns)`. Must contain
        `prev_n_rewarded`, measured in rewarded trials in the previous block.
    output_column : str
        Name of the output centered column. Values are in rewarded trials
        relative to the mean of all numeric `prev_n_rewarded` rows.

    Returns
    -------
    pandas.DataFrame
        Copy of `block_performance` with `output_column` added. Numeric rows
        contain floats; nonnumeric rows contain the string sentinel `"None"`.

    Raises
    ------
    ValueError
        If `prev_n_rewarded` is absent.
    """
    if "prev_n_rewarded" not in block_performance.columns:
        raise ValueError("block_performance is missing required column: prev_n_rewarded")

    output_df = block_performance.copy()
    prev_rewards = pd.to_numeric(output_df["prev_n_rewarded"], errors="coerce")
    numeric_rows = prev_rewards.notna()
    centered_values = np.full(output_df.shape[0], "None", dtype=object)
    if numeric_rows.any():
        centered_values[numeric_rows.to_numpy()] = (
            prev_rewards.loc[numeric_rows] - prev_rewards.loc[numeric_rows].mean()
        ).to_numpy(dtype=float)
    output_df[output_column] = centered_values
    return output_df


def add_block_side_code(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Add a signed side-code column from block labels.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Block table with shape `(n_blocks, n_columns)`. Must contain
        `block_type`, with left blocks beginning with `"left_"` and right
        blocks beginning with `"right_"`.

    Returns
    -------
    pandas.DataFrame
        Copy of `block_performance` with `block_side_code` added. Values are
        `-0.5` for right blocks and `+0.5` for left blocks, matching the task
        convention `0=right`, `1=left`. Other block types receive `"None"`.

    Raises
    ------
    ValueError
        If `block_type` is absent.
    """
    if "block_type" not in block_performance.columns:
        raise ValueError("block_performance is missing required column: block_type")

    output_df = block_performance.copy()
    side_codes = np.full(output_df.shape[0], "None", dtype=object)
    block_types = output_df["block_type"].astype(str)
    side_codes[block_types.str.startswith("right_").to_numpy()] = -0.5
    side_codes[block_types.str.startswith("left_").to_numpy()] = 0.5
    output_df["block_side_code"] = side_codes
    return output_df


def add_session_block_collection_variables(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Add block variables needed for downstream multisession collection.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Single-session block table with shape `(n_blocks, n_columns)`. Required
        columns are `prev_n_rewarded` and `block_type`.

    Returns
    -------
    pandas.DataFrame
        Copy of `block_performance` with `prev_rewards_session_centered` and
        `block_side_code`. Centering uses all numeric `prev_n_rewarded` rows in
        the session.
    """
    output_df = add_prev_rewards_centered(
        block_performance,
        output_column="prev_rewards_session_centered",
    )
    output_df = add_block_side_code(output_df)
    return output_df


def compute_session_side_bias_metrics(
    augmented_trial_df: pd.DataFrame,
    ideal_value_column: str = "observer_value",
    initial_tie_choice: int = ideal_observer.RIGHT_CHOICE,
) -> dict[str, int | float | str]:
    """Compute whole-session side-bias ratios against true and ideal choices.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action` and `state`. If `ideal_value_column` is absent,
        mouse-history ideal values are computed from the columns accepted by
        `ideal_observer.compute_history_ideal_values`. Actions use `0=right`,
        `1=left`; state values are normalized to left/right with the same task
        convention used by switch-persistence diagnostics. Rows marked as
        no-choice or experimenter reward are excluded by
        `make_behavioral_choice_mask`.
    ideal_value_column : str, default="observer_value"
        Signed left-positive ideal-agent value. Positive values greedily choose
        left, negative values choose right, and exact zeroes repeat the
        previous greedy ideal action.
    initial_tie_choice : int, default=ideal_observer.RIGHT_CHOICE
        Greedy choice used when the first valid ideal row is an exact tie.

    Returns
    -------
    dict[str, int or float or str]
        Session-level counts and ratios. Count values are in trials. Ratio
        values are unitless and are not bounded by 1. Ratio fields with zero
        denominators contain the string sentinel `"None"`.

    Raises
    ------
    ValueError
        If required columns are absent or valid animal-choice actions cannot
        be parsed as numeric side choices.
    """
    required_columns = {"action", "state"}
    missing_columns = sorted(required_columns.difference(augmented_trial_df.columns))
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required columns: {missing_columns}")

    behavioral_choice_ix = make_behavioral_choice_mask(augmented_trial_df)
    mouse_actions = pd.to_numeric(augmented_trial_df["action"], errors="coerce")
    valid_action_rows = behavioral_choice_ix & mouse_actions.notna().to_numpy()
    invalid_actions = augmented_trial_df.loc[
        behavioral_choice_ix & mouse_actions.isna().to_numpy(),
        "action",
    ]
    if not invalid_actions.empty:
        invalid_summary = sorted(invalid_actions.astype(str).unique())
        raise ValueError(f"action values must be numeric side choices, got: {invalid_summary}")

    normalized_state_sides = augmented_trial_df["state"].map(
        switch_persistence.normalize_context_side
    )
    left_choice_rows = valid_action_rows & mouse_actions.eq(ideal_observer.LEFT_CHOICE).to_numpy()
    right_choice_rows = valid_action_rows & mouse_actions.eq(ideal_observer.RIGHT_CHOICE).to_numpy()
    true_left_rows = valid_action_rows & normalized_state_sides.eq("left").to_numpy()
    true_right_rows = valid_action_rows & normalized_state_sides.eq("right").to_numpy()

    if ideal_value_column in augmented_trial_df.columns:
        ideal_values = pd.to_numeric(augmented_trial_df[ideal_value_column], errors="coerce")
    else:
        computed_ideal_values = ideal_observer.compute_history_ideal_values(
            augmented_trial_df,
            params=ideal_observer.IdealObserverParams(
                initial_tie_choice=initial_tie_choice
            ),
        )
        ideal_values = pd.to_numeric(pd.Series(computed_ideal_values), errors="coerce")
    valid_ideal_rows = valid_action_rows & ideal_values.notna().to_numpy()
    ideal_left_rows = np.zeros(augmented_trial_df.shape[0], dtype=bool)
    ideal_right_rows = np.zeros(augmented_trial_df.shape[0], dtype=bool)
    if valid_ideal_rows.any():
        ideal_actions = ideal_observer.choose_greedy_left_positive_actions(
            ideal_values.loc[valid_ideal_rows].to_numpy(dtype=float),
            initial_choice=initial_tie_choice,
        )
        valid_ideal_indices = np.flatnonzero(valid_ideal_rows)
        ideal_left_rows[valid_ideal_indices] = ideal_actions == ideal_observer.LEFT_CHOICE
        ideal_right_rows[valid_ideal_indices] = ideal_actions == ideal_observer.RIGHT_CHOICE

    n_left_choices = int(np.sum(left_choice_rows))
    n_right_choices = int(np.sum(right_choice_rows))
    n_true_left_trials = int(np.sum(true_left_rows))
    n_true_right_trials = int(np.sum(true_right_rows))
    n_ideal_left_trials = int(np.sum(ideal_left_rows))
    n_ideal_right_trials = int(np.sum(ideal_right_rows))
    n_valid_action_trials = n_left_choices + n_right_choices
    ideal_valid_left_choice_rows = left_choice_rows & valid_ideal_rows
    n_valid_ideal_trials = int(np.sum(valid_ideal_rows))
    n_valid_ideal_left_choices = int(np.sum(ideal_valid_left_choice_rows))

    return {
        "n_left_choices": n_left_choices,
        "n_right_choices": n_right_choices,
        "n_true_left_trials": n_true_left_trials,
        "n_true_right_trials": n_true_right_trials,
        "n_ideal_left_trials": n_ideal_left_trials,
        "n_ideal_right_trials": n_ideal_right_trials,
        "left_choice_per_true_left": safe_side_bias_ratio(n_left_choices, n_true_left_trials),
        "right_choice_per_true_right": safe_side_bias_ratio(n_right_choices, n_true_right_trials),
        "left_choice_per_ideal_left": safe_side_bias_ratio(n_left_choices, n_ideal_left_trials),
        "right_choice_per_ideal_right": safe_side_bias_ratio(n_right_choices, n_ideal_right_trials),
        "bias_oracle": safe_signed_bias(
            n_left_choices - n_true_left_trials,
            n_valid_action_trials,
        ),
        "bias_ideal": safe_signed_bias(
            n_valid_ideal_left_choices - n_ideal_left_trials,
            n_valid_ideal_trials,
        ),
        "raw_side_bias": safe_signed_bias(
            n_left_choices - n_right_choices,
            n_valid_action_trials,
        ),
    }


def _trial_block_mask(trial_block_values: pd.Series, block_id) -> np.ndarray:
    """Return trial rows belonging to one block id.

    Parameters
    ----------
    trial_block_values : pd.Series
        Trial-level block ids with shape `(n_trials,)`.
    block_id : object
        Block id from the block-performance table.

    Returns
    -------
    np.ndarray
        Boolean mask with shape `(n_trials,)`.
    """
    numeric_trial_blocks = pd.to_numeric(trial_block_values, errors="coerce")
    numeric_block_id = pd.to_numeric(pd.Series([block_id]), errors="coerce").iloc[0]
    if not pd.isna(numeric_block_id):
        return numeric_trial_blocks.eq(float(numeric_block_id)).to_numpy()
    return trial_block_values.astype(str).eq(str(block_id)).to_numpy()


def add_block_agent_mouse_agreement_columns(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    agent_value_columns: dict[str, str] | None = None,
    initial_tie_choice: int = ideal_observer.RIGHT_CHOICE,
) -> pd.DataFrame:
    """Add blockwise mouse-agent agreement summaries to block performance.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_block_columns)`. If `block_ix`
        is present, it is used to align rows to `augmented_trial_df.cur_block`;
        otherwise rows are aligned to the first-seen unique `cur_block` values.
    augmented_trial_df : pd.DataFrame
        Trialwise table with shape `(n_trials, n_trial_columns)`. Required
        columns are `cur_block`, `action`, and the value columns requested in
        `agent_value_columns`.
    agent_value_columns : dict[str, str] or None, default=None
        Mapping from output agreement column name to signed left-positive agent
        value column name. None uses the default Q-learning, forgetting
        Q-learning, HMM log-odds, HMM log-odds with decay, simple heuristic,
        and observer values.
    initial_tie_choice : int, default=ideal_observer.RIGHT_CHOICE
        Greedy choice used when an agent's first valid comparison row is an
        exact value tie.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with one agreement-fraction column per
        requested agent. Agreement fractions are unitless values in [0, 1];
        blocks without valid comparisons receive the string sentinel `"None"`.

    Raises
    ------
    ValueError
        If `augmented_trial_df` lacks `cur_block`, or if any requested agent
        value column is missing.
    """
    if agent_value_columns is None:
        agent_value_columns = DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS
    if "cur_block" not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df is missing required column: cur_block")

    missing_value_columns = [
        value_column
        for value_column in agent_value_columns.values()
        if value_column not in augmented_trial_df.columns
    ]
    if missing_value_columns:
        raise ValueError(
            "augmented_trial_df is missing requested agent value columns: "
            f"{sorted(set(missing_value_columns))}"
        )

    output_df = block_performance.copy()
    if "block_ix" in output_df.columns:
        block_ids = output_df["block_ix"].tolist()
    else:
        block_ids = pd.unique(augmented_trial_df["cur_block"]).tolist()
        if len(block_ids) != output_df.shape[0]:
            raise ValueError(
                "Cannot align block_performance rows to augmented_trial_df.cur_block; "
                "provide a block_ix column or matching block row count."
            )

    for output_column, value_column in agent_value_columns.items():
        trial_agreement = compute_agent_mouse_agreement_by_trial(
            augmented_trial_df,
            value_column=value_column,
            initial_tie_choice=initial_tie_choice,
        )
        block_agreement = []
        for block_id in block_ids:
            block_mask = _trial_block_mask(augmented_trial_df["cur_block"], block_id)
            agreement_values = pd.to_numeric(
                pd.Series(trial_agreement[block_mask]),
                errors="coerce",
            ).dropna()
            if agreement_values.empty:
                block_agreement.append("None")
            else:
                block_agreement.append(float(agreement_values.mean()))
        output_df[output_column] = block_agreement
    return output_df


def make_behavioral_choice_mask(trial_df: pd.DataFrame) -> np.ndarray:
    """Return rows with an animal left/right choice, excluding manual rewards."""
    experimenter_reward_given = _get_experimenter_reward_given_array(trial_df)
    return (
        is_zero_flag(experimenter_reward_given).to_numpy()
        & ~make_no_choice_action_mask(trial_df["action"]).to_numpy()
    )


def mean_or_nan(values: np.ndarray) -> float:
    """Return the mean of present numeric values, or NaN when none are present."""
    values = np.asarray(values, dtype=float)
    present_values = values[~np.isnan(values)]
    if present_values.size == 0:
        return np.nan
    return float(np.mean(present_values))


def median_or_nan(values: np.ndarray) -> float:
    """Return the median of present numeric values, or NaN when none exist."""
    values = np.asarray(values, dtype=float)
    present_values = values[~np.isnan(values)]
    if present_values.size == 0:
        return np.nan
    return float(np.median(present_values))


def std_or_nan(values: np.ndarray) -> float:
    """Return the standard deviation of present numeric values, or NaN when none exist."""
    values = np.asarray(values, dtype=float)
    present_values = values[~np.isnan(values)]
    if present_values.size == 0:
        return np.nan
    return float(np.std(present_values))


def add_numeric_trials_to_correct(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric trials-to-correct values.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `trials_to_correct`.

    Returns
    -------
    pd.DataFrame
        Copy with `trials_to_correct_numeric`; invalid values are `NaN`.
    """
    summary_df = block_performance.copy()
    summary_df["trials_to_correct_numeric"] = pd.to_numeric(
        summary_df["trials_to_correct"],
        errors="coerce",
    )
    return summary_df


def get_nonfinal_missing_trials_to_correct_block_ix(summary_df: pd.DataFrame) -> list[int]:
    """Return non-final block IDs with missing trials-to-correct values.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `trials_to_correct_numeric` and optionally `block_ix`.

    Returns
    -------
    list[int]
        Block IDs for missing `trials_to_correct_numeric` rows, excluding the
        final row.
    """
    missing_trials_to_correct = summary_df["trials_to_correct_numeric"].isna()
    nonfinal_missing = missing_trials_to_correct.copy()
    if nonfinal_missing.size:
        nonfinal_missing.iloc[-1] = False

    if "block_ix" in summary_df.columns:
        return summary_df.loc[nonfinal_missing, "block_ix"].astype(int).tolist()

    return summary_df.index[nonfinal_missing].astype(int).tolist()


def summarize_missing_trials_to_correct_blocks(nonfinal_block_ix: list[int]) -> dict:
    """Return warning metadata for non-final missing trials-to-correct blocks.

    Parameters
    ----------
    nonfinal_block_ix : list[int]
        Non-final block IDs with missing trials-to-correct values.

    Returns
    -------
    dict
        Warning metadata using the public session-summary key names.
    """
    warning_flag = len(nonfinal_block_ix) > 0
    if warning_flag:
        warning_message = (
            "Non-final invalid trials_to_correct entries detected; "
            f"excluding them from summaries. block_ix={nonfinal_block_ix}"
        )
        warnings.warn(warning_message, UserWarning, stacklevel=3)
        nonfinal_block_ix_str = str(nonfinal_block_ix)
    else:
        warning_message = "None"
        nonfinal_block_ix_str = "None"

    return dict(
        trials_to_correct_warning_flag=warning_flag,
        trials_to_correct_warning_message=warning_message,
        trials_to_correct_nonfinal_invalid_block_ix=nonfinal_block_ix_str,
    )


def mean_trials_to_correct_for_block_type(summary_df: pd.DataFrame, block_type: str) -> float | str:
    """Return mean trials-to-correct for one block type.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `block_type` and `trials_to_correct_numeric`.
    block_type : str
        Block type to summarize.

    Returns
    -------
    float or str
        Mean valid trials-to-correct value, or `"None"` when no valid values
        exist for `block_type`.
    """
    values = summary_df.loc[
        summary_df["block_type"] == block_type,
        "trials_to_correct_numeric",
    ].dropna()
    return values.mean() if not values.empty else "None"


def mean_trials_to_correct(summary_df: pd.DataFrame) -> float | str:
    """Return overall mean trials-to-correct across valid blocks.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `trials_to_correct_numeric`.

    Returns
    -------
    float or str
        Mean valid trials-to-correct value, or `"None"` when no valid values
        exist.
    """
    valid_trials = summary_df["trials_to_correct_numeric"].dropna()
    return valid_trials.mean() if not valid_trials.empty else "None"


def summarize_trials_to_correct(block_performance: pd.DataFrame) -> dict:
    """Summarize blockwise trials-to-correct while tolerating invalid sentinels.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise summary dataframe with shape `(n_blocks, n_columns)`
        containing:
        - `block_type` : str, block category for each block
        - `trials_to_correct` : int-like or sentinel values such as `'None'`
        - `block_ix` : optional integer-like block identifier used for
          warning metadata

    Returns
    -------
    dict
        Mean trials-to-correct by block type and overall. Invalid
        `trials_to_correct` values are ignored for numeric summaries.
        Conditions with no valid numeric values return `'None'`.
        Non-final invalid values emit a warning and are reported via:
        - `trials_to_correct_warning_flag` : bool
        - `trials_to_correct_warning_message` : str or `'None'`
        - `trials_to_correct_nonfinal_invalid_block_ix` : str or `'None'`
    """
    summary_df = add_numeric_trials_to_correct(block_performance)
    nonfinal_missing_block_ix = get_nonfinal_missing_trials_to_correct_block_ix(summary_df)
    missing_summary = summarize_missing_trials_to_correct_blocks(nonfinal_missing_block_ix)
    return dict(
        left_cued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "left_cued"),
        left_uncued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "left_uncued"),
        right_cued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "right_cued"),
        right_uncued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "right_uncued"),
        overall_trials_to_correct=mean_trials_to_correct(summary_df),
        **missing_summary,
    )


def safe_zero_ttc_fraction(n_zero_blocks: int, n_blocks: int) -> float | str:
    """Return a zero-trials-to-correct fraction or the missing sentinel.

    Parameters
    ----------
    n_zero_blocks : int
        Number of valid block changes with `trials_to_correct == 0`, in blocks.
    n_blocks : int
        Number of valid block changes entering the denominator, in blocks.

    Returns
    -------
    float or str
        `n_zero_blocks / n_blocks` when `n_blocks > 0`; otherwise `"None"`.
        The fraction is unitless and bounded from 0 to 1.
    """
    if n_blocks == 0:
        return "None"
    return float(n_zero_blocks / n_blocks)


def compute_zero_trials_to_correct_metrics(block_performance: pd.DataFrame) -> dict[str, int | float | str]:
    """Summarize immediate-correction block changes for one session.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        columns are `block_type` and `trials_to_correct`. The first row is
        excluded because it is not a block change. Dark-period rows and rows
        with missing/non-numeric `trials_to_correct` are excluded.

    Returns
    -------
    dict[str, int or float or str]
        Session-level zero-trials-to-correct fractions for `left`, `right`,
        and `overall`, plus matching count fields. Fractions are unitless;
        groups with zero valid block changes return `"None"` for the fraction.
    """
    required_columns = {"block_type", "trials_to_correct"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    metric_df = block_performance.copy().reset_index(drop=True)
    metric_df = metric_df.iloc[1:].copy()
    metric_df["rewarded_side"] = metric_df["block_type"].map(
        lambda block_type: "left"
        if str(block_type).startswith("left_")
        else "right"
        if str(block_type).startswith("right_")
        else "None"
    )
    metric_df["trials_to_correct_numeric"] = pd.to_numeric(
        metric_df["trials_to_correct"],
        errors="coerce",
    )
    valid_blocks = metric_df[
        metric_df["rewarded_side"].isin(["left", "right"])
        & metric_df["trials_to_correct_numeric"].notna()
    ].copy()
    valid_blocks["zero_trials_to_correct_flag"] = (
        valid_blocks["trials_to_correct_numeric"].eq(0).astype(int)
    )

    metric_summary = {}
    for side in ("left", "right"):
        side_flags = valid_blocks.loc[
            valid_blocks["rewarded_side"] == side,
            "zero_trials_to_correct_flag",
        ]
        n_blocks = int(side_flags.shape[0])
        n_zero = int(side_flags.sum()) if n_blocks else 0
        metric_summary[f"{side}_zero_ttc_fraction"] = safe_zero_ttc_fraction(
            n_zero,
            n_blocks,
        )
        metric_summary[f"{side}_zero_ttc_n_blocks"] = n_blocks
        metric_summary[f"{side}_zero_ttc_n_zero"] = n_zero

    overall_flags = valid_blocks["zero_trials_to_correct_flag"]
    overall_n_blocks = int(overall_flags.shape[0])
    overall_n_zero = int(overall_flags.sum()) if overall_n_blocks else 0
    metric_summary["overall_zero_ttc_fraction"] = safe_zero_ttc_fraction(
        overall_n_zero,
        overall_n_blocks,
    )
    metric_summary["overall_zero_ttc_n_blocks"] = overall_n_blocks
    metric_summary["overall_zero_ttc_n_zero"] = overall_n_zero
    return metric_summary


def _numeric_metric_values(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return valid numeric values for one metric column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table with shape `(n_rows, n_columns)`.
    column : str
        Column to coerce to numeric values. Values such as `"None"` are treated
        as missing and dropped.

    Returns
    -------
    pd.Series
        One-dimensional numeric series containing only eligible metric values.
        Values are unitless unless the source column has documented units.
    """
    if column not in dataframe.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce").dropna()


def _safe_median(values: pd.Series) -> float | str:
    """Return the median of valid metric values, or `"None"` when empty.

    Parameters
    ----------
    values : pd.Series
        Numeric metric values with shape `(n_values,)`.

    Returns
    -------
    float or str
        Median value, unitless unless the source metric has units. Empty inputs
        return the CSV missing-value sentinel `"None"`.
    """
    if values.empty:
        return "None"
    return float(values.median())


def _safe_quantile(values: pd.Series, quantile: float) -> float | str:
    """Return a quantile of valid metric values, or `"None"` when empty.

    Parameters
    ----------
    values : pd.Series
        Numeric metric values with shape `(n_values,)`.
    quantile : float
        Quantile in the closed interval `[0, 1]`.

    Returns
    -------
    float or str
        Requested quantile, unitless unless the source metric has units. Empty
        inputs return the CSV missing-value sentinel `"None"`.
    """
    if values.empty:
        return "None"
    return float(values.quantile(quantile))


def _safe_fraction(values: pd.Series, condition: pd.Series | np.ndarray) -> float | str:
    """Return the fraction of valid metric values satisfying a condition.

    Parameters
    ----------
    values : pd.Series
        Numeric metric values with shape `(n_values,)`.
    condition : pd.Series or np.ndarray
        Boolean array with shape `(n_values,)`, aligned to `values`.

    Returns
    -------
    float or str
        Unitless fraction of values satisfying `condition`, or `"None"` when
        no eligible values are present.
    """
    if values.empty:
        return "None"
    return float(np.mean(np.asarray(condition, dtype=bool)))


def compute_session_block_quality_metrics(
    block_performance: pd.DataFrame,
    tts_threshold: float = 5,
    post_switch_correct_threshold: float = 0.7,
    ideal_agreement_threshold: float = 0.6,
) -> dict[str, float | str]:
    """Summarize block-level adaptation and maintenance metrics for one session.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Expected columns are
        `trials_to_correct`, `percent_correct_after_first_correct`, and
        `block_history_ideal_mouse_agreement`. Missing or nonnumeric entries
        are excluded from the relevant metric denominator.
    tts_threshold : float, default=5
        Threshold in trials for the `frac_blocks_TTS_gt_5` summary.
    post_switch_correct_threshold : float, default=0.7
        Unitless correctness threshold for post-first-correct maintenance.
    ideal_agreement_threshold : float, default=0.6
        Unitless agreement threshold for mouse-history ideal-agent agreement.

    Returns
    -------
    dict[str, float or str]
        Session-level scalar summaries. Count-like fractions are unitless.
        Missing summaries use the CSV sentinel `"None"`.
    """
    tts_values = _numeric_metric_values(block_performance, "trials_to_correct")
    post_switch_correct_values = _numeric_metric_values(
        block_performance,
        "percent_correct_after_first_correct",
    )
    ideal_agreement_values = _numeric_metric_values(
        block_performance,
        "block_history_ideal_mouse_agreement",
    )

    return {
        "median_TTS": _safe_median(tts_values),
        "q3_TTS": _safe_quantile(tts_values, 0.75),
        "frac_blocks_TTS_gt_5": _safe_fraction(
            tts_values,
            tts_values > tts_threshold,
        ),
        "median_post_switch_correct": _safe_median(post_switch_correct_values),
        "q1_post_switch_correct": _safe_quantile(post_switch_correct_values, 0.25),
        "frac_blocks_post_switch_correct_lt_0p7": _safe_fraction(
            post_switch_correct_values,
            post_switch_correct_values < post_switch_correct_threshold,
        ),
        "median_ideal_agreement": _safe_median(ideal_agreement_values),
        "q1_ideal_agreement": _safe_quantile(ideal_agreement_values, 0.25),
        "frac_blocks_ideal_agreement_lt_0p6": _safe_fraction(
            ideal_agreement_values,
            ideal_agreement_values < ideal_agreement_threshold,
        ),
    }


def compute_post_switch_ideal_agreement_summary(
    augmented_trial_df: pd.DataFrame,
    block_performance: pd.DataFrame,
) -> dict[str, float | str]:
    """Summarize ideal-agent agreement after the first correct choice per block.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise table with shape `(n_trials, n_columns)`. Required columns are
        `cur_block`, `action`, and `correct`. If `observer_value` is present it
        is used as the signed left-positive ideal-agent value; otherwise values
        are computed from columns accepted by
        `compute_history_ideal_mouse_agreement_by_trial`. Actions use
        `0=right`, `1=left`; correctness values are 0/1. No-choice and
        experimenter-reward rows are excluded through existing behavioral
        choice masks.
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. If present,
        `block_ix` is used to align blocks to `augmented_trial_df.cur_block`;
        otherwise rows align to first-seen `cur_block` values.

    Returns
    -------
    dict[str, float or str]
        `median_post_switch_ideal_agreement` and
        `q1_post_switch_ideal_agreement`, unitless fractions over eligible
        blocks. Blocks without a first correct behavioral choice, without
        post-switch valid trials, or without valid ideal comparisons are
        excluded. Empty summaries use the CSV sentinel `"None"`.
    """
    if "cur_block" not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df is missing required column: cur_block")

    if "block_ix" in block_performance.columns:
        block_ids = block_performance["block_ix"].to_numpy()
    else:
        block_ids = augmented_trial_df["cur_block"].drop_duplicates().to_numpy()
        if len(block_ids) != len(block_performance):
            raise ValueError(
                "Cannot align block_performance rows to augmented_trial_df.cur_block; "
                "provide a block_ix column or matching block row count."
            )

    if "observer_value" in augmented_trial_df.columns:
        trial_agreement = compute_agent_mouse_agreement_by_trial(
            augmented_trial_df,
            value_column="observer_value",
        )
    else:
        trial_agreement = compute_history_ideal_mouse_agreement_by_trial(augmented_trial_df)
    block_post_switch_agreement = []

    for block_id in block_ids:
        block_mask = _trial_block_mask(augmented_trial_df["cur_block"], block_id)
        cur_block_df = augmented_trial_df.loc[block_mask]
        if cur_block_df.empty:
            continue

        behavioral_choice_mask = make_behavioral_choice_mask(cur_block_df)
        if not behavioral_choice_mask.any():
            continue

        correct_values = get_numeric_correct_values(cur_block_df, behavioral_choice_mask)
        behavioral_correct_values = correct_values[np.asarray(behavioral_choice_mask)]
        first_correct_positions = np.nonzero(behavioral_correct_values)[0]
        if first_correct_positions.size == 0:
            continue

        behavioral_trial_indices = np.flatnonzero(block_mask)[np.asarray(behavioral_choice_mask)]
        post_switch_trial_indices = behavioral_trial_indices[first_correct_positions[0]:]
        agreement_values = pd.to_numeric(
            pd.Series(trial_agreement[post_switch_trial_indices]),
            errors="coerce",
        ).dropna()
        if agreement_values.empty:
            continue
        block_post_switch_agreement.append(float(agreement_values.mean()))

    summary_values = pd.Series(block_post_switch_agreement, dtype=float)
    return {
        "median_post_switch_ideal_agreement": _safe_median(summary_values),
        "q1_post_switch_ideal_agreement": _safe_quantile(summary_values, 0.25),
    }


def missing_post_switch_ideal_agreement_summary() -> dict[str, str]:
    """Return missing sentinels for post-switch ideal-agreement summaries.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, str]
        Summary dictionary containing `"None"` for
        `median_post_switch_ideal_agreement` and
        `q1_post_switch_ideal_agreement`.
    """
    return {
        "median_post_switch_ideal_agreement": "None",
        "q1_post_switch_ideal_agreement": "None",
    }


def get_block_switches(trial_df: pd.DataFrame) -> tuple[int, int]:
    """Count left-right action switches while ignoring no-choice rows."""
    actions = np.copy(trial_df['action'].values)
    experimenter_reward_given = _get_experimenter_reward_given_array(trial_df)
    skip_mask = (
        ~is_zero_flag(experimenter_reward_given).to_numpy()
        | make_no_choice_action_mask(actions).to_numpy()
    )
    valid_indices = np.flatnonzero(~skip_mask)
    if valid_indices.size == 0:
        return 0, 0

    for i in np.flatnonzero(skip_mask):
        previous_valid = valid_indices[valid_indices < i]
        next_valid = valid_indices[valid_indices > i]
        if previous_valid.size:
            actions[i] = actions[previous_valid[-1]]
        elif next_valid.size:
            actions[i] = actions[next_valid[0]]

    actions = actions.astype(int)
    n_switches = np.sum(np.abs(np.diff(actions)))
    if trial_df.shape[0] > 1:
        normalized_switches = n_switches / (trial_df.shape[0] - 1)  # must subtract 1 for n-1 opportunites to switch
    else:
        normalized_switches = 0
    return n_switches, normalized_switches


def count_explore_trials(trial_df: pd.DataFrame) -> int:
    """Count explore-tagged trials in one block.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. When present,
        `explore_trial` is interpreted as a boolean/count flag where true,
        positive numeric, or CSV-loaded string values (`"true"`, `"1"`,
        `"1.0"`) mark one explore trial. Missing columns are treated as zero
        explore trials for compatibility with older saved analyses.

    Returns
    -------
    int
        Number of explore-tagged trials in `trial_df`, in trials.
    """
    if "explore_trial" not in trial_df.columns:
        return 0

    explore_values = trial_df["explore_trial"]
    text_values = explore_values.astype(str).str.strip().str.lower()
    true_text_mask = text_values.isin(["true", "1", "1.0"])
    numeric_values = pd.to_numeric(explore_values, errors="coerce")
    positive_numeric_mask = numeric_values.fillna(0) > 0
    return int((true_text_mask | positive_numeric_mask).sum())


def count_terminal_omission_streak(
    first_new_trial_index: int,
    global_reward_values: np.ndarray,
) -> int:
    """Count unrewarded behavioral choices immediately before one trial.

    Parameters
    ----------
    first_new_trial_index : int
        Row index of the first new-side/correct choice in the full session
        trial table, using 0-based row indexing.
    global_reward_values : np.ndarray
        Numeric reward values with shape `(n_trials,)`. Valid behavioral-choice
        rows contain 0.0 or 1.0; rows excluded by behavioral-choice conventions
        contain NaN.

    Returns
    -------
    int
        Number of consecutive unrewarded behavioral-choice trials before
        `first_new_trial_index`. Invalid rows are skipped, preserving the
        existing behavioral-choice convention.
    """
    reward_values = np.asarray(global_reward_values, dtype=float)
    omission_count = 0
    trial_index = first_new_trial_index - 1
    while trial_index >= 0:
        reward_value = reward_values[trial_index]
        if np.isnan(reward_value):
            trial_index -= 1
            continue
        if reward_value == 0:
            omission_count += 1
            trial_index -= 1
            continue
        break
    return omission_count


def compute_block_transition_metrics(
    behavioral_correct_values: np.ndarray,
    behavioral_trial_indices: np.ndarray,
    global_reward_values: np.ndarray,
) -> dict[str, bool | int | str]:
    """Compute transition-structure metrics for one block.

    Parameters
    ----------
    behavioral_correct_values : np.ndarray
        Correctness values for valid behavioral choices in one block, shape
        `(n_valid_block_choices,)`. Values use 1 for new-side/correct choices
        and 0 for old-side/incorrect choices.
    behavioral_trial_indices : np.ndarray
        Full-session row indices for the same valid behavioral choices, shape
        `(n_valid_block_choices,)`.
    global_reward_values : np.ndarray
        Numeric reward values for the full session, shape `(n_trials,)`, with
        NaN on rows excluded by behavioral-choice conventions.

    Returns
    -------
    dict[str, bool | int | str]
        Block-level metrics. Missing metrics use the project CSV sentinel
        `"None"`. Counts are in trials.
    """
    correct_values = np.asarray(behavioral_correct_values, dtype=float)
    correct_positions = np.flatnonzero(correct_values == 1)
    if correct_positions.size == 0:
        return {
            "no_switch": True,
            "transition_width": "None",
            "reversion_choices": "None",
            "reversion_events": "None",
            "terminal_omission_streak_inclusive": "None",
            "short_or_low_postswitch_trials": True,
        }

    first_new_position = int(correct_positions[0])
    post_switch_values = correct_values[first_new_position:]
    old_positions_after_first_new = np.flatnonzero(post_switch_values == 0)
    reversion_choices = int(old_positions_after_first_new.shape[0])
    if reversion_choices == 0:
        transition_width = 0
    else:
        transition_width = int(old_positions_after_first_new[-1] + 1)

    if post_switch_values.shape[0] < 2:
        reversion_events = 0
    else:
        reversion_events = int(
            np.sum((post_switch_values[:-1] == 1) & (post_switch_values[1:] == 0))
        )

    first_new_trial_index = int(np.asarray(behavioral_trial_indices)[first_new_position])
    return {
        "no_switch": False,
        "transition_width": transition_width,
        "reversion_choices": reversion_choices,
        "reversion_events": reversion_events,
        "terminal_omission_streak_inclusive": count_terminal_omission_streak(
            first_new_trial_index=first_new_trial_index,
            global_reward_values=global_reward_values,
        ),
        "short_or_low_postswitch_trials": bool(post_switch_values.shape[0] <= 2),
    }


def make_augmented_trial_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    trial_df = normalize_experimenter_reward_column(trial_df)
    augmented_trial_df = trial_df.copy(deep=True)
    block_types = get_block_types(trial_df)
    augmented_trial_df['block_type'] = block_types
    augmented_trial_df['time_to_choice'] = _get_choice_latency(trial_df)

    prev_action = np.zeros(trial_df.shape[0], dtype='object')
    prev_reward = np.zeros(trial_df.shape[0], dtype='object')
    prev_action[0] = 'None'
    prev_reward[0] = 'None'
    prev_action[1:] = trial_df['action'].to_numpy()[:-1]
    prev_reward[1:] = trial_df['reward'].to_numpy()[:-1]
    augmented_trial_df['prev_action'] = prev_action
    augmented_trial_df['prev_reward'] = prev_reward

    decision_vars = count_decision_variables(trial_df)
    augmented_trial_df = pd.concat([augmented_trial_df, decision_vars], axis=1)

    return augmented_trial_df


def summarize_block_performance(augmented_trial_df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Summarize performance for each task block in one session.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns include
        `cur_block`, `block_type`, `correct`, `reward`, `time_to_choice`, and
        decision-variable columns from `make_augmented_trial_df`.
    session_id : str
        Session identifier copied into the `session_ID` output column.

    Returns
    -------
    pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Count columns are in trials. Post-first-correct metrics include the
        first correct behavioral choice and all later behavioral choices in
        the block; blocks without any correct behavioral choice receive the
        string sentinel `"None"`. `block_history_ideal_mouse_agreement` is the
        fraction of valid behavioral-choice trials in a block where the mouse
        action matches the greedy mouse-history ideal-observer action.
    """
    choice_latency = augmented_trial_df['time_to_choice'].to_numpy(dtype=float)
    history_ideal_mouse_agreement = compute_history_ideal_mouse_agreement_by_trial(
        augmented_trial_df
    )
    global_behavioral_choice_mask = make_behavioral_choice_mask(augmented_trial_df)
    global_reward_values = get_numeric_reward_values(
        augmented_trial_df,
        global_behavioral_choice_mask,
    )
    global_reward_values[~global_behavioral_choice_mask] = np.nan
    blocks = np.unique(augmented_trial_df['cur_block'])
    block_performance = []
    previous_block_length = "None"
    previous_block_reward_fraction = "None"

    for ix, b in enumerate(blocks):
        cur_block_ix = augmented_trial_df['cur_block'] == b
        cur_block_df = augmented_trial_df[cur_block_ix]

        if ix == 0:
            prev_n_correct = 0
            prev_n_rewarded = 0
            prev_consecutive_rewards = 0
            prev_consecutive_rewards_memory = 0
        else:
            prev_n_correct = performance['n_correct']
            prev_n_rewarded = performance['n_rewarded']
            prev_consecutive_rewards = cur_block_df['consecutive_rewards'].values[0]
            prev_consecutive_rewards_memory = cur_block_df['consecutive_rewards_memory'].values[0]

        performance = percent_correct(cur_block_df)
        n_switches, normalized_switches = get_block_switches(cur_block_df)
        n_explore_trials = count_explore_trials(cur_block_df)
        behavioral_choice_ix = make_behavioral_choice_mask(cur_block_df)
        correct_values = get_numeric_correct_values(cur_block_df, behavioral_choice_ix)
        reward_values = get_numeric_reward_values(cur_block_df, behavioral_choice_ix)
        behavioral_correct_values = correct_values[np.asarray(behavioral_choice_ix)]
        behavioral_reward_values = reward_values[np.asarray(behavioral_choice_ix)]
        behavioral_trial_indices = np.flatnonzero(cur_block_ix.to_numpy())[
            np.asarray(behavioral_choice_ix)
        ]
        transition_metrics = compute_block_transition_metrics(
            behavioral_correct_values=behavioral_correct_values,
            behavioral_trial_indices=behavioral_trial_indices,
            global_reward_values=global_reward_values,
        )
        block_history_ideal_values = pd.to_numeric(
            pd.Series(history_ideal_mouse_agreement[cur_block_ix.to_numpy()]),
            errors="coerce",
        ).dropna()
        if block_history_ideal_values.empty:
            block_history_ideal_mouse_agreement = "None"
        else:
            block_history_ideal_mouse_agreement = float(block_history_ideal_values.mean())

        correct_ix = np.nonzero(behavioral_correct_values)[0]
        if correct_ix.size:
            trials_to_correct = correct_ix[0]
            first_correct_trial_in_block = correct_ix[0]
            correct_after_first = behavioral_correct_values[first_correct_trial_in_block:]
            n_trials_after_first_correct = correct_after_first.shape[0]
            percent_correct_after_first_correct = float(np.mean(correct_after_first))
        else:
            trials_to_correct = 'None' #np.nan  # could occur on last block in session, or when the task switches to dark mode
            first_correct_trial_in_block = 'None'
            n_trials_after_first_correct = 'None'
            percent_correct_after_first_correct = 'None'

        performance = dict(block_ix=b, block_type=cur_block_df['block_type'].values[0],
                           trials_to_correct=trials_to_correct,
                           first_correct_trial_in_block=first_correct_trial_in_block,
                           n_trials_after_first_correct=n_trials_after_first_correct,
                           percent_correct_after_first_correct=percent_correct_after_first_correct,
                           block_history_ideal_mouse_agreement=block_history_ideal_mouse_agreement,
                           prev_consecutive_rewards=prev_consecutive_rewards,
                           prev_consecutive_rewards_memory=prev_consecutive_rewards_memory,
                           prev_n_correct=prev_n_correct,
                           prev_n_rewarded=prev_n_rewarded,
                           previous_block_length=previous_block_length,
                           previous_block_reward_fraction=previous_block_reward_fraction,
                           **transition_metrics,
                           n_switches=n_switches,
                           n_explore_trials=n_explore_trials,
                           normalized_switches=normalized_switches,
                           confusion_flag=n_switches > 3,
                           n_correct=int(np.sum(behavioral_correct_values)),
                           percent_correct=performance['overall_correct'],
                           n_rewarded=float(np.sum(behavioral_reward_values)),
                           mean_choice_time=mean_or_nan(choice_latency[cur_block_ix.to_numpy()]),
                           median_choice_time=median_or_nan(choice_latency[cur_block_ix.to_numpy()]),
                           std_choice_time=std_or_nan(choice_latency[cur_block_ix.to_numpy()]),
                           session_ID=session_id)

        block_performance.append(performance)
        current_block_length = int(behavioral_reward_values.shape[0])
        previous_block_length = current_block_length
        if current_block_length == 0:
            previous_block_reward_fraction = "None"
        else:
            previous_block_reward_fraction = float(np.sum(behavioral_reward_values) / current_block_length)

    block_performance = pd.DataFrame(block_performance)
    block_performance['block_ix'] = np.arange(len(block_performance))
    return block_performance


def summarize_session_performance(
    augmented_trial_df: pd.DataFrame,
    block_performance: pd.DataFrame,
    date: str,
    ideal_observer_n_replays: int = 100,
    ideal_observer_seed: int | None = 12345,
) -> pd.DataFrame:
    """Summarize whole-session performance from trial and block tables.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`, including `block_type`
        and `correct`.
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`,
        including `trials_to_correct` and `block_type`.
    date : str
        Session date copied into the `date` output column.
    ideal_observer_n_replays : int, default=100
        Number of fixed-state ideal-observer replay samples to run.
    ideal_observer_seed : int or None, default=12345
        Seed for fixed-state ideal-observer replay sampling.

    Returns
    -------
    pd.DataFrame
        One-row session performance table.
    """
    session_performance = percent_correct(augmented_trial_df)
    session_performance = session_performance | summarize_trials_to_correct(block_performance)
    session_performance = session_performance | compute_zero_trials_to_correct_metrics(
        block_performance
    )
    session_performance = session_performance | compute_session_block_quality_metrics(
        block_performance
    )
    session_performance = session_performance | general_behavior_assessment.summarize_oracle_behavior(
        augmented_trial_df
    )
    session_performance = session_performance | general_behavior_assessment.summarize_rewarded_choice_switching(
        augmented_trial_df
    )
    session_performance = session_performance | ideal_observer.summarize_ideal_observer_behavior(
        augmented_trial_df,
        params=ideal_observer.IdealObserverParams(
            n_fixed_replays=ideal_observer_n_replays,
            random_seed=ideal_observer_seed,
        ),
    )
    session_performance = session_performance | compute_session_side_bias_metrics(
        augmented_trial_df
    )
    if {"cur_block", "correct"}.issubset(augmented_trial_df.columns):
        session_performance = session_performance | compute_post_switch_ideal_agreement_summary(
            augmented_trial_df,
            block_performance,
        )
    else:
        session_performance = session_performance | missing_post_switch_ideal_agreement_summary()
    session_performance['date'] = date
    return pd.DataFrame(session_performance, index=[0])


def analyze_session(
    trial_df: pd.DataFrame,
    mouse: str,
    date: str,
    ideal_observer_n_replays: int = 100,
    ideal_observer_seed: int | None = 12345,
) -> tuple:
    """Build trial, block, and session performance summaries.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns
        include `state`, `action`, `reward`, `correct`, and `cur_block`.
    mouse : str
        Mouse or simulated-agent identifier copied into the block-level
        `session_ID` prefix.
    date : str
        Session date copied into the session summary and block-level
        `session_ID`.
    ideal_observer_n_replays : int, default=100
        Number of fixed-state ideal-observer replay samples to run.
    ideal_observer_seed : int or None, default=12345
        Seed for fixed-state ideal-observer replay sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(session_performance, block_performance, augmented_trial_df)`, where
        `session_performance` has one row, `block_performance` has shape
        `(n_blocks, n_columns)`, and `augmented_trial_df` has shape
        `(n_trials, n_columns)`.
    """
    session_id = mouse + '_' + date
    augmented_trial_df = make_augmented_trial_df(trial_df)
    block_performance = summarize_block_performance(augmented_trial_df, session_id=session_id)
    session_performance = summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date=date,
        ideal_observer_n_replays=ideal_observer_n_replays,
        ideal_observer_seed=ideal_observer_seed,
    )

    return session_performance, block_performance, augmented_trial_df


def append_decision_variables(
    decision_variable_dict: dict[str, list],
    state: DecisionVariableState,
) -> None:
    """Append the current pre-trial decision-variable state.

    Parameters
    ----------
    decision_variable_dict : dict[str, list]
        Output accumulator with one list per decision-variable column.
    state : DecisionVariableState
        Current trial-history state before the current trial outcome is used.

    Returns
    -------
    None
        Mutates `decision_variable_dict` in place.
    """
    decision_variable_dict['negative_value'].append(state.negative_value)
    decision_variable_dict['consecutive_rewards_memory'].append(state.consecutive_rewards_memory)
    decision_variable_dict['consecutive_omissions_memory'].append(state.consecutive_omissions_memory)
    decision_variable_dict['consecutive_rewards'].append(state.consecutive_rewards)
    decision_variable_dict['consecutive_omissions'].append(state.consecutive_omissions)
    decision_variable_dict['left_value'].append(state.left_value)
    decision_variable_dict['right_value'].append(state.right_value)
    decision_variable_dict['relative_value'].append(state.left_value - state.right_value)
    decision_variable_dict['left_omissions'].append(state.left_omissions)
    decision_variable_dict['right_omissions'].append(state.right_omissions)
    decision_variable_dict['relative_omissions'].append(state.left_omissions - state.right_omissions)
    decision_variable_dict['left_cf_value'].append(state.left_value_cf)
    decision_variable_dict['right_cf_value'].append(state.right_value_cf)
    decision_variable_dict['relative_cf_value'].append(state.left_value_cf - state.right_value_cf)
    decision_variable_dict['left_cf_omissions'].append(state.left_omissions_cf)
    decision_variable_dict['right_cf_omissions'].append(state.right_omissions_cf)
    decision_variable_dict['relative_cf_omissions'].append(state.left_omissions_cf - state.right_omissions_cf)
    decision_variable_dict['left_monotonic_cf_value'].append(state.left_monotonic_cf_value)
    decision_variable_dict['right_monotonic_cf_value'].append(state.right_monotonic_cf_value)
    decision_variable_dict['relative_monotonic_cf_value'].append(
        state.left_monotonic_cf_value - state.right_monotonic_cf_value
    )


def should_skip_decision_variable_update(experimenter_reward_given: int | str, action: int | str) -> bool:
    """Return whether a trial should leave decision-variable state unchanged.

    Parameters
    ----------
    experimenter_reward_given : int or str
        Experimenter-reward flag for one trial.
    action : int or str
        Animal action for one trial; `"None"` or `"no_choice"` marks no animal
        choice.

    Returns
    -------
    bool
        True when the trial should not update history counters.
    """
    return (
        not bool(is_zero_flag([experimenter_reward_given]).iloc[0])
        or bool(make_no_choice_action_mask([action]).iloc[0])
    )


def parse_decision_variable_update_values(action, reward) -> tuple[int, int]:
    """Parse one non-skipped trial outcome for decision-variable counters.

    Parameters
    ----------
    action : int, float, or str
        Valid animal choice after no-choice rows have been skipped. Values use
        task coding: `0` for right and `1` for left.
    reward : int, float, or str
        Trial reward magnitude. Positive values are treated as rewarded
        (`1`); zero or negative values are treated as unrewarded (`0`).

    Returns
    -------
    tuple[int, int]
        `(action_int, reward_int)`, where `action_int` is `0` or `1`, and
        `reward_int` is binary reward status in `{0, 1}`.
    """
    try:
        action_int = int(float(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"action must be 0 or 1 after skip filtering, got {action!r}."
        ) from exc
    if action_int not in (0, 1):
        raise ValueError(
            f"action must be 0 or 1 after skip filtering, got {action!r}."
        )

    try:
        reward_int = int(float(reward) > 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reward must be numeric, got {reward!r}.") from exc

    return action_int, reward_int


def increment_reward_history_decision_vars(
    state: DecisionVariableState,
    reward: int,
) -> DecisionVariableState:
    """Update reward-history counters from one trial outcome.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after reward-history updates.
    """
    state.negative_value = counters.negative_value_counter(state.negative_value, reward)
    state.consecutive_rewards = counters.consecutive_reward_counter(state.consecutive_rewards, reward)
    state.consecutive_omissions = counters.consecutive_fail_counter(state.consecutive_omissions, reward)
    state.consecutive_rewards_memory = counters.consecutive_reward_renewal_counter(
        state.consecutive_rewards_memory,
        reward,
        state.previous_trial_rewarded,
    )
    state.consecutive_omissions_memory = counters.consecutive_fail_renewal_counter(
        state.consecutive_omissions_memory,
        reward,
        state.previous_trial_rewarded,
    )
    state.previous_trial_rewarded = reward > 0
    return state


def increment_choice_value_decision_vars(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update non-counterfactual side-specific counters.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after side-specific updates.
    """
    state.left_value, state.right_value = counters.sided_value_counter(
        state.left_value,
        state.right_value,
        action,
        reward,
        zero_min=True,
        counterfactual=False,
    )
    state.left_omissions, state.right_omissions = counters.sided_omissions_counter(
        state.left_omissions,
        state.right_omissions,
        action,
        reward,
        counterfactual=False,
    )
    return state


def increment_counterfactual_decision_vars(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update counterfactual side-specific counters.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after counterfactual updates.
    """
    state.left_value_cf, state.right_value_cf = counters.sided_value_counter(
        state.left_value_cf,
        state.right_value_cf,
        action,
        reward,
        zero_min=True,
        counterfactual=True,
    )
    state.left_omissions_cf, state.right_omissions_cf = counters.sided_omissions_counter(
        state.left_omissions_cf,
        state.right_omissions_cf,
        action,
        reward,
        counterfactual=True,
    )
    return state


def increment_monotonic_counterfactual_decision_vars(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update monotonic counterfactual reward counters.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1. Rewarded trials increment the
        chosen side and reset the unchosen side. Omission trials leave both
        monotonic counters unchanged.

    Returns
    -------
    DecisionVariableState
        The same state object after monotonic counterfactual updates.
    """
    state.left_monotonic_cf_value, state.right_monotonic_cf_value = counters.sided_value_counter(
        state.left_monotonic_cf_value,
        state.right_monotonic_cf_value,
        action,
        reward,
        zero_min=True,
        counterfactual=True,
        monotonic=True,
    )
    return state


def update_decision_variable_state(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update all decision-variable counters from one non-skipped trial.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after all counter updates.
    """
    state = increment_reward_history_decision_vars(state, reward)
    state = increment_choice_value_decision_vars(state, action, reward)
    state = increment_counterfactual_decision_vars(state, action, reward)
    state = increment_monotonic_counterfactual_decision_vars(state, action, reward)
    return state


def count_decision_variables(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Collect pre-trial decision variables for each trial.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial dataframe with shape `(n_trials, n_columns)`, including `action`,
        `reward`, and optionally `experimenter_reward_given`.

    Returns
    -------
    pd.DataFrame
        Decision-variable dataframe with shape `(n_trials, n_decision_vars)`.
        Each row describes trial history before the current trial outcome.
    """
    state = DecisionVariableState()
    decision_variable_dict = defaultdict(list)
    trial_df = normalize_experimenter_reward_column(trial_df)
    experimenter_reward_flags = _get_experimenter_reward_given_array(trial_df)

    for i in range(trial_df.shape[0]):
        append_decision_variables(decision_variable_dict, state)

        action = trial_df.loc[i, 'action']
        if should_skip_decision_variable_update(experimenter_reward_flags[i], action):
            continue

        action, reward = parse_decision_variable_update_values(
            action=action,
            reward=trial_df.loc[i, 'reward'],
        )
        state = update_decision_variable_state(
            state,
            action=action,
            reward=reward,
        )

    return pd.DataFrame(decision_variable_dict)


def summarize_block_switches(block_performance: pd.DataFrame, min_counts=0) -> tuple:
    """This function is not used anywhere, I may delete it"""

    grouped_switches = block_performance.groupby(['consecutive_rewards'])['trials_to_correct']
    counts = grouped_switches.count()
    # ic(counts)

    ix = counts > min_counts
    mean = grouped_switches.mean()[ix]
    std = grouped_switches.std()[ix]
    sem = grouped_switches.sem()[ix]
    rewards = counts.index[ix]
    return rewards, mean, std, sem


def session_stats(dependent_var, independent_var) -> tuple[float | str, float | str, float | str, float | str]:
    """Fit a simple linear model for paired numeric session summary values.

    Parameters
    ----------
    dependent_var : array-like
        One-dimensional numeric values with shape `(n_observations,)`.
    independent_var : array-like
        One-dimensional numeric values with shape `(n_observations,)`, aligned
        row-wise to `dependent_var`.

    Returns
    -------
    tuple[float or str, float or str, float or str, float or str]
        `(slope, intercept, r_value, p_value)` for `dependent_var ~
        independent_var`. Returns string `"None"` for all outputs when fewer
        than two paired observations are available. Input filtering and numeric
        conversion remain the caller's responsibility.
    """
    stats_df = pd.DataFrame({
        'y': dependent_var,
        'a': independent_var,
    })
    if stats_df.shape[0] < 2:
        return "None", "None", "None", "None"

    y, X = model_matrix("y ~ a", stats_df)
    model = sm.OLS(y, X)
    results = model.fit()
    r_value, p_value = results.rsquared, results.pvalues['a']
    intercept, slope = results.params['Intercept'], results.params['a']
    return slope, intercept, r_value, p_value


def _get_valid_trials_to_correct_mask(
    trials_to_correct: pd.Series,
    prev_n_correct: pd.Series,
) -> pd.Series:
    """Identify block rows usable for trials-to-correct regression summaries.

    Parameters
    ----------
    trials_to_correct : pd.Series
        Blockwise trials-to-correct values with shape `(n_blocks,)`. Valid rows
        contain integer-like values; invalid rows use real missing values or
        string missing-value sentinels such as `"None"`.
    prev_n_correct : pd.Series
        Number of correct trials in the previous block, shape `(n_blocks,)`.
        The first block uses a real missing value or string missing-value
        sentinel when no previous block exists.

    Returns
    -------
    pd.Series
        Boolean mask with shape `(n_blocks,)`, aligned to the input index.
    """
    return is_present_value(trials_to_correct) & is_present_value(prev_n_correct)


def _add_regression_stats(
    session_performance: pd.DataFrame,
    column_prefix: str,
    dependent_var: pd.Series,
    independent_var: pd.Series,
) -> pd.DataFrame:
    """Add one simple-regression result to a one-row session summary.

    Parameters
    ----------
    session_performance : pd.DataFrame
        One-row session summary dataframe.
    column_prefix : str
        Prefix naming the independent variable in generated columns.
    dependent_var : pd.Series
        Numeric dependent values with shape `(n_observations,)`.
    independent_var : pd.Series
        Numeric independent values with shape `(n_observations,)`, aligned
        row-wise to `dependent_var`.

    Returns
    -------
    pd.DataFrame
        Copy of `session_performance` with `{column_prefix}_slope`,
        `{column_prefix}_intercept`, `{column_prefix}_r_value`, and
        `{column_prefix}_p_value` columns.
    """
    slope, intercept, r_value, p_value = session_stats(
        dependent_var=dependent_var,
        independent_var=independent_var,
    )
    session_performance[f"{column_prefix}_slope"] = slope
    session_performance[f"{column_prefix}_intercept"] = intercept
    session_performance[f"{column_prefix}_r_value"] = r_value
    session_performance[f"{column_prefix}_p_value"] = p_value
    return session_performance


def _coerce_regression_values(values: pd.Series) -> pd.Series:
    """Coerce a regression input series to finite int64-safe numeric values.

    Parameters
    ----------
    values : pd.Series
        One-dimensional regression input with shape `(n_blocks,)`. Values are
        expected to be block-level counts in trials, but may contain CSV
        sentinels such as `"None"` or stale malformed string values.

    Returns
    -------
    pd.Series
        Float series with shape `(n_blocks,)`. Values that cannot be parsed,
        are nonfinite, or exceed signed int64 bounds are set to NaN so the
        corresponding regression rows can be excluded before fitting.
    """
    numeric_values = pd.to_numeric(values, errors="coerce")
    int64_limit = np.iinfo(np.int64).max
    valid_values = numeric_values.notna() & np.isfinite(numeric_values)
    valid_values = valid_values & numeric_values.abs().le(int64_limit)
    return numeric_values.where(valid_values)


def _numeric_regression_inputs(
    dependent_values: pd.Series,
    independent_values: pd.Series,
    base_valid_rows: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return paired numeric regression inputs after row filtering.

    Parameters
    ----------
    dependent_values : pd.Series
        Numeric dependent values with shape `(n_blocks,)`, in trials.
    independent_values : pd.Series
        Numeric independent values with shape `(n_blocks,)`, in trials.
    base_valid_rows : pd.Series
        Boolean mask with shape `(n_blocks,)` marking rows that are valid under
        the shared trials-to-correct convention.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        `(dependent_var, independent_var)` after excluding rows where either
        series is missing or malformed. Both outputs are float-valued trial
        counts with matching shape `(n_valid_rows,)`.
    """
    valid_rows = base_valid_rows & dependent_values.notna() & independent_values.notna()
    return dependent_values[valid_rows], independent_values[valid_rows]


def _single_numeric_session_value(session_performance: pd.DataFrame, column_name: str) -> float:
    """Return one numeric session-summary value or NaN when unavailable.

    Parameters
    ----------
    session_performance : pd.DataFrame
        Session summary table with shape `(1, n_columns)` or more. The first
        row is used.
    column_name : str
        Column to read.

    Returns
    -------
    float
        Numeric value from the first row, or NaN when the column/value is
        missing or nonnumeric.
    """
    if column_name not in session_performance.columns or session_performance.empty:
        return float("nan")
    numeric_values = pd.to_numeric(session_performance[column_name], errors="coerce")
    return float(numeric_values.iloc[0])


def _tts_percentile_values(trials_to_correct: pd.Series) -> pd.Series:
    """Return within-session TTS percentile values on a 0-to-1 scale.

    Parameters
    ----------
    trials_to_correct : pd.Series
        Numeric TTS values with shape `(n_blocks,)`. Missing values are NaN.

    Returns
    -------
    pd.Series
        Percentile ranks with shape `(n_blocks,)`. The fastest valid block is
        0.0 and the slowest valid block is 1.0. Missing TTS values remain NaN.
    """
    percentiles = pd.Series(np.nan, index=trials_to_correct.index, dtype=float)
    valid_values = trials_to_correct.dropna()
    if valid_values.empty:
        return percentiles
    if valid_values.shape[0] == 1:
        percentiles.loc[valid_values.index] = 0.0
        return percentiles

    ranks = valid_values.rank(method="average", ascending=True)
    percentiles.loc[valid_values.index] = (ranks - 1) / (valid_values.shape[0] - 1)
    return percentiles


def add_relative_tts_metrics(
    block_performance: pd.DataFrame,
    session_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Add session-relative TTS predictions, residuals, and percentiles.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required columns
        are `trials_to_correct` and `prev_n_rewarded`; values are trial counts
        or the project missing-value sentinel `"None"`.
    session_performance : pd.DataFrame
        Session summary table with shape `(1, n_columns)`, including
        `prev_n_rewarded_slope` and `prev_n_rewarded_intercept` from the
        existing `TTS ~ prev_n_rewarded` regression.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with `predicted_TTS_from_prev_rewards`,
        `residual_TTS`, and `TTS_percentile_within_session`. Missing values use
        the project sentinel `"None"` so CSV output matches existing block
        metrics.
    """
    required_columns = {"trials_to_correct", "prev_n_rewarded"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    output_df = block_performance.copy()
    trials_to_correct = pd.to_numeric(output_df["trials_to_correct"], errors="coerce")
    previous_rewards = pd.to_numeric(output_df["prev_n_rewarded"], errors="coerce")
    slope = _single_numeric_session_value(session_performance, "prev_n_rewarded_slope")
    intercept = _single_numeric_session_value(session_performance, "prev_n_rewarded_intercept")

    prediction_values = np.full(output_df.shape[0], "None", dtype=object)
    residual_values = np.full(output_df.shape[0], "None", dtype=object)
    valid_prediction_rows = (
        trials_to_correct.notna()
        & previous_rewards.notna()
        & np.isfinite(slope)
        & np.isfinite(intercept)
    )
    if valid_prediction_rows.any():
        predicted = intercept + slope * previous_rewards.loc[valid_prediction_rows]
        residual = trials_to_correct.loc[valid_prediction_rows] - predicted
        prediction_values[valid_prediction_rows.to_numpy()] = predicted.to_numpy(dtype=float)
        residual_values[valid_prediction_rows.to_numpy()] = residual.to_numpy(dtype=float)

    percentile_values = np.full(output_df.shape[0], "None", dtype=object)
    numeric_percentiles = _tts_percentile_values(trials_to_correct)
    valid_percentiles = numeric_percentiles.notna()
    percentile_values[valid_percentiles.to_numpy()] = numeric_percentiles.loc[
        valid_percentiles
    ].to_numpy(dtype=float)

    output_df["predicted_TTS_from_prev_rewards"] = prediction_values
    output_df["residual_TTS"] = residual_values
    output_df["TTS_percentile_within_session"] = percentile_values
    return output_df


def add_regression_stats_to_session_performance(
    session_performance: pd.DataFrame,
    trials_to_correct: pd.Series,
    prev_n_correct: pd.Series,
    prev_consecutive_rewards: pd.Series,
    prev_n_rewarded: pd.Series,
    n_blocks: int,
) -> pd.DataFrame:
    """Add block-learning regression summaries to a session summary.

    Parameters
    ----------
    session_performance : pd.DataFrame
        One-row session summary dataframe.
    trials_to_correct : pd.Series
        Blockwise trials-to-correct values with shape `(n_blocks,)`.
    prev_n_correct : pd.Series
        Previous-block correct-trial counts with shape `(n_blocks,)`.
    prev_consecutive_rewards : pd.Series
        Previous-block consecutive-reward counts with shape `(n_blocks,)`.
    prev_n_rewarded : pd.Series
        Previous-block rewarded-trial counts with shape `(n_blocks,)`.
    n_blocks : int
        Number of analyzed blocks in the session.

    Returns
    -------
    pd.DataFrame
        Copy of `session_performance` with explicit regression-stat columns
        for `prev_n_rewarded`, `prev_consecutive_rewards`, and
        `prev_n_correct`, plus `n_blocks`.
    """
    session_performance = session_performance.copy()
    trials_to_correct_numeric = _coerce_regression_values(trials_to_correct)
    prev_n_correct_numeric = _coerce_regression_values(prev_n_correct)
    prev_consecutive_rewards_numeric = _coerce_regression_values(prev_consecutive_rewards)
    prev_n_rewarded_numeric = _coerce_regression_values(prev_n_rewarded)
    valid_rows = (
        pd.Series(
            _get_valid_trials_to_correct_mask(trials_to_correct, prev_n_correct),
            index=trials_to_correct.index,
        )
        & trials_to_correct_numeric.notna()
        & prev_n_correct_numeric.notna()
    )

    dependent_var, independent_var = _numeric_regression_inputs(
        dependent_values=trials_to_correct_numeric,
        independent_values=prev_n_rewarded_numeric,
        base_valid_rows=valid_rows,
    )
    session_performance = _add_regression_stats(
        session_performance=session_performance,
        column_prefix="prev_n_rewarded",
        dependent_var=dependent_var,
        independent_var=independent_var,
    )
    dependent_var, independent_var = _numeric_regression_inputs(
        dependent_values=trials_to_correct_numeric,
        independent_values=prev_consecutive_rewards_numeric,
        base_valid_rows=valid_rows,
    )
    session_performance = _add_regression_stats(
        session_performance=session_performance,
        column_prefix="prev_consecutive_rewards",
        dependent_var=dependent_var,
        independent_var=independent_var,
    )
    dependent_var, independent_var = _numeric_regression_inputs(
        dependent_values=trials_to_correct_numeric,
        independent_values=prev_n_correct_numeric,
        base_valid_rows=valid_rows,
    )
    session_performance = _add_regression_stats(
        session_performance=session_performance,
        column_prefix="prev_n_correct",
        dependent_var=dependent_var,
        independent_var=independent_var,
    )
    session_performance["n_blocks"] = n_blocks
    return session_performance


def add_block_bias_columns(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Add legacy block-bias summaries to a block-performance table.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance dataframe with shape `(n_blocks, n_columns)`.
        Required columns are `trials_to_correct` and `prev_n_correct`.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with `bias_rl`, `bias_inf`,
        `min_value_bias`, `bias_rl_flag`, `bias_inf_flag`, and
        `bias_full_flag` columns, plus RL status columns using a one-trial
        grace period when `prev_n_correct == 0`. Invalid block rows receive
        the string sentinel `"None"`.
    """
    block_performance = block_performance.copy()
    valid_rows = _get_valid_trials_to_correct_mask(
        block_performance["trials_to_correct"],
        block_performance["prev_n_correct"],
    )

    base_array = np.zeros(block_performance.shape[0], dtype=object)
    base_array[:] = 0
    base_array[~valid_rows] = "None"

    trials_to_correct = block_performance.loc[valid_rows, "trials_to_correct"].astype(int)
    prev_n_correct = block_performance.loc[valid_rows, "prev_n_correct"].astype(int)

    bias_rl_values = (
        (trials_to_correct - prev_n_correct)
        / (trials_to_correct + prev_n_correct + eps)
    )
    bias_rl = base_array.copy()
    bias_rl[valid_rows] = bias_rl_values.to_numpy()

    inf_standard = 4
    bias_inf_values = (trials_to_correct - inf_standard) / (trials_to_correct + inf_standard)
    bias_inf = base_array.copy()
    bias_inf[valid_rows] = bias_inf_values.to_numpy()

    min_value_bias_values = pd.concat([bias_rl_values, bias_inf_values], axis=1).min(axis=1)
    min_value_bias = base_array.copy()
    min_value_bias[valid_rows] = min_value_bias_values.to_numpy()

    bias_thresh = .2
    bias_rl_flag_values = bias_rl_values > bias_thresh
    bias_inf_flag_values = bias_inf_values > bias_thresh
    bias_full_flag_values = bias_rl_flag_values & bias_inf_flag_values

    rl_effective_prev_n_correct_values = prev_n_correct.clip(lower=1)
    rl_thresh_values = np.floor(rl_effective_prev_n_correct_values * .5)
    rl_thresh_flag_values = trials_to_correct >= rl_thresh_values
    bias_rl_status_values = (
        (trials_to_correct - rl_effective_prev_n_correct_values)
        / (trials_to_correct + rl_effective_prev_n_correct_values + eps)
    )
    rl_biased_flag_values = bias_rl_status_values > bias_thresh
    rl_status_values = pd.Series("valid_rl", index=trials_to_correct.index, dtype=object)
    rl_status_values.loc[~rl_thresh_flag_values] = "below_rl_threshold"
    rl_status_values.loc[rl_thresh_flag_values & rl_biased_flag_values] = "biased_rl"

    bias_rl_flag = base_array.copy()
    bias_rl_flag[valid_rows] = bias_rl_flag_values.to_numpy()
    bias_inf_flag = base_array.copy()
    bias_inf_flag[valid_rows] = bias_inf_flag_values.to_numpy()
    bias_full_flag = base_array.copy()
    bias_full_flag[valid_rows] = bias_full_flag_values.to_numpy()
    rl_effective_prev_n_correct = base_array.copy()
    rl_effective_prev_n_correct[valid_rows] = rl_effective_prev_n_correct_values.to_numpy()
    rl_thresh = base_array.copy()
    rl_thresh[valid_rows] = rl_thresh_values.to_numpy()
    rl_thresh_flag = base_array.copy()
    rl_thresh_flag[valid_rows] = rl_thresh_flag_values.to_numpy()
    bias_rl_status_value = base_array.copy()
    bias_rl_status_value[valid_rows] = bias_rl_status_values.to_numpy()
    rl_status = base_array.copy()
    rl_status[valid_rows] = rl_status_values.to_numpy()

    block_performance["bias_rl"] = bias_rl
    block_performance["bias_inf"] = bias_inf
    block_performance["min_value_bias"] = min_value_bias
    block_performance["bias_rl_flag"] = bias_rl_flag
    block_performance["bias_inf_flag"] = bias_inf_flag
    block_performance["bias_full_flag"] = bias_full_flag
    block_performance["rl_effective_prev_n_correct"] = rl_effective_prev_n_correct
    block_performance["rl_thresh"] = rl_thresh
    block_performance["rl_thresh_flag"] = rl_thresh_flag
    block_performance["bias_rl_status_value"] = bias_rl_status_value
    block_performance["rl_status"] = rl_status
    return block_performance


def assert_saved_csv(path: Path) -> None:
    """Verify that a CSV save produced a non-empty file.

    Parameters
    ----------
    path : Path
        Expected CSV output path.

    Returns
    -------
    None
        Returns None when the file exists and has nonzero byte size.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    OSError
        If `path` exists but is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected saved CSV was not found: {path}")
    if path.stat().st_size == 0:
        raise OSError(f"Expected saved CSV is empty: {path}")


def add_session_metadata_columns(
    table: pd.DataFrame,
    mouse: str,
    session_id: str,
    date: str,
) -> pd.DataFrame:
    """Attach session identifiers to a saved analysis table.

    Parameters
    ----------
    table : pandas.DataFrame
        Analysis table with shape `(n_rows, n_columns)`. Rows may be block,
        trial, or session summaries.
    mouse : str
        Mouse identifier parsed from the session id.
    session_id : str
        Full behavior session identifier, usually
        `{mouse}_{YYYY-MM-DD}_{HHMMSS}`.
    date : str
        Session date formatted as `YYYY-MM-DD`.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with `mouse`, `session_id`, and `date` columns. Existing
        columns with these names are overwritten with the supplied metadata.
    """
    table_with_metadata = table.copy()
    metadata_values = {
        "mouse": mouse,
        "session_id": session_id,
        "date": date,
    }
    for column_name, column_value in metadata_values.items():
        if column_name in table_with_metadata.columns:
            table_with_metadata[column_name] = column_value
        else:
            insert_at = min(len(metadata_values), table_with_metadata.shape[1])
            table_with_metadata.insert(insert_at, column_name, column_value)
    leading_columns = [column for column in metadata_values if column in table_with_metadata.columns]
    remaining_columns = [
        column for column in table_with_metadata.columns if column not in leading_columns
    ]
    return table_with_metadata.loc[:, leading_columns + remaining_columns]


def prepare_block_model_outputs_for_session(
    block_performance: pd.DataFrame,
    session_performance: pd.DataFrame,
    session_id: str,
    mouse: str,
    date: str,
    residual_model_summary: pd.DataFrame | None = None,
    exemplar_model_parameters: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add block-model outputs needed by saved and returned session tables.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Expected columns
        for full model outputs are `block_type`, `prev_n_rewarded`, and
        `trials_to_correct`; residual models also use the same columns.
    session_performance : pandas.DataFrame
        One-row session summary table with shape `(1, n_columns)`.
    session_id : str
        Full behavior session identifier copied into residual-model summary
        rows.
    mouse : str
        Mouse identifier copied into residual-model summary rows.
    date : str
        Session date formatted as `YYYY-MM-DD`, copied into summary rows.
    residual_model_summary : pandas.DataFrame or None, default=None
        Precomputed residual-model summary table. If None, it is computed from
        `block_performance`.
    exemplar_model_parameters : pandas.DataFrame or None, default=None
        Precomputed exemplar parameter table. If None, it is created from the
        configured exemplar specs; block-level exemplar residual columns are
        added only when required input columns are present.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        `(updated_block_performance, updated_session_performance,
        residual_model_summary, exemplar_model_parameters)`. Block residuals
        and exemplar residuals are in trials; normalized exemplar residuals are
        in exemplar residual-SD units.
    """
    updated_blocks = add_session_block_collection_variables(block_performance)
    updated_session = session_performance.copy()

    if residual_model_summary is None:
        updated_blocks, updated_session, residual_model_summary = (
            block_residual_models.add_block_residual_model_outputs(
                block_performance=updated_blocks,
                session_performance=updated_session,
                session_id=session_id,
                mouse=mouse,
                date=date,
            )
        )

    if exemplar_model_parameters is None:
        exemplar_input_columns = {"trials_to_correct", "prev_n_rewarded"}
        if exemplar_input_columns.issubset(updated_blocks.columns):
            updated_blocks, exemplar_model_parameters = (
                block_exemplar_models.add_block_exemplar_model_outputs(updated_blocks)
            )
        else:
            exemplar_model_parameters = (
                block_exemplar_models.exemplar_model_parameters_dataframe(
                    block_exemplar_models.DEFAULT_BLOCK_EXEMPLAR_MODELS
                )
            )

    return updated_blocks, updated_session, residual_model_summary, exemplar_model_parameters


def save_analysis(
    session_performance: pd.DataFrame,
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    sess_id: str,
    session_save_path: Path,
    multisession_save_path: Path=None,
    residual_model_summary: pd.DataFrame | None = None,
    exemplar_model_parameters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Save within-session outputs and optionally update multisession summaries.

    Parameters
    ----------
    session_performance : pd.DataFrame
        Single-row session summary dataframe.
    block_performance : pd.DataFrame
        Blockwise summary dataframe.
    augmented_trial_df : pd.DataFrame
        Trialwise augmented dataframe.
    sess_id : str
        Full session identifier.
    session_save_path : Path
        Directory where within-session CSV outputs are written.
    multisession_save_path : Path or None, default=None
        Multisession summary directory. If None, only within-session outputs
        are written and `session_performance` is returned unchanged.
    residual_model_summary : pandas.DataFrame or None, default=None
        Precomputed residual-model summary table. Passing this avoids refitting
        residual models when `block_performance` has already been enriched.
    exemplar_model_parameters : pandas.DataFrame or None, default=None
        Precomputed exemplar parameter table. Passing this avoids rebuilding
        parameter output when `block_performance` has already been enriched.

    Returns
    -------
    pd.DataFrame
        Updated multisession dataframe if `multisession_save_path` is provided,
        otherwise the input `session_performance`.
    """
    assert session_save_path.exists(), "within-session data save path does not exist"

    match = re.search(r"(.+?)_(\d{4}-\d{2}-\d{2})_(\d{6})", sess_id)
    if match is None:
        raise ValueError(f"sess_id does not match expected pattern: {sess_id}")
    mouse, date, _time = match.groups()
    block_performance_path = session_save_path / (sess_id + '_block_performance.csv')
    augmented_trial_path = session_save_path / (sess_id + '_augmented_trials.csv')

    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    block_performance, session_performance, residual_model_summary, exemplar_model_parameters = (
        prepare_block_model_outputs_for_session(
            block_performance=block_performance,
            session_performance=session_performance,
            session_id=sess_id,
            mouse=mouse,
            date=date,
            residual_model_summary=residual_model_summary,
            exemplar_model_parameters=exemplar_model_parameters,
        )
    )
    session_performance = add_session_metadata_columns(
        session_performance,
        mouse=mouse,
        session_id=sess_id,
        date=date,
    )
    block_performance = add_session_metadata_columns(
        block_performance,
        mouse=mouse,
        session_id=sess_id,
        date=date,
    )
    block_performance.to_csv(block_performance_path, index=False, na_rep='None')
    augmented_trial_df.to_csv(augmented_trial_path, index=False, na_rep='None')
    residual_model_summary_paths = block_residual_models.save_block_residual_model_summaries(
        residual_model_summary,
        session_save_path=session_save_path,
        sess_id=sess_id,
    )
    exemplar_model_parameter_paths = block_exemplar_models.save_block_exemplar_model_parameters(
        exemplar_model_parameters,
        session_save_path=session_save_path,
        sess_id=sess_id,
    )
    assert_saved_csv(block_performance_path)
    assert_saved_csv(augmented_trial_path)
    for residual_model_summary_path in residual_model_summary_paths.values():
        assert_saved_csv(residual_model_summary_path)
    for exemplar_model_parameter_path in exemplar_model_parameter_paths.values():
        assert_saved_csv(exemplar_model_parameter_path)

    if multisession_save_path is None:
        return session_performance

    if not multisession_save_path.exists():#, "between-session data save path does not exist"
        multisession_save_path.mkdir(parents=True, exist_ok=False)

    multisession_summary_path = multisession_save_path / (mouse + '_overall_performance.csv')
    if multisession_summary_path.exists():
        multisession_df = pd.read_csv(multisession_summary_path, na_filter=False)
        current_session_date = multisession_df['date'] == date
        multisession_df = multisession_df[~current_session_date]
        multisession_df = pd.concat([multisession_df, session_performance], axis=0)
    else:
        multisession_df = session_performance.copy()

    multisession_df.sort_values(by='date', inplace=True)
    multisession_df.reset_index(drop=True, inplace=True)
    multisession_df.to_csv(multisession_summary_path, index=False, na_rep='None')
    assert_saved_csv(multisession_summary_path)
    return multisession_df


def load_analysis(sess_id_full: str, session_data_folder: Path, multisession_data_folder: Path) -> tuple:
    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    mouse, date, timestamp = match.groups()
    # print(f"Mouse id: {mouse}")  # abc
    # print(f"Date: {date}")  # YYYY-MM-DD
    # print(f"Time: {timestamp}")  # HHMMSS

    block_performance = pd.read_csv(session_data_folder / (sess_id_full + '_block_performance.csv'), sep=',', na_filter=False)
    augmented_trial_df = pd.read_csv(session_data_folder / (sess_id_full + '_augmented_trials.csv'), sep=',', na_filter=False)
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    multisession_df = pd.read_csv(multisession_data_folder / (mouse + '_overall_performance.csv'), sep=',', na_filter=False)
    return multisession_df, block_performance, augmented_trial_df


def run_analysis(
    trial_df: pd.DataFrame,
    session: Session,
    ideal_observer_n_replays: int = 100,
    ideal_observer_seed: int | None = 12345,
):
    """Run single-session analysis with optional multisession persistence.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns match
        `analyze_session`.
    session : Session
        Session metadata. Required attributes are `mouse`, `date`,
        `sess_id_full`, `processed_data_path`, and `multi_session_save_path`.
        If `multi_session_save_path` is None, only within-session CSV outputs
        are saved.
    ideal_observer_n_replays : int, default=100
        Number of fixed-state ideal-observer replay samples to run.
    ideal_observer_seed : int or None, default=12345
        Seed for fixed-state ideal-observer replay sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(augmented_trial_df, block_performance, multisession_df)`. When
        multisession saving is disabled, the third dataframe is the one-row
        session summary instead of a loaded cross-session table.
    """
    session_performance, block_performance, augmented_trial_df = analyze_session(
        trial_df,
        mouse=session.mouse,
        date=session.date,
        ideal_observer_n_replays=ideal_observer_n_replays,
        ideal_observer_seed=ideal_observer_seed,
    )
    session_performance = add_regression_stats_to_session_performance(
        session_performance=session_performance,
        trials_to_correct=block_performance["trials_to_correct"],
        prev_n_correct=block_performance["prev_n_correct"],
        prev_consecutive_rewards=block_performance["prev_consecutive_rewards"],
        prev_n_rewarded=block_performance["prev_n_rewarded"],
        n_blocks=block_performance.shape[0],
    )
    block_performance = add_relative_tts_metrics(
        block_performance=block_performance,
        session_performance=session_performance,
    )
    block_performance = add_block_bias_columns(block_performance)
    block_performance = switch_persistence.add_previous_block_omission_metrics(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
    )
    block_performance, session_performance, residual_model_summary, exemplar_model_parameters = (
        prepare_block_model_outputs_for_session(
            block_performance=block_performance,
            session_performance=session_performance,
            session_id=session.sess_id_full,
            mouse=session.mouse,
            date=session.date,
        )
    )

    print(
        "prev_consecutive_rewards slope: {}, intercept: {}, r_value: {}, p_value: {}".format(
            session_performance.loc[0, "prev_consecutive_rewards_slope"],
            session_performance.loc[0, "prev_consecutive_rewards_intercept"],
            session_performance.loc[0, "prev_consecutive_rewards_r_value"],
            session_performance.loc[0, "prev_consecutive_rewards_p_value"],
        )
    )
    print(
        "prev_n_correct slope: {}, intercept: {}, r_value: {}, p_value: {}".format(
            session_performance.loc[0, "prev_n_correct_slope"],
            session_performance.loc[0, "prev_n_correct_intercept"],
            session_performance.loc[0, "prev_n_correct_r_value"],
            session_performance.loc[0, "prev_n_correct_p_value"],
        )
    )

    multisession_df = save_analysis(
        session_performance,
        block_performance,
        augmented_trial_df,
        sess_id=session.sess_id_full,
        session_save_path=session.processed_data_path,
        multisession_save_path=getattr(session, "multi_session_save_path", None),
        residual_model_summary=residual_model_summary,
        exemplar_model_parameters=exemplar_model_parameters,
    )
    switch_persistence.save_switch_persistence_outputs(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        processed_data_path=session.processed_data_path,
        sess_id_full=session.sess_id_full,
    )
    return augmented_trial_df, block_performance, multisession_df


def main():
    """Analyze the data from a single mouse session"""

    data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = data_home / 'rpi' / sess_id_full
    processed_data_path = data_home / 'processed'
    figure_path = data_home / 'figures'
    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
    else:
        print("Double-check the session name!")
        return

    sess_id_abbreviated = mouse + '_' + date
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',', na_filter=False)
    sess = type("SessionConfig", (), {})()
    sess.mouse = mouse
    sess.date = date
    sess.sess_id_full = sess_id_full
    sess.processed_data_path = processed_data_path
    sess.multi_session_save_path = multi_session_save_path
    augmented_trial_df, block_performance, multisession_df = run_analysis(trial_df, session=sess)


if __name__ == '__main__':
    main()
