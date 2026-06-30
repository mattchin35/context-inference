import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
import src.behavior_analysis.trial_features as trial_features
from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    get_experimenter_reward_flags,
    is_zero_flag,
    make_no_choice_action_mask,
    normalize_experimenter_reward_column,
)
import json
from typing import Optional
from src.behavior_analysis import residualization


@dataclass
class TaskParams:
    n_states: int = 2  # could be 4 if I use the stimulus version, but that could also be 2 states but in some cases having information
    n_actions: int = 2  # in practice I am not planning more than 2 choices, could I remove this?

    # Probability parameters
    # For HHM inference model, you can play with using model parameters that are different from the true task parameters
    p_cue: float = 0  # 0 for uncued task, 1 for cued, between 0 and 1 for mixed. Must be between 0 and 1
    state_transition_prob: float = .2
    active_reward_probability: float = .8
    inactive_reward_probability: float = 0
    correct_reward_size: float = 1
    incorrect_reward_size: float = 0
    tanh_scale: float = 1
    hmm_reward_decay_lambda: float = 0.1
    # value_mode: str = 'bayesian_log_odds'  # 'expected_reward' or 'bayesian_log_odds'

    # Reinforcement learning parameters
    FQL_decay: float = .7  # for forgetting Q-learning agent
    FQL_reward_update_rate_fast_learn: float = .5
    QL_learning_rate: float = .3  # for standard Q-learning agent
    omission_lam: float = 0.5
    hazard_lam: float = 0.5
    perseveration_decay: float = 0.25
    max_explore_run_length: int = 5


LEFT_RIGHT_VALUE_COLUMNS_FOR_SIDE_EQUIVALENCE = (
    "Qlearning_rel_value",
    "FQlearning_rel_value",
    "FQlearning_rel_value_fast_learn",
    "HMM_rel_value_logodds",
    "HMM_rel_value_logodds_decay",
    "relative_omissions_index",
    "signed_omission_regressor",
    "relative_doubt_index",
    "relative_hazard_index",
    "perseveration_regressor",
    "doubt_perseveration_value",
    "wsls_regressor",
    "observer_value",
    "HMM_decay_res",
    "rel_hazard_res",
)

TRIAL_TYPE_FLAG_COLUMNS = (
    "prev_correct",
    "block_entry_trial",
    "switch_trial",
    "stay_trial",
    "first_switch_in_block",
    "explore_trial",
    "block_entry_explore_trial",
    "explore_run_start",
    "explore_run_trial",
    "explore_run_return",
    "explore_run_id",
    "explore_run_length",
)

def assert_saved_file(path: Path) -> None:
    """Verify that a save produced a non-empty file.

    Parameters
    ----------
    path : Path
        Expected output path.

    Returns
    -------
    None
        Returns None when the file exists and has nonzero byte size.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected saved file was not found: {path}")
    if path.stat().st_size == 0:
        raise OSError(f"Expected saved file is empty: {path}")


def save_trial_features(augmented_trial_df: pd.DataFrame, params: TaskParams, processed_data_path: Path,
                        sess_id_full: str) -> None:
    """Save augmented trial features and feature-generation parameters.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with feature columns, shape `(n_trials, n_columns)`.
    params : TaskParams
        Parameters used to generate feature columns.
    processed_data_path : Path
        Session processed-data directory.
    sess_id_full : str
        Full session identifier used in the augmented-trials filename.

    Returns
    -------
    None
        Writes `{sess_id_full}_augmented_trials.csv` and
        `trial_feature_params.json`.
    """
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df.to_csv(augmented_trial_df_path, index=False, na_rep='None')

    json_fname = processed_data_path / 'trial_feature_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)

    assert_saved_file(augmented_trial_df_path)
    assert_saved_file(json_fname)


def add_observer_value_feature(augmented_trial_df: pd.DataFrame) -> pd.DataFrame:
    """Add a left-positive observer value combining HMM belief and doubt.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `HMM_rel_value_logodds_decay` and `relative_doubt_index`.
        Both are unitless left-minus-right regressors.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with `observer_value` added. Values are
        unitless and use the same left-positive sign convention:
        `HMM_rel_value_logodds_decay - relative_doubt_index`. Rows with
        missing/non-numeric source values receive None.
    """
    required_columns = ["HMM_rel_value_logodds_decay", "relative_doubt_index"]
    missing_columns = [column for column in required_columns if column not in augmented_trial_df.columns]
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required observer columns: {missing_columns}")

    source_values = augmented_trial_df[required_columns].apply(pd.to_numeric, errors="coerce")
    observer_value = np.full(augmented_trial_df.shape[0], None, dtype=object)
    valid_rows = source_values.notna().all(axis=1)
    observer_value[valid_rows.to_numpy()] = (
        source_values.loc[valid_rows, "HMM_rel_value_logodds_decay"]
        - source_values.loc[valid_rows, "relative_doubt_index"]
    ).to_numpy(dtype=float)

    output_df = augmented_trial_df.copy()
    output_df["observer_value"] = observer_value
    return output_df


def _previous_action_side_sign(prev_action) -> float | None:
    """Return the sign that maps left-positive values to previous-action side.

    Parameters
    ----------
    prev_action : int, float, or str
        Previous trial action. Task coding is `0=right`, `1=left`; no-choice
        sentinels such as `"None"` and `"no_choice"` return None.

    Returns
    -------
    float or None
        `1.0` for previous-left, `-1.0` for previous-right, and None when no
        previous side is available.
    """
    if bool(make_no_choice_action_mask([prev_action]).iloc[0]):
        return None

    if isinstance(prev_action, str):
        normalized = prev_action.strip().lower()
        if normalized == "left":
            return 1.0
        if normalized == "right":
            return -1.0

    try:
        action_int = int(float(prev_action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prev_action must be 0/1, left/right, or no-choice; got {prev_action!r}.") from exc

    if action_int == 1:
        return 1.0
    if action_int == 0:
        return -1.0
    raise ValueError(f"prev_action must be 0 or 1 for side choices; got {prev_action!r}.")


def _choice_side_or_none(action) -> int | None:
    """Return a binary side choice from a raw action value.

    Parameters
    ----------
    action : int, float, or str
        Raw side action. Task coding is `0=right`, `1=left`; no-choice
        sentinels such as `"None"` and `"no_choice"` return None.

    Returns
    -------
    int or None
        `1` for left, `0` for right, and None when no side choice is available.
    """
    if bool(make_no_choice_action_mask([action]).iloc[0]):
        return None

    if isinstance(action, str):
        normalized = action.strip().lower()
        if normalized == "left":
            return 1
        if normalized == "right":
            return 0

    try:
        action_int = int(float(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be 0/1, left/right, or no-choice; got {action!r}.") from exc

    if action_int in (0, 1):
        return action_int
    raise ValueError(f"action must be 0 or 1 for side choices; got {action!r}.")


def make_leave_stay_action(raw_actions, prev_actions) -> np.ndarray:
    """Convert raw side choices into stay/leave outcomes.

    Parameters
    ----------
    raw_actions : array-like
        Current trial raw choices with shape `(n_trials,)`, coded as
        `0=right`, `1=left`, or a recognized no-choice sentinel.
    prev_actions : array-like
        Previous trial raw choices with shape `(n_trials,)`, using the same
        side convention as `raw_actions`.

    Returns
    -------
    np.ndarray
        Object array with shape `(n_trials,)`. Values are `1` for staying with
        the previous action, `0` for switching away from it, and `"None"` when
        either side choice is missing.
    """
    raw_action_array = np.asarray(raw_actions, dtype=object)
    prev_action_array = np.asarray(prev_actions, dtype=object)
    if raw_action_array.shape[0] != prev_action_array.shape[0]:
        raise ValueError("raw_actions and prev_actions must have the same length.")

    leave_stay_actions = np.full(raw_action_array.shape[0], "None", dtype=object)
    for trial_index, (raw_action, prev_action) in enumerate(zip(raw_action_array, prev_action_array)):
        raw_side = _choice_side_or_none(raw_action)
        prev_side = _choice_side_or_none(prev_action)
        if raw_side is None or prev_side is None:
            continue
        leave_stay_actions[trial_index] = int(raw_side == prev_side)
    return leave_stay_actions


def _previous_values(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Shift values by one row, optionally within session groups.

    Parameters
    ----------
    values : pd.Series
        One-dimensional values with shape `(n_trials,)`.
    groups : pd.Series or None, default=None
        Optional group labels with shape `(n_trials,)`. When provided, previous
        values do not cross group boundaries.

    Returns
    -------
    pd.Series
        Values shifted by one row, aligned to `values.index`.
    """
    if groups is None:
        return values.shift(1)
    return values.groupby(groups, sort=False).shift(1)


def _numeric_equals_one(values: pd.Series) -> pd.Series:
    """Return values that are numerically equal to one.

    Parameters
    ----------
    values : pd.Series
        One-dimensional values with shape `(n_trials,)`.

    Returns
    -------
    pd.Series
        Boolean values with shape `(n_trials,)`; missing or nonnumeric entries
        are False.
    """
    return pd.to_numeric(values, errors="coerce").eq(1).fillna(False)


def add_trial_type_flags(augmented_trial_df: pd.DataFrame) -> pd.DataFrame:
    """Add trial-type flags for selecting behaviorally interesting trials.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required column
        is `action`; optional columns `cur_block`, `prev_action`, `prev_reward`,
        `correct`, `reward`, `session_ID`, and experimenter-reward flags are
        used when present. Actions use the task convention `0=right`, `1=left`.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with boolean trial-type flags:
        `prev_correct`, `block_entry_trial`, `switch_trial`, `stay_trial`,
        `first_switch_in_block`, `explore_trial`, and
        `block_entry_explore_trial`.
    """
    if "action" not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df must contain 'action' to add trial-type flags.")

    output_df = normalize_experimenter_reward_column(augmented_trial_df)
    output_df = output_df.copy()

    current_sides = pd.Series(
        [_choice_side_or_none(action) for action in output_df["action"]],
        index=output_df.index,
        dtype=object,
    )
    experimenter_reward_flags = get_experimenter_reward_flags(output_df)
    experimenter_reward_is_zero = is_zero_flag(
        pd.Series(experimenter_reward_flags, index=output_df.index)
    )
    valid_choice = current_sides.notna() & experimenter_reward_is_zero

    if "prev_action" in output_df.columns:
        previous_sides = pd.Series(
            [_choice_side_or_none(action) for action in output_df["prev_action"]],
            index=output_df.index,
            dtype=object,
        )
    else:
        previous_sides = pd.Series([None] * output_df.shape[0], index=output_df.index, dtype=object)

    has_previous_side = previous_sides.notna()
    switch_trial = valid_choice & has_previous_side & current_sides.ne(previous_sides)
    stay_trial = valid_choice & has_previous_side & current_sides.eq(previous_sides)

    block_entry_trial = pd.Series(False, index=output_df.index)
    first_switch_in_block = pd.Series(False, index=output_df.index)
    if "cur_block" in output_df.columns:
        first_valid_indices = output_df.loc[valid_choice].groupby("cur_block", sort=False).head(1).index
        block_entry_trial.loc[first_valid_indices] = True

        first_switch_indices = output_df.loc[switch_trial].groupby("cur_block", sort=False).head(1).index
        first_switch_in_block.loc[first_switch_indices] = True

    session_groups = output_df["session_ID"] if "session_ID" in output_df.columns else None
    if "correct" in output_df.columns:
        correct_trial = _numeric_equals_one(output_df["correct"])
        prev_correct = (
            _previous_values(correct_trial, session_groups).fillna(False).astype(bool)
            & _previous_values(valid_choice, session_groups).fillna(False).astype(bool)
        )
    else:
        prev_correct = pd.Series(False, index=output_df.index)

    if "prev_reward" in output_df.columns:
        prev_rewarded = _numeric_equals_one(output_df["prev_reward"])
    elif "reward" in output_df.columns:
        prev_rewarded = (
            _previous_values(_numeric_equals_one(output_df["reward"]), session_groups)
            .fillna(False)
            .astype(bool)
            & _previous_values(valid_choice, session_groups).fillna(False).astype(bool)
        )
    else:
        prev_rewarded = pd.Series(False, index=output_df.index)

    explore_trial = switch_trial & prev_correct & prev_rewarded

    output_df["prev_correct"] = prev_correct.astype(bool).to_numpy()
    output_df["block_entry_trial"] = block_entry_trial.astype(bool).to_numpy()
    output_df["switch_trial"] = switch_trial.astype(bool).to_numpy()
    output_df["stay_trial"] = stay_trial.astype(bool).to_numpy()
    output_df["first_switch_in_block"] = first_switch_in_block.astype(bool).to_numpy()
    output_df["explore_trial"] = explore_trial.astype(bool).to_numpy()
    output_df["block_entry_explore_trial"] = (
        first_switch_in_block & explore_trial
    ).astype(bool).to_numpy()
    return output_df


def add_explore_run_flags(
    augmented_trial_df: pd.DataFrame,
    max_explore_run_length: int = 5,
) -> pd.DataFrame:
    """Tag short exploratory leave-return runs within each block.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action`, `cur_block`, `correct`, and `reward`. Actions use
        the task convention `0=right`, `1=left`; no-choice rows are ignored for
        run length and do not break a candidate run. Experimenter-reward rows
        are ignored when `experimenter_reward_given` or a legacy equivalent is
        present.
    max_explore_run_length : int, default=5
        Maximum number of exploratory-side valid choice trials allowed in one
        run. The initiating switch trial is included in this count.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with added columns:
        `explore_run_start`, `explore_run_trial`, `explore_run_return`,
        `explore_run_id`, and `explore_run_length`. Run lengths are counts of
        exploratory-side valid choice trials. Rows outside runs receive
        `"None"` for id/length.
    """
    if max_explore_run_length < 1:
        raise ValueError("max_explore_run_length must be at least 1.")

    required_columns = ["action", "cur_block", "correct", "reward"]
    missing_columns = [column for column in required_columns if column not in augmented_trial_df.columns]
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required explore-run columns: {missing_columns}")

    output_df = normalize_experimenter_reward_column(augmented_trial_df).copy()
    current_sides = pd.Series(
        [_choice_side_or_none(action) for action in output_df["action"]],
        index=output_df.index,
        dtype=object,
    )
    experimenter_reward_flags = get_experimenter_reward_flags(output_df)
    valid_choice = current_sides.notna() & pd.Series(experimenter_reward_flags, index=output_df.index).eq(0)
    correct_trial = _numeric_equals_one(output_df["correct"])
    rewarded_trial = _numeric_equals_one(output_df["reward"])

    explore_run_start = np.full(output_df.shape[0], False, dtype=bool)
    explore_run_trial = np.full(output_df.shape[0], False, dtype=bool)
    explore_run_return = np.full(output_df.shape[0], False, dtype=bool)
    explore_run_id = np.full(output_df.shape[0], "None", dtype=object)
    explore_run_length = np.full(output_df.shape[0], "None", dtype=object)
    index_positions = {index_label: position for position, index_label in enumerate(output_df.index)}

    next_run_id = 0
    for _block_id, block_df in output_df.groupby("cur_block", sort=False):
        block_indices = block_df.index.to_numpy()
        valid_indices = [index for index in block_indices if bool(valid_choice.loc[index])]
        valid_position = 1
        while valid_position < len(valid_indices):
            previous_index = valid_indices[valid_position - 1]
            start_index = valid_indices[valid_position]
            previous_side = current_sides.loc[previous_index]
            start_side = current_sides.loc[start_index]

            is_candidate_start = (
                start_side != previous_side
                and bool(correct_trial.loc[previous_index])
                and bool(rewarded_trial.loc[previous_index])
            )
            if not is_candidate_start:
                valid_position += 1
                continue

            exploratory_indices = []
            scan_position = valid_position
            accepted_return_index = None
            while scan_position < len(valid_indices):
                scan_index = valid_indices[scan_position]
                scan_side = current_sides.loc[scan_index]
                if scan_side == start_side:
                    exploratory_indices.append(scan_index)
                    if len(exploratory_indices) > max_explore_run_length:
                        break
                    scan_position += 1
                    continue

                if scan_side == previous_side and bool(correct_trial.loc[scan_index]):
                    accepted_return_index = scan_index
                break

            if (
                accepted_return_index is not None
                and 0 < len(exploratory_indices) <= max_explore_run_length
            ):
                start_position = index_positions[start_index]
                return_position = index_positions[accepted_return_index]
                exploratory_positions = [index_positions[index] for index in exploratory_indices]
                explore_run_start[start_position] = True
                explore_run_trial[exploratory_positions] = True
                explore_run_return[return_position] = True
                tagged_indices = exploratory_indices + [accepted_return_index]
                tagged_positions = [index_positions[index] for index in tagged_indices]
                explore_run_id[tagged_positions] = next_run_id
                explore_run_length[tagged_positions] = len(exploratory_indices)
                next_run_id += 1
                valid_position = scan_position + 1
            else:
                valid_position += 1

    output_df["explore_run_start"] = explore_run_start
    output_df["explore_run_trial"] = explore_run_trial
    output_df["explore_run_return"] = explore_run_return
    output_df["explore_run_id"] = explore_run_id
    output_df["explore_run_length"] = explore_run_length
    return output_df


def add_empty_explore_run_flags(augmented_trial_df: pd.DataFrame) -> pd.DataFrame:
    """Add neutral explore-run columns when required run inputs are unavailable.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with explore-run boolean columns set to
        False and id/length columns set to `"None"`.
    """
    output_df = augmented_trial_df.copy()
    output_df["explore_run_start"] = False
    output_df["explore_run_trial"] = False
    output_df["explore_run_return"] = False
    output_df["explore_run_id"] = "None"
    output_df["explore_run_length"] = "None"
    return output_df


def make_prev_action_side_equivalent_trial_values(
    augmented_trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a leave-stay oriented table using previous action as reference.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action`, `prev_action`, and `prev_reward`. Optional
        left/right value columns listed in
        `LEFT_RIGHT_VALUE_COLUMNS_FOR_SIDE_EQUIVALENCE` are transformed when
        present.

    Returns
    -------
    pd.DataFrame
        Side-equivalent table with shape `(n_trials, n_output_columns)`.
        `raw_action` preserves the original left/right choice, while `action`
        is recoded to `1=stay` and `0=leave/switch` relative to `prev_action`.
        For each value column, left-positive values are unchanged when
        `prev_action` is left and sign-flipped when `prev_action` is right.
        Rows without a previous side receive `"None"`.
    """
    required_columns = [
        "action",
        "prev_action",
        "prev_reward",
    ]
    missing_columns = [column for column in required_columns if column not in augmented_trial_df.columns]
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required leave-stay columns: {missing_columns}")

    output_df = pd.DataFrame(index=augmented_trial_df.index)
    for column_name in ("state", "state_int", "cur_trial", "cur_trial_in_block", "cur_block"):
        if column_name in augmented_trial_df.columns:
            output_df[column_name] = augmented_trial_df[column_name]
    output_df["raw_action"] = augmented_trial_df["action"]
    output_df["action"] = make_leave_stay_action(
        raw_actions=augmented_trial_df["action"],
        prev_actions=augmented_trial_df["prev_action"],
    )
    for column_name in (
        "correct",
        "reward",
        EXPERIMENTER_REWARD_GIVEN_COLUMN,
        "session_ID",
        "block_type",
        "time_to_choice",
        "prev_action",
        "prev_reward",
        "inherited_block_strategy",
        "inherited_block_bias",
    ):
        if column_name in augmented_trial_df.columns:
            output_df[column_name] = augmented_trial_df[column_name]
    for column_name in TRIAL_TYPE_FLAG_COLUMNS:
        if column_name in augmented_trial_df.columns:
            output_df[column_name] = augmented_trial_df[column_name]

    side_signs = np.array(
        [_previous_action_side_sign(prev_action) for prev_action in augmented_trial_df["prev_action"]],
        dtype=object,
    )
    has_reference_side = np.array([sign is not None for sign in side_signs], dtype=bool)
    numeric_signs = np.zeros(augmented_trial_df.shape[0], dtype=float)
    numeric_signs[has_reference_side] = np.asarray(side_signs[has_reference_side], dtype=float)

    for column_name in LEFT_RIGHT_VALUE_COLUMNS_FOR_SIDE_EQUIVALENCE:
        if column_name not in augmented_trial_df.columns:
            continue
        source_values = pd.to_numeric(augmented_trial_df[column_name], errors="coerce")
        valid_rows = has_reference_side & source_values.notna().to_numpy()
        side_values = np.full(augmented_trial_df.shape[0], "None", dtype=object)
        side_values[valid_rows] = source_values.to_numpy(dtype=float)[valid_rows] * numeric_signs[valid_rows]
        output_df[f"{column_name}_prev_action_side"] = side_values

    return output_df


def save_leave_stay_trial_values(
    augmented_trial_df: pd.DataFrame,
    processed_data_path: Path,
    sess_id_full: str,
) -> pd.DataFrame:
    """Save previous-action side-equivalent trial values to CSV.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`, including
        `prev_action` and the side-equivalent source value columns.
    processed_data_path : Path
        Session processed-data directory.
    sess_id_full : str
        Full session identifier used in the output filename.

    Returns
    -------
    pd.DataFrame
        Saved side-equivalent table, reloaded with `na_filter=False` so the
        returned dataframe matches CSV-loaded downstream behavior.
    """
    side_equivalent_df = make_prev_action_side_equivalent_trial_values(augmented_trial_df)
    save_path = processed_data_path / f"{sess_id_full}_leave_stay_trial_values.csv"
    side_equivalent_df.to_csv(save_path, index=False, na_rep="None")
    assert_saved_file(save_path)
    return pd.read_csv(save_path, na_filter=False)


def collect_and_save_trial_features(
    augmented_trial_df: pd.DataFrame,
    processed_data_path: Path,
    sess_id_full: str,
    params: Optional[TaskParams] = None,
    max_explore_run_length: int | None = None,
) -> tuple[pd.DataFrame, TaskParams]:
    """Collect trial features from an already loaded dataframe and save outputs.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe already loaded by the caller. Rows index trials and
        must include the columns required by `collect_trial_features`.
    processed_data_path : Path
        Session-local processed-data directory where the updated augmented trial
        CSV and feature parameter JSON will be written.
    sess_id_full : str
        Full session identifier used to name the augmented trial CSV.
    params : Optional[TaskParams], default=None
        Optional feature-generation parameters. If omitted, defaults are used.
    max_explore_run_length : int or None, default=None
        Maximum number of exploratory-side valid choice trials allowed in one
        explore run. If None, `params.max_explore_run_length` is used.

    Returns
    -------
    tuple[pd.DataFrame, TaskParams]
        - augmented_trial_df with feature columns added
        - parameters used to compute and save those features
    """
    augmented_trial_df, params = collect_trial_features(
        augmented_trial_df,
        params=params,
        max_explore_run_length=max_explore_run_length,
    )
    save_trial_features(augmented_trial_df, params, processed_data_path, sess_id_full)
    if "prev_action" in augmented_trial_df.columns:
        save_leave_stay_trial_values(augmented_trial_df, processed_data_path, sess_id_full)
    return augmented_trial_df, params


def _get_column_or_raise(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    raise ValueError(f"augmented_trial_df must contain one of columns: {candidates}")


def validate_trial_feature_inputs(augmented_trial_df: pd.DataFrame) -> None:
    """Validate columns required for model-value feature generation.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.

    Returns
    -------
    None
        Raises ValueError if required columns are missing.
    """
    if 'action' not in augmented_trial_df.columns or 'reward' not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df must contain 'action' and 'reward' columns.")


def collect_trial_index_features(
    augmented_trial_df: pd.DataFrame,
    omission_lam: float = 0.5,
    hazard_lam: float = 0.5,
    perseveration_decay: float = 0.25,
) -> pd.DataFrame:
    """Add trial index/regressor features derived from trial-history counters.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns include side-specific omission counters, `consecutive_omissions`,
        `relative_monotonic_cf_value`, `action`, `reward`, and optional
        `experimenter_reward_given`.
    omission_lam : float, default=0.5
        Positive saturation parameter for omission-derived regressors.
    hazard_lam : float, default=0.5
        Positive saturation parameter for `relative_hazard_index`.
    perseveration_decay : float, default=0.25
        Exponential decay parameter for the perseveration regressor.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with omission, hazard, and perseveration
        regressor columns added. Regressors use left-positive sign convention.
        `doubt_perseveration_value` is the unfit sum of
        `relative_doubt_index + perseveration_regressor`; `wsls_regressor` is
        a signed win-stay/lose-switch heuristic based on the last valid choice.
    """
    right_omissions = _get_column_or_raise(augmented_trial_df, ("right_omissions",)).to_numpy()
    left_omissions = _get_column_or_raise(augmented_trial_df, ("left_omissions",)).to_numpy()
    right_omissions_counterfactual = _get_column_or_raise(augmented_trial_df, ("right_cf_omissions",)).to_numpy()
    left_omissions_counterfactual = _get_column_or_raise(augmented_trial_df, ("left_cf_omissions",)).to_numpy()
    loss_streak = _get_column_or_raise(augmented_trial_df, ("consecutive_omissions",)).to_numpy()
    relative_monotonic_cf_value = _get_column_or_raise(
        augmented_trial_df,
        ("relative_monotonic_cf_value",),
    ).to_numpy()
    actions = _get_column_or_raise(augmented_trial_df, ("action",)).to_numpy()
    rewards = _get_column_or_raise(augmented_trial_df, ("reward",)).to_numpy()

    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    n_trials = augmented_trial_df.shape[0]
    experimenter_reward_given = get_experimenter_reward_flags(augmented_trial_df)

    skip_mask = trial_features.make_skip_trial_mask(experimenter_reward_given=experimenter_reward_given, actions=actions)
    if skip_mask is None:
        skip_mask = np.zeros(n_trials, dtype=bool)
    valid_mask = ~skip_mask

    relative_omissions_index = np.full(n_trials, None, dtype=object)
    signed_omission_regressor = np.full(n_trials, None, dtype=object)
    relative_doubt_index = np.full(n_trials, None, dtype=object)
    relative_hazard_index = np.full(n_trials, None, dtype=object)
    perseveration_regressor = np.full(n_trials, None, dtype=object)
    doubt_perseveration_value = np.full(n_trials, None, dtype=object)

    rel_omission_valid = trial_features.relative_omissions_index(
        R_omissions=right_omissions_counterfactual[valid_mask],
        L_omissions=left_omissions_counterfactual[valid_mask],
    )
    signed_omission_valid = trial_features.signed_omission_regressor(
        loss_streak=loss_streak[valid_mask],
        lam=omission_lam,
        choice_side=actions[valid_mask],
    )
    rel_doubt_valid = trial_features.relative_doubt_index(
        R_omissions=right_omissions_counterfactual[valid_mask],
        L_omissions=left_omissions_counterfactual[valid_mask],
        lam=omission_lam,
    )
    rel_hazard_valid = trial_features.relative_hazard_index(
        relative_monotonic_cf_value=relative_monotonic_cf_value[valid_mask],
        lam=hazard_lam,
    )
    perseveration_valid = trial_features.perseveration_regressor(
        choices=actions[valid_mask],
        decay=perseveration_decay,
    )

    relative_omissions_index[valid_mask] = rel_omission_valid
    signed_omission_regressor[valid_mask] = signed_omission_valid
    relative_doubt_index[valid_mask] = rel_doubt_valid
    relative_hazard_index[valid_mask] = rel_hazard_valid
    perseveration_regressor[valid_mask] = perseveration_valid

    doubt_perseveration_value[valid_mask] = (
        np.asarray(rel_doubt_valid, dtype=float)
        + np.asarray(perseveration_valid, dtype=float)
    )
    wsls_regressor = trial_features.win_stay_lose_switch_regressor(
        actions=actions,
        rewards=rewards,
        experimenter_reward_given=experimenter_reward_given,
    )

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df["relative_omissions_index"] = relative_omissions_index
    augmented_trial_df["signed_omission_regressor"] = signed_omission_regressor
    augmented_trial_df["relative_doubt_index"] = relative_doubt_index
    augmented_trial_df["relative_hazard_index"] = relative_hazard_index
    augmented_trial_df["perseveration_regressor"] = perseveration_regressor
    augmented_trial_df["doubt_perseveration_value"] = doubt_perseveration_value
    augmented_trial_df["wsls_regressor"] = wsls_regressor
    return augmented_trial_df


def collect_model_value_features(
    augmented_trial_df: pd.DataFrame,
    params: TaskParams,
) -> pd.DataFrame:
    """Add model-derived relative-value features to an augmented trial table.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`, including
        `action`, `reward`, and optionally `experimenter_reward_given`.
    params : TaskParams
        Parameters used by Q-learning and HMM feature functions.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with model-value feature columns added.
    """
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    validate_trial_feature_inputs(augmented_trial_df)
    actions = augmented_trial_df['action'].values
    rewards = augmented_trial_df['reward'].values
    experimenter_reward_given = (
        augmented_trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN].values
        if EXPERIMENTER_REWARD_GIVEN_COLUMN in augmented_trial_df.columns
        else None
    )

    ql_rel_value = trial_features.qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        learning_rate=params.QL_learning_rate,
        n_actions=params.n_actions,
        experimenter_reward_given=experimenter_reward_given,
    )
    # Standard forgetting-Q: decay=.7 and default reward update rate (1 - decay).
    fql_rel_value = trial_features.forgetting_qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        decay=params.FQL_decay,
        n_actions=params.n_actions,
        experimenter_reward_given=experimenter_reward_given,
    )
    # Fast-learn forgetting-Q: same decay but explicit reward update rate.
    fql_rel_value_fast_learn = trial_features.forgetting_qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        decay=params.FQL_decay,
        reward_update_rate=params.FQL_reward_update_rate_fast_learn,
        n_actions=params.n_actions,
        experimenter_reward_given=experimenter_reward_given,
    )
    hmm_rel_value_logodds = trial_features.hmm_relative_value(
        actions=actions,
        rewards=rewards,
        state_transition_prob=params.state_transition_prob,
        active_reward_probability=params.active_reward_probability,
        inactive_reward_probability=params.inactive_reward_probability,
        correct_reward_size=params.correct_reward_size,
        incorrect_reward_size=params.incorrect_reward_size,
        value_mode='bayesian_log_odds',
        tanh_scale=params.tanh_scale,
        experimenter_reward_given=experimenter_reward_given,
    )
    hmm_rel_value_logodds_decay = trial_features.hmm_relative_value_reward_decay(
        actions=actions,
        rewards=rewards,
        state_transition_prob=params.state_transition_prob,
        active_reward_probability=params.active_reward_probability,
        inactive_reward_probability=params.inactive_reward_probability,
        correct_reward_size=params.correct_reward_size,
        incorrect_reward_size=params.incorrect_reward_size,
        lambda_decay=params.hmm_reward_decay_lambda,
        value_mode='bayesian_log_odds',
        tanh_scale=params.tanh_scale,
        experimenter_reward_given=experimenter_reward_given,
    )

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df['Qlearning_rel_value'] = ql_rel_value
    augmented_trial_df['FQlearning_rel_value'] = fql_rel_value
    augmented_trial_df['FQlearning_rel_value_fast_learn'] = fql_rel_value_fast_learn
    augmented_trial_df['HMM_rel_value_logodds'] = hmm_rel_value_logodds
    augmented_trial_df['HMM_rel_value_logodds_decay'] = hmm_rel_value_logodds_decay
    return augmented_trial_df


def collect_residualized_trial_features(augmented_trial_df: pd.DataFrame) -> pd.DataFrame:
    """Add residualized value predictors to an augmented trial table.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `FQlearning_rel_value`, `HMM_rel_value_logodds_decay`,
        and `relative_hazard_index`.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with `HMM_decay_res` and
        `rel_hazard_res` columns added. Rows with missing/non-numeric source
        values are kept and receive `None` in residualized columns.
    """
    required_columns = [
        "FQlearning_rel_value",
        "HMM_rel_value_logodds_decay",
        "relative_hazard_index",
    ]
    missing_columns = [column for column in required_columns if column not in augmented_trial_df.columns]
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required residualization columns: {missing_columns}")

    numeric_features = augmented_trial_df[required_columns].apply(pd.to_numeric, errors="coerce")
    n_trials = augmented_trial_df.shape[0]
    hmm_decay_res = np.full(n_trials, None, dtype=object)
    rel_hazard_res = np.full(n_trials, None, dtype=object)

    # Residualize overlapping value representations so trial GLMs can compare
    # HMM-unique and hazard-unique components without replacing the raw predictors.
    valid_hmm_rows = numeric_features[["FQlearning_rel_value", "HMM_rel_value_logodds_decay"]].notna().all(axis=1)
    if valid_hmm_rows.any():
        hmm_decay_res[valid_hmm_rows.to_numpy()] = residualization.residualize_values(
            target_values=numeric_features.loc[valid_hmm_rows, "HMM_rel_value_logodds_decay"],
            control_regressors=numeric_features.loc[valid_hmm_rows, ["FQlearning_rel_value"]],
        )

    residualized_feature_frame = numeric_features.copy()
    residualized_feature_frame["HMM_decay_res"] = pd.to_numeric(hmm_decay_res, errors="coerce")
    valid_hazard_rows = residualized_feature_frame[
        ["FQlearning_rel_value", "HMM_decay_res", "relative_hazard_index"]
    ].notna().all(axis=1)
    if valid_hazard_rows.any():
        rel_hazard_res[valid_hazard_rows.to_numpy()] = residualization.residualize_values(
            target_values=residualized_feature_frame.loc[valid_hazard_rows, "relative_hazard_index"],
            control_regressors=residualized_feature_frame.loc[
                valid_hazard_rows,
                ["FQlearning_rel_value", "HMM_decay_res"],
            ],
        )

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df["HMM_decay_res"] = hmm_decay_res
    augmented_trial_df["rel_hazard_res"] = rel_hazard_res
    return augmented_trial_df


def collect_trial_features(
    augmented_trial_df: pd.DataFrame,
    params: Optional[TaskParams] = None,
    max_explore_run_length: int | None = None,
) -> tuple[pd.DataFrame, TaskParams]:
    """Add model-derived and trial-type features to an augmented trial table.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. It must contain
        the columns required by model-value features and trial-type/run flags.
    params : Optional[TaskParams], default=None
        Feature-generation parameters. If omitted, defaults are used.
    max_explore_run_length : int or None, default=None
        Maximum number of exploratory-side valid choice trials allowed in one
        explore run. If None, `params.max_explore_run_length` is used.

    Returns
    -------
    tuple[pd.DataFrame, TaskParams]
        Feature dataframe with shape `(n_trials, n_columns + features)` and
        the parameters used to compute the features.
    """
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    if params is None:
        params = TaskParams()
    if max_explore_run_length is not None:
        params.max_explore_run_length = max_explore_run_length

    augmented_trial_df = collect_model_value_features(augmented_trial_df, params=params)
    augmented_trial_df = collect_trial_index_features(
        augmented_trial_df,
        omission_lam=params.omission_lam,
        hazard_lam=params.hazard_lam,
        perseveration_decay=params.perseveration_decay,
    )
    augmented_trial_df = add_observer_value_feature(augmented_trial_df)
    augmented_trial_df = collect_residualized_trial_features(augmented_trial_df)
    augmented_trial_df = add_trial_type_flags(augmented_trial_df)
    explore_run_required_columns = {"action", "cur_block", "correct", "reward"}
    if explore_run_required_columns.issubset(augmented_trial_df.columns):
        augmented_trial_df = add_explore_run_flags(
            augmented_trial_df,
            max_explore_run_length=params.max_explore_run_length,
        )
    else:
        augmented_trial_df = add_empty_explore_run_flags(augmented_trial_df)
    return augmented_trial_df, params


def main():
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference/')
    # sess_id_full = 'CT014_2025-12-16_153200'
    # sess_id_full = 'CT014_2025-12-05_165240'
    sess_id_full = 'CT014_2025-12-23_163505'
    processed_data_path = session_data_home / 'processed'

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)

    params = TaskParams()
    params.p_cue = 0
    params.correct_reward_size = 1
    params.incorrect_reward_size = 0
    params.QL_learning_rate = .3

    augmented_trial_df, params = collect_and_save_trial_features(
        augmented_trial_df,
        processed_data_path=processed_data_path,
        sess_id_full=sess_id_full,
        params=params,
    )


if __name__ == '__main__':
    main()
