"""Augmented-table validation and chronological task-variable target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    is_present_value,
    is_zero_flag,
    make_no_choice_action_mask,
)
from src.neural_analysis.task_decoding.config import (
    TARGET_DEFINITION_BY_IDENTIFIER,
    TaskDecodingConfig,
)


_SHARED_COLUMNS = ("cur_trial", "cur_block", "action", EXPERIMENTER_REWARD_GIVEN_COLUMN)
_CATEGORICAL_CLASS_LABELS = {
    "current_state": {"0": "right", "1": "left"},
    "current_action": {"0": "right", "1": "left"},
    "current_action_is_correct": {"0": "incorrect", "1": "correct"},
    "previous_action": {"0": "right", "1": "left"},
    "previous_action_was_rewarded": {"0": "unrewarded", "1": "rewarded"},
    "next_action": {"0": "right", "1": "left"},
    "current_choice_switch_stay": {"0": "stay", "1": "switch"},
    "next_choice_switch_stay": {"0": "stay", "1": "switch"},
}


def _required_columns(config: TaskDecodingConfig) -> set[str]:
    """Collect required columns for one selected task-decoding run.

    Parameters
    ----------
    config : TaskDecodingConfig
        Validated configuration naming one alignment and target subset.

    Returns
    -------
    set[str]
        Shared identity/baseline, alignment, and selected target-source names.
    """
    source_columns = {
        TARGET_DEFINITION_BY_IDENTIFIER[target_name].source_column
        for target_name in config.target_names
    }
    return set(_SHARED_COLUMNS) | {config.alignment} | source_columns


def _present_numeric(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Apply project missing-sentinel semantics before numeric coercion.

    Parameters
    ----------
    values : pd.Series
        One-dimensional source column with shape ``(n_trials,)`` and native
        table units.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Boolean project-present mask and float numeric values, each shape
        ``(n_trials,)``. Missing sentinels become NaN without mutating input.
    """
    present = is_present_value(values)
    numeric = pd.to_numeric(values.where(present), errors="coerce")
    return present, numeric


def _raise_if_invalid_numeric_column(values: pd.Series, column_name: str) -> pd.Series:
    """Validate present numeric source values and return float table values.

    Parameters
    ----------
    values : pd.Series
        Source column with shape ``(n_trials,)``.
    column_name : str
        Column name included in validation errors.

    Returns
    -------
    pd.Series
        Float values with shape ``(n_trials,)``; project missing sentinels are
        NaN and all present entries are finite.

    Raises
    ------
    ValueError
        If a present value is malformed or non-finite.
    """
    present, numeric = _present_numeric(values)
    invalid = present & (~np.isfinite(numeric))
    if invalid.any():
        raise ValueError(f"{column_name} has malformed or non-finite present values.")
    return numeric


def _validate_binary_column(values: pd.Series, column_name: str) -> pd.Series:
    """Validate exact present 0/1 labels while retaining missing row values.

    Parameters
    ----------
    values : pd.Series
        Source-coded categorical values with shape ``(n_trials,)``.
    column_name : str
        Source name used in errors.

    Returns
    -------
    pd.Series
        Float-coded 0/1/NaN Series with shape ``(n_trials,)``.
    """
    numeric = _raise_if_invalid_numeric_column(values, column_name)
    present = is_present_value(values)
    invalid = present & ~numeric.isin([0.0, 1.0])
    if invalid.any():
        raise ValueError(f"{column_name} must contain exact numeric 0 or 1 labels.")
    return numeric


def _action_values_and_validity(action_values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Validate task-coded actions and identify valid animal choices.

    Parameters
    ----------
    action_values : pd.Series
        Action column with shape ``(n_trials,)``. Valid animal choices are
        exactly numeric 0 (right) and 1 (left); project no-choice sentinels are
        allowed row-level missingness.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Numeric action values and Boolean valid-choice mask, each shape
        ``(n_trials,)``.

    Raises
    ------
    ValueError
        If a present non-no-choice action is malformed, fractional, or nonfinite.
    """
    no_choice = make_no_choice_action_mask(action_values) | ~is_present_value(action_values)
    numeric = pd.to_numeric(action_values.where(~no_choice), errors="coerce")
    invalid = (~no_choice) & (~np.isfinite(numeric) | ~numeric.isin([0.0, 1.0]))
    if invalid.any():
        raise ValueError("action must contain exact numeric 0 or 1 labels when present.")
    return numeric, ~no_choice


def _manual_reward_mask(values: pd.Series) -> pd.Series:
    """Identify nonzero experimenter/manual reward flags.

    Parameters
    ----------
    values : pd.Series
        Experimenter flag column with shape ``(n_trials,)``. Numeric zero means
        animal-controlled; nonzero or project-missing entries are manual.

    Returns
    -------
    pd.Series
        Boolean manual-reward mask with shape ``(n_trials,)``.

    Raises
    ------
    ValueError
        If a present flag is malformed or non-finite.
    """
    _raise_if_invalid_numeric_column(values, EXPERIMENTER_REWARD_GIVEN_COLUMN)
    try:
        return ~is_zero_flag(values)
    except ValueError as exc:
        raise ValueError("experimenter_reward_given must contain numeric flag values.") from exc


def _validate_cur_trial(trial_df: pd.DataFrame) -> None:
    """Validate the complete zero-based chronological trial-ID sequence.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Augmented table with shape ``(n_trials, n_columns)`` containing
        ``cur_trial``.

    Returns
    -------
    None
        Raises ValueError unless ``cur_trial`` equals ``0, ..., n_trials - 1``.
    """
    if trial_df.empty:
        raise ValueError("Augmented trial table must not be empty.")
    if not is_present_value(trial_df["cur_trial"]).all():
        raise ValueError("cur_trial must be present for every row.")
    numeric = _raise_if_invalid_numeric_column(trial_df["cur_trial"], "cur_trial")
    numeric_array = numeric.to_numpy(dtype=float)
    if not np.all(np.equal(numeric_array, np.floor(numeric_array))):
        raise ValueError("cur_trial must be integer-valued.")
    expected = np.arange(trial_df.shape[0], dtype=float)
    if not np.array_equal(numeric_array, expected):
        raise ValueError("cur_trial must equal the complete zero-based row sequence.")


def _validate_blocks(block_values: pd.Series) -> None:
    """Validate present block labels occur in one chronological segment each.

    Parameters
    ----------
    block_values : pd.Series
        Block identity column with shape ``(n_trials,)`` in chronological order.

    Returns
    -------
    None
        Raises ValueError for project-missing labels or noncontiguous reuse.
    """
    if not is_present_value(block_values).all():
        raise ValueError("cur_block must be present for every row.")
    seen_blocks: set[tuple[str, str]] = set()
    previous_block: tuple[str, str] | None = None
    for block_value in block_values:
        block_key = (type(block_value).__name__, repr(block_value))
        if block_key != previous_block:
            if block_key in seen_blocks:
                raise ValueError("cur_block labels must occupy contiguous row segments.")
            seen_blocks.add(block_key)
            previous_block = block_key


def validate_augmented_trials(trial_df: pd.DataFrame, config: TaskDecodingConfig) -> None:
    """Validate a selected run's small augmented behavioral input table.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Chronological augmented trial table with shape ``(n_trials, n_columns)``.
        It is not modified. Required columns depend on ``config.target_names``
        and ``config.alignment``.
    config : TaskDecodingConfig
        Validated decoding configuration defining selected target sources.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        For schema, identity, grouping, numeric, or category violations before
        neural arrays are loaded.
    """
    if not isinstance(trial_df, pd.DataFrame):
        raise ValueError("trial_df must be a pandas DataFrame.")
    missing_columns = sorted(_required_columns(config) - set(trial_df.columns))
    if missing_columns:
        raise ValueError(f"Augmented trial table is missing required columns: {missing_columns}")
    _validate_cur_trial(trial_df)
    _validate_blocks(trial_df["cur_block"])
    _action_values_and_validity(trial_df["action"])
    _manual_reward_mask(trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN])
    alignment_values = _raise_if_invalid_numeric_column(
        trial_df[config.alignment],
        config.alignment,
    )
    if not np.isfinite(alignment_values).any():
        raise ValueError(f"{config.alignment} must contain at least one finite value.")

    for target_name in config.target_names:
        definition = TARGET_DEFINITION_BY_IDENTIFIER[target_name]
        source = trial_df[definition.source_column]
        if target_name in {
            "current_action",
            "previous_action",
            "next_action",
            "current_choice_switch_stay",
            "next_choice_switch_stay",
        }:
            continue
        if target_name == "current_state":
            numeric = _raise_if_invalid_numeric_column(source, definition.source_column)
            present = is_present_value(source)
            if (present & ~numeric.isin([0.0, 1.0, 2.0])).any():
                raise ValueError("state_int must contain 0, 1, or known dark state 2.")
            continue
        if target_name == "current_action_is_correct":
            _validate_binary_column(source, definition.source_column)
            continue
        if target_name == "previous_action_was_rewarded":
            _raise_if_invalid_numeric_column(source, definition.source_column)
            continue
        numeric = _raise_if_invalid_numeric_column(source, definition.source_column)
        if definition.family == "numerical" and not np.isfinite(numeric).any():
            raise ValueError(f"{definition.source_column} must contain at least one finite value.")


def _target_metadata(config: TaskDecodingConfig) -> dict[str, dict[str, object]]:
    """Build selected-target metadata for result provenance.

    Parameters
    ----------
    config : TaskDecodingConfig
        Validated configuration naming a nonempty target subset.

    Returns
    -------
    dict[str, dict[str, object]]
        JSON-safe metadata keyed in canonical target order. Categorical entries
        include their original binary labels, numeric mapping, and positive
        class; numerical entries retain source and derivation metadata.
    """
    metadata: dict[str, dict[str, object]] = {}
    for target_name in config.target_names:
        definition = TARGET_DEFINITION_BY_IDENTIFIER[target_name]
        entry: dict[str, object] = {
            "family": definition.family,
            "display_label": definition.display_label,
            "source_column": definition.source_column,
            "derivation": definition.derivation,
        }
        if definition.family == "categorical":
            entry["positive_class"] = 1
            entry["positive_label"] = definition.positive_label
            entry["class_labels"] = dict(_CATEGORICAL_CLASS_LABELS[target_name])
        metadata[target_name] = entry
    return metadata


def _numeric_source_column(
    trial_df: pd.DataFrame,
    numeric_cache: dict[str, pd.Series],
    column_name: str,
) -> pd.Series:
    """Return cached validated numeric values for one augmented-table source.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Range-indexed working view with shape ``(n_trials, n_columns)``.
    numeric_cache : dict[str, pd.Series]
        Per-source cache storing validated float Series of shape ``(n_trials,)``.
    column_name : str
        Existing source-column name to validate and retrieve.

    Returns
    -------
    pd.Series
        Float source values with shape ``(n_trials,)``; missing sentinels are
        NaN and present malformed values raise ValueError.

    Raises
    ------
    ValueError
        If a present source value is malformed or non-finite.
    """
    if column_name not in numeric_cache:
        numeric_cache[column_name] = _raise_if_invalid_numeric_column(
            trial_df[column_name],
            column_name,
        )
    return numeric_cache[column_name]


def build_target_table(trial_df: pd.DataFrame, config: TaskDecodingConfig) -> pd.DataFrame:
    """Build unfiltered chronological encoded targets and target-specific masks.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Chronological augmented table with shape ``(n_trials, n_columns)``.
        The caller's index labels are not used as trial positions, and the input
        dataframe is not modified.
    config : TaskDecodingConfig
        Validated configuration naming targets and alignment source.

    Returns
    -------
    pd.DataFrame
        Range-indexed table with shape ``(n_trials, 5 + 2 * n_targets)``. It
        contains zero-based ``row_position``, ``trial_id``, ``block_id``, one
        encoded numeric column and ``<target>_valid`` mask per selected target,
        plus target-independent ``baseline_valid`` and
        ``baseline_invalid_reason`` fields. Numerical target values retain
        their source units without normalization.
    """
    validate_augmented_trials(trial_df, config)
    working_df = trial_df.reset_index(drop=True).copy(deep=True)
    n_trials = working_df.shape[0]
    action_values, action_is_choice = _action_values_and_validity(working_df["action"])
    manual_reward = _manual_reward_mask(working_df[EXPERIMENTER_REWARD_GIVEN_COLUMN])
    alignment_values = _raise_if_invalid_numeric_column(
        working_df[config.alignment],
        config.alignment,
    )
    alignment_valid = np.isfinite(alignment_values)
    baseline_valid = action_is_choice & ~manual_reward & alignment_valid

    baseline_reasons: list[str] = []
    for row_position in range(n_trials):
        reasons: list[str] = []
        if manual_reward.iloc[row_position]:
            reasons.append("manual_reward")
        if not action_is_choice.iloc[row_position]:
            reasons.append("no_animal_choice")
        if not alignment_valid.iloc[row_position]:
            reasons.append("missing_alignment")
        baseline_reasons.append(";".join(reasons))

    target_table = pd.DataFrame(
        {
            "row_position": np.arange(n_trials, dtype=int),
            "trial_id": working_df["cur_trial"].to_numpy(copy=True),
            "block_id": working_df["cur_block"].to_numpy(copy=True),
            "baseline_valid": baseline_valid.to_numpy(dtype=bool),
            "baseline_invalid_reason": baseline_reasons,
        }
    )
    valid_choice = action_is_choice & ~manual_reward
    numeric_cache: dict[str, pd.Series] = {}

    for target_name in config.target_names:
        definition = TARGET_DEFINITION_BY_IDENTIFIER[target_name]
        values = pd.Series(np.nan, index=target_table.index, dtype=float)
        valid = pd.Series(False, index=target_table.index, dtype=bool)

        if target_name == "current_state":
            source = _numeric_source_column(
                working_df,
                numeric_cache,
                definition.source_column,
            )
            valid = source.isin([0.0, 1.0])
            values.loc[valid] = source.loc[valid]
        elif target_name == "current_action":
            valid = action_is_choice
            values.loc[valid] = action_values.loc[valid]
        elif target_name == "current_action_is_correct":
            source = _numeric_source_column(
                working_df,
                numeric_cache,
                definition.source_column,
            )
            valid = source.isin([0.0, 1.0])
            values.loc[valid] = source.loc[valid]
        elif target_name == "previous_action":
            for row_position in range(1, n_trials):
                if valid_choice.iloc[row_position - 1]:
                    values.iloc[row_position] = action_values.iloc[row_position - 1]
                    valid.iloc[row_position] = True
        elif target_name == "previous_action_was_rewarded":
            reward_values = _numeric_source_column(
                working_df,
                numeric_cache,
                definition.source_column,
            )
            for row_position in range(1, n_trials):
                previous_position = row_position - 1
                if valid_choice.iloc[previous_position] and np.isfinite(
                    reward_values.iloc[previous_position]
                ):
                    values.iloc[row_position] = int(reward_values.iloc[previous_position] > 0)
                    valid.iloc[row_position] = True
        elif target_name == "next_action":
            for row_position in range(n_trials - 1):
                next_position = row_position + 1
                if valid_choice.iloc[next_position]:
                    values.iloc[row_position] = action_values.iloc[next_position]
                    valid.iloc[row_position] = True
        elif target_name == "current_choice_switch_stay":
            for row_position in range(1, n_trials):
                previous_position = row_position - 1
                if valid_choice.iloc[row_position] and valid_choice.iloc[previous_position]:
                    values.iloc[row_position] = int(
                        action_values.iloc[row_position] != action_values.iloc[previous_position]
                    )
                    valid.iloc[row_position] = True
        elif target_name == "next_choice_switch_stay":
            for row_position in range(n_trials - 1):
                next_position = row_position + 1
                if valid_choice.iloc[row_position] and valid_choice.iloc[next_position]:
                    values.iloc[row_position] = int(
                        action_values.iloc[next_position] != action_values.iloc[row_position]
                    )
                    valid.iloc[row_position] = True
        else:
            source = _numeric_source_column(
                working_df,
                numeric_cache,
                definition.source_column,
            )
            valid = pd.Series(np.isfinite(source), index=target_table.index)
            values.loc[valid] = source.loc[valid]

        target_table[target_name] = values
        target_table[f"{target_name}_valid"] = valid.to_numpy(dtype=bool)

    target_table.attrs["target_metadata"] = _target_metadata(config)
    return target_table
