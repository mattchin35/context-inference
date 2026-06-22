"""Post-context-switch persistence metrics for single behavior sessions."""

from pathlib import Path

import pandas as pd

from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    is_zero_flag,
    make_no_choice_action_mask,
    normalize_experimenter_reward_column,
)


SIDE_RIGHT = "right"
SIDE_LEFT = "left"
SWITCH_GROUP_COMBINED = "combined"
SWITCH_DIRECTION_LABELS = {
    (SIDE_LEFT, SIDE_RIGHT): "L_to_R",
    (SIDE_RIGHT, SIDE_LEFT): "R_to_L",
}
PREVIOUS_BLOCK_COLUMNS = (
    "prev_consecutive_rewards",
    "prev_n_omissions",
    "prev_consecutive_omissions",
    "prev_n_rewarded",
    "prev_n_correct",
)


def normalize_context_side(value) -> str:
    """Normalize one context-state value to a left/right side label.

    Parameters
    ----------
    value : int, float, or str
        Context state value from `augmented_trial_df.state`. Accepted values
        are `0`, `1`, `"0"`, `"1"`, `"right"`, `"left"`,
        `"right_patch"`, and `"left_patch"`. Numeric convention is
        `0=right`, `1=left`.

    Returns
    -------
    str
        Normalized side label, either `"right"` or `"left"`.
    """
    if isinstance(value, str):
        text_value = value.strip().lower()
        if text_value in {"right", "right_patch", "0", "0.0"}:
            return SIDE_RIGHT
        if text_value in {"left", "left_patch", "1", "1.0"}:
            return SIDE_LEFT

    try:
        numeric_value = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unrecognized context state value: {value!r}") from exc

    if numeric_value == 0:
        return SIDE_RIGHT
    if numeric_value == 1:
        return SIDE_LEFT
    raise ValueError(f"Unrecognized context state value: {value!r}")


def parse_choice_side(action) -> str | None:
    """Normalize one animal action to a left/right side label.

    Parameters
    ----------
    action : int, float, or str
        Trial action value. `0` means right, `1` means left. Values recognized
        by `make_no_choice_action_mask` are returned as None.

    Returns
    -------
    str or None
        `"right"` or `"left"` for side choices, and None for no-choice rows.
    """
    if bool(make_no_choice_action_mask([action]).iloc[0]):
        return None

    try:
        action_int = int(float(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be 0 or 1 for side choices, got {action!r}.") from exc

    if action_int == 0:
        return SIDE_RIGHT
    if action_int == 1:
        return SIDE_LEFT
    raise ValueError(f"action must be 0 or 1 for side choices, got {action!r}.")


def get_block_context_sides(augmented_trial_df: pd.DataFrame) -> dict[int, str]:
    """Return one normalized context side per block.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `cur_block` and `state`. Each `cur_block` group must
        contain exactly one normalized left/right context state.

    Returns
    -------
    dict[int, str]
        Mapping from integer block id to normalized side label.
    """
    required_columns = {"cur_block", "state"}
    missing_columns = sorted(required_columns.difference(augmented_trial_df.columns))
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required columns: {missing_columns}")

    block_sides = {}
    for block_id, block_df in augmented_trial_df.groupby("cur_block", sort=True):
        normalized_states = block_df["state"].map(normalize_context_side)
        unique_states = sorted(normalized_states.unique())
        if len(unique_states) != 1:
            raise ValueError(
                f"Block {block_id} has multiple context states: {unique_states}"
            )
        block_sides[int(block_id)] = unique_states[0]
    return block_sides


def build_block_trial_id_map(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
) -> dict[int, int]:
    """Map block-summary ids to raw trial block ids.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        column is `block_ix`, a unitless block identifier. In current
        session-level summaries this may be a contiguous row id rather than
        the original trial block id.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        column is `cur_block`, a unitless raw trial block identifier in
        session order.

    Returns
    -------
    dict[int, int]
        Mapping from `block_performance.block_ix` to
        `augmented_trial_df.cur_block`. Direct id matches are preserved;
        otherwise rows are aligned by block order when both tables have the
        same number of blocks. If neither rule applies, ids are returned as
        direct candidates so callers can decide whether missing trial blocks
        are acceptable for their use case.
    """
    if "block_ix" not in block_performance.columns:
        raise ValueError("block_performance must contain a 'block_ix' column.")
    if "cur_block" not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df must contain a 'cur_block' column.")

    block_ids = pd.to_numeric(block_performance["block_ix"], errors="raise").astype(int).tolist()
    trial_block_ids = (
        pd.to_numeric(augmented_trial_df["cur_block"], errors="raise")
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    trial_block_id_set = set(trial_block_ids)
    if set(block_ids).issubset(trial_block_id_set):
        return {block_id: block_id for block_id in block_ids}

    if len(block_ids) == len(trial_block_ids):
        return dict(zip(block_ids, trial_block_ids))

    return {block_id: block_id for block_id in block_ids}


def make_valid_choice_mask(trial_df: pd.DataFrame) -> pd.Series:
    """Return rows with an animal side choice and no experimenter reward.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        column is `action`; optional `experimenter_reward_given` is normalized
        from legacy `give_reward` when present and otherwise defaults to zero.

    Returns
    -------
    pd.Series
        Boolean Series with shape `(n_trials,)`, aligned to `trial_df.index`.
    """
    if "action" not in trial_df.columns:
        raise ValueError("trial_df must contain an 'action' column.")

    normalized = normalize_experimenter_reward_column(trial_df)
    no_choice = make_no_choice_action_mask(normalized["action"])
    if EXPERIMENTER_REWARD_GIVEN_COLUMN in normalized.columns:
        no_experimenter_reward = is_zero_flag(normalized[EXPERIMENTER_REWARD_GIVEN_COLUMN])
    else:
        no_experimenter_reward = pd.Series(True, index=normalized.index)
    return ~no_choice & no_experimenter_reward


def get_correct_choice_omission_mask(trial_df: pd.DataFrame) -> pd.Series:
    """Return correct, unrewarded animal choices with no manual reward.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `action`, `correct`, and `reward`. Correct and reward are
        unitless task outcomes; reward values greater than zero are treated as
        rewarded.

    Returns
    -------
    pd.Series
        Boolean Series with shape `(n_trials,)`, aligned to `trial_df.index`.
    """
    required_columns = {"action", "correct", "reward"}
    missing_columns = sorted(required_columns.difference(trial_df.columns))
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {missing_columns}")

    valid_choice = make_valid_choice_mask(trial_df)
    omission_mask = pd.Series(False, index=trial_df.index)
    if not bool(valid_choice.any()):
        return omission_mask

    valid_correct = pd.to_numeric(
        trial_df.loc[valid_choice, "correct"],
        errors="coerce",
    )
    invalid_correct = trial_df.loc[valid_choice, "correct"].loc[valid_correct.isna()]
    if not invalid_correct.empty:
        invalid_summary = sorted(invalid_correct.astype(str).unique())
        raise ValueError(
            f"correct values must be numeric for valid animal choices, got: {invalid_summary}"
        )

    valid_reward = pd.to_numeric(
        trial_df.loc[valid_choice, "reward"],
        errors="coerce",
    )
    invalid_reward = trial_df.loc[valid_choice, "reward"].loc[valid_reward.isna()]
    if not invalid_reward.empty:
        invalid_summary = sorted(invalid_reward.astype(str).unique())
        raise ValueError(
            f"reward values must be numeric for valid animal choices, got: {invalid_summary}"
        )

    omission_mask.loc[valid_choice] = (
        valid_correct.astype(float).gt(0) & ~valid_reward.astype(float).gt(0)
    )
    return omission_mask


def count_terminal_correct_choice_omissions(block_df: pd.DataFrame) -> int:
    """Count terminal correct-choice omissions in one block.

    Parameters
    ----------
    block_df : pd.DataFrame
        Trialwise dataframe for one block, shape `(n_block_trials,
        n_columns)`. Required columns match `get_correct_choice_omission_mask`.
        Incorrect choices, no-choice rows, and manual rewards are excluded
        from the terminal correct-choice sequence.

    Returns
    -------
    int
        Number of consecutive correct-but-unrewarded choices at the end of the
        block's correct-choice sequence, in trials.
    """
    omission_mask = get_correct_choice_omission_mask(block_df)
    valid_choice = make_valid_choice_mask(block_df)
    valid_correct = pd.to_numeric(
        block_df.loc[valid_choice, "correct"],
        errors="coerce",
    ).astype(float).gt(0)
    correct_choice_df = block_df.loc[valid_choice].loc[valid_correct]
    if correct_choice_df.empty:
        return 0

    correct_choice_omitted = omission_mask.loc[correct_choice_df.index].to_numpy(dtype=bool)
    terminal_count = 0
    for was_omitted in correct_choice_omitted[::-1]:
        if not was_omitted:
            break
        terminal_count += 1
    return terminal_count


def add_previous_block_omission_metrics(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add previous-block correct-choice omission metrics to block rows.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        column is `block_ix`, a unitless block identifier. When this does not
        directly match `augmented_trial_df.cur_block`, rows are aligned by
        block order if both tables contain the same number of blocks.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `cur_block`, `action`, `correct`, and `reward`; optional
        manual-reward columns are normalized before omission counting.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with `prev_n_omissions` and
        `prev_consecutive_omissions` columns added. Counts are unitless trial
        counts from the previous block; the first block receives zeros.
    """
    if "block_ix" not in block_performance.columns:
        raise ValueError("block_performance must contain a 'block_ix' column.")
    if "cur_block" not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df must contain a 'cur_block' column.")

    trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    block_trial_id_map = build_block_trial_id_map(block_performance, trial_df)
    trial_block_ids = pd.to_numeric(trial_df["cur_block"], errors="raise").astype(int)
    omission_mask = get_correct_choice_omission_mask(trial_df)
    enriched = block_performance.copy()
    prev_n_omissions = []
    prev_consecutive_omissions = []
    previous_block_id = None

    for block_id in enriched["block_ix"].astype(int):
        if previous_block_id is None:
            prev_n_omissions.append(0)
            prev_consecutive_omissions.append(0)
        else:
            previous_trial_block_id = block_trial_id_map[previous_block_id]
            previous_block_df = trial_df[trial_block_ids == previous_trial_block_id]
            previous_omissions = omission_mask.loc[previous_block_df.index]
            prev_n_omissions.append(int(previous_omissions.sum()))
            prev_consecutive_omissions.append(
                count_terminal_correct_choice_omissions(previous_block_df)
            )
        previous_block_id = block_id

    enriched["prev_n_omissions"] = prev_n_omissions
    enriched["prev_consecutive_omissions"] = prev_consecutive_omissions
    return enriched


def get_block_exclusion_reason(
    block_df: pd.DataFrame,
    first_switch_position: int | None,
) -> str:
    """Return the block-level exclusion reason for summary metrics.

    Parameters
    ----------
    block_df : pd.DataFrame
        Trialwise dataframe for a post-switch block, shape `(n_block_trials,
        n_columns)`.
    first_switch_position : int or None
        Zero-based raw row position within `block_df` for the first new-context
        side choice. None indicates no switch before the block ended.

    Returns
    -------
    str
        `"None"` when the block is included in session summaries; otherwise
        one of the diagnostic exclusion labels.
    """
    if first_switch_position is None:
        return "no_switch_before_block_end"

    rows_before_switch = block_df.iloc[:first_switch_position]
    if make_no_choice_action_mask(rows_before_switch["action"]).any():
        return "no_choice_before_switch"

    normalized = normalize_experimenter_reward_column(rows_before_switch)
    if EXPERIMENTER_REWARD_GIVEN_COLUMN in normalized.columns:
        no_experimenter_reward = is_zero_flag(normalized[EXPERIMENTER_REWARD_GIVEN_COLUMN])
        if not bool(no_experimenter_reward.all()):
            return "experimenter_reward_before_switch"

    return "None"


def build_switch_persistence_rows_for_block(
    block_row: pd.Series,
    block_df: pd.DataFrame,
    previous_side: str,
    current_side: str,
) -> list[dict]:
    """Build post-switch diagnostic rows for one switched block.

    Parameters
    ----------
    block_row : pd.Series
        Blockwise metadata row for the current block. Required item is
        `block_ix`; previous-block metric columns are copied when present.
    block_df : pd.DataFrame
        Trialwise rows from the current block, shape `(n_block_trials,
        n_columns)`, in session order.
    previous_side : str
        Previous context side label, `"left"` or `"right"`.
    current_side : str
        Current context side label, `"left"` or `"right"`.

    Returns
    -------
    list[dict]
        One dictionary per trial row in `block_df`. Rows preserve no-choice and
        post-switch trials for diagnostics.
    """
    if block_df.empty:
        raise ValueError(f"No trial rows found for block_ix {block_row['block_ix']}.")

    trial_cur_block = int(
        pd.to_numeric(block_df["cur_block"], errors="raise").astype(int).iloc[0]
    )
    choice_sides = [parse_choice_side(action) for action in block_df["action"]]
    first_switch_position = next(
        (position for position, side in enumerate(choice_sides) if side == current_side),
        None,
    )
    exclusion_reason = get_block_exclusion_reason(block_df, first_switch_position)
    include_block = exclusion_reason == "None"
    switch_direction = SWITCH_DIRECTION_LABELS[(previous_side, current_side)]
    valid_choice_counter = 0
    rows = []

    for position, (trial_index, trial_row) in enumerate(block_df.iterrows()):
        side = choice_sides[position]
        if side is not None:
            valid_choice_counter += 1

        if side is None:
            choice_status = "no_choice"
            stay_value = False
        elif first_switch_position is not None and position > first_switch_position:
            choice_status = "post_switch"
            stay_value = False
        elif side == previous_side:
            choice_status = "stay"
            stay_value = True
        elif side == current_side:
            choice_status = "switch"
            stay_value = False
        else:  # pragma: no cover - parse_choice_side restricts values to left/right.
            choice_status = "unknown"
            stay_value = False

        include_trial = include_block and choice_status in {"stay", "switch"}
        row = {
            "block_ix": int(block_row["block_ix"]),
            "trial_cur_block": trial_cur_block,
            "raw_trial_index": int(trial_index),
            "trial_index_after_block_switch": position + 1,
            "choice_trial_after_switch": valid_choice_counter if side is not None else "None",
            "previous_context_side": previous_side,
            "current_context_side": current_side,
            "switch_direction": switch_direction,
            "action_side": side if side is not None else "None",
            "choice_status": choice_status,
            "stay": bool(stay_value),
            "include_block_in_session_metric": bool(include_block),
            "include_trial_in_session_metric": bool(include_trial),
            "exclusion_reason": exclusion_reason,
        }
        if "cur_trial" in trial_row.index:
            row["cur_trial"] = trial_row["cur_trial"]
        for column_name in PREVIOUS_BLOCK_COLUMNS:
            if column_name in block_row.index:
                row[column_name] = block_row[column_name]
        rows.append(row)

    return rows


def compute_switch_persistence_trials(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute trial-level persistence rows after context switches.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        column is `block_ix`; previous-block metrics are copied when present.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `cur_block`, `state`, and `action`.

    Returns
    -------
    pd.DataFrame
        Long dataframe with one row per post-switch block trial. The x-axis
        for summary metrics is `choice_trial_after_switch`, counting only
        animal side choices. Diagnostic rows are retained even when excluded
        from session summaries.
    """
    if "block_ix" not in block_performance.columns:
        raise ValueError("block_performance must contain a 'block_ix' column.")
    for column_name in ("cur_block", "state", "action"):
        if column_name not in augmented_trial_df.columns:
            raise ValueError(f"augmented_trial_df must contain a '{column_name}' column.")

    enriched_blocks = add_previous_block_omission_metrics(block_performance, augmented_trial_df)
    trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    block_trial_id_map = build_block_trial_id_map(enriched_blocks, trial_df)
    trial_block_ids = pd.to_numeric(trial_df["cur_block"], errors="raise").astype(int)
    block_sides = get_block_context_sides(trial_df)
    rows = []
    previous_block_id = None
    previous_side = None

    for _, block_row in enriched_blocks.iterrows():
        block_id = int(block_row["block_ix"])
        trial_block_id = block_trial_id_map[block_id]
        if trial_block_id not in block_sides:
            missing_direct_ids = sorted(
                set(enriched_blocks["block_ix"].astype(int)).difference(block_sides)
            )
            raise ValueError(
                "Cannot align block_performance block_ix to augmented_trial_df cur_block: "
                f"block_ix={block_id}, trial_cur_block={trial_block_id}, "
                f"missing_direct_block_ix={missing_direct_ids}."
            )
        current_side = block_sides[trial_block_id]
        if previous_block_id is None:
            previous_block_id = block_id
            previous_side = current_side
            continue

        if current_side != previous_side:
            block_df = trial_df[trial_block_ids == trial_block_id]
            rows.extend(
                build_switch_persistence_rows_for_block(
                    block_row=block_row,
                    block_df=block_df,
                    previous_side=previous_side,
                    current_side=current_side,
                )
            )

        previous_block_id = block_id
        previous_side = current_side

    return pd.DataFrame(rows)


def summarize_switch_persistence(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize stay probability by post-switch choice index.

    Parameters
    ----------
    detail_df : pd.DataFrame
        Output from `compute_switch_persistence_trials`, shape
        `(n_rows, n_columns)`. Required columns are `switch_direction`,
        `choice_trial_after_switch`, `stay`, and
        `include_trial_in_session_metric`.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with shape `(n_groups * n_trial_indices,
        n_columns)`. Rows include `combined`, `L_to_R`, and `R_to_L` groups
        when data are present. `proportion_stay` is unitless; counts are block
        counts.
    """
    columns = [
        "switch_group",
        "choice_trial_after_switch",
        "n_blocks",
        "n_stay",
        "n_switch",
        "proportion_stay",
    ]
    if detail_df.empty:
        return pd.DataFrame(columns=columns)

    included = detail_df[detail_df["include_trial_in_session_metric"]].copy()
    if included.empty:
        return pd.DataFrame(columns=columns)

    included["choice_trial_after_switch"] = pd.to_numeric(
        included["choice_trial_after_switch"],
        errors="raise",
    ).astype(int)
    summary_parts = []
    for switch_group, group_df in (
        [(SWITCH_GROUP_COMBINED, included)]
        + [(label, included[included["switch_direction"] == label]) for label in ("L_to_R", "R_to_L")]
    ):
        if group_df.empty:
            continue
        grouped = group_df.groupby("choice_trial_after_switch", sort=True)
        summary = grouped["stay"].agg(n_blocks="size", n_stay="sum").reset_index()
        summary["switch_group"] = switch_group
        summary["n_stay"] = summary["n_stay"].astype(int)
        summary["n_switch"] = summary["n_blocks"] - summary["n_stay"]
        summary["proportion_stay"] = summary["n_stay"] / summary["n_blocks"]
        summary_parts.append(summary[columns])

    if not summary_parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(summary_parts, axis=0, ignore_index=True)


def save_switch_persistence_outputs(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    processed_data_path: Path,
    sess_id_full: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and save switch-persistence detail and summary CSVs.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.
    processed_data_path : pathlib.Path
        Session processed-data directory where CSV outputs are written.
    sess_id_full : str
        Full session identifier used as the CSV filename prefix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        `(detail_df, summary_df)` where `detail_df` is one row per
        post-switch trial and `summary_df` is one row per group and valid
        post-switch choice index.
    """
    processed_data_path.mkdir(parents=True, exist_ok=True)
    detail_df = compute_switch_persistence_trials(block_performance, augmented_trial_df)
    summary_df = summarize_switch_persistence(detail_df)

    detail_path = processed_data_path / f"{sess_id_full}_switch_persistence_trials.csv"
    summary_path = processed_data_path / f"{sess_id_full}_switch_persistence_summary.csv"
    detail_df.to_csv(detail_path, index=False, na_rep="None")
    summary_df.to_csv(summary_path, index=False, na_rep="None")
    return detail_df, summary_df
