"""Compare candidate expectant models with the mouse-history ideal observer.

This module is descriptive: it groups existing trial flags and agreement
values without changing model state, model parameters, or trial definitions.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import session_analysis


ANALYSIS_VERSION = "1.0.0"
DEFAULT_SELECTIVITY_MODEL_VALUES = {
    "Probe+P+D": "probe_persistence_doubt_value",
    "Home E+D": "home_referenced_expectancy_doubt_value",
}
EVENT_TYPES = (
    "reward_to_switch",
    "reward_to_stay",
    "omission_to_switch",
    "other_valid",
)


def _coerce_binary_flag(values: pd.Series, column_name: str) -> np.ndarray:
    """Return one boolean value per row from a saved binary flag column.

    Parameters
    ----------
    values : pandas.Series, shape (n_trials,)
        Boolean, numeric 0/1, or string-equivalent flag values. Missing values
        are treated as false.
    column_name : str
        Source name included in validation errors.

    Returns
    -------
    numpy.ndarray, shape (n_trials,)
        Boolean flag values aligned to ``values``.
    """
    text = values.astype(str).str.strip().str.lower()
    normalized = values.copy()
    normalized = normalized.mask(text.eq("true"), 1)
    normalized = normalized.mask(text.eq("false"), 0)
    numeric = pd.to_numeric(normalized, errors="coerce")
    missing_text = text.isin(["", "nan", "none", "null"])
    invalid = ~missing_text & values.notna() & (~numeric.isin([0, 1]) | numeric.isna())
    if invalid.any():
        invalid_values = sorted(values.loc[invalid].astype(str).unique().tolist())
        raise ValueError(
            f"{column_name} must contain boolean or numeric zero/one values; "
            f"found {invalid_values}."
        )
    return numeric.fillna(0).astype(bool).to_numpy()


def _validate_inputs(
    trial_df: pd.DataFrame,
    model_value_columns: Mapping[str, str],
) -> None:
    """Validate columns shared by block and event selectivity calculations."""
    required_columns = {
        "cur_block",
        "action",
        "prev_reward",
        "prev_correct",
        "switch_trial",
        "explore_trial",
        "block_entry_trial",
        "observer_value",
        *model_value_columns.values(),
    }
    missing_columns = sorted(required_columns.difference(trial_df.columns))
    if missing_columns:
        raise ValueError(
            f"trial_df is missing expectant-selectivity columns: {missing_columns}"
        )
    if trial_df["cur_block"].isna().any():
        raise ValueError("trial_df cur_block must be present for every trial.")


def _behavior_masks(
    trial_df: pd.DataFrame,
    exclude_block_entry: bool,
) -> dict[str, np.ndarray]:
    """Return mutually exclusive existing-event masks and their opportunity set."""
    valid_choice = session_analysis.make_behavioral_choice_mask(trial_df)
    prev_correct = _coerce_binary_flag(trial_df["prev_correct"], "prev_correct")
    switch_trial = _coerce_binary_flag(trial_df["switch_trial"], "switch_trial")
    explore_trial = _coerce_binary_flag(trial_df["explore_trial"], "explore_trial")
    block_entry = _coerce_binary_flag(
        trial_df["block_entry_trial"], "block_entry_trial"
    )
    prev_rewarded = pd.to_numeric(
        trial_df["prev_reward"], errors="coerce"
    ).eq(1).to_numpy()
    within_selection = ~block_entry if exclude_block_entry else np.ones(
        trial_df.shape[0], dtype=bool
    )

    reward_opportunity = (
        valid_choice & prev_correct & prev_rewarded & within_selection
    )
    reward_to_switch = valid_choice & explore_trial & within_selection
    reward_to_stay = reward_opportunity & ~switch_trial
    omission_to_switch = (
        valid_choice
        & switch_trial
        & prev_correct
        & ~explore_trial
        & within_selection
    )
    return {
        "valid_choice": valid_choice,
        "reward_opportunity": reward_opportunity,
        "reward_to_switch": reward_to_switch,
        "reward_to_stay": reward_to_stay,
        "omission_to_switch": omission_to_switch,
    }


def _agreement_arrays(
    trial_df: pd.DataFrame,
    model_value_columns: Mapping[str, str],
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Compute numeric trialwise agreement using the existing greedy policy."""
    ideal_agreement = pd.to_numeric(
        pd.Series(
            session_analysis.compute_agent_mouse_agreement_by_trial(
                trial_df,
                value_column="observer_value",
            )
        ),
        errors="coerce",
    )
    model_agreements = {
        model_label: pd.to_numeric(
            pd.Series(
                session_analysis.compute_agent_mouse_agreement_by_trial(
                    trial_df,
                    value_column=value_column,
                )
            ),
            errors="coerce",
        )
        for model_label, value_column in model_value_columns.items()
    }
    return ideal_agreement, model_agreements


def compute_block_model_selectivity(
    trial_df: pd.DataFrame,
    model_value_columns: Mapping[str, str] | None = None,
    min_valid_choices: int = 5,
    exclude_block_entry: bool = True,
) -> pd.DataFrame:
    """Summarize candidate-versus-ideal agreement and rewarded switching.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Chronological trial table containing block ids, mouse actions, existing
        trial-type flags, ``prev_reward``, ``observer_value``, and configured
        candidate signed values. Actions use ``0=right`` and ``1=left``.
    model_value_columns : mapping[str, str] or None, default=None
        Display label to signed left-positive value-column mapping. None uses
        Probe+P+D and Home E+D.
    min_valid_choices : int, default=5
        Minimum non-manual left/right choices for ``summary_eligible=True``.
        All blocks remain in the returned table.
    exclude_block_entry : bool, default=True
        If true, reward-switch opportunities and events exclude rows already
        marked by the existing ``block_entry_trial`` flag.

    Returns
    -------
    pandas.DataFrame
        Long table with shape ``(n_blocks * n_models, 13)``. Agreement and
        reward-switch-rate columns are unitless fractions. Count columns are
        trials. ``agreement_delta`` is model minus ideal agreement calculated
        on the same valid comparison rows.
    """
    if min_valid_choices < 1:
        raise ValueError("min_valid_choices must be at least one.")
    models = dict(
        DEFAULT_SELECTIVITY_MODEL_VALUES
        if model_value_columns is None
        else model_value_columns
    )
    if not models:
        raise ValueError("model_value_columns must contain at least one model.")
    _validate_inputs(trial_df, models)
    masks = _behavior_masks(trial_df, exclude_block_entry=exclude_block_entry)
    ideal_agreement, model_agreements = _agreement_arrays(trial_df, models)

    rows = []
    block_values = trial_df["cur_block"]
    for block_position, block_id in enumerate(pd.unique(block_values)):
        in_block = block_values.eq(block_id).to_numpy()
        n_valid_choices = int(np.sum(in_block & masks["valid_choice"]))
        n_opportunities = int(np.sum(in_block & masks["reward_opportunity"]))
        n_events = int(np.sum(in_block & masks["reward_to_switch"]))
        reward_switch_rate = (
            float(n_events / n_opportunities) if n_opportunities else np.nan
        )

        for model_label, model_agreement in model_agreements.items():
            paired = (
                in_block
                & ideal_agreement.notna().to_numpy()
                & model_agreement.notna().to_numpy()
            )
            n_comparisons = int(np.sum(paired))
            if n_comparisons:
                ideal_score = float(ideal_agreement.loc[paired].mean())
                model_score = float(model_agreement.loc[paired].mean())
                agreement_delta = model_score - ideal_score
            else:
                ideal_score = np.nan
                model_score = np.nan
                agreement_delta = np.nan
            rows.append(
                {
                    "block_id": block_id,
                    "block_position": block_position,
                    "model": model_label,
                    "model_value_column": models[model_label],
                    "model_agreement": model_score,
                    "ideal_agreement": ideal_score,
                    "agreement_delta": agreement_delta,
                    "n_valid_choices": n_valid_choices,
                    "n_valid_comparisons": n_comparisons,
                    "reward_switch_opportunities": n_opportunities,
                    "reward_switch_events": n_events,
                    "reward_switch_rate": reward_switch_rate,
                    "summary_eligible": n_valid_choices >= min_valid_choices,
                }
            )
    return pd.DataFrame(rows)


def compute_event_model_selectivity(
    trial_df: pd.DataFrame,
    model_value_columns: Mapping[str, str] | None = None,
    exclude_block_entry: bool = True,
) -> pd.DataFrame:
    """Count paired candidate and ideal outcomes within behavioral events.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Same aligned trial contract as :func:`compute_block_model_selectivity`.
    model_value_columns : mapping[str, str] or None, default=None
        Display label to signed left-positive candidate value mapping.
    exclude_block_entry : bool, default=True
        Whether reward/omission switch categories exclude existing block-entry
        rows. Excluded entry trials remain in ``other_valid`` when agreement is
        available, so event counts partition all paired valid comparisons.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per model and event type. Counts are trials;
        ``mean_agreement_delta`` is the unitless mean of paired trial outcomes
        in ``{-1, 0, 1}``.
    """
    models = dict(
        DEFAULT_SELECTIVITY_MODEL_VALUES
        if model_value_columns is None
        else model_value_columns
    )
    if not models:
        raise ValueError("model_value_columns must contain at least one model.")
    _validate_inputs(trial_df, models)
    masks = _behavior_masks(trial_df, exclude_block_entry=exclude_block_entry)
    ideal_agreement, model_agreements = _agreement_arrays(trial_df, models)

    event_masks = {
        "reward_to_switch": masks["reward_to_switch"],
        "reward_to_stay": masks["reward_to_stay"],
        "omission_to_switch": masks["omission_to_switch"],
    }
    rows = []
    for model_label, model_agreement in model_agreements.items():
        paired = ideal_agreement.notna().to_numpy() & model_agreement.notna().to_numpy()
        categorized = np.zeros(trial_df.shape[0], dtype=bool)
        for event_mask in event_masks.values():
            categorized |= event_mask
        model_event_masks = dict(event_masks)
        model_event_masks["other_valid"] = paired & ~categorized

        for event_type in EVENT_TYPES:
            selected = paired & model_event_masks[event_type]
            model_values = model_agreement.loc[selected].to_numpy(dtype=float)
            ideal_values = ideal_agreement.loc[selected].to_numpy(dtype=float)
            deltas = model_values - ideal_values
            rows.append(
                {
                    "model": model_label,
                    "model_value_column": models[model_label],
                    "event_type": event_type,
                    "n_trials": int(selected.sum()),
                    "model_only_correct": int(
                        np.sum((model_values == 1) & (ideal_values == 0))
                    ),
                    "ideal_only_correct": int(
                        np.sum((model_values == 0) & (ideal_values == 1))
                    ),
                    "both_correct": int(
                        np.sum((model_values == 1) & (ideal_values == 1))
                    ),
                    "both_incorrect": int(
                        np.sum((model_values == 0) & (ideal_values == 0))
                    ),
                    "mean_agreement_delta": (
                        float(np.mean(deltas)) if deltas.size else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)
