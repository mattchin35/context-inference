"""Trial-local count, mask, fold, window, and history preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pynapple as nap
from sklearn.model_selection import GroupKFold

from src.behavior_analysis.project_utils import get_experimenter_reward_flags
from src.neural_analysis.population.pca import build_trial_unit_rate_tensor_pynapple
from src.neural_analysis.spike_behavior.trials import make_trial_type_masks

from .configuration import (
    CV_GROUP_COLUMN,
    N_CV_FOLDS,
    AnalysisWindows,
    FilterConfig,
    ResolvedRegionalPopulation,
    TemporalConfig,
)
from .records import FoldAssignment, HistoryMatrices, RegionalCountTensor


@dataclass(frozen=True)
class AnalysisTrialMasks:
    """Position-indexed scientific, condition, and CV trial masks.

    Every Boolean array has shape ``(n_trials,)`` and is indexed by zero-based
    trial-table row position. Alignment values, when inspected, are in seconds.
    """

    scientific_eligible: np.ndarray
    condition_masks: Mapping[str, np.ndarray]
    cv_eligible: np.ndarray
    reward_status_valid: np.ndarray
    alignment_valid: np.ndarray
    choice_match: np.ndarray
    context_match: np.ndarray
    user_included: np.ndarray
    block_present: np.ndarray
    original_index_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Copy masks into immutable one-dimensional Boolean arrays."""
        n_trials = len(self.original_index_labels)
        array_fields = (
            "scientific_eligible",
            "cv_eligible",
            "reward_status_valid",
            "alignment_valid",
            "choice_match",
            "context_match",
            "user_included",
            "block_present",
        )
        for field_name in array_fields:
            values = np.asarray(getattr(self, field_name))
            if values.dtype != np.dtype(bool) or values.shape != (n_trials,):
                raise ValueError(f"{field_name} must be Boolean with shape (n_trials,).")
            copy = np.array(values, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, field_name, copy)
        condition_masks: dict[str, np.ndarray] = {}
        for condition, values in self.condition_masks.items():
            mask = np.asarray(values)
            if mask.dtype != np.dtype(bool) or mask.shape != (n_trials,):
                raise ValueError(
                    f"condition mask {condition!r} must be Boolean with shape (n_trials,)."
                )
            copy = np.array(mask, copy=True)
            copy.setflags(write=False)
            condition_masks[str(condition)] = copy
        object.__setattr__(
            self, "condition_masks", MappingProxyType(condition_masks)
        )


def _normalized_trial_rows(
    trial_rows: Sequence[object] | np.ndarray, n_trials: int
) -> np.ndarray:
    """Return ascending unique in-range zero-based positional rows."""
    values = np.asarray(trial_rows)
    if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
        raise ValueError("trial_rows must contain one-dimensional integer positions.")
    rows = values.astype(np.int64, copy=True)
    if rows.size == 0:
        raise ValueError("trial_rows must not be empty.")
    if np.any(rows < 0) or np.any(rows >= n_trials):
        raise ValueError("trial_rows contains an out-of-range position.")
    if rows.size > 1 and np.any(np.diff(rows) <= 0):
        raise ValueError("trial_rows must be strictly ascending and unique.")
    return rows


def _whole_bin_edges(windows: AnalysisWindows, temporal: TemporalConfig) -> np.ndarray:
    """Construct the exact configured whole-window edge vector in seconds."""
    duration = windows.whole_stop_s - windows.whole_start_s
    n_bins = int(round(duration / temporal.bin_size_s))
    edges = windows.whole_start_s + np.arange(n_bins + 1, dtype=np.float64) * temporal.bin_size_s
    edges[0] = windows.whole_start_s
    edges[-1] = windows.whole_stop_s
    return edges


def build_regional_count_tensor(
    spike_group: nap.TsGroup,
    population: ResolvedRegionalPopulation,
    trial_df: pd.DataFrame,
    trial_rows: Sequence[object] | np.ndarray,
    alignment: str,
    windows: AnalysisWindows,
    temporal: TemporalConfig,
) -> RegionalCountTensor:
    """Bin one resolved population with Pynapple into integer spike counts.

    Parameters
    ----------
    spike_group : pynapple.TsGroup
        Spike timestamps in seconds, keyed by integer cluster ID.
    population : ResolvedRegionalPopulation
        Ordered cluster and qualified unit identities for one region.
    trial_df : pandas.DataFrame
        One row per trial; ``alignment`` contains absolute seconds.
    trial_rows : one-dimensional integer sequence
        Ascending unique zero-based row positions to include.
    alignment : {"choice_time", "start_time"}
        Trial-table column used as time zero.
    windows : AnalysisWindows
        Whole-window bounds in seconds relative to alignment.
    temporal : TemporalConfig
        Bin width in seconds and history settings.

    Returns
    -------
    RegionalCountTensor
        Counts with shape ``(trial, time_bin, unit)`` and dtype ``int64``;
        edges are seconds relative to alignment.
    """
    if not isinstance(trial_df, pd.DataFrame):
        raise TypeError("trial_df must be a pandas DataFrame.")
    if alignment not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment column {alignment!r}.")
    rows = _normalized_trial_rows(trial_rows, len(trial_df))
    selected_trials = trial_df.iloc[rows]
    alignment_values = pd.to_numeric(
        selected_trials[alignment], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(alignment_values)):
        raise ValueError("Selected trials must have finite alignment times in seconds.")

    # The existing Pynapple builder uses label-based indexing internally. Resetting
    # only this positional subset makes its row labels explicit and local.
    local_trials = selected_trials.reset_index(drop=True)
    rates_hz, _ = build_trial_unit_rate_tensor_pynapple(
        spike_group=spike_group,
        unit_ids=np.asarray(population.cluster_ids, dtype=int),
        trial_df=local_trials,
        trial_indices=np.arange(rows.size, dtype=int),
        alignment_event=alignment,
        window=(windows.whole_start_s, windows.whole_stop_s),
        bin_size_s=temporal.bin_size_s,
    )
    raw_counts = np.asarray(rates_hz, dtype=np.float64) * temporal.bin_size_s
    rounded_counts = np.rint(raw_counts)
    if not np.all(np.isfinite(raw_counts)):
        raise ValueError("Pynapple returned nonfinite spike counts.")
    if np.any(raw_counts < 0) or not np.allclose(
        raw_counts, rounded_counts, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Pynapple returned noninteger or negative spike counts.")
    if np.any(rounded_counts > np.iinfo(np.int64).max):
        raise ValueError("Pynapple spike counts exceed the int64 range.")

    return RegionalCountTensor(
        counts=rounded_counts.astype(np.int64),
        trial_rows=rows,
        original_index_labels=tuple(str(value) for value in selected_trials.index),
        bin_edges_s=_whole_bin_edges(windows, temporal),
        unit_ids=population.unit_ids,
    )


def validate_regional_tensor_axes(
    pfc_tensor: RegionalCountTensor, hpc_tensor: RegionalCountTensor
) -> None:
    """Require identical trial identity and relative-time axes across regions.

    Both tensors contain spike counts with axes ``(trial, time_bin, unit)``;
    unit-axis lengths and identities may differ.
    """
    if not np.array_equal(pfc_tensor.trial_rows, hpc_tensor.trial_rows):
        raise ValueError("Regional tensors must have identical trial rows.")
    if pfc_tensor.original_index_labels != hpc_tensor.original_index_labels:
        raise ValueError("Regional tensors must have identical original trial labels.")
    if not np.array_equal(pfc_tensor.bin_edges_s, hpc_tensor.bin_edges_s):
        raise ValueError("Regional tensors must have byte-identical bin edges.")
    if pfc_tensor.counts.shape[:2] != hpc_tensor.counts.shape[:2]:
        raise ValueError("Regional tensors must have identical trial and time-bin axes.")


def _side_match(
    trial_df: pd.DataFrame, column: str, selection: str
) -> np.ndarray:
    """Return the project's numeric left/right match for one trial column."""
    if selection == "all":
        return np.ones(len(trial_df), dtype=bool)
    if column not in trial_df.columns:
        raise ValueError(f"trial_df is missing required filter column {column!r}.")
    numeric = pd.to_numeric(trial_df[column], errors="coerce").to_numpy(dtype=float)
    expected = 1.0 if selection == "left" else 0.0
    return np.isfinite(numeric) & (numeric == expected)


def build_analysis_trial_masks(
    trial_df: pd.DataFrame,
    alignment: str,
    filters: FilterConfig,
) -> AnalysisTrialMasks:
    """Build scientific, named-condition, and CV masks by trial-row position.

    Parameters
    ----------
    trial_df : pandas.DataFrame
        One row per trial. Alignment values are absolute seconds; ``action``
        and ``state_int`` use 1=left and 0=right when selected.
    alignment : {"choice_time", "start_time"}
        Column used for finite-alignment eligibility.
    filters : FilterConfig
        Requested conditions, choice/context side, and excluded zero-based rows.

    Returns
    -------
    AnalysisTrialMasks
        Boolean arrays with shape ``(n_trials,)``. Missing blocks affect only
        ``cv_eligible`` and do not remove scientifically eligible Granger rows.
    """
    if alignment not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment column {alignment!r}.")
    n_trials = len(trial_df)
    if filters.excluded_trial_rows and filters.excluded_trial_rows[-1] >= n_trials:
        raise ValueError("excluded_trial_rows contains an out-of-range position.")

    reward_flags = get_experimenter_reward_flags(trial_df, default_zero=True)
    reward_status_valid = np.asarray(reward_flags == 0, dtype=bool)
    alignment_values = pd.to_numeric(
        trial_df[alignment], errors="coerce"
    ).to_numpy(dtype=float)
    alignment_valid = np.isfinite(alignment_values)
    choice_match = _side_match(trial_df, "action", filters.choice)
    context_match = _side_match(trial_df, "state_int", filters.context)
    user_included = np.ones(n_trials, dtype=bool)
    if filters.excluded_trial_rows:
        user_included[np.asarray(filters.excluded_trial_rows, dtype=int)] = False
    if CV_GROUP_COLUMN in trial_df.columns:
        block_present = trial_df[CV_GROUP_COLUMN].notna().to_numpy(dtype=bool)
    else:
        block_present = np.zeros(n_trials, dtype=bool)

    scientific_eligible = (
        reward_status_valid
        & alignment_valid
        & choice_match
        & context_match
        & user_included
    )
    existing_masks: Mapping[str, pd.Series] = {}
    if any(condition != "all" for condition in filters.conditions):
        existing_masks = make_trial_type_masks(trial_df)
    condition_masks: dict[str, np.ndarray] = {}
    for condition in filters.conditions:
        if condition == "all":
            condition_masks[condition] = scientific_eligible.copy()
        else:
            condition_values = np.asarray(existing_masks[condition], dtype=bool)
            condition_masks[condition] = scientific_eligible & condition_values

    return AnalysisTrialMasks(
        scientific_eligible=scientific_eligible,
        condition_masks=condition_masks,
        cv_eligible=scientific_eligible & block_present,
        reward_status_valid=reward_status_valid,
        alignment_valid=alignment_valid,
        choice_match=choice_match,
        context_match=context_match,
        user_included=user_included,
        block_present=block_present,
        original_index_labels=tuple(str(value) for value in trial_df.index),
    )


def _canonical_block_json(value: object) -> str:
    """Normalize one supported block scalar and return canonical JSON text."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("cur_block values must not be Boolean.")
    if isinstance(value, Integral):
        normalized: str | int | float = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("cur_block numeric values must be finite.")
        normalized = int(numeric) if numeric.is_integer() else numeric
    elif isinstance(value, str):
        normalized = value
    else:
        raise ValueError("cur_block values must be string or finite numeric scalars.")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_block_fold_assignment(trial_df: pd.DataFrame) -> FoldAssignment:
    """Assign complete nonmissing blocks to five deterministic test folds.

    Parameters
    ----------
    trial_df : pandas.DataFrame
        Complete unfiltered trial table. ``cur_block`` contains string or finite
        numeric scalar group identities; row positions are zero-based.

    Returns
    -------
    FoldAssignment
        One entry per trial row. Missing-block rows have canonical JSON ``null``
        and no fold; all rows sharing a normalized block share one fold ID 0-4.
    """
    if CV_GROUP_COLUMN not in trial_df.columns:
        raise ValueError(f"trial_df is missing CV block column {CV_GROUP_COLUMN!r}.")
    block_values = trial_df[CV_GROUP_COLUMN]
    present = block_values.notna().to_numpy(dtype=bool)
    canonical_values = ["null"] * len(trial_df)
    for position in np.flatnonzero(present):
        canonical_values[int(position)] = _canonical_block_json(
            block_values.iloc[int(position)]
        )
    distinct_groups = {
        value
        for value, keep in zip(canonical_values, present, strict=True)
        if keep
    }
    if len(distinct_groups) < N_CV_FOLDS:
        raise ValueError("CV requires at least five distinct nonmissing cur_block values.")

    present_positions = np.flatnonzero(present)
    groups = np.asarray(
        [canonical_values[int(position)] for position in present_positions], dtype=object
    )
    fold_ids: list[int | None] = [None] * len(trial_df)
    splitter = GroupKFold(n_splits=N_CV_FOLDS, shuffle=False)
    samples = np.zeros((present_positions.size, 1), dtype=float)
    for fold_id, (_, test_local_positions) in enumerate(
        splitter.split(samples, groups=groups)
    ):
        for local_position in test_local_positions:
            fold_ids[int(present_positions[int(local_position)])] = fold_id

    return FoldAssignment(
        trial_rows=np.arange(len(trial_df), dtype=np.int64),
        original_index_labels=tuple(str(value) for value in trial_df.index),
        block_values_json=tuple(canonical_values),
        fold_ids=tuple(fold_ids),
    )


def select_window_bins(
    activity: np.ndarray,
    window: str,
    windows: AnalysisWindows,
    temporal: TemporalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Select before/after/whole bins while retaining whole-window positions.

    Parameters
    ----------
    activity : numpy.ndarray
        Array with shape ``(trial, whole_window_bin, feature)``.
    window : {"before", "after", "whole"}
        Requested relative window.
    windows : AnalysisWindows
        Bounds in seconds relative to alignment.
    temporal : TemporalConfig
        Bin width in seconds.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Selected activity with axes ``(trial, selected_window_bin, feature)``
        and zero-based whole-window bin positions with shape ``(selected_bin,)``.
    """
    values = np.asarray(activity)
    if values.ndim != 3:
        raise ValueError("activity must have shape (trial, whole_window_bin, feature).")
    whole_count = int(
        round((windows.whole_stop_s - windows.whole_start_s) / temporal.bin_size_s)
    )
    before_count = int(round((windows.split_s - windows.whole_start_s) / temporal.bin_size_s))
    if values.shape[1] != whole_count:
        raise ValueError("activity time axis does not match the configured whole window.")
    slices = {
        "before": slice(0, before_count),
        "after": slice(before_count, whole_count),
        "whole": slice(0, whole_count),
    }
    if window not in slices:
        raise ValueError("window must be 'before', 'after', or 'whole'.")
    positions = np.arange(whole_count, dtype=np.int64)[slices[window]]
    return values[:, slices[window], :], positions


def build_history_matrices(
    target_activity: np.ndarray,
    source_activity: np.ndarray,
    trial_rows: np.ndarray,
    target_bin_positions: np.ndarray,
    lag_bins: int,
    order_bins: int,
) -> HistoryMatrices:
    """Construct aligned within-trial target and source history matrices.

    Parameters
    ----------
    target_activity : numpy.ndarray
        Shape ``(trial, selected_window_bin, target_feature)``.
    source_activity : numpy.ndarray
        Shape ``(trial, selected_window_bin, source_feature)`` on the same axes.
    trial_rows : numpy.ndarray
        Zero-based source trial positions with shape ``(trial,)``.
    target_bin_positions : numpy.ndarray
        Whole-window bin identities with shape ``(selected_window_bin,)``.
    lag_bins : int
        Positive gap in bins from response to the most recent history value.
    order_bins : int
        Positive number of history bins. Columns are most-recent lag first,
        then stable feature order.

    Returns
    -------
    HistoryMatrices
        Response/history arrays on a shared ``observation`` axis. Nonfinite
        observations are removed once using the full comparison row mask.
    """
    target = np.asarray(target_activity, dtype=np.float64)
    source = np.asarray(source_activity, dtype=np.float64)
    rows = np.asarray(trial_rows)
    bin_positions = np.asarray(target_bin_positions)
    if target.ndim != 3 or source.ndim != 3:
        raise ValueError("Activity inputs must have shape (trial, selected_window_bin, feature).")
    if target.shape[:2] != source.shape[:2]:
        raise ValueError("Target and source activity must share trial and time-bin axes.")
    if rows.dtype.kind not in {"i", "u"} or rows.shape != (target.shape[0],):
        raise ValueError("trial_rows must be integer with shape (trial,).")
    if bin_positions.dtype.kind not in {"i", "u"} or bin_positions.shape != (target.shape[1],):
        raise ValueError(
            "target_bin_positions must be integer with shape (selected_window_bin,)."
        )
    if (
        isinstance(lag_bins, bool)
        or not isinstance(lag_bins, Integral)
        or lag_bins <= 0
    ):
        raise ValueError("lag_bins must be a positive integer.")
    if (
        isinstance(order_bins, bool)
        or not isinstance(order_bins, Integral)
        or order_bins <= 0
    ):
        raise ValueError("order_bins must be a positive integer.")

    first_response_bin = int(lag_bins + order_bins - 1)
    response_bin_indices = np.arange(
        first_response_bin, target.shape[1], dtype=np.int64
    )
    response_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []
    row_trial_blocks: list[np.ndarray] = []
    row_bin_blocks: list[np.ndarray] = []
    for trial_position, trial_row in enumerate(rows.astype(np.int64)):
        response_blocks.append(target[trial_position, response_bin_indices, :])
        target_lags = [
            target[trial_position, response_bin_indices - offset, :]
            for offset in range(int(lag_bins), int(lag_bins + order_bins))
        ]
        source_lags = [
            source[trial_position, response_bin_indices - offset, :]
            for offset in range(int(lag_bins), int(lag_bins + order_bins))
        ]
        target_blocks.append(np.concatenate(target_lags, axis=1))
        source_blocks.append(np.concatenate(source_lags, axis=1))
        row_trial_blocks.append(
            np.full(response_bin_indices.size, trial_row, dtype=np.int64)
        )
        row_bin_blocks.append(bin_positions[response_bin_indices].astype(np.int64))

    n_target_features = target.shape[2]
    n_source_features = source.shape[2]
    if response_blocks:
        responses = np.concatenate(response_blocks, axis=0)
        target_history = np.concatenate(target_blocks, axis=0)
        source_history = np.concatenate(source_blocks, axis=0)
        row_trial = np.concatenate(row_trial_blocks)
        row_target_bin = np.concatenate(row_bin_blocks)
    else:
        responses = np.empty((0, n_target_features), dtype=np.float64)
        target_history = np.empty(
            (0, int(order_bins) * n_target_features), dtype=np.float64
        )
        source_history = np.empty(
            (0, int(order_bins) * n_source_features), dtype=np.float64
        )
        row_trial = np.empty(0, dtype=np.int64)
        row_target_bin = np.empty(0, dtype=np.int64)

    finite_rows = (
        np.all(np.isfinite(responses), axis=1)
        & np.all(np.isfinite(target_history), axis=1)
        & np.all(np.isfinite(source_history), axis=1)
    )
    return HistoryMatrices(
        responses=responses[finite_rows],
        target_history=target_history[finite_rows],
        source_history=source_history[finite_rows],
        row_trial=row_trial[finite_rows],
        row_target_bin=row_target_bin[finite_rows],
    )


def fingerprint_row_identities(
    trial_rows: np.ndarray, target_bin_positions: np.ndarray
) -> str:
    """Return an order-independent SHA-256 for observation identities.

    Parameters
    ----------
    trial_rows, target_bin_positions : numpy.ndarray
        Equal one-dimensional integer arrays. Values are zero-based trial-row
        and whole-window target-bin positions.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 of sorted compact ASCII JSON pairs.
    """
    trials = np.asarray(trial_rows)
    bins = np.asarray(target_bin_positions)
    if trials.dtype.kind not in {"i", "u"} or bins.dtype.kind not in {"i", "u"}:
        raise ValueError("Row identities must be integer arrays.")
    if trials.ndim != 1 or bins.shape != trials.shape:
        raise ValueError("Row identity arrays must be one-dimensional with equal shape.")
    if np.any(trials < 0) or np.any(bins < 0):
        raise ValueError("Row identities must be nonnegative.")
    pairs = [
        (int(trial), int(bin_position))
        for trial, bin_position in zip(trials, bins, strict=True)
    ]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Row identities must not contain duplicate pairs.")
    payload = json.dumps(sorted(pairs), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
