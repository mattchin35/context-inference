"""Trial-aligned spike and lick binning for behavior analyses."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.spike_behavior.trials import resolve_trial_end

def build_bin_edges(
    trial_start: float,
    trial_end: float,
    bin_size: float,
    pre_time: float,
    post_time: float,
) -> np.ndarray:
    """
    Construct trial-relative bin edges using the legacy floor-division convention.

    Parameters
    ----------
    trial_start : float
        Trial start time in seconds.
    trial_end : float
        Trial end time in seconds.
    bin_size : float
        Width of each time bin in seconds.
    pre_time : float
        Time before trial start to include in seconds.
    post_time : float
        Time after trial end to include in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional float array of bin edges with shape ``(n_bins + 1,)`` and units seconds.
    """

    bin_start = trial_start - pre_time
    bin_end = trial_end + post_time
    n_bins = int((bin_end - bin_start) / bin_size)
    bin_edges = np.arange(bin_start, bin_end, bin_size, dtype=float)
    if bin_edges.size < n_bins + 1:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_size)
    return bin_edges

def _count_tsgroup_in_bins(
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_edges: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    """
    Count spikes for selected units within one trial interval.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps are in seconds.
    cluster_ids : np.ndarray
        One-dimensional integer array with shape ``(n_units,)`` listing the unit order
        for the returned count matrix.
    bin_edges : np.ndarray
        One-dimensional float array of bin edges with shape ``(n_bins + 1,)`` and units seconds.
    bin_size : float
        Width of each time bin in seconds.

    Returns
    -------
    np.ndarray
        Two-dimensional count array with shape ``(n_units, n_bins)``. Values are spike counts per bin.
    """

    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if cluster_ids.size == 0:
        return np.zeros((0, max(bin_edges.size - 1, 0)), dtype=float)

    restricted_group = nap.TsGroup({int(cluster_id): spike_group[int(cluster_id)] for cluster_id in cluster_ids})
    interval = nap.IntervalSet(start=[bin_edges[0]], end=[bin_edges[-1]])
    binned_counts = restricted_group.count(bin_size=bin_size, ep=interval)
    return binned_counts.values.T.astype(float)

def _count_ts_in_bins(
    event_times: nap.Ts,
    bin_edges: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    """
    Count one event stream within one trial interval.

    Parameters
    ----------
    event_times : nap.Ts
        One-dimensional event timestamp series with units seconds.
    bin_edges : np.ndarray
        One-dimensional float array of bin edges with shape ``(n_bins + 1,)`` and units seconds.
    bin_size : float
        Width of each time bin in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional count array with shape ``(n_bins,)``. Values are event counts per bin.
    """

    interval = nap.IntervalSet(start=[bin_edges[0]], end=[bin_edges[-1]])
    return event_times.count(bin_size=bin_size, ep=interval).values.astype(float)

def bin_spikes_to_trial_pynapple(
    trial_start: float,
    trial_end: float,
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_size: float = 0.1,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin selected spike trains around one trial.

    Parameters
    ----------
    trial_start : float
        Trial start time in seconds.
    trial_end : float
        Trial end time in seconds.
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps are in seconds.
    cluster_ids : np.ndarray
        One-dimensional integer array with shape ``(n_units,)`` defining row order in the
        returned count matrix.
    bin_size : float, optional
        Width of each time bin in seconds.
    pre_time : float, optional
        Time before trial start to include in seconds.
    post_time : float, optional
        Time after trial end to include in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(binned_spikes, bin_edges)`` where ``binned_spikes`` has shape ``(n_units, n_bins)``
        and contains spike counts, and ``bin_edges`` has shape ``(n_bins + 1,)`` with units seconds.
    """

    bin_edges = build_bin_edges(
        trial_start=trial_start,
        trial_end=trial_end,
        bin_size=bin_size,
        pre_time=pre_time,
        post_time=post_time,
    )
    binned_spikes = _count_tsgroup_in_bins(
        spike_group=spike_group,
        cluster_ids=cluster_ids,
        bin_edges=bin_edges,
        bin_size=bin_size,
    )
    return binned_spikes, bin_edges

def bin_licks_to_trial_pynapple(
    trial_start: float,
    trial_end: float,
    lick_times: Mapping[str, nap.Ts],
    bin_size: float = 0.1,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin right and left lick events around one trial.

    Parameters
    ----------
    trial_start : float
        Trial start time in seconds.
    trial_end : float
        Trial end time in seconds.
    lick_times : Mapping[str, nap.Ts]
        Mapping containing ``"right_entry"`` and ``"left_entry"`` keys. Each value is a
        one-dimensional ``nap.Ts`` of lick timestamps in seconds.
    bin_size : float, optional
        Width of each time bin in seconds.
    pre_time : float, optional
        Time before trial start to include in seconds.
    post_time : float, optional
        Time after trial end to include in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(binned_licks, bin_edges)`` where ``binned_licks`` has shape ``(n_bins, 2)``
        and columns are ordered as ``[right, left]`` counts, and ``bin_edges`` has shape
        ``(n_bins + 1,)`` with units seconds.
    """

    if "right_entry" not in lick_times or "left_entry" not in lick_times:
        raise ValueError("lick_times must contain 'right_entry' and 'left_entry' keys.")

    bin_edges = build_bin_edges(
        trial_start=trial_start,
        trial_end=trial_end,
        bin_size=bin_size,
        pre_time=pre_time,
        post_time=post_time,
    )
    binned_licks = np.column_stack(
        [
            _count_ts_in_bins(lick_times["right_entry"], bin_edges, bin_size),
            _count_ts_in_bins(lick_times["left_entry"], bin_edges, bin_size),
        ]
    ).astype(float)
    return binned_licks, bin_edges

def bin_region_trials(
    trial_df: pd.DataFrame,
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_size: float = 0.5,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Bin region-selected spikes for every trial while preserving the legacy output structure.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are ``start_time``,
        ``choice_time``, ``reward_time``, ``state_int``, and ``action``.
        Time columns are in seconds.
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps are in seconds.
    cluster_ids : np.ndarray
        One-dimensional integer array with shape ``(n_units,)`` defining row order in each
        trial's ``binned_spikes`` matrix.
    bin_size : float, optional
        Width of each time bin in seconds.
    pre_time : float, optional
        Time before trial start to include in seconds.
    post_time : float, optional
        Time after trial end to include in seconds.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per trial. Each dictionary contains:
        ``trial_ix`` (int),
        ``binned_spikes`` (``np.ndarray`` with shape ``(n_units, n_bins)``),
        ``bin_edges`` (``np.ndarray`` with shape ``(n_bins + 1,)`` in seconds),
        ``bin_states`` (``np.ndarray`` with shape ``(n_bins,)``),
        ``bin_choices`` (``np.ndarray`` with shape ``(n_bins,)``).
    """

    required_columns = {"start_time", "choice_time", "reward_time", "state_int", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    spikes_trial_binned: list[dict[str, Any]] = []
    for trial_ix, trial_row in trial_df.iterrows():
        trial_start = float(trial_row["start_time"])
        trial_end = resolve_trial_end(trial_row)
        binned_spikes, bin_edges = bin_spikes_to_trial_pynapple(
            trial_start=trial_start,
            trial_end=trial_end,
            spike_group=spike_group,
            cluster_ids=cluster_ids,
            bin_size=bin_size,
            pre_time=pre_time,
            post_time=post_time,
        )
        n_bins = binned_spikes.shape[1]
        spikes_trial_binned.append(
            {
                "trial_ix": int(trial_ix),
                "binned_spikes": binned_spikes,
                "bin_edges": bin_edges,
                "bin_states": np.full(n_bins, float(trial_row["state_int"]), dtype=float),
                "bin_choices": np.full(n_bins, float(trial_row["action"]), dtype=float),
            }
        )

    return spikes_trial_binned

def make_classifier_bins(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    trial_mask: pd.Series | np.ndarray,
    event: str = "choice_time",
    bounds: tuple[float, float] = (-0.5, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract classifier inputs from pre-binned trial spike counts.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial. Each dictionary must
        contain ``binned_spikes`` with shape ``(n_units, n_bins)``, ``bin_edges`` with shape
        ``(n_bins + 1,)`` in seconds, ``bin_states`` with shape ``(n_bins,)``, and
        ``bin_choices`` with shape ``(n_bins,)``.
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required event columns are ``start_time`` and/or
        ``choice_time`` depending on ``event``. Time values are in seconds.
    trial_mask : pd.Series | np.ndarray
        One-dimensional boolean selector with shape ``(n_trials,)`` indicating which trials to
        include in the extracted classifier bins.
    event : str, optional
        Alignment event name. Supported values are ``"start_time"`` and ``"choice_time"``.
    bounds : tuple[float, float], optional
        Time window relative to ``event`` in seconds as ``(start_offset, end_offset)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(spike_bins, state_bins, choice_bins)`` where ``spike_bins`` has shape
        ``(n_units, n_selected_bins)``, and ``state_bins`` and ``choice_bins`` have shape
        ``(n_selected_bins,)``.
    """

    if event not in {"start_time", "choice_time"}:
        raise ValueError("event must be 'start_time' or 'choice_time'.")
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly two values.")
    if len(region_trial_binned) != len(trial_df):
        raise ValueError("region_trial_binned and trial_df must have the same number of trials.")

    trial_mask_array = np.asarray(trial_mask, dtype=bool)
    if trial_mask_array.ndim != 1 or trial_mask_array.shape[0] != len(trial_df):
        raise ValueError("trial_mask must be a one-dimensional boolean selector matching trial_df.")

    selected_trial_indices = np.flatnonzero(trial_mask_array)
    if selected_trial_indices.size == 0:
        raise ValueError("No trials were selected for classifier bin extraction.")

    spike_bins = []
    state_bins = []
    choice_bins = []
    for trial_index in selected_trial_indices:
        event_time = trial_df.iloc[trial_index][event]
        if pd.isna(event_time):
            raise ValueError(f"Selected trial {trial_index} is missing {event}.")

        bin_edges = np.asarray(region_trial_binned[trial_index]["bin_edges"], dtype=float)
        trial_bin_mask = (
            (bin_edges >= float(event_time) + bounds[0]) &
            (bin_edges < float(event_time) + bounds[1])
        )[:-1]
        if not np.any(trial_bin_mask):
            raise ValueError(
                f"Selected trial {trial_index} has no bins in bounds {bounds} around {event}."
            )

        spike_bins.append(np.asarray(region_trial_binned[trial_index]["binned_spikes"], dtype=float)[:, trial_bin_mask])
        state_bins.append(np.asarray(region_trial_binned[trial_index]["bin_states"], dtype=float)[trial_bin_mask])
        choice_bins.append(np.asarray(region_trial_binned[trial_index]["bin_choices"], dtype=float)[trial_bin_mask])

    return (
        np.concatenate(spike_bins, axis=1),
        np.concatenate(state_bins),
        np.concatenate(choice_bins),
    )

def collect_condition_classifier_bins(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    trial_masks: Mapping[str, pd.Series],
    condition_names: list[str],
    windows: Mapping[str, tuple[float, float]],
    event: str = "choice_time",
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Collect classifier-ready bins for ordered condition/window combinations.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table with one row per trial. Must contain the requested alignment event column.
    trial_masks : Mapping[str, pd.Series]
        Mapping from condition name to boolean trial masks indexed like ``trial_df``.
    condition_names : list[str]
        Ordered list of condition names to extract.
    windows : Mapping[str, tuple[float, float]]
        Mapping from window label to relative event bounds in seconds.
    event : str, optional
        Alignment event name passed through to ``make_classifier_bins``.

    Returns
    -------
    dict[tuple[str, str], dict[str, Any]]
        Dictionary keyed by ``(condition_name, window_name)``. Each value contains either:
        ``status == "ok"`` with ``spike_bins``, ``state_bins``, and ``choice_bins``,
        or ``status == "failed"`` with a short failure reason.
    """

    collected_bins: dict[tuple[str, str], dict[str, Any]] = {}
    for condition_name in condition_names:
        if condition_name not in trial_masks:
            raise ValueError(f"Requested condition {condition_name!r} is not present in trial_masks.")

        for window_name, bounds in windows.items():
            key = (condition_name, window_name)
            try:
                spike_bins, state_bins, choice_bins = make_classifier_bins(
                    region_trial_binned=region_trial_binned,
                    trial_df=trial_df,
                    trial_mask=trial_masks[condition_name],
                    event=event,
                    bounds=bounds,
                )
            except ValueError as error:
                collected_bins[key] = {
                    "status": "failed",
                    "reason": str(error),
                    "event": event,
                    "bounds": bounds,
                }
                continue

            collected_bins[key] = {
                "status": "ok",
                "event": event,
                "bounds": bounds,
                "spike_bins": spike_bins,
                "state_bins": state_bins,
                "choice_bins": choice_bins,
            }

    return collected_bins
