"""Trial-aligned lick and single-unit spike PETH calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.spike_behavior.trials import select_valid_lick_peth_trials

LEFT_LICK_EVENT = "left_entry"
RIGHT_LICK_EVENT = "right_entry"

@dataclass(frozen=True)
class LickPethData:
    """
    Trial-aligned lick raster and rate data for PETH-style plots.

    Attributes
    ----------
    alignment : str
        Alignment mode. Supported values are ``"trial_start"`` and ``"choice"``.
    trial_indices : np.ndarray
        One-dimensional integer array with shape ``(n_trials,)`` containing source trial indices.
    time_bin_edges : np.ndarray
        One-dimensional float array with shape ``(n_bins + 1,)`` in seconds relative to alignment.
    time_bin_centers : np.ndarray
        One-dimensional float array with shape ``(n_bins,)`` in seconds relative to alignment.
    left_raster_times_by_trial, right_raster_times_by_trial : list[np.ndarray]
        One array per trial. Each array contains lick times in seconds relative to alignment.
    left_rate_by_trial, right_rate_by_trial : np.ndarray
        Lick-rate arrays with shape ``(n_trials, n_bins)`` in events/s. Invalid bins are ``NaN``.
    valid_rate_bins : np.ndarray
        Boolean array with shape ``(n_trials, n_bins)`` marking bins included in mean rates.
    trial_start_offsets, led_offsets, choice_offsets : np.ndarray
        One-dimensional float arrays with shape ``(n_trials,)`` in seconds relative to alignment.
    """

    alignment: str
    trial_indices: np.ndarray
    time_bin_edges: np.ndarray
    time_bin_centers: np.ndarray
    left_raster_times_by_trial: list[np.ndarray]
    right_raster_times_by_trial: list[np.ndarray]
    left_rate_by_trial: np.ndarray
    right_rate_by_trial: np.ndarray
    valid_rate_bins: np.ndarray
    trial_start_offsets: np.ndarray
    led_offsets: np.ndarray
    choice_offsets: np.ndarray

@dataclass(frozen=True)
class SpikePethData:
    """
    Trial-aligned spike raster and firing-rate data for one unit.

    Attributes
    ----------
    unit_cluster_id : int
        Sorter cluster id for the plotted unit.
    alignment : str
        Alignment mode. Supported values are ``"trial_start"`` and ``"choice"``.
    trial_indices : np.ndarray
        One-dimensional integer array with shape ``(n_trials,)`` containing source trial indices.
    time_bin_edges : np.ndarray
        One-dimensional float array with shape ``(n_bins + 1,)`` in seconds relative to alignment.
    time_bin_centers : np.ndarray
        One-dimensional float array with shape ``(n_bins,)`` in seconds relative to alignment.
    spike_raster_times_by_trial : list[np.ndarray]
        One array per trial. Each array contains spike times in seconds relative to alignment.
    spike_rate_by_trial : np.ndarray
        Firing-rate array with shape ``(n_trials, n_bins)`` in spikes/s. Invalid bins are ``NaN``.
    valid_rate_bins : np.ndarray
        Boolean array with shape ``(n_trials, n_bins)`` marking bins included in mean rates.
    trial_start_offsets, led_offsets, choice_offsets : np.ndarray
        One-dimensional float arrays with shape ``(n_trials,)`` in seconds relative to alignment.
    """

    unit_cluster_id: int
    alignment: str
    trial_indices: np.ndarray
    time_bin_edges: np.ndarray
    time_bin_centers: np.ndarray
    spike_raster_times_by_trial: list[np.ndarray]
    spike_rate_by_trial: np.ndarray
    valid_rate_bins: np.ndarray
    trial_start_offsets: np.ndarray
    led_offsets: np.ndarray
    choice_offsets: np.ndarray

def select_first_valid_unit_cluster_id(cluster_info: pd.DataFrame) -> int:
    """
    Select the first sorter unit labeled as usable for single-unit PETH plotting.

    Parameters
    ----------
    cluster_info : pd.DataFrame
        Cluster metadata table with shape ``(n_units, n_columns)``. Required columns are
        ``cluster_id`` and ``group``. ``cluster_id`` contains integer sorter cluster ids.
        ``group`` contains sorter labels, where ``"good"`` and ``"mua"`` are treated as valid.

    Returns
    -------
    int
        First valid ``cluster_id`` in the existing row order.
    """

    required_columns = {"cluster_id", "group"}
    missing_columns = required_columns - set(cluster_info.columns)
    if missing_columns:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing_columns)}")

    normalized_groups = cluster_info["group"].astype(str).str.strip().str.lower()
    valid_unit_mask = normalized_groups.isin({"good", "mua"})
    if not valid_unit_mask.any():
        raise ValueError("No units labeled 'good' or 'mua' are available for spike PETH plotting.")

    return int(cluster_info.loc[valid_unit_mask, "cluster_id"].iloc[0])

def _normalize_trial_mask(trial_mask: pd.Series | np.ndarray, trial_df: pd.DataFrame) -> pd.Series:
    """Validate and align a user-provided trial mask to the trial table."""
    if isinstance(trial_mask, pd.Series):
        if not trial_mask.index.equals(trial_df.index):
            trial_mask = trial_mask.reindex(trial_df.index, fill_value=False)
        return trial_mask.astype(bool)

    mask_array = np.asarray(trial_mask, dtype=bool)
    if mask_array.ndim != 1 or mask_array.shape[0] != trial_df.shape[0]:
        raise ValueError("trial_mask must be a one-dimensional boolean mask matching trial_df rows.")
    return pd.Series(mask_array, index=trial_df.index)

def _make_relative_bin_edges(x_start: float, x_end: float, bin_size: float) -> np.ndarray:
    """Create bin edges that cover the requested relative-time window."""
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    if x_end <= x_start:
        raise ValueError("PETH window end must be greater than window start.")

    n_bins = int(np.ceil((x_end - x_start) / bin_size))
    return x_start + np.arange(n_bins + 1, dtype=float) * bin_size

def _ts_times(event_times: nap.Ts, event_name: str) -> np.ndarray:
    """Extract sorted timestamps from one Pynapple event stream."""
    if not isinstance(event_times, nap.Ts):
        raise TypeError(f"{event_name!r} must be a pynapple Ts.")
    return np.sort(np.asarray(event_times.index.to_numpy(), dtype=float))

def _make_event_rate_row(
    absolute_event_times: np.ndarray,
    reference_time: float,
    time_bin_edges: np.ndarray,
    valid_bin_mask: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative event times and binned event rates for one trial."""
    relative_event_times = absolute_event_times - reference_time
    valid_window_start = time_bin_edges[0]
    valid_window_end = time_bin_edges[:-1][valid_bin_mask][-1] + bin_size
    event_mask = (relative_event_times >= valid_window_start) & (relative_event_times <= valid_window_end)
    plotted_event_times = relative_event_times[event_mask]

    event_counts, _ = np.histogram(plotted_event_times, bins=time_bin_edges)
    event_rate = event_counts.astype(float) / bin_size
    event_rate[~valid_bin_mask] = np.nan
    return plotted_event_times, event_rate

def build_lick_peth_data(
    trial_df: pd.DataFrame,
    lick_times: Mapping[str, nap.Ts],
    alignment: str,
    rate_bin_size: float = 0.5,
    pre_time: float = 2.0,
    post_time: float = 1.0,
    show_led_lines: bool = True,
    max_time_after_trial_start: float | None = None,
    trial_mask: pd.Series | np.ndarray | None = None,
) -> LickPethData:
    """
    Build trial-aligned lick rasters and lick-rate arrays.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Required columns are
        ``start_time`` and ``choice_time`` in seconds. ``led_on_time`` in seconds is required
        when ``show_led_lines=True``.
    lick_times : Mapping[str, nap.Ts]
        Mapping with ``"left_entry"`` and ``"right_entry"`` keys. Each value is a
        one-dimensional Pynapple ``Ts`` of lick timestamps in seconds.
    alignment : str
        Alignment mode. ``"trial_start"`` aligns time zero to ``start_time`` and uses the
        longest selected choice latency plus ``post_time`` as the right x-limit. ``"choice"``
        aligns time zero to ``choice_time`` and uses a fixed ``[-pre_time, post_time]`` window.
    rate_bin_size : float, default=0.5
        Lick-rate bin width in seconds for the lower mean-rate panel.
    pre_time : float, default=2.0
        Seconds before the alignment event to include.
    post_time : float, default=1.0
        Seconds after choice to include for trial-start alignment, or seconds after choice
        alignment for choice-aligned plots.
    show_led_lines : bool, default=True
        Whether to require numeric LED times while preparing event offsets.
    max_time_after_trial_start : float or None, default=None
        Optional positive x-axis cap in seconds after trial start. This applies only to
        ``alignment="trial_start"`` and does not drop long trials; licks and event markers beyond
        the cap are not shown.
    trial_mask : pd.Series, np.ndarray, or None, default=None
        Optional boolean selector with shape ``(n_trials,)``. The selected trials are intersected
        with the validity mask and remain in chronological table order.

    Returns
    -------
    LickPethData
        Dataclass containing trial-wise relative lick rasters, binned lick rates with shape
        ``(n_trials, n_bins)``, and event offsets in seconds.
    """

    if alignment not in {"trial_start", "choice"}:
        raise ValueError("alignment must be 'trial_start' or 'choice'.")
    if pre_time < 0 or post_time < 0:
        raise ValueError("pre_time and post_time must be non-negative.")
    if rate_bin_size <= 0:
        raise ValueError("rate_bin_size must be positive.")
    if max_time_after_trial_start is not None and max_time_after_trial_start <= 0:
        raise ValueError("max_time_after_trial_start must be positive when provided.")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")

    valid_trial_mask = select_valid_lick_peth_trials(trial_df, require_led_time=show_led_lines)
    if trial_mask is not None:
        valid_trial_mask &= _normalize_trial_mask(trial_mask, trial_df)
    selected_trials = trial_df.loc[valid_trial_mask].copy()
    if selected_trials.empty:
        raise ValueError("No valid trials are available for lick PETH plotting.")

    start_times = pd.to_numeric(selected_trials["start_time"], errors="coerce").to_numpy(dtype=float)
    choice_times = pd.to_numeric(selected_trials["choice_time"], errors="coerce").to_numpy(dtype=float)
    if "led_on_time" in selected_trials.columns:
        led_times = pd.to_numeric(selected_trials["led_on_time"], errors="coerce").to_numpy(dtype=float)
    else:
        led_times = np.full(selected_trials.shape[0], np.nan, dtype=float)

    if alignment == "trial_start":
        reference_times = start_times
        trial_start_offsets = np.zeros_like(start_times)
        led_offsets = led_times - start_times
        choice_offsets = choice_times - start_times
        x_start = -float(pre_time)
        x_end = float(np.nanmax(choice_offsets) + post_time)
        if max_time_after_trial_start is not None:
            x_end = min(x_end, float(max_time_after_trial_start))
        trial_end_offsets = choice_offsets + post_time
    else:
        reference_times = choice_times
        trial_start_offsets = start_times - choice_times
        led_offsets = led_times - choice_times
        choice_offsets = np.zeros_like(choice_times)
        x_start = -float(pre_time)
        x_end = float(post_time)
        trial_end_offsets = np.full_like(choice_times, x_end, dtype=float)

    time_bin_edges = _make_relative_bin_edges(x_start=x_start, x_end=x_end, bin_size=rate_bin_size)
    time_bin_centers = time_bin_edges[:-1] + rate_bin_size / 2
    valid_rate_bins = time_bin_centers[None, :] <= trial_end_offsets[:, None]

    left_times = _ts_times(lick_times[LEFT_LICK_EVENT], LEFT_LICK_EVENT)
    right_times = _ts_times(lick_times[RIGHT_LICK_EVENT], RIGHT_LICK_EVENT)
    left_raster_times_by_trial: list[np.ndarray] = []
    right_raster_times_by_trial: list[np.ndarray] = []
    left_rate_rows = []
    right_rate_rows = []
    for trial_idx, reference_time in enumerate(reference_times):
        left_raster_times, left_rate = _make_event_rate_row(
            absolute_event_times=left_times,
            reference_time=float(reference_time),
            time_bin_edges=time_bin_edges,
            valid_bin_mask=valid_rate_bins[trial_idx],
            bin_size=rate_bin_size,
        )
        right_raster_times, right_rate = _make_event_rate_row(
            absolute_event_times=right_times,
            reference_time=float(reference_time),
            time_bin_edges=time_bin_edges,
            valid_bin_mask=valid_rate_bins[trial_idx],
            bin_size=rate_bin_size,
        )
        left_raster_times_by_trial.append(left_raster_times)
        right_raster_times_by_trial.append(right_raster_times)
        left_rate_rows.append(left_rate)
        right_rate_rows.append(right_rate)

    return LickPethData(
        alignment=alignment,
        trial_indices=selected_trials.index.to_numpy(dtype=int),
        time_bin_edges=time_bin_edges,
        time_bin_centers=time_bin_centers,
        left_raster_times_by_trial=left_raster_times_by_trial,
        right_raster_times_by_trial=right_raster_times_by_trial,
        left_rate_by_trial=np.vstack(left_rate_rows),
        right_rate_by_trial=np.vstack(right_rate_rows),
        valid_rate_bins=valid_rate_bins,
        trial_start_offsets=trial_start_offsets,
        led_offsets=led_offsets,
        choice_offsets=choice_offsets,
    )

def build_single_unit_spike_peth_data(
    trial_df: pd.DataFrame,
    unit_spikes: nap.Ts,
    unit_cluster_id: int,
    alignment: str,
    rate_bin_size: float = 0.1,
    pre_time: float = 2.0,
    post_time: float = 1.0,
    show_led_lines: bool = True,
    max_time_after_trial_start: float | None = None,
    trial_mask: pd.Series | np.ndarray | None = None,
) -> SpikePethData:
    """
    Build trial-aligned spike rasters and firing-rate arrays for one unit.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Required columns are
        ``start_time`` and ``choice_time`` in seconds. ``led_on_time`` in seconds is required
        when ``show_led_lines=True``.
    unit_spikes : nap.Ts
        One-dimensional Pynapple ``Ts`` containing spike timestamps for one unit in seconds.
    unit_cluster_id : int
        Sorter cluster id for ``unit_spikes``.
    alignment : str
        Alignment mode. ``"trial_start"`` aligns time zero to ``start_time`` and uses the
        longest selected choice latency plus ``post_time`` as the right x-limit. ``"choice"``
        aligns time zero to ``choice_time`` and uses a fixed ``[-pre_time, post_time]`` window.
    rate_bin_size : float, default=0.1
        Firing-rate bin width in seconds for the lower mean-rate panel.
    pre_time : float, default=2.0
        Seconds before the alignment event to include.
    post_time : float, default=1.0
        Seconds after choice to include for trial-start alignment, or seconds after choice
        alignment for choice-aligned plots.
    show_led_lines : bool, default=True
        Whether to require numeric LED times while preparing event offsets.
    max_time_after_trial_start : float or None, default=None
        Optional positive x-axis cap in seconds after trial start. This applies only to
        ``alignment="trial_start"`` and does not drop long trials; spikes and event markers
        beyond the cap are not shown.
    trial_mask : pd.Series, np.ndarray, or None, default=None
        Optional boolean selector with shape ``(n_trials,)``. The selected trials are intersected
        with the validity mask and remain in chronological table order.

    Returns
    -------
    SpikePethData
        Dataclass containing trial-wise relative spike rasters, binned firing rates with shape
        ``(n_trials, n_bins)``, and event offsets in seconds.
    """

    if alignment not in {"trial_start", "choice"}:
        raise ValueError("alignment must be 'trial_start' or 'choice'.")
    if not isinstance(unit_spikes, nap.Ts):
        raise TypeError("unit_spikes must be a pynapple Ts.")
    if pre_time < 0 or post_time < 0:
        raise ValueError("pre_time and post_time must be non-negative.")
    if rate_bin_size <= 0:
        raise ValueError("rate_bin_size must be positive.")
    if max_time_after_trial_start is not None and max_time_after_trial_start <= 0:
        raise ValueError("max_time_after_trial_start must be positive when provided.")

    valid_trial_mask = select_valid_lick_peth_trials(trial_df, require_led_time=show_led_lines)
    if trial_mask is not None:
        valid_trial_mask &= _normalize_trial_mask(trial_mask, trial_df)
    selected_trials = trial_df.loc[valid_trial_mask].copy()
    if selected_trials.empty:
        raise ValueError("No valid trials are available for spike PETH plotting.")

    start_times = pd.to_numeric(selected_trials["start_time"], errors="coerce").to_numpy(dtype=float)
    choice_times = pd.to_numeric(selected_trials["choice_time"], errors="coerce").to_numpy(dtype=float)
    if "led_on_time" in selected_trials.columns:
        led_times = pd.to_numeric(selected_trials["led_on_time"], errors="coerce").to_numpy(dtype=float)
    else:
        led_times = np.full(selected_trials.shape[0], np.nan, dtype=float)

    if alignment == "trial_start":
        reference_times = start_times
        trial_start_offsets = np.zeros_like(start_times)
        led_offsets = led_times - start_times
        choice_offsets = choice_times - start_times
        x_start = -float(pre_time)
        x_end = float(np.nanmax(choice_offsets) + post_time)
        if max_time_after_trial_start is not None:
            x_end = min(x_end, float(max_time_after_trial_start))
        trial_end_offsets = choice_offsets + post_time
    else:
        reference_times = choice_times
        trial_start_offsets = start_times - choice_times
        led_offsets = led_times - choice_times
        choice_offsets = np.zeros_like(choice_times)
        x_start = -float(pre_time)
        x_end = float(post_time)
        trial_end_offsets = np.full_like(choice_times, x_end, dtype=float)

    time_bin_edges = _make_relative_bin_edges(x_start=x_start, x_end=x_end, bin_size=rate_bin_size)
    time_bin_centers = time_bin_edges[:-1] + rate_bin_size / 2
    valid_rate_bins = time_bin_centers[None, :] <= trial_end_offsets[:, None]

    spike_times = _ts_times(unit_spikes, "unit_spikes")
    spike_raster_times_by_trial: list[np.ndarray] = []
    spike_rate_rows = []
    for trial_idx, reference_time in enumerate(reference_times):
        spike_raster_times, spike_rate = _make_event_rate_row(
            absolute_event_times=spike_times,
            reference_time=float(reference_time),
            time_bin_edges=time_bin_edges,
            valid_bin_mask=valid_rate_bins[trial_idx],
            bin_size=rate_bin_size,
        )
        spike_raster_times_by_trial.append(spike_raster_times)
        spike_rate_rows.append(spike_rate)

    return SpikePethData(
        unit_cluster_id=int(unit_cluster_id),
        alignment=alignment,
        trial_indices=selected_trials.index.to_numpy(dtype=int),
        time_bin_edges=time_bin_edges,
        time_bin_centers=time_bin_centers,
        spike_raster_times_by_trial=spike_raster_times_by_trial,
        spike_rate_by_trial=np.vstack(spike_rate_rows),
        valid_rate_bins=valid_rate_bins,
        trial_start_offsets=trial_start_offsets,
        led_offsets=led_offsets,
        choice_offsets=choice_offsets,
    )
