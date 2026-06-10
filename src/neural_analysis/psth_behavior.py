"""
The goal of this function is to make plots of events around the LED cue or the choice event.
It will be done with licks and with spikes.
Licks should have L and R plots
Spikes should have a plot for each unit
"""

"""
- load a session, its PFC-HPC-V1 spikes, and its L-R licks from the event dict
- reject all the experimenter-given-reward trials
- gather the LED onsets and the choice events
- Gather trial conditions - L/R correct, L/R incorrect, L/R omission, L/R switch, L/R stay 

Lick plots
- plot L/R licks in a plot_peth-like fashion, both on the same superplot. Make superplots for each condition
- ideally sort trials by 
0) All trials in chronological order.
Trial-start alignment
I'd like trials to be perfectly aligned to the same window (say, 2s before trial start to 1s after choice), but the LED-choice intervals may vary.7
I want the plots to have that full window for each trial, even if it doesn't look ideal - so the plot should be as wide as the longest trial, and shorter trials will have blank space after the choice.
Time 0 will be trial start, and have a general vertical line at 0.
Each trial should have a vertical line at the time of the LED cue and a vertical line at the time of the choice. Those LED signals should
all be at the same time, so if things check out then I'll want to be able to turn off the LED lines.

Choice time alignment
Trials should show -2s before choice to 1s after choice. I want vertical lines for trial start, LED on (able to be turned off), 
choice time (should be a general vertical line at 0). Using choice time means that I should be able to use the same window size for every plot.

L licks and R licks should be treated like different units, so L and R will have their own PETH and mean plots.    

Unit-spike plots
Only assess units that are not noise, by user-chosen labels (mua or good is fine)

1) L-correct + R-correct. 
Use the same windows as in the lick trials.

2) L-incorrect + R-incorrect 
3) L-omission + R-omission
4) L-switch + R-switch
5) L-stay + R-stay
This could probably all live on one big figure, with 5 x 4 subplots. That is 5 conditions, each with 2 psth (L/R lick) and their 2 mean plots  

- Each unit needs its own plot... but a superplot would be helpful. but what if I have like 300 units lol
    - I should plot all 300 while getting used to inspecting the data. Either do a long pdf or split across multiple images
    to maintain visibility  
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
from pathlib import Path
from typing import Mapping
from dataclasses import dataclass
from src.behavior_analysis.project_utils import get_experimenter_reward_flags
from src.neural_analysis import spike_behavior_pynapple


SUPPORTED_ANALYSIS_REGIONS = {"HPC", "V1", "PFC"}
LEFT_LICK_EVENT = "left_entry"
RIGHT_LICK_EVENT = "right_entry"
LICK_COLORS = {
    LEFT_LICK_EVENT: "tab:orange",
    RIGHT_LICK_EVENT: "tab:blue",
}


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


def _numeric_present(values: pd.Series) -> pd.Series:
    """Return rows with numeric, non-missing values."""
    return pd.to_numeric(values, errors="coerce").notna()


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
        on the next valid row's action, matching ``spike_behavior_pynapple.make_trial_type_masks``.
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


def _nanmean_rate(rate_by_trial: np.ndarray) -> np.ndarray:
    """Average rates across trials while ignoring invalid bins."""
    valid_counts = np.sum(~np.isnan(rate_by_trial), axis=0)
    summed_rates = np.nansum(rate_by_trial, axis=0)
    return np.divide(
        summed_rates,
        valid_counts,
        out=np.full(rate_by_trial.shape[1], np.nan, dtype=float),
        where=valid_counts > 0,
    )


def _plot_raster_rows(
    ax,
    raster_times_by_trial: list[np.ndarray],
    color: str,
    label: str,
    raster_line_length: float,
    raster_line_width: float,
) -> None:
    """Plot trial-wise event ticks on one raster axis."""
    lineoffsets = np.arange(len(raster_times_by_trial), dtype=float)
    event_collections = ax.eventplot(
        raster_times_by_trial,
        orientation="horizontal",
        lineoffsets=lineoffsets,
        linelengths=raster_line_length,
        linewidths=raster_line_width,
        colors=color,
        label="_nolegend_",
    )
    if event_collections:
        event_collections[0].set_label(label)


def _draw_trial_event_markers(
    ax,
    peth_data: LickPethData | SpikePethData,
    show_led_lines: bool,
    event_marker_length: float,
) -> None:
    """Draw per-trial event markers on a raster axis."""
    y_positions = np.arange(peth_data.trial_indices.shape[0], dtype=float)
    half_marker_length = min(float(event_marker_length) / 2, 0.45)
    if peth_data.alignment == "trial_start":
        ax.vlines(
            peth_data.choice_offsets,
            y_positions - half_marker_length,
            y_positions + half_marker_length,
            colors="tab:purple",
            linewidth=0.8,
            label="Choice",
        )
    else:
        ax.vlines(
            peth_data.trial_start_offsets,
            y_positions - half_marker_length,
            y_positions + half_marker_length,
            colors="black",
            linewidth=0.8,
            label="Trial start",
        )

    if show_led_lines:
        led_mask = ~np.isnan(peth_data.led_offsets)
        ax.vlines(
            peth_data.led_offsets[led_mask],
            y_positions[led_mask] - half_marker_length,
            y_positions[led_mask] + half_marker_length,
            colors="tab:green",
            linewidth=0.8,
            label="LED",
        )


def _format_lick_axes(
    raster_ax,
    mean_ax,
    lick_peth_data: LickPethData,
    show_led_lines: bool,
    title: str,
    event_marker_length: float,
) -> None:
    """Apply shared axis labels and event markers."""
    alignment_label = "Trial start" if lick_peth_data.alignment == "trial_start" else "Choice"
    for ax in (raster_ax, mean_ax):
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, label=alignment_label)
        ax.set_xlim(lick_peth_data.time_bin_edges[0], lick_peth_data.time_bin_edges[-1])

    _draw_trial_event_markers(
        raster_ax,
        lick_peth_data,
        show_led_lines=show_led_lines,
        event_marker_length=event_marker_length,
    )
    n_trials = lick_peth_data.trial_indices.shape[0]
    raster_ax.set_ylim(n_trials - 0.5, -0.5)
    raster_ax.set_ylabel("Trial")
    raster_ax.set_title(title)
    mean_ax.set_ylabel("Licks/s")
    mean_ax.set_xlabel("Time from alignment (s)")


def _format_spike_axes(
    raster_ax,
    mean_ax,
    spike_peth_data: SpikePethData,
    show_led_lines: bool,
    event_marker_length: float,
) -> None:
    """Apply shared axis labels and event markers to one-unit spike plots."""
    alignment_label = "Trial start" if spike_peth_data.alignment == "trial_start" else "Choice"
    for ax in (raster_ax, mean_ax):
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, label=alignment_label)
        ax.set_xlim(spike_peth_data.time_bin_edges[0], spike_peth_data.time_bin_edges[-1])

    _draw_trial_event_markers(
        raster_ax,
        spike_peth_data,
        show_led_lines=show_led_lines,
        event_marker_length=event_marker_length,
    )
    n_trials = spike_peth_data.trial_indices.shape[0]
    raster_ax.set_ylim(n_trials - 0.5, -0.5)
    raster_ax.set_ylabel("Trial")
    raster_ax.set_title(f"Unit {spike_peth_data.unit_cluster_id} spike raster")
    mean_ax.set_ylabel("Spikes/s")
    mean_ax.set_xlabel("Time from alignment (s)")


def _lick_peth_fig_height(n_trials: int, row_height_inches: float, min_fig_height: float) -> float:
    """Scale raster figure height with trial count while keeping small plots readable."""
    if row_height_inches <= 0:
        raise ValueError("raster_row_height_inches must be positive.")
    if min_fig_height <= 0:
        raise ValueError("min_fig_height must be positive.")
    return max(float(min_fig_height), 2.0 + int(n_trials) * float(row_height_inches))


def plot_lick_peth(
    lick_peth_data: LickPethData,
    lick_layout: str = "overlay",
    show_led_lines: bool = True,
    raster_line_length: float = 0.6,
    raster_line_width: float = 0.7,
    raster_row_height_inches: float = 0.08,
    min_fig_height: float = 6.0,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot lick rasters and mean lick rates.
z
    Parameters
    ----------
    lick_peth_data : LickPethData
        Trial-aligned lick data. Lick-rate arrays have shape ``(n_trials, n_bins)`` in events/s.
    lick_layout : str, default="overlay"
        ``"overlay"`` plots left and right licks on shared raster and mean axes. ``"separate"``
        plots left and right licks in separate columns.
    show_led_lines : bool, default=True
        Whether to draw per-trial LED markers on the raster axes.
    raster_line_length : float, default=0.6
        Height of each raster lick tick in trial-row units.
    raster_line_width : float, default=0.7
        Width of each raster lick tick in points.
    raster_row_height_inches : float, default=0.08
        Figure-height contribution for each trial row, in inches.
    min_fig_height : float, default=6.0
        Minimum figure height in inches.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and axes. Overlay mode returns two axes with shape ``(2,)``; separate mode returns
        four axes with shape ``(2, 2)``.
    """

    if lick_layout not in {"overlay", "separate"}:
        raise ValueError("lick_layout must be 'overlay' or 'separate'.")
    if raster_line_length <= 0:
        raise ValueError("raster_line_length must be positive.")
    if raster_line_width <= 0:
        raise ValueError("raster_line_width must be positive.")

    left_mean_rate = _nanmean_rate(lick_peth_data.left_rate_by_trial)
    right_mean_rate = _nanmean_rate(lick_peth_data.right_rate_by_trial)
    time_centers = lick_peth_data.time_bin_centers
    n_trials = lick_peth_data.trial_indices.shape[0]
    fig_height = _lick_peth_fig_height(
        n_trials=n_trials,
        row_height_inches=raster_row_height_inches,
        min_fig_height=min_fig_height,
    )

    if lick_layout == "overlay":
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, fig_height), height_ratios=[3, 1])
        raster_ax, mean_ax = axes
        _plot_raster_rows(
            raster_ax,
            lick_peth_data.left_raster_times_by_trial,
            LICK_COLORS[LEFT_LICK_EVENT],
            "Left lick",
            raster_line_length=raster_line_length,
            raster_line_width=raster_line_width,
        )
        _plot_raster_rows(
            raster_ax,
            lick_peth_data.right_raster_times_by_trial,
            LICK_COLORS[RIGHT_LICK_EVENT],
            "Right lick",
            raster_line_length=raster_line_length,
            raster_line_width=raster_line_width,
        )
        mean_ax.plot(time_centers, left_mean_rate, color=LICK_COLORS[LEFT_LICK_EVENT], label="Left lick")
        mean_ax.plot(time_centers, right_mean_rate, color=LICK_COLORS[RIGHT_LICK_EVENT], label="Right lick")
        _format_lick_axes(
            raster_ax,
            mean_ax,
            lick_peth_data,
            show_led_lines,
            title="Lick raster",
            event_marker_length=raster_line_length,
        )
        for ax in axes:
            ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, axes

    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(12, fig_height), height_ratios=[3, 1])
    left_raster_ax, right_raster_ax = axes[0]
    left_mean_ax, right_mean_ax = axes[1]

    _plot_raster_rows(
        left_raster_ax,
        lick_peth_data.left_raster_times_by_trial,
        LICK_COLORS[LEFT_LICK_EVENT],
        "Left lick",
        raster_line_length=raster_line_length,
        raster_line_width=raster_line_width,
    )
    _plot_raster_rows(
        right_raster_ax,
        lick_peth_data.right_raster_times_by_trial,
        LICK_COLORS[RIGHT_LICK_EVENT],
        "Right lick",
        raster_line_length=raster_line_length,
        raster_line_width=raster_line_width,
    )
    left_mean_ax.plot(time_centers, left_mean_rate, color=LICK_COLORS[LEFT_LICK_EVENT], label="Left lick")
    right_mean_ax.plot(time_centers, right_mean_rate, color=LICK_COLORS[RIGHT_LICK_EVENT], label="Right lick")

    _format_lick_axes(
        left_raster_ax,
        left_mean_ax,
        lick_peth_data,
        show_led_lines,
        title="Left licks",
        event_marker_length=raster_line_length,
    )
    _format_lick_axes(
        right_raster_ax,
        right_mean_ax,
        lick_peth_data,
        show_led_lines,
        title="Right licks",
        event_marker_length=raster_line_length,
    )
    for ax in np.ravel(axes):
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, axes


def plot_single_unit_spike_peth(
    spike_peth_data: SpikePethData,
    show_led_lines: bool = True,
    raster_line_length: float = 0.6,
    raster_line_width: float = 0.7,
    raster_row_height_inches: float = 0.08,
    min_fig_height: float = 6.0,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one unit's spike raster and mean firing rate.

    Parameters
    ----------
    spike_peth_data : SpikePethData
        Trial-aligned spike data. Firing-rate array has shape ``(n_trials, n_bins)`` in spikes/s.
    show_led_lines : bool, default=True
        Whether to draw per-trial LED markers on the raster axis.
    raster_line_length : float, default=0.6
        Height of each raster spike tick in trial-row units.
    raster_line_width : float, default=0.7
        Width of each raster spike tick in points.
    raster_row_height_inches : float, default=0.08
        Figure-height contribution for each trial row, in inches.
    min_fig_height : float, default=6.0
        Minimum figure height in inches.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and two axes with shape ``(2,)``: raster axis followed by mean-rate axis.
    """

    if raster_line_length <= 0:
        raise ValueError("raster_line_length must be positive.")
    if raster_line_width <= 0:
        raise ValueError("raster_line_width must be positive.")

    mean_spike_rate = _nanmean_rate(spike_peth_data.spike_rate_by_trial)
    n_trials = spike_peth_data.trial_indices.shape[0]
    fig_height = _lick_peth_fig_height(
        n_trials=n_trials,
        row_height_inches=raster_row_height_inches,
        min_fig_height=min_fig_height,
    )

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, fig_height), height_ratios=[3, 1])
    raster_ax, mean_ax = axes
    _plot_raster_rows(
        raster_ax,
        spike_peth_data.spike_raster_times_by_trial,
        color="black",
        label=f"Unit {spike_peth_data.unit_cluster_id}",
        raster_line_length=raster_line_length,
        raster_line_width=raster_line_width,
    )
    mean_ax.plot(
        spike_peth_data.time_bin_centers,
        mean_spike_rate,
        color="black",
        label=f"Unit {spike_peth_data.unit_cluster_id}",
    )
    _format_spike_axes(
        raster_ax,
        mean_ax,
        spike_peth_data,
        show_led_lines=show_led_lines,
        event_marker_length=raster_line_length,
    )
    for ax in axes:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, axes


def save_figure_with_message(fig: plt.Figure, save_path: Path) -> None:
    """
    Save one matplotlib figure and print its output path.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object to save.
    save_path : Path
        Output path for the PNG file. Units: filesystem path.

    Returns
    -------
    None
        Saves ``fig`` to ``save_path`` and prints a short status message.
    """

    fig.savefig(save_path, format="png", dpi=300)
    print(f"Saved figure {save_path.name} to {save_path}")


def plot_peth(unit_peth, unit_peth_counts, ax_mean, ax_spikes,
              bin_size, color=None):
    mean = np.mean(unit_peth_counts / bin_size, axis=1)
    ax_mean.plot(mean, color=color)
    ax_mean.set_ylabel("spikes/s")
    ax_mean.axvline(0.0, color="gray", linestyle="--")
    ax_spikes.plot(unit_peth.to_tsd(), "|", markersize=5, color=color)
    ax_mean.set_xlabel("time from event (s)")
    ax_spikes.set_ylabel("event")
    ax_spikes.axvline(0.0, color="gray", linestyle="--")



def main() -> None:
    """
    Run one hard-coded session example for user-defined region spike binning.

    The script loads processed behavior tables, loads aligned spike times plus sorter cluster
    metadata for the current example HPC probe, builds a Pynapple spike group, bins
    region-selected spikes by trial, and prints a short summary. Times are handled in seconds throughout.
    """

    multi_session_save_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis")
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference")
    sess_id_full = "CT014_2025-12-23_163505"
    pfc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-19_180103/sorting_mchin_20260330"
    hpc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-19_183540/sorting_mchin_20260331"

    raw_behavior_folder = session_data_home / "rpi" / sess_id_full
    processed_data_path = session_data_home / "processed"
    figure_path = session_data_home / "figures" / "peth"

    aligned_spike_path = session_data_home / 'ephys/aligned/aligned_imec'
    aligned_pfc_spike_path = aligned_spike_path / 'imec0_sync.npz'
    aligned_hpc_spike_path = aligned_spike_path / 'imec1_sync.npz'

    probe_json_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/probe_json.json"

    if not hpc_spike_path.exists():
        raise FileNotFoundError(f"HPC sorter output not found at {hpc_spike_path}")
    assert probe_json_path.exists(), f"Probe JSON file not found at {probe_json_path}"
    assert aligned_pfc_spike_path.exists(), f"Aligned PFC spike file not found at {aligned_pfc_spike_path}"
    assert aligned_hpc_spike_path.exists(), f"Aligned HPC spike file not found at {aligned_hpc_spike_path}"

    decode_target = "action"  # state_int or action
    n_decoder_runs = 5
    # print(", ".join(map(str, "your_array")))
    hpc_channels_shank0 = np.array([
        4, 3, 2, 1, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181,
        180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167,
        166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153,
        152, 151, 150, 149, 148, 147, 146, 145, 96, 95, 94, 93, 92, 91, 90, 89,
        88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71,
        70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53,
        52, 51, 50, 49, 384, 383, 382, 381
    ])
    hpc_channels_shank3 = np.array([
        255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242,
        241, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326, 325, 324,
        323, 322, 321, 320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310,
        309, 308, 307, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296,
        295, 294, 293, 292, 291, 290, 289, 240, 239, 238, 237, 236, 235, 234,
        233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
        219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206,
        205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 144,
        143, 142, 141
    ])
    hpc_channels = np.concatenate([hpc_channels_shank0, hpc_channels_shank3])

    v1_channels_shank0 = np.array([
        138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125,
        124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,
        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
        14, 13, 12, 11, 10, 9, 8, 7, 6, 5
    ])
    v1_channels_shank3 = np.array([
        78, 377, 376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365,
        364, 363, 362, 361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351,
        350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 340, 339, 338, 337,
        288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275,
        274, 273, 272, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261,
        260, 259, 258, 257, 256
    ])
    v1_channels = np.concatenate([v1_channels_shank0, v1_channels_shank3])

    _pfc_channels = ("55  54  53  52  51  50  49  48 383 382 381 380 379 378 377 376 375 374 "
                    "373 372 371 370 369 368 367 366 365 364 363 362 361 360 359 358 357 356 "
                    "355 354 353 352 351 350 349 348 347 346 345 344 343 342 341 340 339 338 "
                    "337 336 287 286 285 284 283 282 281 280 279 278 277 276 275 274 273 272 "
                    "271 270 269 268 267 266 265 264 263 262 261 260 259 258 257 256 255 254 "
                    "253 252 251 250 249 248 247 246 245 244 243 242 241 240 335 334 333 332 "
                    "331 330 329 328 327 326 325 324 323 322 321 320 319 318 317 316 315 314 "
                    "313 312 311 310 309 308 307 306 305 304 303 302 301 300 299 298 297 296 "
                    "295 294 293 292 291 290 289 288 239 238 237 236 235 234 233 232 231 230 "
                    "229 228 227 226 225 224 223 222 221 220 219 218 217 216 215 214 213 212 "
                    "211 210 209 208 207 206 205 204 203 202 201 200 199 198 197 196 195 194 "
                    "193 192 143 142 141 140 139 138 137 136 135 134 133 132 131 130 129 128 "
                    "127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 111 110 "
                    "109 108 107 106 105 104 103 102 101 100  99  98  97  96  47  46  45  44 "
                    "43  42  41  40  39  38  37  36  35  34  33  32  31  30  29  28  27  26 "
                    "25  24  23  22  21  20  19  18  17  16  15  14  13  12  11  10 9  8 "
                    "7   6   5   4   3   2   1   0")
    pfc_channels = np.sort(np.array(list(map(int, [val for val in _pfc_channels.split(" ") if val != '']))))

    mouse, date, timestamp = spike_behavior_pynapple.parse_session_id(sess_id_full)
    session_info_path = raw_behavior_folder / f"{sess_id_full}_session_info.pkl"

    session = spike_behavior_pynapple.Session(
        multi_session_save_path=multi_session_save_path,
        session_data_home=session_data_home,
        sess_id_full=sess_id_full,
        sess_id_abbreviated=f"{mouse}_{date}",
        raw_behavior_folder=raw_behavior_folder,
        processed_data_path=processed_data_path,
        figure_path=figure_path,
        mouse=mouse,
        date=date,
        timestamp=timestamp,
        session_info_path=session_info_path,
        session_info=spike_behavior_pynapple.load_session_info(session_info_path)
        if session_info_path.exists()
        else None,
        pfc_spike_path=pfc_spike_path,
        hpc_spike_path=hpc_spike_path,
    )

    region_name = "PFC"
    if region_name in SUPPORTED_ANALYSIS_REGIONS:
        if region_name == "HPC":
            region_channels = spike_behavior_pynapple.normalize_region_channels(hpc_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "V1":
            region_channels = spike_behavior_pynapple.normalize_region_channels(v1_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "PFC":
            region_channels = spike_behavior_pynapple.normalize_region_channels(pfc_channels)
            sorter_output_path = session.pfc_spike_path
            spike_path = aligned_pfc_spike_path
    else:
        exit("Region not supported")

    event_df, trial_df = spike_behavior_pynapple.load_session_tables(session)
    lick_times = spike_behavior_pynapple.build_lick_time_dict(event_df)
    figure_path.mkdir(parents=True, exist_ok=True)
    lick_rate_bin_size = 0.5
    max_time_after_trial_start = 5.0
    trial_type_masks = make_lick_peth_trial_type_masks(trial_df, require_led_time=True)
    conditions_to_plot = {
        "all_valid": None,
        "left_correct": trial_type_masks["left_correct"],
        "right_correct": trial_type_masks["right_correct"],
        "left_incorrect": trial_type_masks["left_incorrect"],
        "right_incorrect": trial_type_masks["right_incorrect"],
        "left_omission": trial_type_masks["left_omission"],
        "right_omission": trial_type_masks["right_omission"],
        "left_switch": trial_type_masks["left_switch"],
        "right_switch": trial_type_masks["right_switch"],
        "left_stay": trial_type_masks["left_stay"],
        "right_stay": trial_type_masks["right_stay"],
    }
    for condition_name, trial_mask in conditions_to_plot.items():
        if trial_mask is not None and not np.any(np.asarray(trial_mask, dtype=bool)):
            print(f"Skipping {condition_name}: no valid trials")
            continue

        for alignment in ("trial_start", "choice"):
            lick_peth_data = build_lick_peth_data(
                trial_df=trial_df,
                lick_times=lick_times,
                alignment=alignment,
                rate_bin_size=lick_rate_bin_size,
                pre_time=2.0,
                post_time=1.0,
                show_led_lines=True,
                max_time_after_trial_start=max_time_after_trial_start,
                trial_mask=trial_mask,
            )
            for lick_layout in ("overlay", "separate"):
                fig, _ = plot_lick_peth(
                    lick_peth_data,
                    lick_layout=lick_layout,
                    show_led_lines=True,
                )
                save_path = figure_path / f"{sess_id_full}_{condition_name}_lick_peth_{alignment}_{lick_layout}.png"
                save_figure_with_message(fig, save_path)
                plt.close(fig)

    aligned_spike_times = spike_behavior_pynapple.load_aligned_spikes(spike_path)
    spike_clusters, cluster_info = spike_behavior_pynapple.load_sorter_metadata(
        sorter_output_path=sorter_output_path,
    )
    spike_behavior_pynapple.validate_aligned_spike_inputs(
        aligned_spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
    )
    region_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info=cluster_info,
        region_channels=region_channels,
    )
    region_cluster_info = cluster_info.loc[cluster_info["cluster_id"].isin(region_cluster_ids)].copy()
    unit_cluster_id = select_first_valid_unit_cluster_id(region_cluster_info)
    unit_spike_group = spike_behavior_pynapple.build_spike_tsgroup(
        spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=np.array([unit_cluster_id], dtype=int),
    )
    unit_spikes = unit_spike_group[unit_cluster_id]

    spike_rate_bin_size = 0.1
    for condition_name, trial_mask in conditions_to_plot.items():
        if trial_mask is not None and not np.any(np.asarray(trial_mask, dtype=bool)):
            print(f"Skipping {condition_name} spike plot: no valid trials")
            continue

        for alignment in ("trial_start", "choice"):
            spike_peth_data = build_single_unit_spike_peth_data(
                trial_df=trial_df,
                unit_spikes=unit_spikes,
                unit_cluster_id=unit_cluster_id,
                alignment=alignment,
                rate_bin_size=spike_rate_bin_size,
                pre_time=2.0,
                post_time=1.0,
                show_led_lines=True,
                max_time_after_trial_start=max_time_after_trial_start,
                trial_mask=trial_mask,
            )
            fig, _ = plot_single_unit_spike_peth(
                spike_peth_data,
                show_led_lines=True,
            )
            save_path = (
                figure_path
                / f"{sess_id_full}_{region_name}_unit-{unit_cluster_id}_{condition_name}_spike_peth_{alignment}.png"
            )
            save_figure_with_message(fig, save_path)
            plt.close(fig)


if __name__ == "__main__":
    main()
