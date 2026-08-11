from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.spike_behavior_pynapple import make_trial_type_masks


LEFT_LICK_EVENT = "left_entry"
RIGHT_LICK_EVENT = "right_entry"
LEFT_CHOICE_ACTION = 1
RIGHT_CHOICE_ACTION = 0
LICK_RASTER_STYLES = {
    LEFT_LICK_EVENT: {"label": "Left licks", "color": "tab:orange"},
    RIGHT_LICK_EVENT: {"label": "Right licks", "color": "tab:blue"},
}


def filter_trials_for_unit_plot(
    trial_df: pd.DataFrame,
    condition: str = "all",
    action: int | str | None = "all",
) -> np.ndarray:
    """
    Select trial indices for unit raster/PSTH plotting.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Condition filters use
        columns required by ``make_trial_type_masks``: ``give_reward``,
        ``correct``, ``reward``, and ``action``.
    condition : str, default="all"
        Trial condition. Supported values are ``"all"`` and keys returned by
        ``make_trial_type_masks`` such as ``"correct_rewarded"``, ``"incorrect"``,
        ``"omission"``, ``"switch"``, and ``"stay"``.
    action : int | str | None, default="all"
        Optional action selector. ``"all"`` or ``None`` disables action
        filtering; otherwise values are compared numerically to ``trial_df["action"]``.

    Returns
    -------
    np.ndarray
        Chronologically ordered trial indices with shape ``(n_selected_trials,)``.
    """

    if condition == "all":
        selected_mask = pd.Series(True, index=trial_df.index)
    else:
        trial_masks = make_trial_type_masks(trial_df)
        if condition not in trial_masks:
            raise ValueError(f"Unknown trial condition {condition!r}.")
        selected_mask = trial_masks[condition].copy()

    if action is not None and str(action).lower() != "all":
        action_value = int(action)
        selected_mask = selected_mask & pd.to_numeric(trial_df["action"], errors="coerce").eq(action_value)

    return trial_df.index[np.asarray(selected_mask, dtype=bool)].to_numpy(dtype=int)


def paginate_trial_indices(trial_indices: np.ndarray, page_index: int, page_size: int) -> np.ndarray:
    """
    Return one chronological page of trial indices.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``.
    page_index : int
        Zero-based page number.
    page_size : int
        Number of trials per page.

    Returns
    -------
    np.ndarray
        Trial indices for the requested page.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if page_index < 0:
        raise ValueError("page_index must be nonnegative.")
    start_index = int(page_index) * int(page_size)
    end_index = start_index + int(page_size)
    return np.asarray(trial_indices, dtype=int)[start_index:end_index]


def split_trial_indices_by_action(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split trial indices into left-choice and right-choice arrays.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and an ``action`` column. The project
        convention is ``1=left`` and ``0=right``.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. These trials
        are assumed to have already passed any condition filters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(left_trial_indices, right_trial_indices)``. Both arrays are
        one-dimensional, chronologically ordered, and contain integer trial
        indices. Trials with missing or non-left/right action values are
        excluded from both outputs.
    """

    if "action" not in trial_df.columns:
        raise ValueError("trial_df is missing action column.")
    selected_trials = trial_df.loc[np.asarray(trial_indices, dtype=int)]
    actions = pd.to_numeric(selected_trials["action"], errors="coerce")
    left_mask = actions.eq(LEFT_CHOICE_ACTION).to_numpy(dtype=bool)
    right_mask = actions.eq(RIGHT_CHOICE_ACTION).to_numpy(dtype=bool)
    trial_indices = np.asarray(trial_indices, dtype=int)
    return trial_indices[left_mask], trial_indices[right_mask]


def extract_relative_unit_spikes(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
) -> list[np.ndarray]:
    """
    Extract spike times relative to an event for selected trials.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table. ``alignment_event`` must name a numeric time column in
        seconds.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``.
    alignment_event : str
        Event column used as time zero, typically ``"choice_time"`` or
        ``"start_time"``.
    window : tuple[float, float]
        Inclusive lower and upper plotting bounds in seconds relative to the
        alignment event.

    Returns
    -------
    list[np.ndarray]
        One relative spike-time array per requested trial. Each array is
        one-dimensional and in seconds relative to ``alignment_event``.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

    unit_spike_times = np.asarray(unit_spike_times, dtype=float)
    relative_spikes_by_trial: list[np.ndarray] = []
    for trial_index in np.asarray(trial_indices, dtype=int):
        event_time = pd.to_numeric(pd.Series([trial_df.loc[trial_index, alignment_event]]), errors="coerce").iloc[0]
        if pd.isna(event_time):
            relative_spikes_by_trial.append(np.array([], dtype=float))
            continue
        event_time = float(event_time)
        absolute_start = event_time + float(window[0])
        absolute_end = event_time + float(window[1])
        in_window = (unit_spike_times >= absolute_start) & (unit_spike_times <= absolute_end)
        relative_spikes_by_trial.append(unit_spike_times[in_window] - event_time)
    return relative_spikes_by_trial


def compute_psth_hz(
    relative_spikes_by_trial: list[np.ndarray],
    window: tuple[float, float],
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a peri-event time histogram in firing-rate units.

    Parameters
    ----------
    relative_spikes_by_trial : list[np.ndarray]
        One relative spike-time array per trial. Times are in seconds relative
        to the selected event.
    window : tuple[float, float]
        Histogram bounds in seconds relative to the event.
    bin_size : float
        Histogram bin width in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers, firing_rate_hz)``. Both arrays have shape ``(n_bins,)``.
        Firing rate is normalized by ``n_trials * bin_size``.
    """

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

    bin_edges = np.arange(float(window[0]), float(window[1]) + float(bin_size), float(bin_size))
    if bin_edges[-1] < float(window[1]):
        bin_edges = np.append(bin_edges, float(window[1]))
    all_relative_spikes = (
        np.concatenate(relative_spikes_by_trial)
        if len(relative_spikes_by_trial) > 0 and any(spikes.size for spikes in relative_spikes_by_trial)
        else np.array([], dtype=float)
    )
    spike_counts, _ = np.histogram(all_relative_spikes, bins=bin_edges)
    n_trials = max(len(relative_spikes_by_trial), 1)
    firing_rate_hz = spike_counts.astype(float) / (n_trials * float(bin_size))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    return bin_centers, firing_rate_hz


def _make_full_bin_edges(window: tuple[float, float], bin_size: float) -> np.ndarray:
    """
    Build fixed-width bin edges, trimming any incomplete final bin.

    Parameters
    ----------
    window : tuple[float, float]
        Relative time bounds in seconds. The start is included; the end is used
        only if a full bin fits exactly.
    bin_size : float
        Fixed bin width in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional bin-edge array with shape ``(n_bins + 1,)`` in
        seconds. Raises ``ValueError`` if no full bin fits within ``window``.
    """

    if float(bin_size) <= 0:
        raise ValueError("bin_size must be positive.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

    window_start = float(window[0])
    window_end = float(window[1])
    n_full_bins = int(np.floor((window_end - window_start) / float(bin_size) + 1e-12))
    if n_full_bins < 1:
        raise ValueError("window must contain at least one full bin.")
    return window_start + np.arange(n_full_bins + 1, dtype=float) * float(bin_size)


def compute_binned_firing_rates_hz(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-trial binned firing rates for one unit.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds. ``alignment_event``
        must name one numeric event-time column.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``.
    alignment_event : str
        Event column used as relative time zero, typically ``"choice_time"`` or
        ``"start_time"``.
    window : tuple[float, float]
        Relative time bounds in seconds. Only complete bins of width
        ``bin_size`` are included; an incomplete final bin is trimmed.
    bin_size : float, default=0.1
        Firing-rate bin width in seconds. A value of ``0.1`` gives 100 ms bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers, firing_rates_hz)``. ``bin_centers`` has shape
        ``(n_bins,)`` in seconds. ``firing_rates_hz`` has shape
        ``(n_trials, n_bins)`` in Hz, computed as spike counts divided by
        ``bin_size`` for each trial and bin.
    """

    bin_edges = _make_full_bin_edges(window=window, bin_size=bin_size)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    relative_spikes_by_trial = extract_relative_unit_spikes(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=trial_indices,
        alignment_event=alignment_event,
        window=(float(bin_edges[0]), float(bin_edges[-1])),
    )

    firing_rates_hz = np.zeros((len(relative_spikes_by_trial), bin_centers.size), dtype=float)
    for trial_row_index, relative_spikes in enumerate(relative_spikes_by_trial):
        spike_counts, _ = np.histogram(relative_spikes, bins=bin_edges)
        firing_rates_hz[trial_row_index, :] = spike_counts.astype(float) / float(bin_size)
    return bin_centers, firing_rates_hz


def plot_unit_binned_rate_trial_traces(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size: float,
    unit_id: int,
    title_suffix: str | None = None,
    figure_size: tuple[float, float] = (10.0, 5.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot per-trial binned firing rates with the across-trial mean overlaid.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Every listed
        trial is included in both the faded traces and the mean line.
    alignment_event : str
        Event column used as relative time zero.
    window : tuple[float, float]
        Relative time bounds in seconds. Incomplete final bins are trimmed.
    bin_size : float
        Firing-rate bin width in seconds.
    unit_id : int
        Cluster id shown in the plot title.
    title_suffix : str | None, optional
        Optional suffix appended to the plot title.
    figure_size : tuple[float, float], default=(10.0, 5.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis. X-axis units are seconds relative to
        ``alignment_event`` and y-axis units are Hz.
    """

    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")
    bin_centers, firing_rates_hz = compute_binned_firing_rates_hz(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=trial_indices,
        alignment_event=alignment_event,
        window=window,
        bin_size=bin_size,
    )
    mean_rate_hz = firing_rates_hz.mean(axis=0) if firing_rates_hz.shape[0] > 0 else np.zeros(bin_centers.shape)

    figure, axis = plt.subplots(1, 1, figsize=(float(figure_size[0]), float(figure_size[1])))
    for trial_rates_hz in firing_rates_hz:
        axis.plot(bin_centers, trial_rates_hz, color="tab:blue", alpha=0.18, linewidth=0.8)
    axis.plot(bin_centers, mean_rate_hz, color="black", linewidth=2.0, label="Mean")
    axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    axis.set_xlim(float(window[0]), float(window[1]))
    axis.set_ylabel("Firing rate (Hz)")
    axis.set_xlabel(f"Time from {alignment_event} (s)")
    title = f"Unit {unit_id} trial rates aligned to {alignment_event}"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    axis.set_title(title)
    axis.legend(loc="upper right", fontsize="small")
    figure.tight_layout()
    return figure, axis


def plot_unit_binned_rate_mean_sd(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size: float,
    unit_id: int,
    title_suffix: str | None = None,
    figure_size: tuple[float, float] = (10.0, 5.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot mean binned firing rate with an across-trial standard-deviation band.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Every listed
        trial contributes to the mean and standard deviation.
    alignment_event : str
        Event column used as relative time zero.
    window : tuple[float, float]
        Relative time bounds in seconds. Incomplete final bins are trimmed.
    bin_size : float
        Firing-rate bin width in seconds.
    unit_id : int
        Cluster id shown in the plot title.
    title_suffix : str | None, optional
        Optional suffix appended to the plot title.
    figure_size : tuple[float, float], default=(10.0, 5.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis. X-axis units are seconds relative to
        ``alignment_event`` and y-axis units are Hz.
    """

    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")
    bin_centers, firing_rates_hz = compute_binned_firing_rates_hz(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=trial_indices,
        alignment_event=alignment_event,
        window=window,
        bin_size=bin_size,
    )
    if firing_rates_hz.shape[0] == 0:
        mean_rate_hz = np.zeros(bin_centers.shape)
        sd_rate_hz = np.zeros(bin_centers.shape)
    else:
        mean_rate_hz = firing_rates_hz.mean(axis=0)
        sd_rate_hz = firing_rates_hz.std(axis=0)

    figure, axis = plt.subplots(1, 1, figsize=(float(figure_size[0]), float(figure_size[1])))
    axis.fill_between(
        bin_centers,
        mean_rate_hz - sd_rate_hz,
        mean_rate_hz + sd_rate_hz,
        color="tab:blue",
        alpha=0.22,
        label="Mean +/- SD",
    )
    axis.plot(bin_centers, mean_rate_hz, color="black", linewidth=2.0, label="Mean")
    axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    axis.set_xlim(float(window[0]), float(window[1]))
    axis.set_ylabel("Firing rate (Hz)")
    axis.set_xlabel(f"Time from {alignment_event} (s)")
    title = f"Unit {unit_id} mean +/- SD rate aligned to {alignment_event}"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    axis.set_title(title)
    axis.legend(loc="upper right", fontsize="small")
    figure.tight_layout()
    return figure, axis


def compute_trial_event_offsets(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    event_column: str,
) -> np.ndarray:
    """
    Compute per-trial event times relative to the raster alignment event.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. ``alignment_event`` and
        ``event_column`` must be columns containing event times in seconds.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Values must
        select rows from ``trial_df``.
    alignment_event : str
        Column name for the event used as time zero, in seconds.
    event_column : str
        Column name for the event marker to draw, in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional offset array with shape ``(n_trials,)`` in seconds.
        Values are ``event_column - alignment_event``. Missing or nonnumeric
        event/alignment values are returned as ``np.nan``.
    """

    missing_columns = [column for column in (alignment_event, event_column) if column not in trial_df.columns]
    if missing_columns:
        raise ValueError(f"trial_df is missing event column(s): {missing_columns}.")

    selected_trials = trial_df.loc[np.asarray(trial_indices, dtype=int)]
    alignment_times = pd.to_numeric(selected_trials[alignment_event], errors="coerce").to_numpy(dtype=float)
    event_times = pd.to_numeric(selected_trials[event_column], errors="coerce").to_numpy(dtype=float)
    return event_times - alignment_times


def paginate_unit_ids(unit_ids: np.ndarray, page_index: int, page_size: int) -> np.ndarray:
    """
    Return one page of unit ids while preserving caller-provided order.

    Parameters
    ----------
    unit_ids : np.ndarray
        One-dimensional unit/cluster id array with shape ``(n_units,)``.
    page_index : int
        Zero-based page number.
    page_size : int
        Number of units per page.

    Returns
    -------
    np.ndarray
        Unit ids for the requested page with shape ``(n_page_units,)``.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if page_index < 0:
        raise ValueError("page_index must be nonnegative.")
    start_index = int(page_index) * int(page_size)
    end_index = start_index + int(page_size)
    return np.asarray(unit_ids, dtype=int)[start_index:end_index]


def get_trial_alignment_time(trial_df: pd.DataFrame, trial_index: int, alignment_event: str) -> float:
    """
    Return the absolute alignment time for one trial.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. ``alignment_event`` must be a
        numeric event-time column in seconds.
    trial_index : int
        Trial row index to plot.
    alignment_event : str
        Event column used as relative time zero, typically ``"start_time"`` or
        ``"choice_time"``.

    Returns
    -------
    float
        Absolute alignment time in seconds.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    if int(trial_index) not in trial_df.index:
        raise ValueError(f"trial_index {trial_index} is not present in trial_df.")
    event_time = pd.to_numeric(pd.Series([trial_df.loc[int(trial_index), alignment_event]]), errors="coerce").iloc[0]
    if pd.isna(event_time):
        raise ValueError(f"Trial {trial_index} has missing alignment time for {alignment_event!r}.")
    return float(event_time)


def extract_relative_events_for_trial(
    event_times: nap.Ts | np.ndarray,
    reference_time: float,
    window: tuple[float, float],
) -> np.ndarray:
    """
    Extract event times relative to one trial alignment time.

    Parameters
    ----------
    event_times : nap.Ts | np.ndarray
        One-dimensional event times in seconds.
    reference_time : float
        Absolute alignment time in seconds.
    window : tuple[float, float]
        Inclusive lower and upper bounds in seconds relative to
        ``reference_time``.

    Returns
    -------
    np.ndarray
        Relative event times within ``window`` with shape ``(n_events,)`` in
        seconds.
    """

    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if isinstance(event_times, nap.Ts):
        absolute_event_times = np.asarray(event_times.index.to_numpy(), dtype=float)
    else:
        absolute_event_times = np.asarray(event_times, dtype=float).reshape(-1)
    relative_event_times = absolute_event_times - float(reference_time)
    in_window = (relative_event_times >= float(window[0])) & (relative_event_times <= float(window[1]))
    return np.sort(relative_event_times[in_window])


def extract_relative_spikes_for_units(
    spike_group: nap.TsGroup,
    unit_ids: np.ndarray,
    reference_time: float,
    window: tuple[float, float],
) -> dict[int, np.ndarray]:
    """
    Extract one trial's relative spikes for visible units.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Times are in seconds.
    unit_ids : np.ndarray
        One-dimensional visible unit ids with shape ``(n_units,)``.
    reference_time : float
        Absolute alignment time in seconds.
    window : tuple[float, float]
        Inclusive lower and upper bounds in seconds relative to
        ``reference_time``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from cluster id to relative spike times in seconds.
    """

    if not isinstance(spike_group, nap.TsGroup):
        raise TypeError("spike_group must be a pynapple TsGroup.")
    relative_spikes_by_unit: dict[int, np.ndarray] = {}
    for unit_id in np.asarray(unit_ids, dtype=int):
        if int(unit_id) not in spike_group:
            raise ValueError(f"Unit {unit_id} is not present in the spike group.")
        relative_spikes_by_unit[int(unit_id)] = extract_relative_events_for_trial(
            event_times=spike_group[int(unit_id)],
            reference_time=reference_time,
            window=window,
        )
    return relative_spikes_by_unit


def compute_population_psth_hz(
    relative_spikes_by_unit: dict[int, np.ndarray],
    window: tuple[float, float],
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a one-trial population PSTH normalized per unit.

    Parameters
    ----------
    relative_spikes_by_unit : dict[int, np.ndarray]
        Mapping from cluster id to one-dimensional relative spike-time arrays.
        Spike times are in seconds relative to the selected trial alignment.
    window : tuple[float, float]
        Histogram bounds in seconds relative to the alignment event.
    bin_size : float
        Histogram bin width in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers, population_rate_hz_per_unit)``. Both arrays have shape
        ``(n_bins,)``. Rate is normalized by ``n_units * bin_size``.
    """

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

    bin_edges = np.arange(float(window[0]), float(window[1]) + float(bin_size), float(bin_size))
    if bin_edges[-1] < float(window[1]):
        bin_edges = np.append(bin_edges, float(window[1]))
    all_relative_spikes = (
        np.concatenate(list(relative_spikes_by_unit.values()))
        if relative_spikes_by_unit and any(spikes.size for spikes in relative_spikes_by_unit.values())
        else np.array([], dtype=float)
    )
    spike_counts, _ = np.histogram(all_relative_spikes, bins=bin_edges)
    n_units = max(len(relative_spikes_by_unit), 1)
    population_rate_hz = spike_counts.astype(float) / (n_units * float(bin_size))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    return bin_centers, population_rate_hz


def _draw_trial_event_markers(
    raster_axis: plt.Axes,
    trial_event_offsets: np.ndarray,
    y_positions: np.ndarray,
    color: str,
    label: str,
    row_spacing: float,
) -> None:
    """
    Draw short vertical event markers on individual raster rows.

    Parameters
    ----------
    raster_axis : plt.Axes
        Matplotlib axis containing the raster plot.
    trial_event_offsets : np.ndarray
        One-dimensional event offsets with shape ``(n_trials,)`` in seconds.
    y_positions : np.ndarray
        One-dimensional y positions with shape ``(n_trials,)`` in arbitrary
        raster-row units.
    color : str
        Matplotlib color for this event type.
    label : str
        Legend label for this event type.
    row_spacing : float
        Vertical spacing between neighboring raster rows, in raster-row units.

    Returns
    -------
    None
        The raster axis is modified in place.
    """

    half_marker_length = min(0.35 * float(row_spacing), 0.45)
    label_has_been_drawn = False
    for event_offset, y_position in zip(trial_event_offsets, y_positions, strict=False):
        if np.isnan(event_offset):
            continue
        marker_label = label if not label_has_been_drawn else None
        raster_axis.vlines(
            event_offset,
            y_position - half_marker_length,
            y_position + half_marker_length,
            color=color,
            linewidth=1.0,
            label=marker_label,
        )
        label_has_been_drawn = True


def plot_trial_behavior_and_spike_raster(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts],
    spike_group: nap.TsGroup,
    raster_unit_ids: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    psth_unit_ids: np.ndarray | None = None,
    psth_bin_size: float | None = 0.05,
    lfp_time_s: np.ndarray | None = None,
    lfp_uv: np.ndarray | None = None,
    lfp_label: str | None = None,
    lfp_y_label: str = "LFP (uV)",
    figure_size: tuple[float, float] = (12.0, 10.0),
    spike_row_spacing: float = 1.0,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """
    Plot one trial's left/right licks, choice, LED, and visible unit spikes.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``start_time``, ``choice_time``, ``led_on_time``, and ``action``.
        Times are in seconds. The project action convention is ``0=right``
        and ``1=left``.
    trial_index : int
        Trial row index to plot.
    lick_times : Mapping[str, nap.Ts]
        Mapping with ``"left_entry"`` and ``"right_entry"`` keys. Each value
        is a one-dimensional Pynapple ``Ts`` of lick timestamps in seconds.
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike times are in
        seconds.
    raster_unit_ids : np.ndarray
        One-dimensional visible unit ids with shape ``(n_raster_units,)``. The
        order controls spike raster row order.
    alignment_event : str
        Event column used as relative time zero, typically ``"start_time"`` or
        ``"choice_time"``.
    window : tuple[float, float]
        Inclusive lower and upper x-axis bounds in seconds relative to
        ``alignment_event``.
    psth_unit_ids : np.ndarray | None, optional
        Unit ids included in the population PSTH. If ``None``, the PSTH uses
        ``raster_unit_ids``. This can differ from visible raster units to
        summarize all currently selected units.
    psth_bin_size : float | None, default=0.05
        Population PSTH bin width in seconds. ``None`` disables the PSTH and
        returns a raster-only figure.
    lfp_time_s : np.ndarray | None, optional
        One-dimensional LFP time axis in seconds relative to alignment, shape
        ``(n_lfp_samples,)``. If provided, ``lfp_uv`` must also be provided.
    lfp_uv : np.ndarray | None, optional
        One-dimensional gain-corrected LFP trace in microvolts, shape
        ``(n_lfp_samples,)``.
    lfp_label : str | None, optional
        Label shown on the LFP axis title, such as the saved channel index.
    lfp_y_label : str, default="LFP (uV)"
        Y-axis label for the LFP trace. Use neutral labels such as ``"LFP"``
        when the loaded values are not guaranteed to be microvolts.
    figure_size : tuple[float, float], default=(12.0, 10.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.
    spike_row_spacing : float, default=1.0
        Vertical spacing between neighboring spike rows, in arbitrary raster
        row units. Behavior rows are placed above the paginated spike rows.

    Returns
    -------
    tuple[plt.Figure, plt.Axes | np.ndarray]
        Matplotlib figure and axes. Raster-only mode without LFP returns one
        ``Axes``. Multi-axis modes return axes in top-to-bottom plotting order:
        optional LFP, raster, optional population PSTH. X-axis units are
        seconds relative to ``alignment_event``.
    """

    required_columns = {"start_time", "choice_time", "led_on_time", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")
    if float(spike_row_spacing) <= 0:
        raise ValueError("spike_row_spacing must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    visible_unit_ids = np.asarray(raster_unit_ids, dtype=int).reshape(-1)
    population_unit_ids = (
        visible_unit_ids
        if psth_unit_ids is None
        else np.asarray(psth_unit_ids, dtype=int).reshape(-1)
    )
    if psth_bin_size is not None and float(psth_bin_size) <= 0:
        raise ValueError("psth_bin_size must be positive when provided.")
    show_lfp = lfp_time_s is not None or lfp_uv is not None
    if show_lfp:
        if lfp_time_s is None or lfp_uv is None:
            raise ValueError("lfp_time_s and lfp_uv must be provided together.")
        lfp_time_s = np.asarray(lfp_time_s, dtype=float).reshape(-1)
        lfp_uv = np.asarray(lfp_uv, dtype=float).reshape(-1)
        if lfp_time_s.shape != lfp_uv.shape:
            raise ValueError("lfp_time_s and lfp_uv must have the same one-dimensional shape.")
    reference_time = get_trial_alignment_time(
        trial_df=trial_df,
        trial_index=int(trial_index),
        alignment_event=alignment_event,
    )
    left_licks = extract_relative_events_for_trial(
        event_times=lick_times[LEFT_LICK_EVENT],
        reference_time=reference_time,
        window=window,
    )
    right_licks = extract_relative_events_for_trial(
        event_times=lick_times[RIGHT_LICK_EVENT],
        reference_time=reference_time,
        window=window,
    )
    relative_raster_spikes_by_unit = extract_relative_spikes_for_units(
        spike_group=spike_group,
        unit_ids=visible_unit_ids,
        reference_time=reference_time,
        window=window,
    )
    relative_psth_spikes_by_unit = (
        relative_raster_spikes_by_unit
        if np.array_equal(population_unit_ids, visible_unit_ids)
        else extract_relative_spikes_for_units(
            spike_group=spike_group,
            unit_ids=population_unit_ids,
            reference_time=reference_time,
            window=window,
        )
    )

    spike_y_positions = np.arange(visible_unit_ids.size, dtype=float) * float(spike_row_spacing)
    behavior_gap = float(spike_row_spacing)
    right_lick_y = (spike_y_positions[-1] if spike_y_positions.size else 0.0) + behavior_gap + float(spike_row_spacing)
    left_lick_y = right_lick_y + float(spike_row_spacing)

    show_psth = psth_bin_size is not None
    n_axes = 1 + int(show_lfp) + int(show_psth)
    if n_axes == 1:
        figure, axis = plt.subplots(
            1,
            1,
            figsize=(float(figure_size[0]), float(figure_size[1])),
        )
        lfp_axis = None
        raster_axis = axis
        psth_axis = None
    else:
        height_ratios = []
        if show_lfp:
            height_ratios.append(1)
        height_ratios.append(4)
        if show_psth:
            height_ratios.append(1)
        figure, axes = plt.subplots(
            n_axes,
            1,
            sharex=True,
            figsize=(float(figure_size[0]), float(figure_size[1])),
            height_ratios=height_ratios,
        )
        axes = np.asarray(axes, dtype=object).reshape(-1)
        axis_index = 0
        lfp_axis = axes[axis_index] if show_lfp else None
        if show_lfp:
            axis_index += 1
        raster_axis = axes[axis_index]
        axis_index += 1
        psth_axis = axes[axis_index] if show_psth else None

    if lfp_axis is not None:
        lfp_axis.plot(lfp_time_s, lfp_uv, color="black", linewidth=0.8)
        lfp_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
        lfp_axis.set_ylabel(str(lfp_y_label))
        lfp_axis.set_xlim(float(window[0]), float(window[1]))
        lfp_axis.set_title(lfp_label or "LFP")

    for row_index, unit_id in enumerate(visible_unit_ids):
        unit_spikes = relative_raster_spikes_by_unit[int(unit_id)]
        if unit_spikes.size == 0:
            continue
        raster_axis.scatter(
            unit_spikes,
            np.full(unit_spikes.shape, spike_y_positions[row_index], dtype=float),
            marker="|",
            color="black",
            s=80,
        )

    raster_axis.eventplot(
        [right_licks],
        orientation="horizontal",
        lineoffsets=[right_lick_y],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["label"],
    )
    raster_axis.eventplot(
        [left_licks],
        orientation="horizontal",
        lineoffsets=[left_lick_y],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["label"],
    )

    trial_row = trial_df.loc[int(trial_index)]
    choice_time = pd.to_numeric(pd.Series([trial_row["choice_time"]]), errors="coerce").iloc[0]
    if not pd.isna(choice_time):
        choice_offset = float(choice_time) - reference_time
        action_value = pd.to_numeric(pd.Series([trial_row["action"]]), errors="coerce").iloc[0]
        if not pd.isna(action_value):
            choice_y = left_lick_y if int(action_value) == LEFT_CHOICE_ACTION else right_lick_y
            raster_axis.vlines(
                choice_offset,
                choice_y - 0.38,
                choice_y + 0.38,
                colors="tab:purple",
                linewidth=1.4,
                label="Choice",
            )

    led_time = pd.to_numeric(pd.Series([trial_row["led_on_time"]]), errors="coerce").iloc[0]
    if not pd.isna(led_time):
        raster_axis.vlines(
            float(led_time) - reference_time,
            right_lick_y - 0.4,
            left_lick_y + 0.4,
            colors="tab:green",
            linewidth=1.1,
            label="LED",
        )

    raster_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, label=alignment_event)
    y_ticks = [*spike_y_positions.tolist(), right_lick_y, left_lick_y]
    y_tick_labels = [f"Unit {int(unit_id)}" for unit_id in visible_unit_ids]
    y_tick_labels.extend(["Right licks", "Left licks"])
    raster_axis.set_yticks(y_ticks)
    raster_axis.set_yticklabels(y_tick_labels)
    raster_axis.set_xlim(float(window[0]), float(window[1]))
    y_padding = max(0.5 * float(spike_row_spacing), 0.5)
    raster_axis.set_ylim(-y_padding, left_lick_y + y_padding)
    raster_axis.set_ylabel("Event / unit")
    raster_axis.set_title(f"Trial {trial_index} behavior and spikes aligned to {alignment_event}")
    raster_axis.legend(loc="upper right", fontsize="small")
    if psth_axis is None:
        raster_axis.set_xlabel(f"Time from {alignment_event} (s)")
    else:
        bin_centers, population_rate_hz = compute_population_psth_hz(
            relative_spikes_by_unit=relative_psth_spikes_by_unit,
            window=window,
            bin_size=float(psth_bin_size),
        )
        if bin_centers.size > 0:
            psth_axis.bar(
                bin_centers,
                population_rate_hz,
                width=float(psth_bin_size),
                align="center",
                color="0.35",
                edgecolor="0.35",
            )
        psth_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
        psth_axis.set_ylabel("Population rate (Hz/unit)")
        psth_axis.set_xlabel(f"Time from {alignment_event} (s)")
        psth_axis.set_xlim(float(window[0]), float(window[1]))
    figure.tight_layout()
    return (figure, axis) if n_axes == 1 else (figure, axes)


def plot_trial_behavior_and_population_pca(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts],
    pca_time_s: np.ndarray,
    pca_scores: np.ndarray,
    trial_position: int,
    alignment_event: str,
    window: tuple[float, float],
    pc_count: int = 5,
    lfp_time_s: np.ndarray | None = None,
    lfp_uv: np.ndarray | None = None,
    lfp_label: str | None = None,
    lfp_y_label: str = "LFP (uV)",
    figure_size: tuple[float, float] = (12.0, 10.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one trial's behavior events and population PCA time courses.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``start_time``, ``choice_time``, ``led_on_time``, and ``action``.
        Times are in seconds.
    trial_index : int
        Trial row index used to extract behavior events from ``trial_df``.
    lick_times : Mapping[str, nap.Ts]
        Mapping with ``"left_entry"`` and ``"right_entry"`` keys. Each value
        is a one-dimensional Pynapple ``Ts`` of lick timestamps in seconds.
    pca_time_s : np.ndarray
        One-dimensional PCA bin centers with shape ``(n_bins,)`` in seconds
        relative to ``alignment_event``.
    pca_scores : np.ndarray
        PCA score tensor with shape ``(n_trials, n_bins, n_components)``.
        ``trial_position`` selects the first axis.
    trial_position : int
        Zero-based position of ``trial_index`` within the PCA fit trial set.
    alignment_event : str
        Event column used as relative time zero, typically ``"start_time"`` or
        ``"choice_time"``.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    pc_count : int, default=5
        Number of leading PCs to plot. The plotted count is capped by the
        number of fitted components.
    lfp_time_s : np.ndarray | None, optional
        Optional LFP time axis with shape ``(n_lfp_samples,)`` in seconds
        relative to alignment. If provided, ``lfp_uv`` must also be provided.
    lfp_uv : np.ndarray | None, optional
        Optional LFP trace with shape ``(n_lfp_samples,)``.
    lfp_label : str | None, optional
        Title label for the optional LFP axis.
    lfp_y_label : str, default="LFP (uV)"
        Y-axis label for the optional LFP trace.
    figure_size : tuple[float, float], default=(12.0, 10.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes in top-to-bottom order: optional LFP,
        behavior events, and PCA traces. X-axis units are seconds relative to
        ``alignment_event``.
    """

    required_columns = {"start_time", "choice_time", "led_on_time", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if int(pc_count) < 1:
        raise ValueError("pc_count must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    pca_time_s = np.asarray(pca_time_s, dtype=float).reshape(-1)
    pca_scores = np.asarray(pca_scores, dtype=float)
    if pca_scores.ndim != 3:
        raise ValueError("pca_scores must have shape (n_trials, n_bins, n_components).")
    if int(trial_position) < 0 or int(trial_position) >= pca_scores.shape[0]:
        raise ValueError("trial_position is outside the PCA score trial axis.")
    if pca_time_s.shape[0] != pca_scores.shape[1]:
        raise ValueError("pca_time_s length must match the PCA score time axis.")

    show_lfp = lfp_time_s is not None or lfp_uv is not None
    if show_lfp:
        if lfp_time_s is None or lfp_uv is None:
            raise ValueError("lfp_time_s and lfp_uv must be provided together.")
        lfp_time_s = np.asarray(lfp_time_s, dtype=float).reshape(-1)
        lfp_uv = np.asarray(lfp_uv, dtype=float).reshape(-1)
        if lfp_time_s.shape != lfp_uv.shape:
            raise ValueError("lfp_time_s and lfp_uv must have the same one-dimensional shape.")

    reference_time = get_trial_alignment_time(
        trial_df=trial_df,
        trial_index=int(trial_index),
        alignment_event=alignment_event,
    )
    left_licks = extract_relative_events_for_trial(
        event_times=lick_times[LEFT_LICK_EVENT],
        reference_time=reference_time,
        window=window,
    )
    right_licks = extract_relative_events_for_trial(
        event_times=lick_times[RIGHT_LICK_EVENT],
        reference_time=reference_time,
        window=window,
    )

    n_axes = 3 if show_lfp else 2
    height_ratios = [1, 1, 3] if show_lfp else [1, 3]
    figure, axes = plt.subplots(
        n_axes,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=height_ratios,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    axis_index = 0
    if show_lfp:
        lfp_axis = axes[axis_index]
        axis_index += 1
        lfp_axis.plot(lfp_time_s, lfp_uv, color="black", linewidth=0.8)
        lfp_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
        lfp_axis.set_ylabel(str(lfp_y_label))
        lfp_axis.set_xlim(float(window[0]), float(window[1]))
        lfp_axis.set_title(lfp_label or "LFP")

    behavior_axis = axes[axis_index]
    pca_axis = axes[axis_index + 1]

    behavior_axis.eventplot(
        [right_licks],
        orientation="horizontal",
        lineoffsets=[0.0],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["label"],
    )
    behavior_axis.eventplot(
        [left_licks],
        orientation="horizontal",
        lineoffsets=[1.0],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["label"],
    )
    trial_row = trial_df.loc[int(trial_index)]
    choice_time = pd.to_numeric(pd.Series([trial_row["choice_time"]]), errors="coerce").iloc[0]
    if not pd.isna(choice_time):
        choice_offset = float(choice_time) - reference_time
        action_value = pd.to_numeric(pd.Series([trial_row["action"]]), errors="coerce").iloc[0]
        choice_y = 0.5
        if not pd.isna(action_value):
            choice_y = 1.0 if int(action_value) == LEFT_CHOICE_ACTION else 0.0
        behavior_axis.vlines(
            choice_offset,
            choice_y - 0.35,
            choice_y + 0.35,
            colors="tab:purple",
            linewidth=1.4,
            label="Choice",
        )
    led_time = pd.to_numeric(pd.Series([trial_row["led_on_time"]]), errors="coerce").iloc[0]
    if not pd.isna(led_time):
        behavior_axis.vlines(
            float(led_time) - reference_time,
            -0.35,
            1.35,
            colors="tab:green",
            linewidth=1.1,
            label="LED",
        )
    behavior_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, label=alignment_event)
    behavior_axis.set_yticks([0.0, 1.0])
    behavior_axis.set_yticklabels(["Right licks", "Left licks"])
    behavior_axis.set_ylim(-0.7, 1.7)
    behavior_axis.set_ylabel("Behavior")
    behavior_axis.set_xlim(float(window[0]), float(window[1]))
    behavior_axis.set_title(f"Trial {trial_index} behavior aligned to {alignment_event}")
    behavior_axis.legend(loc="upper right", fontsize="small")

    plotted_pc_count = min(int(pc_count), pca_scores.shape[2])
    for pc_index in range(plotted_pc_count):
        pca_axis.plot(
            pca_time_s,
            pca_scores[int(trial_position), :, pc_index],
            linewidth=1.2,
            label=f"PC{pc_index + 1}",
        )
    pca_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
    pca_axis.set_xlim(float(window[0]), float(window[1]))
    pca_axis.set_ylabel("PC score")
    pca_axis.set_xlabel(f"Time from {alignment_event} (s)")
    pca_axis.set_title(f"Population PCA trajectory, first {plotted_pc_count} PCs")
    pca_axis.legend(loc="upper right", fontsize="small", ncol=min(plotted_pc_count, 5))

    figure.tight_layout()
    return figure, axes


def plot_pca_cumulative_explained_variance(
    cumulative_explained_variance: np.ndarray,
    pc_count: int | None = None,
    figure_size: tuple[float, float] = (5.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative explained variance against the number of PCs.

    Parameters
    ----------
    cumulative_explained_variance : np.ndarray
        One-dimensional cumulative explained variance ratio with shape
        ``(n_components,)``. Values are fractions between 0 and 1.
    pc_count : int | None, optional
        Maximum number of leading PCs to display. ``None`` displays all
        supplied components. The displayed count is capped by available
        components.
    figure_size : tuple[float, float], default=(5.0, 3.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis. X-axis is one-based PC count; y-axis is
        cumulative explained variance ratio.
    """

    cumulative_explained_variance = np.asarray(cumulative_explained_variance, dtype=float).reshape(-1)
    if cumulative_explained_variance.size == 0:
        raise ValueError("cumulative_explained_variance must contain at least one value.")
    if not np.isfinite(cumulative_explained_variance).all():
        raise ValueError("cumulative_explained_variance must contain only finite values.")
    if pc_count is not None and int(pc_count) < 1:
        raise ValueError("pc_count must be positive when provided.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    displayed_pc_count = cumulative_explained_variance.size
    if pc_count is not None:
        displayed_pc_count = min(displayed_pc_count, int(pc_count))
    cumulative_explained_variance = cumulative_explained_variance[:displayed_pc_count]
    pc_numbers = np.arange(1, cumulative_explained_variance.size + 1, dtype=int)
    figure, axis = plt.subplots(1, 1, figsize=(float(figure_size[0]), float(figure_size[1])))
    axis.plot(pc_numbers, cumulative_explained_variance, marker="o", color="black", linewidth=1.2)
    axis.set_xlabel("Number of PCs")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0.0, min(1.05, max(1.0, float(cumulative_explained_variance.max()) * 1.05)))
    if displayed_pc_count <= 20:
        axis.set_xticks(pc_numbers)
    else:
        tick_interval = max(1, int(np.ceil(displayed_pc_count / 10)))
        sparse_ticks = np.unique(
            np.concatenate(
                [
                    np.array([1], dtype=int),
                    np.arange(tick_interval, displayed_pc_count + 1, tick_interval, dtype=int),
                    np.array([displayed_pc_count], dtype=int),
                ]
            )
        )
        axis.set_xticks(sparse_ticks)
    figure.tight_layout()
    return figure, axis


def _validate_unit_summary_plot_type(summary_plot_type: str) -> None:
    """
    Validate a single-unit lower-panel summary plot type.

    Parameters
    ----------
    summary_plot_type : str
        Lower-panel summary mode. Supported values are ``"psth"``,
        ``"binned-rate-trials-mean"``, and ``"binned-rate-mean-sd"``.

    Returns
    -------
    None
        Raises ``ValueError`` if ``summary_plot_type`` is unsupported.
    """

    valid_summary_plot_types = {"psth", "binned-rate-trials-mean", "binned-rate-mean-sd"}
    if summary_plot_type not in valid_summary_plot_types:
        raise ValueError(f"summary_plot_type must be one of {sorted(valid_summary_plot_types)}.")


def _draw_unit_spike_raster(
    raster_axis: plt.Axes,
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    raster_trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    title: str,
    event_marker_columns: tuple[str, ...] | None = None,
    event_marker_styles: dict[str, dict[str, str]] | None = None,
    raster_row_spacing: float = 1.0,
) -> None:
    """
    Draw one single-unit raw spike raster on an existing axis.

    Parameters
    ----------
    raster_axis : plt.Axes
        Matplotlib axis to modify in place.
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    raster_trial_indices : np.ndarray
        Trial indices displayed as raster rows, shape ``(n_raster_trials,)``.
    alignment_event : str
        Event column used as time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    title : str
        Axis title.
    event_marker_columns : tuple[str, ...] | None, optional
        Event-time columns to draw as per-trial raster markers.
    event_marker_styles : dict[str, dict[str, str]] | None, optional
        Optional style map keyed by event column. Each style may contain
        ``"label"`` and ``"color"`` strings.
    raster_row_spacing : float, default=1.0
        Vertical spacing between neighboring raster rows in arbitrary raster
        row units.

    Returns
    -------
    None
        ``raster_axis`` is modified in place. X-axis units are seconds relative
        to ``alignment_event``.
    """

    if float(raster_row_spacing) <= 0:
        raise ValueError("raster_row_spacing must be positive.")
    raster_spikes = extract_relative_unit_spikes(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=raster_trial_indices,
        alignment_event=alignment_event,
        window=window,
    )
    y_positions = np.arange(len(raster_spikes), dtype=float) * float(raster_row_spacing)
    for row_index, relative_spikes in enumerate(raster_spikes):
        if relative_spikes.size == 0:
            continue
        raster_axis.scatter(
            relative_spikes,
            np.full(relative_spikes.shape, y_positions[row_index], dtype=float),
            marker="|",
            color="black",
            s=80,
        )
    marker_styles = event_marker_styles or {}
    for event_column in event_marker_columns or ():
        event_offsets = compute_trial_event_offsets(
            trial_df=trial_df,
            trial_indices=raster_trial_indices,
            alignment_event=alignment_event,
            event_column=event_column,
        )
        event_style = marker_styles.get(event_column, {})
        _draw_trial_event_markers(
            raster_axis=raster_axis,
            trial_event_offsets=event_offsets,
            y_positions=y_positions,
            color=event_style.get("color", "0.4"),
            label=event_style.get("label", event_column),
            row_spacing=float(raster_row_spacing),
        )
    raster_axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    raster_axis.set_ylabel("Trial")
    y_padding = max(0.5 * float(raster_row_spacing), 0.5)
    max_y_position = y_positions[-1] if y_positions.size else 0.0
    raster_axis.set_ylim(-y_padding, max(max_y_position + y_padding, y_padding))
    raster_axis.set_xlim(float(window[0]), float(window[1]))
    raster_axis.set_title(title)
    if event_marker_columns:
        raster_axis.legend(loc="upper right", fontsize="small")


def _draw_unit_summary_panel(
    summary_axis: plt.Axes,
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    summary_trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    psth_bin_size: float,
    summary_plot_type: str,
    binned_rate_bin_size: float = 0.1,
) -> None:
    """
    Draw a single-unit firing-rate summary on an existing axis.

    Parameters
    ----------
    summary_axis : plt.Axes
        Matplotlib axis to modify in place.
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    summary_trial_indices : np.ndarray
        Trial indices included in the summary, shape ``(n_summary_trials,)``.
    alignment_event : str
        Event column used as time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    psth_bin_size : float
        PSTH bin width in seconds. Used only when ``summary_plot_type`` is
        ``"psth"``.
    summary_plot_type : str
        Lower-panel summary mode. Supported values are ``"psth"``,
        ``"binned-rate-trials-mean"``, and ``"binned-rate-mean-sd"``.
    binned_rate_bin_size : float, default=0.1
        Bin width in seconds for binned-rate summary modes.

    Returns
    -------
    None
        ``summary_axis`` is modified in place. X-axis units are seconds
        relative to ``alignment_event`` and y-axis units are Hz.
    """

    _validate_unit_summary_plot_type(summary_plot_type)
    if summary_plot_type == "psth":
        psth_spikes = extract_relative_unit_spikes(
            unit_spike_times=unit_spike_times,
            trial_df=trial_df,
            trial_indices=summary_trial_indices,
            alignment_event=alignment_event,
            window=window,
        )
        bin_centers, firing_rate_hz = compute_psth_hz(
            psth_spikes,
            window=window,
            bin_size=psth_bin_size,
        )
        if bin_centers.size > 0:
            summary_axis.bar(
                bin_centers,
                firing_rate_hz,
                width=float(psth_bin_size),
                align="center",
                color="0.35",
                edgecolor="0.35",
            )
    else:
        bin_centers, firing_rates_hz = compute_binned_firing_rates_hz(
            unit_spike_times=unit_spike_times,
            trial_df=trial_df,
            trial_indices=summary_trial_indices,
            alignment_event=alignment_event,
            window=window,
            bin_size=float(binned_rate_bin_size),
        )
        if firing_rates_hz.shape[0] == 0:
            mean_rate_hz = np.zeros(bin_centers.shape)
            sd_rate_hz = np.zeros(bin_centers.shape)
        else:
            mean_rate_hz = firing_rates_hz.mean(axis=0)
            sd_rate_hz = firing_rates_hz.std(axis=0)
        if summary_plot_type == "binned-rate-trials-mean":
            for trial_rates_hz in firing_rates_hz:
                summary_axis.plot(bin_centers, trial_rates_hz, color="tab:blue", alpha=0.18, linewidth=0.8)
            summary_axis.plot(bin_centers, mean_rate_hz, color="black", linewidth=2.0, label="Mean")
            summary_axis.legend(loc="upper right", fontsize="small")
        else:
            summary_axis.fill_between(
                bin_centers,
                mean_rate_hz - sd_rate_hz,
                mean_rate_hz + sd_rate_hz,
                color="tab:blue",
                alpha=0.22,
                label="Mean +/- SD",
            )
            summary_axis.plot(bin_centers, mean_rate_hz, color="black", linewidth=2.0, label="Mean")
            summary_axis.legend(loc="upper right", fontsize="small")
    summary_axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    summary_axis.set_xlim(float(window[0]), float(window[1]))
    summary_axis.set_ylabel("Firing rate (Hz)")
    summary_axis.set_xlabel(f"Time from {alignment_event} (s)")


def plot_unit_raster_and_psth(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    raster_trial_indices: np.ndarray,
    psth_trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size: float,
    unit_id: int,
    title_suffix: str | None = None,
    event_marker_columns: tuple[str, ...] | None = None,
    event_marker_styles: dict[str, dict[str, str]] | None = None,
    raster_row_spacing: float = 1.0,
    figure_size: tuple[float, float] = (10.0, 7.0),
    summary_plot_type: str = "psth",
    binned_rate_bin_size: float = 0.1,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one unit's page-level raster with a selectable lower summary panel.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    raster_trial_indices : np.ndarray
        Trial indices for the displayed raster page, shape ``(n_page_trials,)``.
    psth_trial_indices : np.ndarray
        Trial indices used for the lower summary panel, shape
        ``(n_filtered_trials,)``. The raster page can differ from these summary
        trials.
    alignment_event : str
        Event column used as time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    bin_size : float
        PSTH bin width in seconds. Used only when ``summary_plot_type`` is
        ``"psth"``.
    unit_id : int
        Cluster id shown in the plot title.
    title_suffix : str | None, optional
        Optional suffix appended to the title.
    event_marker_columns : tuple[str, ...] | None, optional
        Event-time columns to draw as per-trial raster markers. Times are in
        seconds and are plotted relative to ``alignment_event``.
    event_marker_styles : dict[str, dict[str, str]] | None, optional
        Optional style map keyed by event column. Each style may contain
        ``"label"`` and ``"color"`` strings.
    raster_row_spacing : float, default=1.0
        Vertical spacing between raster rows, in arbitrary raster-row units.
    figure_size : tuple[float, float], default=(10.0, 7.0)
        Matplotlib figure size as ``(width_inches, height_inches)``. This
        controls the rendered physical size of the combined raster/PSTH plot
        without changing any time units or trial indexing.
    summary_plot_type : str, default="psth"
        Lower-panel summary type. Supported values are ``"psth"``,
        ``"binned-rate-trials-mean"``, and ``"binned-rate-mean-sd"``.
    binned_rate_bin_size : float, default=0.1
        Bin width in seconds for the two binned-rate summary modes. A value of
        ``0.1`` gives 100 ms bins. Incomplete final bins are trimmed.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes array with shape ``(2,)``. Axis 0 is the
        raw spike raster and axis 1 is the selected firing-rate summary.
    """

    if float(raster_row_spacing) <= 0:
        raise ValueError("raster_row_spacing must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")
    _validate_unit_summary_plot_type(summary_plot_type)

    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[3, 1],
    )
    raster_axis, psth_axis = axes
    title = f"Unit {unit_id} aligned to {alignment_event}"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    _draw_unit_spike_raster(
        raster_axis=raster_axis,
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        raster_trial_indices=raster_trial_indices,
        alignment_event=alignment_event,
        window=window,
        title=title,
        event_marker_columns=event_marker_columns,
        event_marker_styles=event_marker_styles,
        raster_row_spacing=float(raster_row_spacing),
    )
    _draw_unit_summary_panel(
        summary_axis=psth_axis,
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        summary_trial_indices=psth_trial_indices,
        alignment_event=alignment_event,
        window=window,
        psth_bin_size=float(bin_size),
        summary_plot_type=summary_plot_type,
        binned_rate_bin_size=float(binned_rate_bin_size),
    )
    figure.tight_layout()
    return figure, axes


def plot_unit_left_right_choice_comparison(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    left_raster_trial_indices: np.ndarray,
    right_raster_trial_indices: np.ndarray,
    left_summary_trial_indices: np.ndarray,
    right_summary_trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size: float,
    unit_id: int,
    summary_plot_type: str = "psth",
    binned_rate_bin_size: float = 0.1,
    title_suffix: str | None = None,
    event_marker_columns: tuple[str, ...] | None = None,
    event_marker_styles: dict[str, dict[str, str]] | None = None,
    raster_row_spacing: float = 1.0,
    figure_size: tuple[float, float] = (14.0, 7.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one unit as side-by-side left-choice and right-choice raster summaries.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    left_raster_trial_indices : np.ndarray
        Left-choice trial indices displayed in the left raster, shape
        ``(n_left_raster_trials,)``.
    right_raster_trial_indices : np.ndarray
        Right-choice trial indices displayed in the right raster, shape
        ``(n_right_raster_trials,)``.
    left_summary_trial_indices : np.ndarray
        Left-choice trial indices included in the lower-left summary, shape
        ``(n_left_summary_trials,)``.
    right_summary_trial_indices : np.ndarray
        Right-choice trial indices included in the lower-right summary, shape
        ``(n_right_summary_trials,)``.
    alignment_event : str
        Event column used as time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    bin_size : float
        PSTH bin width in seconds. Used only when ``summary_plot_type`` is
        ``"psth"``.
    unit_id : int
        Cluster id shown in the plot titles.
    summary_plot_type : str, default="psth"
        Lower-panel summary type. Supported values are ``"psth"``,
        ``"binned-rate-trials-mean"``, and ``"binned-rate-mean-sd"``.
    binned_rate_bin_size : float, default=0.1
        Bin width in seconds for the two binned-rate summary modes.
    title_suffix : str | None, optional
        Optional suffix appended to each top-panel title.
    event_marker_columns : tuple[str, ...] | None, optional
        Event-time columns to draw as per-trial raster markers.
    event_marker_styles : dict[str, dict[str, str]] | None, optional
        Optional style map keyed by event column. Each style may contain
        ``"label"`` and ``"color"`` strings.
    raster_row_spacing : float, default=1.0
        Vertical spacing between neighboring raster rows in arbitrary raster
        row units.
    figure_size : tuple[float, float], default=(14.0, 7.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes array with shape ``(2, 2)``. Top row axes
        are raw rasters; bottom row axes are the selected firing-rate summary.
        Left column is left choices and right column is right choices.
    """

    if float(raster_row_spacing) <= 0:
        raise ValueError("raster_row_spacing must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")
    _validate_unit_summary_plot_type(summary_plot_type)

    figure, axes = plt.subplots(
        2,
        2,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[3, 1],
    )
    figure_title = f"Unit {int(unit_id)} aligned to {alignment_event}"
    if title_suffix:
        figure_title = f"{figure_title} ({title_suffix})"
    figure.suptitle(figure_title)
    side_configs = [
        (
            "Left choices",
            np.asarray(left_raster_trial_indices, dtype=int),
            np.asarray(left_summary_trial_indices, dtype=int),
            axes[0, 0],
            axes[1, 0],
        ),
        (
            "Right choices",
            np.asarray(right_raster_trial_indices, dtype=int),
            np.asarray(right_summary_trial_indices, dtype=int),
            axes[0, 1],
            axes[1, 1],
        ),
    ]
    for side_label, raster_indices, summary_indices, raster_axis, summary_axis in side_configs:
        _draw_unit_spike_raster(
            raster_axis=raster_axis,
            unit_spike_times=unit_spike_times,
            trial_df=trial_df,
            raster_trial_indices=raster_indices,
            alignment_event=alignment_event,
            window=window,
            title=side_label,
            event_marker_columns=event_marker_columns,
            event_marker_styles=event_marker_styles,
            raster_row_spacing=float(raster_row_spacing),
        )
        raster_axis.text(
            0.02,
            0.94,
            f"raster n={raster_indices.size}; summary n={summary_indices.size}",
            transform=raster_axis.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        _draw_unit_summary_panel(
            summary_axis=summary_axis,
            unit_spike_times=unit_spike_times,
            trial_df=trial_df,
            summary_trial_indices=summary_indices,
            alignment_event=alignment_event,
            window=window,
            psth_bin_size=float(bin_size),
            summary_plot_type=summary_plot_type,
            binned_rate_bin_size=float(binned_rate_bin_size),
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure, axes


def _sanitize_filename_part(value: str | int | float | None) -> str:
    """Return one filesystem-safe filename token."""

    if value is None:
        return "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def save_unit_plot_figure(
    figure: plt.Figure,
    figure_path: Path | str,
    session_id: str,
    unit_id: int,
    region_name: str,
    condition: str,
    action_label: str,
    alignment_event: str,
    page_index: int,
    plot_type: str = "raster-psth",
) -> Path:
    """
    Save the current single-unit viewer figure in the viewer output folder.

    Parameters
    ----------
    figure : plt.Figure
        Matplotlib figure to save.
    figure_path : Path | str
        Session figure directory.
    session_id : str
        Full session identifier.
    unit_id : int
        Cluster id shown in the figure.
    region_name : str
        Active region label.
    condition : str
        Active condition filter.
    action_label : str
        Active action filter label.
    alignment_event : str
        Alignment event column.
    page_index : int
        Zero-based raster page index.
    plot_type : str, default="raster-psth"
        Single-unit plot type token. The default preserves the historical
        raster/PSTH filename without adding an explicit plot-type token.

    Returns
    -------
    Path
        Saved PNG path.
    """

    output_dir = Path(figure_path) / "unit_spike_viewer"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_parts = [
        session_id,
        region_name,
        f"unit{int(unit_id)}",
        condition,
        action_label,
        alignment_event,
    ]
    if plot_type != "raster-psth":
        filename_parts.append(plot_type)
    filename_parts.append(f"page{int(page_index)}")
    filename = "_".join(_sanitize_filename_part(part) for part in filename_parts) + ".png"
    save_path = output_dir / filename
    figure.savefig(save_path, format="png", dpi=300)
    return save_path
