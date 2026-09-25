from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.spike_behavior.trials import make_trial_type_masks

if TYPE_CHECKING:
    from src.neural_analysis.lfp.phase import SingleTrialRelativePhaseResult, WithinTrialPLVResult
    from src.neural_analysis.spike_lfp.hilbert import SingleTrialSpikeLFPHilbertResult
    from src.neural_analysis.spike_lfp.phase_locking import SpikePhaseLockingResult


LEFT_LICK_EVENT = "left_entry"
RIGHT_LICK_EVENT = "right_entry"
LEFT_CHOICE_ACTION = 1
RIGHT_CHOICE_ACTION = 0
LICK_RASTER_STYLES = {
    LEFT_LICK_EVENT: {"label": "Left licks", "color": "tab:orange"},
    RIGHT_LICK_EVENT: {"label": "Right licks", "color": "tab:blue"},
}


def plot_spike_lfp_phase_locking(
    result: SpikePhaseLockingResult,
    polar_frequency_hz: float,
    polar_bin_count: int = 24,
    figure_size: tuple[float, float] = (11.0, 6.0),
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    Plot frequency-resolved locking metrics and one phase tuning curve.

    Parameters
    ----------
    result : SpikePhaseLockingResult
        Frequency metrics with shape ``(frequency,)``, sampled complex phase
        arrays with shape ``(frequency, spike)``, and occupancy-normalized
        phase tuning arrays with shape ``(frequency, phase_bin)``. Frequencies
        are in Hz, phase edges and preferred phase are in radians, occupancy is
        in seconds, and phase firing rates are in Hz.
    polar_frequency_hz : float
        Requested positive frequency in Hz. The nearest computed frequency is
        displayed and identified in the polar title.
    polar_bin_count : int, default=24
        Positive number of phase bins. It must match the stored phase tuning
        arrays and the number of intervals in ``result.phase_bin_edges_rad``.
    figure_size : tuple[float, float], default=(11.0, 6.0)
        Figure dimensions in inches.

    Returns
    -------
    tuple[plt.Figure, dict[str, plt.Axes]]
        Figure and axes named ``ppc``, ``resultant_length``, and ``polar``.
    """

    frequencies_hz = np.asarray(result.frequencies_hz, dtype=float).reshape(-1)
    ppc = np.asarray(result.ppc, dtype=float).reshape(-1)
    resultant_length = np.asarray(result.resultant_length, dtype=float).reshape(-1)
    preferred_phase = np.asarray(result.preferred_phase_rad, dtype=float).reshape(-1)
    spike_counts = np.asarray(result.n_spikes, dtype=int).reshape(-1)
    phase_vectors = np.asarray(result.spike_phase_vectors)
    phase_valid = np.asarray(result.spike_phase_valid, dtype=bool)
    phase_bin_edges = np.asarray(result.phase_bin_edges_rad, dtype=float)
    phase_spike_counts = np.asarray(result.phase_spike_counts)
    phase_occupancy_s = np.asarray(result.phase_occupancy_s, dtype=float)
    phase_firing_rate_hz = np.asarray(result.phase_firing_rate_hz, dtype=float)
    if frequencies_hz.size == 0 or np.any(frequencies_hz <= 0.0):
        raise ValueError("result frequencies_hz must contain positive values.")
    if any(values.shape != frequencies_hz.shape for values in (ppc, resultant_length, preferred_phase, spike_counts)):
        raise ValueError("Frequency metrics must match frequencies_hz.")
    if phase_vectors.shape != phase_valid.shape or phase_vectors.shape[0] != frequencies_hz.size:
        raise ValueError("Sampled phase arrays must have shape (frequency, spike).")
    if not np.isfinite(float(polar_frequency_hz)) or float(polar_frequency_hz) <= 0.0:
        raise ValueError("polar_frequency_hz must be positive and finite.")
    if int(polar_bin_count) != polar_bin_count or int(polar_bin_count) < 1:
        raise ValueError("polar_bin_count must be a positive integer.")
    expected_tuning_shape = (frequencies_hz.size, int(polar_bin_count))
    if phase_bin_edges.shape != (int(polar_bin_count) + 1,):
        raise ValueError("phase_bin_edges_rad must contain polar_bin_count + 1 edges.")
    if not np.all(np.isfinite(phase_bin_edges)) or np.any(np.diff(phase_bin_edges) <= 0.0):
        raise ValueError("phase_bin_edges_rad must be finite and strictly increasing.")
    if any(
        values.shape != expected_tuning_shape
        for values in (phase_spike_counts, phase_occupancy_s, phase_firing_rate_hz)
    ):
        raise ValueError("Phase tuning arrays must have shape (frequency, phase_bin_count).")

    figure = plt.figure(figsize=(float(figure_size[0]), float(figure_size[1])))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.3, wspace=0.3)
    ppc_axis = figure.add_subplot(grid[0, 0])
    resultant_axis = figure.add_subplot(grid[1, 0], sharex=ppc_axis)
    polar_axis = figure.add_subplot(grid[:, 1], projection="polar")
    axes = {"ppc": ppc_axis, "resultant_length": resultant_axis, "polar": polar_axis}

    ppc_axis.plot(frequencies_hz, ppc, color="tab:blue", marker=".", linewidth=1.2)
    ppc_axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    ppc_axis.set_xscale("log")
    ppc_axis.set_ylabel("PPC")
    ppc_axis.set_title(f"Unit {result.unit_id} to {result.lfp_site_label}")
    resultant_axis.plot(frequencies_hz, resultant_length, color="tab:orange", marker=".", linewidth=1.2)
    resultant_axis.set_xscale("log")
    resultant_axis.set_ylim(0.0, 1.0)
    resultant_axis.set_xlabel("Frequency (Hz)")
    resultant_axis.set_ylabel("Mean resultant length")

    frequency_index = int(np.argmin(np.abs(frequencies_hz - float(polar_frequency_hz))))
    selected_frequency_hz = float(frequencies_hz[frequency_index])
    selected_rates_hz = phase_firing_rate_hz[frequency_index]
    bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2.0
    bin_widths = np.diff(phase_bin_edges)
    finite_rate_mask = np.isfinite(selected_rates_hz)
    polar_axis.bar(
        bin_centers[finite_rate_mask],
        selected_rates_hz[finite_rate_mask],
        width=bin_widths[finite_rate_mask],
        color="tab:blue",
        alpha=0.55,
        edgecolor="white",
    )
    finite_rates_hz = selected_rates_hz[np.isfinite(selected_rates_hz)]
    maximum_rate_hz = float(np.max(finite_rates_hz)) if finite_rates_hz.size else 0.0
    radial_maximum_hz = maximum_rate_hz * 1.10 if maximum_rate_hz > 0.0 else 1.0
    polar_axis.set_ylim(0.0, radial_maximum_hz)
    radial_ticks_hz = np.linspace(0.0, radial_maximum_hz, 4)[1:]
    polar_axis.set_yticks(radial_ticks_hz)
    polar_axis.set_yticklabels([f"{tick:g}" for tick in radial_ticks_hz])
    polar_axis.set_rlabel_position(22.5)

    selected_preferred_phase = float(preferred_phase[frequency_index])
    selected_resultant_length = float(resultant_length[frequency_index])
    if np.isfinite(selected_preferred_phase) and np.isfinite(selected_resultant_length):
        arrow_radius_hz = selected_resultant_length * radial_maximum_hz
        polar_axis.annotate(
            "",
            xy=(selected_preferred_phase, arrow_radius_hz),
            xytext=(selected_preferred_phase, 0.0),
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.8},
        )
    polar_axis.set_theta_zero_location("E")
    polar_axis.set_theta_direction(1)
    polar_axis.set_ylabel("Firing rate (Hz)")
    exposure_s = float(np.nansum(phase_occupancy_s[frequency_index]))
    selected_phase_spike_counts = np.asarray(phase_spike_counts[frequency_index], dtype=float)
    valid_spike_count = int(np.nansum(selected_phase_spike_counts))
    mean_spikes_per_bin = float(np.nanmean(selected_phase_spike_counts))
    maximum_bin_count = int(np.nanmax(selected_phase_spike_counts))
    polar_axis.text(
        0.02,
        0.02,
        (
            f"Valid spikes: {valid_spike_count}\n"
            f"Mean spikes/bin: {mean_spikes_per_bin:.2f}\n"
            f"Maximum bin count: {maximum_bin_count}\n"
            "Arrow length encodes R, not Hz."
        ),
        transform=polar_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize="small",
    )
    preferred_phase_text = selected_preferred_phase
    preferred_phase_label = (
        f"{preferred_phase_text:.2f} rad" if np.isfinite(preferred_phase_text) else "n/a"
    )
    resultant_length_text = selected_resultant_length
    resultant_length_label = (
        f"{resultant_length_text:.2f}" if np.isfinite(resultant_length_text) else "n/a"
    )
    polar_axis.set_title(
        f"Firing rate by phase at {selected_frequency_hz:g} Hz\n"
        f"n={int(spike_counts[frequency_index])}, exposure={exposure_s:.2f} s, "
        f"preferred={preferred_phase_label}, R={resultant_length_label}",
        pad=20.0,
    )
    figure.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.1)
    return figure, axes


def break_wrapped_phase_trace(
    phase_rad: np.ndarray,
    jump_threshold_rad: float = np.pi,
) -> np.ndarray:
    """Break a wrapped phase trace at discontinuities for line plotting.

    Parameters
    ----------
    phase_rad : np.ndarray
        One-dimensional wrapped phase samples with shape ``(n_samples,)`` in
        radians. Finite values normally lie in ``[-pi, pi]``; existing NaNs
        mark invalid phase estimates and are preserved.
    jump_threshold_rad : float, default=pi
        Positive finite angular jump threshold in radians. The later sample of
        every adjacent finite pair whose absolute difference exceeds this
        threshold is replaced by NaN.

    Returns
    -------
    np.ndarray
        Float copy with shape ``(n_samples,)`` in radians. It preserves input
        finite values except for later samples at detected wrap discontinuities
        and preserves existing NaNs.
    """

    phase_values = np.asarray(phase_rad, dtype=float)
    if phase_values.ndim != 1:
        raise ValueError("phase_rad must be one-dimensional.")
    threshold = float(jump_threshold_rad)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("jump_threshold_rad must be finite and positive.")

    broken_phase = phase_values.copy()
    if broken_phase.size < 2:
        return broken_phase
    adjacent_finite = np.isfinite(phase_values[:-1]) & np.isfinite(phase_values[1:])
    wrap_jumps = adjacent_finite & (np.abs(np.diff(phase_values)) > threshold)
    broken_phase[1:][wrap_jumps] = np.nan
    return broken_phase


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


def draw_single_trial_behavior_axis(
    axis: plt.Axes,
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts],
    alignment_event: str,
    window: tuple[float, float],
) -> plt.Axes:
    """
    Draw lick, LED, choice, reward, and alignment events for one trial.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis receiving the event raster.
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``alignment_event``, ``choice_time``, ``led_on_time``, and ``action``;
        optional ``reward_time`` values are plotted when finite. Times are in
        seconds on the same absolute time base as ``lick_times``.
    trial_index : int
        Integer row index selecting the displayed trial.
    lick_times : Mapping[str, nap.Ts]
        Mapping containing ``"left_entry"`` and ``"right_entry"`` Pynapple
        timestamp series in seconds.
    alignment_event : str
        Trial timestamp column used as relative time zero.
    window : tuple[float, float]
        Visible relative bounds as ``(start_s, end_s)`` in seconds.

    Returns
    -------
    plt.Axes
        The same axis, configured with behavior events and seconds on x.
    """

    required_columns = {alignment_event, "choice_time", "led_on_time", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")
    if int(trial_index) not in trial_df.index:
        raise ValueError(f"trial_index {trial_index} is not present in trial_df.")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

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
    axis.eventplot(
        [right_licks],
        orientation="horizontal",
        lineoffsets=[0.0],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["label"],
    )
    axis.eventplot(
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
        action_value = pd.to_numeric(pd.Series([trial_row["action"]]), errors="coerce").iloc[0]
        choice_y = 0.5
        if not pd.isna(action_value):
            choice_y = 1.0 if int(action_value) == LEFT_CHOICE_ACTION else 0.0
        axis.vlines(
            float(choice_time) - reference_time,
            choice_y - 0.35,
            choice_y + 0.35,
            colors="tab:purple",
            linewidth=1.4,
            label="Choice",
        )

    led_time = pd.to_numeric(pd.Series([trial_row["led_on_time"]]), errors="coerce").iloc[0]
    if not pd.isna(led_time):
        axis.vlines(
            float(led_time) - reference_time,
            -0.35,
            1.35,
            colors="tab:green",
            linewidth=1.1,
            label="LED",
        )
    if "reward_time" in trial_df.columns:
        reward_time = pd.to_numeric(pd.Series([trial_row["reward_time"]]), errors="coerce").iloc[0]
        if not pd.isna(reward_time):
            axis.vlines(
                float(reward_time) - reference_time,
                -0.35,
                1.35,
                colors="tab:red",
                linewidth=1.2,
                label="Reward",
            )

    axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, label=alignment_event)
    axis.set_yticks([0.0, 1.0])
    axis.set_yticklabels(["Right licks", "Left licks"])
    axis.set_ylim(-0.7, 1.7)
    axis.set_ylabel("Behavior")
    axis.set_xlim(float(window[0]), float(window[1]))
    axis.set_title(f"Trial {trial_index} behavior aligned to {alignment_event}")
    axis.legend(loc="upper right", fontsize="small", ncol=3)
    return axis


def plot_trial_lfp_spectrogram_and_behavior(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts],
    spectrogram_time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    log_power_db: np.ndarray,
    lfp_time_s: np.ndarray,
    lfp_values: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    power_limits_db: tuple[float, float],
    lfp_label: str,
    power_unit_label: str,
    reference_trial_count: int,
    lfp_y_label: str = "LFP (uV)",
    figure_size: tuple[float, float] = (11.0, 7.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one trial's Morlet spectrogram, LFP trace, and behavior events.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and behavior columns documented by
        ``draw_single_trial_behavior_axis``. Event times are in seconds.
    trial_index : int
        Integer row index selecting the displayed trial.
    lick_times : Mapping[str, nap.Ts]
        Left and right lick timestamps in seconds.
    spectrogram_time_s : np.ndarray
        Spectrogram sample times with shape ``(n_times,)`` in seconds relative
        to ``alignment_event``.
    frequencies_hz : np.ndarray
        Wavelet frequencies with shape ``(n_frequencies,)`` in Hz.
    log_power_db : np.ndarray
        Absolute log power with shape ``(n_times, n_frequencies)`` in dB
        relative to one squared source-unit.
    lfp_time_s : np.ndarray
        Trace times with shape ``(n_trace_samples,)`` in relative seconds.
    lfp_values : np.ndarray
        LFP trace with shape ``(n_trace_samples,)`` in source voltage units.
    alignment_event : str
        Trial event column used as relative time zero.
    window : tuple[float, float]
        Shared x bounds in relative seconds.
    power_limits_db : tuple[float, float]
        Shared spectrogram color limits as ``(minimum_db, maximum_db)``.
    lfp_label : str
        Source and channel title for the LFP trace.
    power_unit_label : str
        Colorbar unit text, such as ``"dB re 1 uV^2"``.
    reference_trial_count : int
        Number of session reference trials used to estimate color limits.
    lfp_y_label : str, default="LFP (uV)"
        LFP trace y-axis label with source units.
    figure_size : tuple[float, float], default=(11.0, 7.0)
        Figure dimensions as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and three axes in order: spectrogram, LFP trace, behavior.
        All x axes use seconds relative to ``alignment_event``.
    """

    spectrogram_time_s = np.asarray(spectrogram_time_s, dtype=float).reshape(-1)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    log_power_db = np.asarray(log_power_db, dtype=float)
    lfp_time_s = np.asarray(lfp_time_s, dtype=float).reshape(-1)
    lfp_values = np.asarray(lfp_values, dtype=float).reshape(-1)
    if log_power_db.shape != (spectrogram_time_s.size, frequencies_hz.size):
        raise ValueError("log_power_db must have shape (n_times, n_frequencies).")
    if lfp_time_s.shape != lfp_values.shape:
        raise ValueError("lfp_time_s and lfp_values must have matching one-dimensional shapes.")
    if frequencies_hz.size == 0 or np.any(frequencies_hz <= 0):
        raise ValueError("frequencies_hz must contain positive values.")
    if len(power_limits_db) != 2 or float(power_limits_db[0]) >= float(power_limits_db[1]):
        raise ValueError("power_limits_db must contain increasing limits.")
    if int(reference_trial_count) < 1:
        raise ValueError("reference_trial_count must be positive.")

    figure, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[2.5, 0.8, 1.0],
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    spectrogram_axis, lfp_axis, behavior_axis = axes

    power_mesh = spectrogram_axis.pcolormesh(
        spectrogram_time_s,
        frequencies_hz,
        log_power_db.T,
        shading="auto",
        cmap="magma",
        vmin=float(power_limits_db[0]),
        vmax=float(power_limits_db[1]),
    )
    spectrogram_axis.axvline(0.0, color="white", linestyle="--", linewidth=1.0)
    spectrogram_axis.set_yscale("log")
    spectrogram_axis.set_ylim(float(np.min(frequencies_hz)), float(np.max(frequencies_hz)))
    frequency_ticks = np.geomspace(float(np.min(frequencies_hz)), float(np.max(frequencies_hz)), 5)
    spectrogram_axis.set_yticks(frequency_ticks)
    spectrogram_axis.set_yticklabels([f"{frequency:g}" for frequency in frequency_ticks])
    spectrogram_axis.set_ylabel("Frequency (Hz)")
    spectrogram_axis.set_title(f"Morlet power, {lfp_label}")
    colorbar = figure.colorbar(power_mesh, ax=spectrogram_axis, pad=0.015)
    colorbar.set_label(str(power_unit_label))

    lfp_axis.plot(lfp_time_s, lfp_values, color="black", linewidth=0.8)
    lfp_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    lfp_axis.set_ylabel(str(lfp_y_label))
    lfp_axis.set_title(str(lfp_label))
    draw_single_trial_behavior_axis(
        axis=behavior_axis,
        trial_df=trial_df,
        trial_index=int(trial_index),
        lick_times=lick_times,
        alignment_event=alignment_event,
        window=window,
    )
    behavior_axis.set_xlabel(f"Time from {alignment_event} (s)")
    for axis in axes:
        axis.set_xlim(float(window[0]), float(window[1]))

    figure.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.12, hspace=0.35)
    figure.text(
        0.5,
        0.025,
        (
            f"Absolute log power uses shared {power_limits_db[0]:.1f} to "
            f"{power_limits_db[1]:.1f} dB limits estimated from "
            f"{int(reference_trial_count)} reference trials."
        ),
        ha="center",
        va="bottom",
        fontsize="small",
    )
    return figure, axes


def plot_trial_spike_lfp_hilbert_phase_and_behavior(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts | np.ndarray],
    result: SingleTrialSpikeLFPHilbertResult,
    alignment_event: str,
    window: tuple[float, float],
    lfp_y_label: str = "LFP",
    figure_size: tuple[float, float] = (11.0, 8.0),
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot one trial's raw LFP, Hilbert phase, unit spikes, and behavior.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with behavior columns required by
        ``draw_single_trial_behavior_axis``. Event times are synchronized
        absolute seconds.
    trial_index : int
        Integer trial-table row identifier for the displayed trial.
    lick_times : Mapping[str, nap.Ts | np.ndarray]
        Absolute left and right lick timestamps in seconds, keyed by
        ``"left_entry"`` and ``"right_entry"``.
    result : SingleTrialSpikeLFPHilbertResult
        Visible-trial output. Time-indexed arrays have shape
        ``(n_visible_samples,)``; times are seconds relative to the alignment
        event and LFP arrays retain their source voltage units. Spike-indexed
        arrays have shape ``(n_visible_spikes,)`` with spike phases in
        wrapped radians.
    alignment_event : str
        Trial timestamp column defining relative time zero.
    window : tuple[float, float]
        Finite increasing visible relative-time bounds ``(start_s, end_s)``.
    lfp_y_label : str, default="LFP"
        Y-axis label for raw and bandpassed source-voltage traces.
    figure_size : tuple[float, float], default=(11.0, 8.0)
        Figure dimensions in inches.

    Returns
    -------
    tuple[plt.Figure, dict[str, plt.Axes]]
        Figure and shared-x axes named ``raw_lfp``, ``bandpassed_lfp``,
        ``phase``, and ``behavior``. The phase axis displays radians in
        ``[-pi, pi]``; invalid spike-phase samples are excluded only from its
        scatter points and remain represented by filtered-trace spike rugs.
    """

    if len(window) != 2:
        raise ValueError("window must contain exactly two bounds.")
    window_start_s, window_end_s = (float(window[0]), float(window[1]))
    if not np.isfinite(window_start_s) or not np.isfinite(window_end_s) or window_start_s >= window_end_s:
        raise ValueError("window must contain finite increasing bounds.")
    if len(figure_size) != 2 or any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in figure_size):
        raise ValueError("figure_size must contain two positive finite values.")

    relative_time_s = np.asarray(result.relative_time_s, dtype=float)
    raw_lfp = np.asarray(result.raw_lfp, dtype=float)
    bandpassed_lfp = np.asarray(result.bandpassed_lfp, dtype=float)
    phase_rad = np.asarray(result.phase_rad, dtype=float)
    phase_valid = np.asarray(result.phase_valid, dtype=bool)
    time_indexed_values = (raw_lfp, bandpassed_lfp, phase_rad, phase_valid)
    if relative_time_s.ndim != 1 or relative_time_s.size == 0:
        raise ValueError("result.relative_time_s must be a nonempty one-dimensional array.")
    if any(values.ndim != 1 or values.shape != relative_time_s.shape for values in time_indexed_values):
        raise ValueError("Visible LFP and phase arrays must be one-dimensional and match relative_time_s.")
    if not np.all(np.isfinite(relative_time_s)) or np.any(np.diff(relative_time_s) <= 0.0):
        raise ValueError("result.relative_time_s must be finite and strictly increasing.")
    if not np.all(np.isfinite(raw_lfp)) or not np.all(np.isfinite(bandpassed_lfp)):
        raise ValueError("Visible raw and bandpassed LFP arrays must be finite.")

    spike_times_s = np.asarray(result.spike_times_relative_s, dtype=float)
    spike_phase_rad = np.asarray(result.spike_phase_rad, dtype=float)
    spike_phase_valid = np.asarray(result.spike_phase_valid, dtype=bool)
    if (
        spike_times_s.ndim != 1
        or spike_phase_rad.ndim != 1
        or spike_phase_valid.ndim != 1
        or spike_phase_rad.shape != spike_times_s.shape
        or spike_phase_valid.shape != spike_times_s.shape
    ):
        raise ValueError("Spike time, phase, and validity arrays must share one spike dimension.")
    if not np.all(np.isfinite(spike_times_s)):
        raise ValueError("result.spike_times_relative_s must be finite.")
    if len(result.frequency_band_hz) != 2:
        raise ValueError("result.frequency_band_hz must contain two cutoff frequencies.")
    band_low_hz, band_high_hz = (float(result.frequency_band_hz[0]), float(result.frequency_band_hz[1]))
    if not np.isfinite(band_low_hz) or not np.isfinite(band_high_hz) or band_low_hz >= band_high_hz:
        raise ValueError("result.frequency_band_hz must contain finite increasing values.")

    figure, axis_array = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[1.05, 1.05, 1.0, 0.8],
    )
    raw_axis, bandpassed_axis, phase_axis, behavior_axis = np.asarray(axis_array, dtype=object).reshape(-1)
    axes = {
        "raw_lfp": raw_axis,
        "bandpassed_lfp": bandpassed_axis,
        "phase": phase_axis,
        "behavior": behavior_axis,
    }

    raw_axis.plot(relative_time_s, raw_lfp, color="0.15", linewidth=0.75)
    raw_axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    raw_axis.set_ylabel(str(lfp_y_label))
    raw_axis.set_title("Raw LFP")

    bandpassed_axis.plot(relative_time_s, bandpassed_lfp, color="tab:blue", linewidth=0.85)
    bandpassed_axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    if spike_times_s.size:
        bandpassed_axis.vlines(
            spike_times_s,
            0.0,
            0.10,
            color="black",
            linewidth=0.8,
            transform=bandpassed_axis.get_xaxis_transform(),
            label=f"Spikes (n={spike_times_s.size})",
        )
        bandpassed_axis.legend(loc="upper right", fontsize="small")
    bandpassed_axis.set_ylabel(str(lfp_y_label))
    bandpassed_axis.set_title(f"{band_low_hz:g}-{band_high_hz:g} Hz bandpass")

    phase_axis.plot(
        relative_time_s,
        break_wrapped_phase_trace(phase_rad),
        color="tab:purple",
        linewidth=0.85,
        label="Hilbert phase",
    )
    phase_axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    valid_spike_phase = spike_phase_valid & np.isfinite(spike_phase_rad)
    if np.any(valid_spike_phase):
        phase_axis.scatter(
            spike_times_s[valid_spike_phase],
            spike_phase_rad[valid_spike_phase],
            color="black",
            s=20.0,
            zorder=3,
            label=f"Spike phase (n={int(np.count_nonzero(valid_spike_phase))})",
        )
    phase_axis.set_ylim(-np.pi, np.pi)
    phase_axis.set_yticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    phase_axis.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    phase_axis.set_ylabel("Phase (rad)")
    phase_axis.set_title("Hilbert phase at selected-unit spikes")
    phase_axis.legend(loc="upper right", fontsize="small")

    draw_single_trial_behavior_axis(
        axis=behavior_axis,
        trial_df=trial_df,
        trial_index=int(trial_index),
        lick_times=lick_times,
        alignment_event=alignment_event,
        window=(window_start_s, window_end_s),
    )
    behavior_axis.set_xlabel(f"Time from {alignment_event} (s)")
    for axis in axes.values():
        axis.set_xlim(window_start_s, window_end_s)

    figure.suptitle(
        f"Trial {int(trial_index)}: unit {int(result.unit_id)} and {result.lfp_site_label}",
        y=0.99,
    )
    figure.subplots_adjust(left=0.1, right=0.94, top=0.93, bottom=0.08, hspace=0.45)
    return figure, axes


def plot_trial_lfp_relative_phase_and_behavior(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts | np.ndarray],
    result: SingleTrialRelativePhaseResult,
    display_valid_mask: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    lfp_y_label: str = "LFP",
    figure_size: tuple[float, float] = (11.0, 8.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one trial's relative phase, source-rate traces, and behavior events.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with behavior columns required by
        ``draw_single_trial_behavior_axis`` and event times in seconds.
    trial_index : int
        Integer trial-table row identifier.
    lick_times : Mapping[str, nap.Ts | np.ndarray]
        Left/right lick timestamps in absolute seconds.
    result : SingleTrialRelativePhaseResult
        Relative phase with shape ``(frequency, time)``, axes in Hz/seconds,
        and two source-rate LFP traces in source-dependent voltage units.
    display_valid_mask : np.ndarray
        Boolean frequency-by-time mask matching ``result.phase_angle_rad``.
    alignment_event : str
        Trial event column defining relative time zero.
    window : tuple[float, float]
        Shared visible x bounds in event-relative seconds.
    lfp_y_label : str, default="LFP"
        Source-trace y-axis label.
    figure_size : tuple[float, float], default=(11.0, 8.0)
        Figure dimensions in inches.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and four primary axes ordered as relative phase, source A,
        source B, and behavior. The colorbar is an additional figure axis.
    """

    phase_angle = np.asarray(result.phase_angle_rad, dtype=float)
    display_valid = np.asarray(display_valid_mask, dtype=bool)
    frequencies_hz = np.asarray(result.frequencies_hz, dtype=float).reshape(-1)
    relative_time_s = np.asarray(result.relative_time_s, dtype=float).reshape(-1)
    if phase_angle.shape != (frequencies_hz.size, relative_time_s.size):
        raise ValueError("phase_angle_rad must have shape (frequency, time).")
    if display_valid.shape != phase_angle.shape:
        raise ValueError("display_valid_mask must match phase_angle_rad.")
    if frequencies_hz.size == 0 or np.any(frequencies_hz <= 0.0):
        raise ValueError("frequencies_hz must contain positive values.")

    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[3.0, 0.8, 0.8, 1.1],
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    phase_axis, site_a_axis, site_b_axis, behavior_axis = axes
    cyclic_colormap = plt.get_cmap("twilight_shifted").copy()
    cyclic_colormap.set_bad(color="0.82", alpha=1.0)
    displayed_angles = np.ma.array(phase_angle, mask=~display_valid)
    phase_mesh = phase_axis.pcolormesh(
        relative_time_s,
        frequencies_hz,
        displayed_angles,
        shading="auto",
        cmap=cyclic_colormap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    phase_axis.axvline(0.0, color="white", linestyle="--", linewidth=1.0)
    phase_axis.set_yscale("log")
    phase_axis.set_ylim(float(np.min(frequencies_hz)), float(np.max(frequencies_hz)))
    frequency_ticks = np.geomspace(float(np.min(frequencies_hz)), float(np.max(frequencies_hz)), 5)
    phase_axis.set_yticks(frequency_ticks)
    phase_axis.set_yticklabels([f"{frequency:g}" for frequency in frequency_ticks])
    phase_axis.minorticks_off()
    phase_axis.set_ylabel("Frequency (Hz)")
    phase_axis.set_title(
        f"Trial {int(trial_index)} relative phase: {result.site_a_label} - {result.site_b_label}"
    )
    colorbar = figure.colorbar(
        phase_mesh,
        ax=phase_axis,
        pad=0.015,
        ticks=[-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
    )
    colorbar.set_ticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    colorbar.set_label("Phase difference A - B (rad)")

    site_a_axis.plot(result.source_time_a_s, result.source_lfp_a, color="tab:blue", linewidth=0.7)
    site_a_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    site_a_axis.set_ylabel(str(lfp_y_label))
    site_a_axis.set_title(f"{result.site_a_label} (unprocessed)")
    site_b_axis.plot(result.source_time_b_s, result.source_lfp_b, color="tab:orange", linewidth=0.7)
    site_b_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    site_b_axis.set_ylabel(str(lfp_y_label))
    site_b_axis.set_title(f"{result.site_b_label} (unprocessed)")

    draw_single_trial_behavior_axis(
        axis=behavior_axis,
        trial_df=trial_df,
        trial_index=int(trial_index),
        lick_times=lick_times,
        alignment_event=alignment_event,
        window=window,
    )
    behavior_axis.set_xlabel(f"Time from {alignment_event} (s)")
    for axis in axes:
        axis.set_xlim(float(window[0]), float(window[1]))
    figure.subplots_adjust(left=0.1, right=0.9, top=0.94, bottom=0.08, hspace=0.45)
    return figure, axes


def plot_trial_lfp_phase_analysis_and_behavior(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts | np.ndarray],
    phase_result: SingleTrialRelativePhaseResult,
    plv_result: WithinTrialPLVResult,
    phase_display_valid_mask: np.ndarray,
    display_mode: str,
    alignment_event: str,
    window: tuple[float, float],
    lfp_y_label: str = "LFP",
    figure_size: tuple[float, float] = (11.0, 8.0),
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    Plot phase and/or PLV above source-rate LFP traces and trial behavior.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with behavior columns required by
        ``draw_single_trial_behavior_axis`` and absolute event times in seconds.
    trial_index : int
        Integer trial-table row identifier.
    lick_times : Mapping[str, nap.Ts | np.ndarray]
        Left/right lick timestamps in absolute seconds.
    phase_result : SingleTrialRelativePhaseResult
        Visible phase arrays with shape ``(frequency, time)``, axes in
        Hz/seconds, and source-rate LFP traces in source-dependent units.
    plv_result : WithinTrialPLVResult
        Dimensionless PLV with shape ``(frequency, time)`` and values in
        ``[0, 1]`` or NaN.
    phase_display_valid_mask : np.ndarray
        Boolean frequency-by-time mask matching
        ``phase_result.phase_angle_rad``.
    display_mode : str
        One of ``"Phase difference"``, ``"Within-trial PLV"``, or
        ``"Phase + PLV"``.
    alignment_event : str
        Trial event column defining relative time zero.
    window : tuple[float, float]
        Shared visible x bounds in event-relative seconds.
    lfp_y_label : str, default="LFP"
        Source-trace y-axis label.
    figure_size : tuple[float, float], default=(11.0, 8.0)
        Figure dimensions in inches.

    Returns
    -------
    tuple[plt.Figure, dict[str, plt.Axes]]
        Figure and named primary axes. Heatmap keys depend on ``display_mode``;
        ``site_a``, ``site_b``, and ``behavior`` are always present.
    """

    display_options = {"Phase difference", "Within-trial PLV", "Phase + PLV"}
    if display_mode not in display_options:
        raise ValueError(f"Unknown phase-analysis display mode: {display_mode!r}.")
    phase_angle = np.asarray(phase_result.phase_angle_rad, dtype=float)
    phase_valid = np.asarray(phase_display_valid_mask, dtype=bool)
    frequencies_hz = np.asarray(phase_result.frequencies_hz, dtype=float).reshape(-1)
    relative_time_s = np.asarray(phase_result.relative_time_s, dtype=float).reshape(-1)
    if phase_angle.shape != (frequencies_hz.size, relative_time_s.size):
        raise ValueError("phase_angle_rad must have shape (frequency, time).")
    if phase_valid.shape != phase_angle.shape:
        raise ValueError("phase_display_valid_mask must match phase_angle_rad.")
    if frequencies_hz.size == 0 or np.any(frequencies_hz <= 0.0):
        raise ValueError("frequencies_hz must contain positive values.")

    combined = display_mode == "Phase + PLV"
    figure = plt.figure(figsize=(float(figure_size[0]), float(figure_size[1])))
    grid = figure.add_gridspec(
        4,
        2 if combined else 1,
        height_ratios=[3.0, 0.8, 0.8, 1.1],
        hspace=0.55,
        wspace=0.35,
    )
    axes: dict[str, plt.Axes] = {}
    if display_mode in {"Phase difference", "Phase + PLV"}:
        phase_axis = figure.add_subplot(grid[0, 0])
        axes["phase"] = phase_axis
        cyclic_colormap = plt.get_cmap("twilight_shifted").copy()
        cyclic_colormap.set_bad(color="0.82", alpha=1.0)
        phase_mesh = phase_axis.pcolormesh(
            relative_time_s,
            frequencies_hz,
            np.ma.array(phase_angle, mask=~phase_valid),
            shading="auto",
            cmap=cyclic_colormap,
            vmin=-np.pi,
            vmax=np.pi,
        )
        phase_colorbar = figure.colorbar(
            phase_mesh,
            ax=phase_axis,
            pad=0.015,
            ticks=[-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
        )
        phase_colorbar.set_ticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
        phase_colorbar.set_label("Phase difference A - B (rad)")
        phase_axis.set_title("Instantaneous relative phase")
    if display_mode in {"Within-trial PLV", "Phase + PLV"}:
        plv_column = 1 if combined else 0
        plv_axis = figure.add_subplot(grid[0, plv_column])
        axes["plv"] = plv_axis
        plv_values = np.asarray(plv_result.plv, dtype=float)
        plv_time_s = np.asarray(plv_result.relative_time_s, dtype=float).reshape(-1)
        if plv_values.shape != (frequencies_hz.size, plv_time_s.size):
            raise ValueError("plv must have shape (frequency, time).")
        plv_colormap = plt.get_cmap("viridis").copy()
        plv_colormap.set_bad(color="0.82", alpha=1.0)
        plv_mesh = plv_axis.pcolormesh(
            plv_time_s,
            frequencies_hz,
            np.ma.masked_invalid(plv_values),
            shading="auto",
            cmap=plv_colormap,
            vmin=0.0,
            vmax=1.0,
        )
        plv_colorbar = figure.colorbar(plv_mesh, ax=plv_axis, pad=0.015)
        plv_colorbar.set_label("Within-trial PLV")
        plv_axis.set_title(f"Local PLV ({plv_result.window_cycles:g} cycles)")

    for heatmap_axis in [axes[key] for key in ("phase", "plv") if key in axes]:
        heatmap_axis.axvline(0.0, color="white", linestyle="--", linewidth=1.0)
        heatmap_axis.set_yscale("log")
        heatmap_axis.set_ylim(float(np.min(frequencies_hz)), float(np.max(frequencies_hz)))
        frequency_ticks = np.geomspace(float(np.min(frequencies_hz)), float(np.max(frequencies_hz)), 5)
        heatmap_axis.set_yticks(frequency_ticks)
        heatmap_axis.set_yticklabels([f"{frequency:g}" for frequency in frequency_ticks])
        heatmap_axis.minorticks_off()
        heatmap_axis.set_ylabel("Frequency (Hz)")

    site_a_axis = figure.add_subplot(grid[1, :])
    site_b_axis = figure.add_subplot(grid[2, :], sharex=site_a_axis)
    behavior_axis = figure.add_subplot(grid[3, :], sharex=site_a_axis)
    axes.update(site_a=site_a_axis, site_b=site_b_axis, behavior=behavior_axis)
    site_a_axis.plot(phase_result.source_time_a_s, phase_result.source_lfp_a, color="tab:blue", linewidth=0.7)
    site_a_axis.set_ylabel(str(lfp_y_label))
    site_a_axis.set_title(f"{phase_result.site_a_label} (unprocessed)")
    site_b_axis.plot(phase_result.source_time_b_s, phase_result.source_lfp_b, color="tab:orange", linewidth=0.7)
    site_b_axis.set_ylabel(str(lfp_y_label))
    site_b_axis.set_title(f"{phase_result.site_b_label} (unprocessed)")
    draw_single_trial_behavior_axis(
        axis=behavior_axis,
        trial_df=trial_df,
        trial_index=int(trial_index),
        lick_times=lick_times,
        alignment_event=alignment_event,
        window=window,
    )
    behavior_axis.set_xlabel(f"Time from {alignment_event} (s)")
    for axis in axes.values():
        axis.axvline(0.0, color="gray" if axis not in (axes.get("phase"), axes.get("plv")) else "white", linestyle="--", linewidth=1.0)
        axis.set_xlim(float(window[0]), float(window[1]))
    figure.suptitle(
        f"Trial {int(trial_index)}: {phase_result.site_a_label} - {phase_result.site_b_label}",
        y=0.99,
    )
    figure.subplots_adjust(left=0.09, right=0.92, top=0.93, bottom=0.08)
    return figure, axes


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


def build_concatenated_trial_time_axis(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    pca_time_s: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    axis_mode: str = "auto",
) -> dict[str, object]:
    """
    Build x-axis coordinates for concatenated trial-window PCA inspection.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and an ``alignment_event`` column in
        seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)``.
    pca_time_s : np.ndarray
        One-dimensional PCA bin centers with shape ``(n_bins,)`` in seconds
        relative to ``alignment_event``.
    alignment_event : str
        Trial time column used as time zero.
    window : tuple[float, float]
        Relative window bounds in seconds as ``(start_s, end_s)``.
    axis_mode : str, default="auto"
        X-axis layout. ``"auto"`` preserves real elapsed time when trial
        windows are non-overlapping and otherwise falls back to pseudo-time.
        ``"pseudo_time"`` forces equal-width side-by-side trial windows.
        ``"real_time"`` requires finite, non-overlapping trial windows.

    Returns
    -------
    dict[str, object]
        Dictionary containing ``x_by_trial`` with shape ``(n_trials, n_bins)``,
        ``trial_start_x`` and ``trial_end_x`` with shape ``(n_trials,)``,
        ``trial_break_x`` with shape ``(max(n_trials - 1, 0),)``, and ``mode``
        as either ``"real_time"`` or ``"pseudo_time"``. X values are seconds
        from the first visible trial-window start in real-time mode, or seconds
        in concatenated trial-window coordinates in pseudo-time mode.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if axis_mode not in {"auto", "real_time", "pseudo_time"}:
        raise ValueError("axis_mode must be 'auto', 'real_time', or 'pseudo_time'.")
    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if normalized_trial_indices.size == 0:
        raise ValueError("trial_indices must contain at least one trial.")
    pca_time_s = np.asarray(pca_time_s, dtype=float).reshape(-1)
    if pca_time_s.size == 0:
        raise ValueError("pca_time_s must contain at least one time bin.")

    alignment_times = pd.to_numeric(
        trial_df.loc[normalized_trial_indices, alignment_event],
        errors="coerce",
    ).to_numpy(dtype=float)
    window_start_s = float(window[0])
    window_end_s = float(window[1])
    window_width_s = window_end_s - window_start_s
    absolute_start_s = alignment_times + window_start_s
    absolute_end_s = alignment_times + window_end_s

    can_use_real_time = np.isfinite(alignment_times).all()
    if can_use_real_time and absolute_start_s.size > 1:
        can_use_real_time = bool(np.all(absolute_start_s[1:] >= absolute_end_s[:-1]))
    if axis_mode == "real_time" and not can_use_real_time:
        raise ValueError("axis_mode='real_time' requires finite, non-overlapping trial windows.")

    if axis_mode != "pseudo_time" and can_use_real_time:
        first_window_start_s = float(absolute_start_s[0])
        x_by_trial = alignment_times[:, np.newaxis] + pca_time_s[np.newaxis, :] - first_window_start_s
        trial_start_x = absolute_start_s - first_window_start_s
        trial_end_x = absolute_end_s - first_window_start_s
        mode = "real_time"
    else:
        trial_start_x = np.arange(normalized_trial_indices.size, dtype=float) * window_width_s
        trial_end_x = trial_start_x + window_width_s
        x_by_trial = trial_start_x[:, np.newaxis] + (pca_time_s[np.newaxis, :] - window_start_s)
        mode = "pseudo_time"
    trial_break_x = trial_start_x[1:].copy()

    return {
        "x_by_trial": x_by_trial,
        "trial_start_x": trial_start_x,
        "trial_end_x": trial_end_x,
        "trial_break_x": trial_break_x,
        "mode": mode,
    }


def plot_concatenated_trial_behavior_and_population_pca(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    lick_times: Mapping[str, nap.Ts],
    pca_time_s: np.ndarray,
    pca_scores: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    pc_count: int = 5,
    figure_size: tuple[float, float] = (11.0, 5.0),
    axis_mode: str = "auto",
) -> tuple[plt.Figure, np.ndarray, dict[str, object]]:
    """
    Plot behavior events and PCA trajectories across concatenated trial windows.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``start_time``, ``choice_time``, ``reward_time``, ``led_on_time``,
        ``action``, and ``reward``. Times are in seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)`` matching
        the first axis of ``pca_scores``.
    lick_times : Mapping[str, nap.Ts]
        Mapping with ``"left_entry"`` and ``"right_entry"`` keys. Each value is
        a one-dimensional Pynapple ``Ts`` of lick timestamps in seconds.
    pca_time_s : np.ndarray
        One-dimensional PCA bin centers with shape ``(n_bins,)`` in seconds
        relative to ``alignment_event``.
    pca_scores : np.ndarray
        PCA score tensor with shape ``(n_trials, n_bins, n_components)``.
    alignment_event : str
        Event column used as relative time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    pc_count : int, default=5
        Number of leading PCs to plot. The plotted count is capped by fitted
        components.
    figure_size : tuple[float, float], default=(11.0, 5.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.
    axis_mode : str, default="auto"
        X-axis layout passed to ``build_concatenated_trial_time_axis``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray, dict[str, object]]
        Figure, two axes in top-to-bottom order, and axis-coordinate metadata
        from ``build_concatenated_trial_time_axis``.
    """

    required_columns = {"start_time", "choice_time", "reward_time", "led_on_time", "action", "reward"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")
    if int(pc_count) < 1:
        raise ValueError("pc_count must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    pca_time_s = np.asarray(pca_time_s, dtype=float).reshape(-1)
    pca_scores = np.asarray(pca_scores, dtype=float)
    if pca_scores.ndim != 3:
        raise ValueError("pca_scores must have shape (n_trials, n_bins, n_components).")
    if pca_scores.shape[0] != normalized_trial_indices.size:
        raise ValueError("pca_scores trial axis must match trial_indices length.")
    if pca_scores.shape[1] != pca_time_s.size:
        raise ValueError("pca_scores time axis must match pca_time_s length.")

    axis_data = build_concatenated_trial_time_axis(
        trial_df=trial_df,
        trial_indices=normalized_trial_indices,
        pca_time_s=pca_time_s,
        alignment_event=alignment_event,
        window=window,
        axis_mode=axis_mode,
    )
    x_by_trial = np.asarray(axis_data["x_by_trial"], dtype=float)
    trial_start_x = np.asarray(axis_data["trial_start_x"], dtype=float)
    trial_end_x = np.asarray(axis_data["trial_end_x"], dtype=float)
    trial_break_x = np.asarray(axis_data["trial_break_x"], dtype=float)

    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[0.9, 2.6],
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    behavior_axis = axes[0]
    pca_axis = axes[1]

    for trial_position, trial_index in enumerate(normalized_trial_indices):
        trial_row = trial_df.loc[int(trial_index)]
        reward_value = pd.to_numeric(pd.Series([trial_row["reward"]]), errors="coerce").iloc[0]
        shade_color = "tab:green" if not pd.isna(reward_value) and int(reward_value) == 1 else "tab:red"
        for axis in (behavior_axis, pca_axis):
            axis.axvspan(
                trial_start_x[trial_position],
                trial_end_x[trial_position],
                color=shade_color,
                alpha=0.08,
                linewidth=0,
            )
            axis.axvline(trial_start_x[trial_position], color="0.6", linewidth=0.8, alpha=0.35)
        pca_axis.axvline(trial_end_x[trial_position], color="0.6", linewidth=0.8, alpha=0.2)

        reference_time = get_trial_alignment_time(
            trial_df=trial_df,
            trial_index=int(trial_index),
            alignment_event=alignment_event,
        )
        trial_offset_x = x_by_trial[trial_position, 0] - pca_time_s[0]
        for event_column, y_value, color, label in (
            ("led_on_time", 0.5, "tab:green", "LED"),
            ("choice_time", 1.5, "tab:purple", "Choice"),
            ("reward_time", 2.5, "tab:green", "Reward"),
        ):
            event_time = pd.to_numeric(pd.Series([trial_row[event_column]]), errors="coerce").iloc[0]
            if pd.isna(event_time):
                continue
            relative_event_time = float(event_time) - reference_time
            if float(window[0]) <= relative_event_time <= float(window[1]):
                event_x = trial_offset_x + relative_event_time
                behavior_axis.plot(
                    [event_x, event_x],
                    [y_value - 0.3, y_value + 0.3],
                    color=color,
                    linewidth=1.2,
                    label=label,
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
        if right_licks.size:
            behavior_axis.eventplot(
                [trial_offset_x + right_licks],
                orientation="horizontal",
                lineoffsets=[3.5],
                linelengths=0.4,
                linewidths=0.8,
                colors=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["color"],
            )
        if left_licks.size:
            behavior_axis.eventplot(
                [trial_offset_x + left_licks],
                orientation="horizontal",
                lineoffsets=[4.5],
                linelengths=0.4,
                linewidths=0.8,
                colors=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["color"],
            )

    plotted_pc_count = min(int(pc_count), pca_scores.shape[2])
    pc_color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    if not pc_color_cycle:
        pc_color_cycle = ["C0"]
    for pc_index in range(plotted_pc_count):
        pc_color = pc_color_cycle[pc_index % len(pc_color_cycle)]
        for trial_position in range(normalized_trial_indices.size):
            pca_axis.plot(
                x_by_trial[trial_position],
                pca_scores[trial_position, :, pc_index],
                color=pc_color,
                linewidth=1.0,
                label=f"PC{pc_index + 1}",
            )

    for break_position_x in trial_break_x:
        for axis in (behavior_axis, pca_axis):
            axis.axvline(
                float(break_position_x),
                color="black",
                linewidth=1.6,
                alpha=0.25,
                label="Trial break",
            )

    behavior_axis.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    behavior_axis.set_yticklabels(["LED", "Choice", "Reward", "Right licks", "Left licks"])
    behavior_axis.set_ylim(0.0, 5.0)
    behavior_axis.set_ylabel("Behavior")
    behavior_axis.set_title("Concatenated trial behavior and PCA")

    handles, labels = behavior_axis.get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique_labels.setdefault(label, handle)
    if unique_labels:
        behavior_axis.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize="small")

    pca_axis.set_xlim(float(trial_start_x[0]), float(trial_end_x[-1]))
    pca_axis.set_ylabel("PC score")
    pca_axis.set_xlabel(
        "Time from first visible trial window start (s)"
        if axis_data["mode"] == "real_time"
        else "Concatenated trial-window time (s)"
    )
    pca_axis.set_title(f"Population PCA trajectories, first {plotted_pc_count} PCs")
    handles, labels = pca_axis.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        unique_labels.setdefault(label, handle)
    pca_axis.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize="small", ncol=min(plotted_pc_count, 5))

    figure.tight_layout()
    return figure, axes, axis_data


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
