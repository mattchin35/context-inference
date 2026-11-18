"""Reusable Spike-LFP phase-locking and single-trial plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.lfp.plotting import break_wrapped_phase_trace
from src.neural_analysis.spike_behavior.plotting import draw_single_trial_behavior_axis

if TYPE_CHECKING:
    from src.neural_analysis.spike_lfp.hilbert import SingleTrialSpikeLFPHilbertResult
    from src.neural_analysis.spike_lfp.phase_locking import SpikePhaseLockingResult


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
