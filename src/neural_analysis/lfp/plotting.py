"""Reusable LFP phase and single-trial plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.spike_behavior.plotting import draw_single_trial_behavior_axis

if TYPE_CHECKING:
    from src.neural_analysis.lfp.phase import SingleTrialRelativePhaseResult, WithinTrialPLVResult


def plot_phase_clustering(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    relative_time_s: np.ndarray,
    metric_label: str,
    site_label: str,
    condition_label: str,
    n_trials: int,
    figure_size: tuple[float, float] = (10.0, 5.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one frequency-by-time ITPC or ISPC heatmap.

    Parameters
    ----------
    values : np.ndarray
        Matrix with shape ``(n_frequencies, n_times)`` and values in ``[0, 1]``.
    frequencies_hz : np.ndarray
        Positive frequencies with shape ``(n_frequencies,)`` in Hz.
    relative_time_s : np.ndarray
        Event-relative times with shape ``(n_times,)`` in seconds.
    metric_label : str
        Colorbar label, expected to be ``"ITPC"`` or ``"ISPC"``.
    site_label : str
        Human-readable site or ordered site-pair label.
    condition_label : str
        Trial-condition label.
    n_trials : int
        Nominal contributing trial count shown in the title.
    figure_size : tuple[float, float], default=(10.0, 5.0)
        Figure dimensions in inches.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and heatmap axis. Time is in seconds, frequency in Hz, and color
        is dimensionless phase consistency.
    """

    matrix = np.asarray(values, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    times = np.asarray(relative_time_s, dtype=float).reshape(-1)
    if matrix.shape != (frequencies.size, times.size):
        raise ValueError("values must have shape (frequency, time).")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0) or not np.isfinite(matrix).all():
        raise ValueError("Phase-clustering values must be finite and in [0, 1].")
    if frequencies.size == 0 or np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must contain positive values.")
    if int(n_trials) < 0:
        raise ValueError("n_trials must be nonnegative.")

    figure, axis = plt.subplots(figsize=(float(figure_size[0]), float(figure_size[1])))
    mesh = axis.pcolormesh(times, frequencies, matrix, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.axvline(0.0, color="white", linestyle="--", linewidth=1.2)
    axis.set_yscale("log")
    frequency_ticks = np.geomspace(float(np.min(frequencies)), float(np.max(frequencies)), 5)
    axis.set_yticks(frequency_ticks)
    axis.set_yticklabels([f"{frequency:g}" for frequency in frequency_ticks])
    axis.set_xlabel("Time from alignment event (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(f"{metric_label}: {site_label}; {condition_label}; n={int(n_trials)}")
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label(str(metric_label))
    figure.tight_layout()
    return figure, axis


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
