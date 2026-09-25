"""Legacy phase API with plotting retained outside the canonical science module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.lfp.phase import (
    ANALYSIS_VERSION,
    PHASE_TENSOR_AXIS_ORDER,
    RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS,
    ContinuousProcessingBlock,
    PhaseClusteringResult,
    PhaseTrialTensor,
    SingleTrialRelativePhaseResult,
    WaveletCoefficientResult,
    WaveletPhaseResult,
    WithinTrialPLVResult,
    combine_phase_trial_tensors,
    compute_ispc,
    compute_itpc,
    compute_plv_window_samples,
    compute_single_trial_relative_phase,
    compute_site_phase_trial_tensor,
    compute_wavelet_coefficients,
    compute_wavelet_phase,
    compute_within_trial_plv,
    make_phase_condition_masks,
    make_phase_trial_tensor,
    make_relative_phase_display_mask,
    make_relative_phase_support_mask,
    normalize_wavelet_phase,
    plan_continuous_processing_blocks,
    save_phase_clustering_result,
    save_single_trial_phase_analysis_result,
    save_single_trial_relative_phase_result,
    select_tensor_trial_mask,
)


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
