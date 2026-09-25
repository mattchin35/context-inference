"""Morlet spectrogram computation and shared trial color scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pynapple as nap
from scipy import signal

from src.neural_analysis.spike_behavior.trials import make_trial_type_masks


MORLET_SUPPORT_CUTOFF = 8.0


@dataclass(frozen=True)
class LFPSpectrogramResult:
    """One trial's decimated LFP trace and time-frequency power."""

    time_s: np.ndarray
    frequencies_hz: np.ndarray
    log_power_db: np.ndarray
    lfp_values: np.ndarray
    sample_rate_hz: float


def compute_wavelet_padding_s(
    minimum_frequency_hz: float,
    window_length: float,
) -> float:
    """
    Compute padding required by the longest Pynapple Morlet wavelet.

    Parameters
    ----------
    minimum_frequency_hz : float
        Lowest requested wavelet frequency in Hz.
    window_length : float
        Pynapple Morlet ``window_length`` parameter, dimensionless.

    Returns
    -------
    float
        Required padding on each side of the visible interval in seconds.
    """

    if float(minimum_frequency_hz) <= 0:
        raise ValueError("minimum_frequency_hz must be positive.")
    if float(window_length) <= 0:
        raise ValueError("window_length must be positive.")
    return MORLET_SUPPORT_CUTOFF * float(window_length) / float(minimum_frequency_hz)


def compute_decimation_factor(
    sample_rate_hz: float,
    target_sample_rate_hz: float,
    maximum_frequency_hz: float,
) -> int:
    """
    Select integer anti-aliased decimation while preserving requested frequencies.

    Parameters
    ----------
    sample_rate_hz : float
        Input LFP sample rate in Hz.
    target_sample_rate_hz : float
        Preferred output sample rate in Hz.
    maximum_frequency_hz : float
        Highest requested spectrogram frequency in Hz.

    Returns
    -------
    int
        Integer downsampling factor. A value of one leaves the data at its
        original sample rate. The resulting Nyquist frequency remains above
        ``maximum_frequency_hz``.
    """

    sample_rate_hz = float(sample_rate_hz)
    target_sample_rate_hz = float(target_sample_rate_hz)
    maximum_frequency_hz = float(maximum_frequency_hz)
    if sample_rate_hz <= 0 or target_sample_rate_hz <= 0 or maximum_frequency_hz <= 0:
        raise ValueError("Sample rates and maximum_frequency_hz must be positive.")
    if sample_rate_hz <= 2.0 * maximum_frequency_hz:
        raise ValueError("Input LFP sample rate does not satisfy Nyquist for the requested frequencies.")

    downsample_factor = max(1, int(np.floor(sample_rate_hz / target_sample_rate_hz)))
    while downsample_factor > 1 and sample_rate_hz / downsample_factor <= 2.0 * maximum_frequency_hz:
        downsample_factor -= 1
    return downsample_factor


def apply_optional_60_hz_notch(
    lfp_values: np.ndarray,
    sample_rate_hz: float,
    enabled: bool = False,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Optionally apply a zero-phase 60 Hz IIR notch filter.

    Parameters
    ----------
    lfp_values : np.ndarray
        One-dimensional LFP vector with shape ``(n_samples,)`` in the input
        voltage units.
    sample_rate_hz : float
        LFP sample rate in Hz.
    enabled : bool, default=False
        Whether to apply the notch. The default returns a float copy unchanged.
    quality_factor : float, default=30.0
        Dimensionless notch quality factor.

    Returns
    -------
    np.ndarray
        Filtered LFP vector with shape ``(n_samples,)`` in unchanged input units.
    """

    values = np.asarray(lfp_values, dtype=float).reshape(-1)
    if not bool(enabled):
        return values.copy()
    if float(sample_rate_hz) <= 120.0:
        raise ValueError("A 60 Hz notch requires a sample rate above 120 Hz.")
    if float(quality_factor) <= 0:
        raise ValueError("quality_factor must be positive.")
    numerator, denominator = signal.iirnotch(
        w0=60.0,
        Q=float(quality_factor),
        fs=float(sample_rate_hz),
    )
    return np.asarray(signal.filtfilt(numerator, denominator, values), dtype=float)


def compute_morlet_log_power(
    relative_time_s: np.ndarray,
    lfp_values: np.ndarray,
    sample_rate_hz: float,
    frequencies_hz: np.ndarray,
    visible_window: tuple[float, float],
    gaussian_width: float = 1.5,
    window_length: float = 1.0,
    precision: int = 16,
    norm: str | None = "l1",
    target_sample_rate_hz: float = 500.0,
    notch_60_hz: bool = False,
    notch_quality_factor: float = 30.0,
) -> LFPSpectrogramResult:
    """
    Compute absolute Morlet log power for one padded trial LFP window.

    Parameters
    ----------
    relative_time_s : np.ndarray
        One-dimensional sample times with shape ``(n_samples,)`` in seconds
        relative to the trial alignment event. This input should include
        wavelet padding outside ``visible_window``.
    lfp_values : np.ndarray
        One-dimensional LFP vector with shape ``(n_samples,)`` in the source
        voltage units.
    sample_rate_hz : float
        Input sample rate in Hz.
    frequencies_hz : np.ndarray
        Positive frequencies with shape ``(n_frequencies,)`` in Hz.
    visible_window : tuple[float, float]
        Returned half-open time interval ``[start_s, end_s)`` in seconds.
    gaussian_width : float, default=1.5
        Pynapple Morlet Gaussian width, dimensionless.
    window_length : float, default=1.0
        Pynapple Morlet window-length parameter, dimensionless.
    precision : int, default=16
        Base-2 wavelet evaluation precision used by Pynapple.
    norm : str | None, default="l1"
        Pynapple wavelet normalization: ``"l1"``, ``"l2"``, or ``None``.
    target_sample_rate_hz : float, default=500.0
        Preferred post-decimation sample rate in Hz.
    notch_60_hz : bool, default=False
        Whether to apply a 60 Hz notch before decimation.
    notch_quality_factor : float, default=30.0
        Dimensionless quality factor for the optional notch.

    Returns
    -------
    LFPSpectrogramResult
        ``time_s`` and ``lfp_values`` have shape ``(n_visible_samples,)``;
        ``frequencies_hz`` has shape ``(n_frequencies,)``; ``log_power_db``
        has shape ``(n_visible_samples, n_frequencies)`` in dB relative to one
        squared source-unit; ``sample_rate_hz`` is the decimated rate in Hz.
    """

    time_s = np.asarray(relative_time_s, dtype=float).reshape(-1)
    values = np.asarray(lfp_values, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if time_s.shape != values.shape or time_s.size < 2:
        raise ValueError("relative_time_s and lfp_values must have matching nontrivial shapes.")
    if not np.isfinite(time_s).all() or not np.isfinite(values).all():
        raise ValueError("LFP times and values must be finite.")
    if frequencies.size == 0 or not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must contain finite positive values.")
    if len(visible_window) != 2 or float(visible_window[0]) >= float(visible_window[1]):
        raise ValueError("visible_window must contain increasing start and end times.")
    if norm not in {"l1", "l2", None}:
        raise ValueError("norm must be 'l1', 'l2', or None.")

    filtered_values = apply_optional_60_hz_notch(
        values,
        sample_rate_hz=float(sample_rate_hz),
        enabled=bool(notch_60_hz),
        quality_factor=float(notch_quality_factor),
    )
    lfp_tsd = nap.Tsd(t=time_s, d=filtered_values, time_units="s")
    decimation_factor = compute_decimation_factor(
        sample_rate_hz=float(sample_rate_hz),
        target_sample_rate_hz=float(target_sample_rate_hz),
        maximum_frequency_hz=float(np.max(frequencies)),
    )
    if decimation_factor > 1:
        lfp_tsd = lfp_tsd.decimate(down=int(decimation_factor))
    output_sample_rate_hz = float(sample_rate_hz) / float(decimation_factor)

    coefficients = nap.compute_wavelet_transform(
        lfp_tsd,
        frequencies,
        fs=output_sample_rate_hz,
        gaussian_width=float(gaussian_width),
        window_length=float(window_length),
        precision=int(precision),
        norm=norm,
    )
    coefficient_values = np.asarray(coefficients.values)
    power = np.abs(coefficient_values) ** 2
    log_power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    decimated_time_s = np.asarray(lfp_tsd.index, dtype=float).reshape(-1)
    visible_mask = (
        (decimated_time_s >= float(visible_window[0]))
        & (decimated_time_s < float(visible_window[1]))
    )
    if not visible_mask.any():
        raise ValueError("No decimated LFP samples fall inside visible_window.")
    return LFPSpectrogramResult(
        time_s=decimated_time_s[visible_mask],
        frequencies_hz=frequencies.copy(),
        log_power_db=np.asarray(log_power_db[visible_mask, :], dtype=float),
        lfp_values=np.asarray(lfp_tsd.values, dtype=float).reshape(-1)[visible_mask],
        sample_rate_hz=output_sample_rate_hz,
    )


def select_reference_trial_indices(
    trial_df: pd.DataFrame,
    alignment_event: str,
    maximum_trial_count: int = 24,
) -> np.ndarray:
    """
    Select deterministic, session-spanning valid trials for color scaling.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)`` and columns required
        by ``make_trial_type_masks`` plus ``alignment_event`` in seconds.
    alignment_event : str
        Trial timestamp column used for LFP alignment.
    maximum_trial_count : int, default=24
        Maximum number of valid reference trials.

    Returns
    -------
    np.ndarray
        Integer trial indices with shape ``(n_reference_trials,)``. Trials are
        evenly distributed through the valid session trials and deterministic.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    if int(maximum_trial_count) < 1:
        raise ValueError("maximum_trial_count must be positive.")
    valid_mask = make_trial_type_masks(trial_df)["valid"]
    alignment_times = pd.to_numeric(trial_df[alignment_event], errors="coerce")
    finite_alignment = alignment_times.replace([np.inf, -np.inf], np.nan).notna()
    valid_indices = trial_df.index[np.asarray(valid_mask & finite_alignment, dtype=bool)].to_numpy(dtype=int)
    if valid_indices.size <= int(maximum_trial_count):
        return valid_indices
    positions = np.linspace(0, valid_indices.size - 1, int(maximum_trial_count), dtype=int)
    return valid_indices[positions]


def estimate_shared_log_power_limits(
    log_power_arrays: list[np.ndarray] | tuple[np.ndarray, ...],
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> tuple[float, float]:
    """
    Estimate shared robust color limits from pooled absolute log power.

    Parameters
    ----------
    log_power_arrays : list[np.ndarray] | tuple[np.ndarray, ...]
        Trial power arrays, each with shape ``(n_times, n_frequencies)`` in dB
        relative to one squared source-unit.
    lower_percentile : float, default=2.0
        Lower pooled percentile in the closed interval ``[0, 100]``.
    upper_percentile : float, default=98.0
        Upper pooled percentile in the closed interval ``[0, 100]``.

    Returns
    -------
    tuple[float, float]
        Shared ``(minimum_db, maximum_db)`` color limits.
    """

    if not 0.0 <= float(lower_percentile) < float(upper_percentile) <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    finite_values = [
        np.asarray(array, dtype=float).reshape(-1)[np.isfinite(np.asarray(array, dtype=float).reshape(-1))]
        for array in log_power_arrays
    ]
    finite_values = [values for values in finite_values if values.size > 0]
    if not finite_values:
        raise ValueError("At least one finite log-power value is required.")
    lower, upper = np.percentile(
        np.concatenate(finite_values),
        [float(lower_percentile), float(upper_percentile)],
    )
    if np.isclose(lower, upper):
        padding = max(1.0, abs(float(lower)) * 0.01)
        lower -= padding
        upper += padding
    return float(lower), float(upper)
