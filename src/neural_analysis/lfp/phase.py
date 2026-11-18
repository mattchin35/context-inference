"""Continuous-wavelet phase tensors and trial-wise phase-clustering metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.lfp import spectrogram as lfp_spectrogram
from src.neural_analysis.spike_behavior import trials as spike_behavior_pynapple


ANALYSIS_VERSION = "0.2.0"
PHASE_TENSOR_AXIS_ORDER = ("site", "frequency", "trial", "time")
RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS = (
    "Off",
    "Per-frequency percentile",
    "Absolute magnitude",
)


@dataclass(frozen=True)
class ContinuousProcessingBlock:
    """One bounded continuous interval and the trials extracted from it."""

    trial_indices: np.ndarray
    event_times_s: np.ndarray
    core_start_s: float
    core_end_s: float
    load_start_s: float
    load_end_s: float


@dataclass(frozen=True)
class WaveletCoefficientResult:
    """Complex wavelet coefficients plus the unprocessed source-rate trace."""

    time_s: np.ndarray
    frequencies_hz: np.ndarray
    coefficients: np.ndarray
    sample_rate_hz: float
    source_time_s: np.ndarray
    source_lfp_values: np.ndarray
    source_sample_rate_hz: float


@dataclass(frozen=True)
class WaveletPhaseResult:
    """Complex unit phase for one continuous LFP site."""

    time_s: np.ndarray
    frequencies_hz: np.ndarray
    phase: np.ndarray
    valid: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class PhaseTrialTensor:
    """Event-aligned phase with explicit site/frequency/trial/time axes."""

    phase: np.ndarray
    valid: np.ndarray
    relative_time_s: np.ndarray
    trial_indices: np.ndarray
    excluded_trial_indices: np.ndarray


@dataclass(frozen=True)
class PhaseClusteringResult:
    """Bounded phase-clustering values and their effective trial counts."""

    values: np.ndarray
    effective_trial_count: np.ndarray
    n_trials: int


@dataclass(frozen=True)
class SingleTrialRelativePhaseResult:
    """One trial's relative phase, amplitude, axes, source traces, and metadata."""

    relative_phase_complex: np.ndarray
    phase_angle_rad: np.ndarray
    amplitude_a: np.ndarray
    amplitude_b: np.ndarray
    numerical_valid: np.ndarray
    support_relative_phase_complex: np.ndarray
    support_amplitude_a: np.ndarray
    support_amplitude_b: np.ndarray
    support_numerical_valid: np.ndarray
    support_relative_time_s: np.ndarray
    amplitude_percentiles_a: np.ndarray
    amplitude_percentiles_b: np.ndarray
    frequencies_hz: np.ndarray
    relative_time_s: np.ndarray
    source_time_a_s: np.ndarray
    source_lfp_a: np.ndarray
    source_time_b_s: np.ndarray
    source_lfp_b: np.ndarray
    trial_index: int
    event_time_s: float
    site_a_label: str
    site_b_label: str


@dataclass(frozen=True)
class WithinTrialPLVResult:
    """Local within-trial phase-locking values and window diagnostics."""

    plv: np.ndarray
    frequencies_hz: np.ndarray
    relative_time_s: np.ndarray
    window_cycles: float
    effective_window_s: np.ndarray
    window_sample_count: np.ndarray
    valid_sample_count: np.ndarray
    valid_fraction: np.ndarray
    trial_index: int
    site_a_label: str
    site_b_label: str


def normalize_wavelet_phase(
    coefficients: np.ndarray,
    minimum_relative_magnitude: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert complex wavelet coefficients to valid unit-phase vectors.

    Parameters
    ----------
    coefficients : np.ndarray
        Complex coefficient array of arbitrary shape. Axis meanings are
        preserved and coefficients may use any voltage-amplitude units.
    minimum_relative_magnitude : float, default=1e-12
        Nonnegative threshold relative to the largest finite coefficient
        magnitude. Coefficients at or below the threshold are invalid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(unit_phase, valid)`` with the same shape as ``coefficients``.
        ``unit_phase`` is ``complex64`` with magnitude one where valid and zero
        elsewhere. ``valid`` is boolean and has no physical units.
    """

    coefficient_array = np.asarray(coefficients)
    if not np.iscomplexobj(coefficient_array):
        coefficient_array = coefficient_array.astype(np.complex128)
    if float(minimum_relative_magnitude) < 0:
        raise ValueError("minimum_relative_magnitude must be nonnegative.")

    magnitudes = np.abs(coefficient_array)
    finite = np.isfinite(coefficient_array.real) & np.isfinite(coefficient_array.imag) & np.isfinite(magnitudes)
    largest_magnitude = float(np.max(magnitudes[finite])) if finite.any() else 0.0
    magnitude_threshold = float(minimum_relative_magnitude) * largest_magnitude
    valid = finite & (magnitudes > magnitude_threshold) & (magnitudes > 0.0)
    unit_phase = np.zeros(coefficient_array.shape, dtype=np.complex64)
    unit_phase[valid] = (coefficient_array[valid] / magnitudes[valid]).astype(np.complex64)
    return unit_phase, valid


def compute_wavelet_coefficients(
    time_s: np.ndarray,
    lfp_values: np.ndarray,
    sample_rate_hz: float,
    frequencies_hz: np.ndarray,
    gaussian_width: float = 1.5,
    window_length: float = 1.0,
    precision: int = 16,
    norm: str | None = "l1",
    target_sample_rate_hz: float = 500.0,
    notch_60_hz: bool = False,
    notch_quality_factor: float = 30.0,
) -> WaveletCoefficientResult:
    """
    Compute continuous Morlet coefficients while retaining the source trace.

    Parameters
    ----------
    time_s : np.ndarray
        Uniform one-dimensional sample times with shape ``(n_samples,)`` in
        seconds. Times may be recording-relative or absolute.
    lfp_values : np.ndarray
        One-dimensional LFP values with shape ``(n_samples,)`` in source units.
    sample_rate_hz : float
        Input sampling rate in Hz.
    frequencies_hz : np.ndarray
        Positive wavelet frequencies with shape ``(n_frequencies,)`` in Hz.
    gaussian_width : float, default=1.5
        Dimensionless Pynapple Morlet Gaussian width.
    window_length : float, default=1.0
        Dimensionless Pynapple Morlet window-length parameter.
    precision : int, default=16
        Base-2 Pynapple wavelet evaluation precision.
    norm : str | None, default="l1"
        Pynapple wavelet normalization: ``"l1"``, ``"l2"``, or ``None``.
    target_sample_rate_hz : float, default=500.0
        Preferred integer-decimated rate in Hz before the transform.
    notch_60_hz : bool, default=False
        Whether to apply the existing zero-phase 60 Hz notch before decimation.
    notch_quality_factor : float, default=30.0
        Dimensionless notch quality factor.
    Returns
    -------
    WaveletCoefficientResult
        ``coefficients`` has shape ``(n_frequencies, n_output_times)`` as
        ``complex64``. ``time_s`` is in seconds at the post-decimation rate.
        ``source_time_s`` and ``source_lfp_values`` preserve the unprocessed
        one-dimensional source-rate input and its original physical units.
    """

    input_time_s = np.asarray(time_s, dtype=float).reshape(-1)
    input_values = np.asarray(lfp_values, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if input_time_s.shape != input_values.shape or input_time_s.size < 2:
        raise ValueError("time_s and lfp_values must have matching nontrivial shapes.")
    if not np.isfinite(input_time_s).all() or not np.isfinite(input_values).all():
        raise ValueError("time_s and lfp_values must contain finite values.")
    if np.any(np.diff(input_time_s) <= 0):
        raise ValueError("time_s must be strictly increasing.")
    if frequencies.size == 0 or not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must contain finite positive values.")

    filtered_values = lfp_spectrogram.apply_optional_60_hz_notch(
        input_values,
        sample_rate_hz=float(sample_rate_hz),
        enabled=bool(notch_60_hz),
        quality_factor=float(notch_quality_factor),
    )
    continuous_tsd = nap.Tsd(t=input_time_s, d=filtered_values, time_units="s")
    decimation_factor = lfp_spectrogram.compute_decimation_factor(
        sample_rate_hz=float(sample_rate_hz),
        target_sample_rate_hz=float(target_sample_rate_hz),
        maximum_frequency_hz=float(np.max(frequencies)),
    )
    if decimation_factor > 1:
        continuous_tsd = continuous_tsd.decimate(down=int(decimation_factor))
    output_sample_rate_hz = float(sample_rate_hz) / float(decimation_factor)
    coefficients = nap.compute_wavelet_transform(
        continuous_tsd,
        frequencies,
        fs=output_sample_rate_hz,
        gaussian_width=float(gaussian_width),
        window_length=float(window_length),
        precision=int(precision),
        norm=norm,
    )
    coefficient_values = np.asarray(coefficients.values).T.astype(np.complex64)
    return WaveletCoefficientResult(
        time_s=np.asarray(continuous_tsd.index, dtype=float).reshape(-1),
        frequencies_hz=frequencies.copy(),
        coefficients=coefficient_values,
        sample_rate_hz=output_sample_rate_hz,
        source_time_s=input_time_s.copy(),
        source_lfp_values=input_values.copy(),
        source_sample_rate_hz=float(sample_rate_hz),
    )


def compute_wavelet_phase(
    time_s: np.ndarray,
    lfp_values: np.ndarray,
    sample_rate_hz: float,
    frequencies_hz: np.ndarray,
    gaussian_width: float = 1.5,
    window_length: float = 1.0,
    precision: int = 16,
    norm: str | None = "l1",
    target_sample_rate_hz: float = 500.0,
    notch_60_hz: bool = False,
    notch_quality_factor: float = 30.0,
    minimum_relative_magnitude: float = 1e-12,
) -> WaveletPhaseResult:
    """
    Compute continuous Morlet unit phase for one LFP site.

    Parameters
    ----------
    time_s : np.ndarray
        Uniform one-dimensional source times with shape ``(n_samples,)`` in
        seconds.
    lfp_values : np.ndarray
        One-dimensional LFP values with shape ``(n_samples,)`` in source units.
    sample_rate_hz : float
        Input sample rate in Hz.
    frequencies_hz : np.ndarray
        Positive frequencies with shape ``(n_frequencies,)`` in Hz.
    gaussian_width, window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int, default=16
        Base-2 Pynapple wavelet evaluation precision.
    norm : str | None, default="l1"
        Pynapple wavelet normalization.
    target_sample_rate_hz : float, default=500.0
        Preferred post-decimation sample rate in Hz.
    notch_60_hz : bool, default=False
        Whether to apply the existing zero-phase 60 Hz notch.
    notch_quality_factor : float, default=30.0
        Dimensionless notch-filter quality factor.
    minimum_relative_magnitude : float, default=1e-12
        Relative coefficient magnitude threshold for numerical validity.

    Returns
    -------
    WaveletPhaseResult
        ``phase`` and ``valid`` have shape ``(frequency, output_time)``;
        phase is unitless complex phase and output times are in seconds.
    """

    coefficient_result = compute_wavelet_coefficients(
        time_s=time_s,
        lfp_values=lfp_values,
        sample_rate_hz=float(sample_rate_hz),
        frequencies_hz=frequencies_hz,
        gaussian_width=float(gaussian_width),
        window_length=float(window_length),
        precision=int(precision),
        norm=norm,
        target_sample_rate_hz=float(target_sample_rate_hz),
        notch_60_hz=bool(notch_60_hz),
        notch_quality_factor=float(notch_quality_factor),
    )
    unit_phase, valid = normalize_wavelet_phase(
        coefficient_result.coefficients,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
    )
    return WaveletPhaseResult(
        time_s=coefficient_result.time_s.copy(),
        frequencies_hz=coefficient_result.frequencies_hz.copy(),
        phase=unit_phase,
        valid=valid,
        sample_rate_hz=coefficient_result.sample_rate_hz,
    )


def plan_continuous_processing_blocks(
    event_times_s: np.ndarray,
    trial_indices: np.ndarray,
    window: tuple[float, float],
    wavelet_padding_s: float,
    maximum_core_duration_s: float,
) -> list[ContinuousProcessingBlock]:
    """
    Group chronological trial windows into bounded continuous transform blocks.

    Parameters
    ----------
    event_times_s : np.ndarray
        Absolute alignment times with shape ``(n_trials,)`` in seconds.
    trial_indices : np.ndarray
        Integer trial identifiers with shape ``(n_trials,)``.
    window : tuple[float, float]
        Event-relative extraction bounds in seconds.
    wavelet_padding_s : float
        Seconds loaded on both sides of each block for Morlet edge support.
    maximum_core_duration_s : float
        Maximum unpadded span in seconds represented by one block.

    Returns
    -------
    list[ContinuousProcessingBlock]
        Chronological blocks. Each trial occurs in exactly one block; load
        bounds include wavelet padding and use absolute seconds.
    """

    event_times = np.asarray(event_times_s, dtype=float).reshape(-1)
    indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if event_times.shape != indices.shape:
        raise ValueError("event_times_s and trial_indices must have matching shapes.")
    if event_times.size == 0:
        return []
    if not np.isfinite(event_times).all() or np.any(np.diff(event_times) < 0):
        raise ValueError("event_times_s must be finite and chronologically ordered.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must contain increasing bounds.")
    if float(wavelet_padding_s) < 0 or float(maximum_core_duration_s) <= 0:
        raise ValueError("Padding must be nonnegative and maximum_core_duration_s must be positive.")

    blocks: list[ContinuousProcessingBlock] = []
    block_start_position = 0
    while block_start_position < event_times.size:
        block_stop_position = block_start_position + 1
        while block_stop_position < event_times.size:
            candidate_core_start = event_times[block_start_position] + float(window[0])
            candidate_core_end = event_times[block_stop_position] + float(window[1])
            if candidate_core_end - candidate_core_start > float(maximum_core_duration_s):
                break
            block_stop_position += 1
        block_events = event_times[block_start_position:block_stop_position]
        block_indices = indices[block_start_position:block_stop_position]
        core_start_s = float(block_events[0] + float(window[0]))
        core_end_s = float(block_events[-1] + float(window[1]))
        blocks.append(
            ContinuousProcessingBlock(
                trial_indices=block_indices.copy(),
                event_times_s=block_events.copy(),
                core_start_s=core_start_s,
                core_end_s=core_end_s,
                load_start_s=core_start_s - float(wavelet_padding_s),
                load_end_s=core_end_s + float(wavelet_padding_s),
            )
        )
        block_start_position = block_stop_position
    return blocks


def make_phase_trial_tensor(
    coefficient_times_s: np.ndarray,
    unit_phase: np.ndarray,
    event_times_s: np.ndarray,
    trial_indices: np.ndarray,
    window: tuple[float, float],
    output_sample_rate_hz: float,
    edge_buffer_s: float = 0.0,
) -> PhaseTrialTensor:
    """
    Interpolate continuous unit phase onto a common event-relative trial grid.

    Parameters
    ----------
    coefficient_times_s : np.ndarray
        Strictly increasing continuous coefficient times with shape
        ``(n_source_times,)`` in seconds.
    unit_phase : np.ndarray
        Complex phase vectors with shape
        ``(n_sites, n_frequencies, n_source_times)``.
    event_times_s : np.ndarray
        Absolute alignment times with shape ``(n_trials,)`` in seconds.
    trial_indices : np.ndarray
        Integer source-table trial indices with shape ``(n_trials,)``.
    window : tuple[float, float]
        Half-open relative output interval in seconds.
    output_sample_rate_hz : float
        Exact common output grid rate in Hz.
    edge_buffer_s : float, default=0.0
        Additional continuous support required outside each trial interval in
        seconds. Trials lacking this support are excluded completely.

    Returns
    -------
    PhaseTrialTensor
        ``phase`` and ``valid`` use axis order
        ``(site, frequency, trial, time)``. Phase is ``complex64`` and relative
        time is in seconds. Excluded source trial indices are reported.
    """

    source_times = np.asarray(coefficient_times_s, dtype=float).reshape(-1)
    phase_values = np.asarray(unit_phase)
    events = np.asarray(event_times_s, dtype=float).reshape(-1)
    indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if phase_values.ndim != 3 or phase_values.shape[2] != source_times.size:
        raise ValueError("unit_phase must have shape (site, frequency, source_time).")
    if events.shape != indices.shape:
        raise ValueError("event_times_s and trial_indices must have matching shapes.")
    if source_times.size < 2 or not np.isfinite(source_times).all() or np.any(np.diff(source_times) <= 0):
        raise ValueError("coefficient_times_s must be finite and strictly increasing.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must contain increasing bounds.")
    if float(output_sample_rate_hz) <= 0 or float(edge_buffer_s) < 0:
        raise ValueError("output_sample_rate_hz must be positive and edge_buffer_s nonnegative.")

    output_sample_count = int(np.round((float(window[1]) - float(window[0])) * float(output_sample_rate_hz)))
    if output_sample_count < 1:
        raise ValueError("The requested window contains no output samples.")
    relative_time_s = float(window[0]) + np.arange(output_sample_count, dtype=float) / float(output_sample_rate_hz)
    source_step_s = float(np.median(np.diff(source_times)))
    included_positions = []
    excluded_positions = []
    for trial_position, event_time_s in enumerate(events):
        required_start_s = float(event_time_s) + float(window[0]) - float(edge_buffer_s)
        required_end_s = float(event_time_s) + float(window[1]) + float(edge_buffer_s)
        source_end_s = float(source_times[-1]) + source_step_s
        if (
            not np.isfinite(event_time_s)
            or required_start_s < float(source_times[0])
            or required_end_s > source_end_s
        ):
            excluded_positions.append(trial_position)
        else:
            included_positions.append(trial_position)

    output_phase = np.zeros(
        (phase_values.shape[0], phase_values.shape[1], len(included_positions), output_sample_count),
        dtype=np.complex64,
    )
    output_valid = np.zeros(output_phase.shape, dtype=bool)
    for output_trial_position, source_trial_position in enumerate(included_positions):
        target_times_s = float(events[source_trial_position]) + relative_time_s
        for site_index in range(phase_values.shape[0]):
            for frequency_index in range(phase_values.shape[1]):
                source_vector = phase_values[site_index, frequency_index]
                source_valid = (
                    np.isfinite(source_vector.real)
                    & np.isfinite(source_vector.imag)
                    & (np.abs(source_vector) > 0.0)
                )
                if source_valid.sum() < 2:
                    continue
                interpolated = np.interp(
                    target_times_s,
                    source_times[source_valid],
                    source_vector.real[source_valid],
                ) + 1j * np.interp(
                    target_times_s,
                    source_times[source_valid],
                    source_vector.imag[source_valid],
                )
                normalized, valid = normalize_wavelet_phase(interpolated, minimum_relative_magnitude=0.0)
                output_phase[site_index, frequency_index, output_trial_position] = normalized
                output_valid[site_index, frequency_index, output_trial_position] = valid
    return PhaseTrialTensor(
        phase=output_phase,
        valid=output_valid,
        relative_time_s=relative_time_s,
        trial_indices=indices[np.asarray(included_positions, dtype=int)],
        excluded_trial_indices=indices[np.asarray(excluded_positions, dtype=int)],
    )


def compute_single_trial_relative_phase(
    site_a: WaveletCoefficientResult,
    site_b: WaveletCoefficientResult,
    event_time_s: float,
    visible_window: tuple[float, float],
    output_sample_rate_hz: float,
    trial_index: int,
    site_a_label: str,
    site_b_label: str,
    minimum_relative_magnitude: float = 1e-12,
    support_window: tuple[float, float] | None = None,
) -> SingleTrialRelativePhaseResult:
    """
    Compare two padded wavelet results on one exact event-relative grid.

    Parameters
    ----------
    site_a, site_b : WaveletCoefficientResult
        Independently synchronized padded transforms. Coefficients use shape
        ``(frequency, transformed_time)`` and times use absolute seconds.
    event_time_s : float
        Absolute behavioral alignment time in seconds.
    visible_window : tuple[float, float]
        Returned half-open event-relative interval in seconds.
    output_sample_rate_hz : float
        Exact comparison-grid sample rate in Hz.
    trial_index : int
        Source trial-table row identifier.
    site_a_label, site_b_label : str
        Human-readable probe/channel labels defining the ordered A-minus-B
        phase difference.
    minimum_relative_magnitude : float, default=1e-12
        Numerical-validity threshold relative to each site's largest padded
        coefficient magnitude.
    support_window : tuple[float, float] | None, default=None
        Half-open event-relative interval in seconds retained for centered
        downstream calculations. It must contain ``visible_window``. ``None``
        retains only the visible interval.

    Returns
    -------
    SingleTrialRelativePhaseResult
        Frequency-by-time relative complex phase, angle in radians, amplitudes,
        validity, percentile references, exact axes, cropped unprocessed source
        traces, and identifying metadata. Support arrays use shape
        ``(frequency, support_time)``; visible arrays use shape
        ``(frequency, visible_time)``. Relative phase is
        ``A * conjugate(B)``.
    """

    if not np.isfinite(float(event_time_s)):
        raise ValueError("event_time_s must be finite.")
    if len(visible_window) != 2 or float(visible_window[0]) >= float(visible_window[1]):
        raise ValueError("visible_window must contain increasing bounds.")
    if float(output_sample_rate_hz) <= 0:
        raise ValueError("output_sample_rate_hz must be positive.")
    if not np.array_equal(site_a.frequencies_hz, site_b.frequencies_hz):
        raise ValueError("Both sites must use an identical frequency vector.")

    support_bounds = visible_window if support_window is None else support_window
    if len(support_bounds) != 2 or float(support_bounds[0]) >= float(support_bounds[1]):
        raise ValueError("support_window must contain increasing bounds.")
    if float(support_bounds[0]) > float(visible_window[0]) or float(support_bounds[1]) < float(
        visible_window[1]
    ):
        raise ValueError("support_window must contain visible_window.")
    output_sample_count = int(
        np.round((float(support_bounds[1]) - float(support_bounds[0])) * float(output_sample_rate_hz))
    )
    if output_sample_count < 1:
        raise ValueError("visible_window contains no output samples.")
    support_relative_time_s = float(support_bounds[0]) + np.arange(output_sample_count) / float(
        output_sample_rate_hz
    )
    absolute_time_s = float(event_time_s) + support_relative_time_s

    phase_a, support_amplitude_a, valid_a = _interpolate_site_coefficients(
        site_a,
        absolute_time_s,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
    )
    phase_b, support_amplitude_b, valid_b = _interpolate_site_coefficients(
        site_b,
        absolute_time_s,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
    )
    support_numerical_valid = valid_a & valid_b
    support_relative_phase = phase_a * np.conjugate(phase_b)
    support_relative_phase[~support_numerical_valid] = 0.0j
    relative_phase_magnitude = np.abs(support_relative_phase)
    np.divide(
        support_relative_phase,
        relative_phase_magnitude,
        out=support_relative_phase,
        where=support_numerical_valid & (relative_phase_magnitude > 0.0),
    )
    visible_positions = (support_relative_time_s >= float(visible_window[0])) & (
        support_relative_time_s < float(visible_window[1])
    )
    expected_visible_count = int(
        np.round((float(visible_window[1]) - float(visible_window[0])) * float(output_sample_rate_hz))
    )
    if int(np.sum(visible_positions)) != expected_visible_count:
        raise ValueError("support_window does not produce the requested exact visible grid.")
    relative_time_s = support_relative_time_s[visible_positions]
    relative_phase = support_relative_phase[:, visible_positions]
    amplitude_a = support_amplitude_a[:, visible_positions]
    amplitude_b = support_amplitude_b[:, visible_positions]
    numerical_valid = support_numerical_valid[:, visible_positions]
    phase_angle = np.angle(relative_phase).astype(np.float32)
    phase_angle[~numerical_valid] = np.nan

    source_time_a_s, source_lfp_a = _crop_source_trace(site_a, float(event_time_s), visible_window)
    source_time_b_s, source_lfp_b = _crop_source_trace(site_b, float(event_time_s), visible_window)
    percentile_axis = np.arange(101, dtype=float)
    amplitude_percentiles_a = np.percentile(np.abs(site_a.coefficients), percentile_axis, axis=1).T
    amplitude_percentiles_b = np.percentile(np.abs(site_b.coefficients), percentile_axis, axis=1).T
    return SingleTrialRelativePhaseResult(
        relative_phase_complex=relative_phase.astype(np.complex64),
        phase_angle_rad=phase_angle,
        amplitude_a=amplitude_a.astype(np.float32),
        amplitude_b=amplitude_b.astype(np.float32),
        numerical_valid=numerical_valid,
        support_relative_phase_complex=support_relative_phase.astype(np.complex64),
        support_amplitude_a=support_amplitude_a.astype(np.float32),
        support_amplitude_b=support_amplitude_b.astype(np.float32),
        support_numerical_valid=support_numerical_valid,
        support_relative_time_s=support_relative_time_s,
        amplitude_percentiles_a=amplitude_percentiles_a.astype(np.float32),
        amplitude_percentiles_b=amplitude_percentiles_b.astype(np.float32),
        frequencies_hz=np.asarray(site_a.frequencies_hz, dtype=float).copy(),
        relative_time_s=relative_time_s,
        source_time_a_s=source_time_a_s,
        source_lfp_a=source_lfp_a,
        source_time_b_s=source_time_b_s,
        source_lfp_b=source_lfp_b,
        trial_index=int(trial_index),
        event_time_s=float(event_time_s),
        site_a_label=str(site_a_label),
        site_b_label=str(site_b_label),
    )


def make_relative_phase_display_mask(
    result: SingleTrialRelativePhaseResult,
    mode: str = "Off",
    percentile: float = 10.0,
    absolute_threshold_a: float = 0.0,
    absolute_threshold_b: float = 0.0,
) -> np.ndarray:
    """
    Build a display mask from numerical validity and optional amplitude rules.

    Parameters
    ----------
    result : SingleTrialRelativePhaseResult
        Single-trial arrays with shape ``(frequency, time)`` and coefficient
        amplitudes in each site's source-dependent wavelet units.
    mode : str, default="Off"
        One of ``RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS``.
    percentile : float, default=10.0
        Padded-segment per-frequency percentile in ``[0, 100]`` used for both
        sites in percentile mode.
    absolute_threshold_a, absolute_threshold_b : float, default=0.0
        Nonnegative site-specific coefficient magnitude thresholds used in
        absolute mode.

    Returns
    -------
    np.ndarray
        Boolean display mask with shape ``(frequency, time)``. A pixel is valid
        only when both sites satisfy numerical and selected amplitude criteria.
    """

    if mode not in RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS:
        raise ValueError(f"Unknown relative-phase amplitude mask mode: {mode!r}.")
    display_valid = np.asarray(result.numerical_valid, dtype=bool).copy()
    if mode == "Off":
        return display_valid
    if mode == "Per-frequency percentile":
        if not 0.0 <= float(percentile) <= 100.0:
            raise ValueError("percentile must be in [0, 100].")
        percentile_axis = np.arange(101, dtype=float)
        threshold_a = np.asarray(
            [np.interp(float(percentile), percentile_axis, row) for row in result.amplitude_percentiles_a]
        )
        threshold_b = np.asarray(
            [np.interp(float(percentile), percentile_axis, row) for row in result.amplitude_percentiles_b]
        )
        return display_valid & (result.amplitude_a >= threshold_a[:, np.newaxis]) & (
            result.amplitude_b >= threshold_b[:, np.newaxis]
        )
    if float(absolute_threshold_a) < 0.0 or float(absolute_threshold_b) < 0.0:
        raise ValueError("Absolute amplitude thresholds must be nonnegative.")
    return display_valid & (result.amplitude_a >= float(absolute_threshold_a)) & (
        result.amplitude_b >= float(absolute_threshold_b)
    )


def make_relative_phase_support_mask(
    result: SingleTrialRelativePhaseResult,
    mode: str = "Off",
    percentile: float = 10.0,
    absolute_threshold_a: float = 0.0,
    absolute_threshold_b: float = 0.0,
) -> np.ndarray:
    """
    Build an amplitude-validity mask over the retained padded support interval.

    Parameters
    ----------
    result : SingleTrialRelativePhaseResult
        Support arrays with shape ``(frequency, support_time)``. Time is in
        event-relative seconds and amplitudes use source wavelet units.
    mode : str, default="Off"
        One of ``RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS``.
    percentile : float, default=10.0
        Per-frequency padded-segment percentile in ``[0, 100]``.
    absolute_threshold_a, absolute_threshold_b : float, default=0.0
        Nonnegative site-specific wavelet-magnitude thresholds.

    Returns
    -------
    np.ndarray
        Boolean array with shape ``(frequency, support_time)``.
    """

    if mode not in RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS:
        raise ValueError(f"Unknown relative-phase amplitude mask mode: {mode!r}.")
    valid = np.asarray(result.support_numerical_valid, dtype=bool).copy()
    if mode == "Off":
        return valid
    if mode == "Per-frequency percentile":
        if not 0.0 <= float(percentile) <= 100.0:
            raise ValueError("percentile must be in [0, 100].")
        percentile_axis = np.arange(101, dtype=float)
        threshold_a = np.asarray(
            [np.interp(float(percentile), percentile_axis, row) for row in result.amplitude_percentiles_a]
        )
        threshold_b = np.asarray(
            [np.interp(float(percentile), percentile_axis, row) for row in result.amplitude_percentiles_b]
        )
        return valid & (result.support_amplitude_a >= threshold_a[:, np.newaxis]) & (
            result.support_amplitude_b >= threshold_b[:, np.newaxis]
        )
    if float(absolute_threshold_a) < 0.0 or float(absolute_threshold_b) < 0.0:
        raise ValueError("Absolute amplitude thresholds must be nonnegative.")
    return valid & (result.support_amplitude_a >= float(absolute_threshold_a)) & (
        result.support_amplitude_b >= float(absolute_threshold_b)
    )


def compute_plv_window_samples(
    frequencies_hz: np.ndarray,
    sample_rate_hz: float,
    window_cycles: float,
    min_window_s: float | None = None,
    max_window_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert cycle-defined PLV windows to centered odd sample counts.

    Parameters
    ----------
    frequencies_hz : np.ndarray
        One-dimensional positive frequency vector in Hz.
    sample_rate_hz : float
        Positive common phase-grid sample rate in samples/second.
    window_cycles : float
        Positive number of oscillatory cycles in each local window.
    min_window_s, max_window_s : float | None, default=None
        Optional positive duration bounds in seconds, applied before sample
        conversion. Bounds are independent; when both are supplied the minimum
        cannot exceed the maximum.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer odd sample counts and their effective durations in seconds,
        both with shape ``(frequency,)``.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if frequencies.size == 0 or np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain positive finite values.")
    if not np.isfinite(float(sample_rate_hz)) or float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite.")
    if not np.isfinite(float(window_cycles)) or float(window_cycles) <= 0.0:
        raise ValueError("window_cycles must be positive and finite.")
    if min_window_s is not None and float(min_window_s) <= 0.0:
        raise ValueError("min_window_s must be positive when supplied.")
    if max_window_s is not None and float(max_window_s) <= 0.0:
        raise ValueError("max_window_s must be positive when supplied.")
    if min_window_s is not None and max_window_s is not None and float(min_window_s) > float(max_window_s):
        raise ValueError("min_window_s cannot exceed max_window_s.")

    durations_s = float(window_cycles) / frequencies
    if min_window_s is not None:
        durations_s = np.maximum(durations_s, float(min_window_s))
    if max_window_s is not None:
        durations_s = np.minimum(durations_s, float(max_window_s))
    sample_counts = np.maximum(1, np.rint(durations_s * float(sample_rate_hz)).astype(int))
    sample_counts += (sample_counts % 2 == 0).astype(int)
    return sample_counts, sample_counts.astype(float) / float(sample_rate_hz)


def compute_within_trial_plv(
    relative_phase_complex: np.ndarray,
    valid_mask: np.ndarray,
    frequencies_hz: np.ndarray,
    relative_time_s: np.ndarray,
    visible_window: tuple[float, float],
    window_cycles: float = 3.0,
    min_window_s: float | None = None,
    max_window_s: float | None = None,
    min_valid_fraction: float = 0.8,
    step_samples: int = 1,
    trial_index: int = -1,
    site_a_label: str = "A",
    site_b_label: str = "B",
) -> WithinTrialPLVResult:
    """
    Compute centered local phase stability within one padded trial segment.

    Parameters
    ----------
    relative_phase_complex : np.ndarray
        Unit complex A-minus-B phase vectors with shape
        ``(frequency, support_time)``.
    valid_mask : np.ndarray
        Boolean array matching ``relative_phase_complex``. Invalid samples are
        excluded without amplitude weighting.
    frequencies_hz : np.ndarray
        Positive frequencies in Hz with shape ``(frequency,)``.
    relative_time_s : np.ndarray
        Strictly increasing, uniformly sampled support times in seconds with
        shape ``(support_time,)``.
    visible_window : tuple[float, float]
        Returned half-open event-relative interval in seconds. Complete windows
        may use support samples outside this interval.
    window_cycles : float, default=3.0
        Number of cycles included in each frequency-specific centered window.
    min_window_s, max_window_s : float | None, default=None
        Optional duration bounds in seconds.
    min_valid_fraction : float, default=0.8
        Required fraction of the complete requested window in ``[0, 1]``.
    step_samples : int, default=1
        Positive evaluation stride on the support grid.
    trial_index : int, default=-1
        Source trial-table row identifier.
    site_a_label, site_b_label : str, default="A", "B"
        Ordered site labels for the underlying A-minus-B phase vectors.

    Returns
    -------
    WithinTrialPLVResult
        ``plv``, valid counts, and valid fractions have shape
        ``(frequency, visible_time)``. PLV is dimensionless float32 in
        ``[0, 1]`` or NaN when a full window is unavailable or insufficiently
        valid. Window metadata has shape ``(frequency,)``.

    Notes
    -----
    Short-window PLV has finite-sample upward bias. This function provides no
    significance threshold and is intended for within-trial inspection.
    """

    phase = np.asarray(relative_phase_complex)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    times = np.asarray(relative_time_s, dtype=float).reshape(-1)
    if phase.ndim != 2 or phase.shape != valid.shape:
        raise ValueError("relative_phase_complex and valid_mask must share shape (frequency, support_time).")
    if phase.shape != (frequencies.size, times.size):
        raise ValueError("Phase arrays must match the frequency and support-time axes.")
    if times.size < 2:
        raise ValueError("relative_time_s must contain at least two samples.")
    time_steps = np.diff(times)
    if np.any(time_steps <= 0.0) or not np.allclose(time_steps, time_steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError("relative_time_s must be strictly increasing and uniformly sampled.")
    if len(visible_window) != 2 or float(visible_window[0]) >= float(visible_window[1]):
        raise ValueError("visible_window must contain increasing bounds.")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1].")
    if int(step_samples) != step_samples or int(step_samples) < 1:
        raise ValueError("step_samples must be a positive integer.")

    sample_rate_hz = 1.0 / float(time_steps[0])
    window_sample_count, effective_window_s = compute_plv_window_samples(
        frequencies_hz=frequencies,
        sample_rate_hz=sample_rate_hz,
        window_cycles=float(window_cycles),
        min_window_s=min_window_s,
        max_window_s=max_window_s,
    )
    finite_phase = np.isfinite(phase.real) & np.isfinite(phase.imag) & (np.abs(phase) > 0.0)
    valid &= finite_phase
    unit_phase = np.zeros(phase.shape, dtype=np.complex128)
    unit_phase[valid] = phase[valid] / np.abs(phase[valid])
    plv_support = np.full(phase.shape, np.nan, dtype=np.float32)
    count_support = np.zeros(phase.shape, dtype=np.int32)

    centers = np.arange(times.size)
    for frequency_index, sample_count in enumerate(window_sample_count):
        half_window = int(sample_count) // 2
        starts = centers - half_window
        stops = centers + half_window + 1
        complete = (starts >= 0) & (stops <= times.size)
        valid_centers = centers[complete]
        valid_starts = starts[complete]
        valid_stops = stops[complete]
        phase_prefix = np.concatenate(([0.0j], np.cumsum(unit_phase[frequency_index])))
        count_prefix = np.concatenate(([0], np.cumsum(valid[frequency_index], dtype=np.int64)))
        local_sums = phase_prefix[valid_stops] - phase_prefix[valid_starts]
        local_counts = count_prefix[valid_stops] - count_prefix[valid_starts]
        accepted = (local_counts / float(sample_count)) >= float(min_valid_fraction)
        accepted &= local_counts > 0
        accepted_centers = valid_centers[accepted]
        plv_support[frequency_index, accepted_centers] = (
            np.abs(local_sums[accepted] / local_counts[accepted]).astype(np.float32)
        )
        count_support[frequency_index, valid_centers] = local_counts.astype(np.int32)

    visible_positions = np.flatnonzero(
        (times >= float(visible_window[0])) & (times < float(visible_window[1]))
    )[:: int(step_samples)]
    if visible_positions.size == 0:
        raise ValueError("visible_window contains no support-grid samples.")
    visible_counts = count_support[:, visible_positions]
    return WithinTrialPLVResult(
        plv=plv_support[:, visible_positions],
        frequencies_hz=frequencies.copy(),
        relative_time_s=times[visible_positions].copy(),
        window_cycles=float(window_cycles),
        effective_window_s=effective_window_s,
        window_sample_count=window_sample_count,
        valid_sample_count=visible_counts,
        valid_fraction=visible_counts.astype(np.float32) / window_sample_count[:, np.newaxis],
        trial_index=int(trial_index),
        site_a_label=str(site_a_label),
        site_b_label=str(site_b_label),
    )


def _interpolate_site_coefficients(
    result: WaveletCoefficientResult,
    target_time_s: np.ndarray,
    minimum_relative_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate one padded coefficient matrix to absolute target seconds."""

    source_time_s = np.asarray(result.time_s, dtype=float).reshape(-1)
    coefficients = np.asarray(result.coefficients)
    target = np.asarray(target_time_s, dtype=float).reshape(-1)
    if coefficients.shape != (result.frequencies_hz.size, source_time_s.size):
        raise ValueError("coefficients must have shape (frequency, transformed_time).")
    if source_time_s.size < 2 or np.any(np.diff(source_time_s) <= 0):
        raise ValueError("Wavelet coefficient times must be strictly increasing.")
    if target[0] < source_time_s[0] or target[-1] > source_time_s[-1]:
        raise ValueError("Padded wavelet coefficients do not cover the requested visible window.")

    source_phase, source_valid = normalize_wavelet_phase(
        coefficients,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
    )
    output_phase = np.zeros((coefficients.shape[0], target.size), dtype=np.complex64)
    output_amplitude = np.zeros((coefficients.shape[0], target.size), dtype=np.float32)
    output_valid = np.zeros((coefficients.shape[0], target.size), dtype=bool)
    for frequency_index in range(coefficients.shape[0]):
        valid = source_valid[frequency_index]
        if valid.sum() < 2:
            continue
        interpolated_phase = np.interp(
            target,
            source_time_s[valid],
            source_phase[frequency_index, valid].real,
        ) + 1j * np.interp(
            target,
            source_time_s[valid],
            source_phase[frequency_index, valid].imag,
        )
        normalized, interpolated_valid = normalize_wavelet_phase(
            interpolated_phase,
            minimum_relative_magnitude=0.0,
        )
        output_phase[frequency_index] = normalized
        output_valid[frequency_index] = interpolated_valid
        output_amplitude[frequency_index] = np.interp(
            target,
            source_time_s[valid],
            np.abs(coefficients[frequency_index, valid]),
        ).astype(np.float32)
    return output_phase, output_amplitude, output_valid


def _crop_source_trace(
    result: WaveletCoefficientResult,
    event_time_s: float,
    visible_window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an unprocessed source-rate trace and convert its time to event-relative seconds."""

    source_time_s = np.asarray(result.source_time_s, dtype=float).reshape(-1)
    source_lfp = np.asarray(result.source_lfp_values, dtype=float).reshape(-1)
    if source_time_s.shape != source_lfp.shape:
        raise ValueError("Source LFP times and values must have matching shapes.")
    relative_time_s = source_time_s - float(event_time_s)
    visible = (relative_time_s >= float(visible_window[0])) & (relative_time_s < float(visible_window[1]))
    if not visible.any():
        raise ValueError("Source trace contains no samples in the visible window.")
    return relative_time_s[visible], source_lfp[visible]


def compute_itpc(
    phase_tensor: np.ndarray,
    trial_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> PhaseClusteringResult:
    """
    Calculate inter-trial phase clustering across the trial axis.

    Parameters
    ----------
    phase_tensor : np.ndarray
        Complex unit phase with shape ``(site, frequency, trial, time)``.
    trial_mask : np.ndarray | None, optional
        Boolean trial selector with shape ``(n_trials,)``. ``None`` selects all.
    valid_mask : np.ndarray | None, optional
        Boolean validity array with the same shape as ``phase_tensor``. When
        omitted, finite nonzero vectors are considered valid.

    Returns
    -------
    PhaseClusteringResult
        Values and effective counts have shape ``(site, frequency, time)``.
        Values are dimensionless in ``[0, 1]`` and ``n_trials`` is the nominal
        number selected before per-pixel validity exclusions.
    """

    phase, selected, valid = _validate_phase_clustering_inputs(phase_tensor, trial_mask, valid_mask)
    selected_indices = np.flatnonzero(selected)
    selected_phase = np.take(phase, selected_indices, axis=2)
    selected_valid = np.take(valid, selected_indices, axis=2)
    selected_magnitude = np.abs(selected_phase)
    normalized_phase = np.zeros(selected_phase.shape, dtype=np.complex128)
    np.divide(selected_phase, selected_magnitude, out=normalized_phase, where=selected_valid)
    vector_sum = np.sum(np.where(selected_valid, normalized_phase, 0.0j), axis=2)
    effective_count = np.sum(selected_valid, axis=2, dtype=int)
    values = np.zeros(effective_count.shape, dtype=float)
    np.divide(np.abs(vector_sum), effective_count, out=values, where=effective_count > 0)
    return PhaseClusteringResult(
        values=np.clip(values, 0.0, 1.0),
        effective_trial_count=effective_count,
        n_trials=int(np.sum(selected)),
    )


def combine_phase_trial_tensors(tensors: list[PhaseTrialTensor]) -> PhaseTrialTensor:
    """
    Stack independently preprocessed sites on their common retained trials.

    Parameters
    ----------
    tensors : list[PhaseTrialTensor]
        Nonempty list of site tensors. Each input must have one site, matching
        frequency/time dimensions, matching relative times in seconds, and
        unique integer trial indices.

    Returns
    -------
    PhaseTrialTensor
        Combined tensor with shape ``(n_sites, frequency, common_trial, time)``.
        Trial order follows the first tensor. Excluded indices include explicit
        source exclusions and trials absent from any site's intersection.
    """

    if not tensors:
        raise ValueError("At least one phase trial tensor is required.")
    first = tensors[0]
    if first.phase.ndim != 4 or first.phase.shape[0] != 1 or first.valid.shape != first.phase.shape:
        raise ValueError("Each input tensor must contain exactly one site with matching validity shape.")
    common_trial_set = set(np.asarray(first.trial_indices, dtype=int).tolist())
    all_retained_trials = set(common_trial_set)
    all_excluded_trials = set(np.asarray(first.excluded_trial_indices, dtype=int).tolist())
    for tensor in tensors[1:]:
        if tensor.phase.ndim != 4 or tensor.phase.shape[0] != 1 or tensor.valid.shape != tensor.phase.shape:
            raise ValueError("Each input tensor must contain exactly one site with matching validity shape.")
        if tensor.phase.shape[1] != first.phase.shape[1] or tensor.phase.shape[3] != first.phase.shape[3]:
            raise ValueError("All site tensors must have matching frequency and time dimensions.")
        if not np.allclose(tensor.relative_time_s, first.relative_time_s):
            raise ValueError("All site tensors must use the same relative time grid.")
        tensor_trials = set(np.asarray(tensor.trial_indices, dtype=int).tolist())
        if len(tensor_trials) != tensor.trial_indices.size:
            raise ValueError("Each site tensor must contain unique trial indices.")
        common_trial_set &= tensor_trials
        all_retained_trials |= tensor_trials
        all_excluded_trials |= set(np.asarray(tensor.excluded_trial_indices, dtype=int).tolist())
    common_trial_indices = np.asarray(
        [trial_index for trial_index in first.trial_indices if int(trial_index) in common_trial_set],
        dtype=int,
    )
    if common_trial_indices.size == 0:
        raise ValueError("Selected sites have no common retained trials.")

    phase_by_site = []
    valid_by_site = []
    for tensor in tensors:
        position_by_trial = {
            int(trial_index): position for position, trial_index in enumerate(tensor.trial_indices)
        }
        positions = np.asarray([position_by_trial[int(index)] for index in common_trial_indices], dtype=int)
        phase_by_site.append(np.take(tensor.phase, positions, axis=2))
        valid_by_site.append(np.take(tensor.valid, positions, axis=2))
    all_excluded_trials |= all_retained_trials - set(common_trial_indices.tolist())
    return PhaseTrialTensor(
        phase=np.concatenate(phase_by_site, axis=0),
        valid=np.concatenate(valid_by_site, axis=0),
        relative_time_s=first.relative_time_s.copy(),
        trial_indices=common_trial_indices,
        excluded_trial_indices=np.asarray(sorted(all_excluded_trials), dtype=int),
    )


def select_tensor_trial_mask(
    tensor_trial_indices: np.ndarray,
    full_trial_mask: np.ndarray,
) -> np.ndarray:
    """
    Map a full trial-table condition mask onto a retained tensor trial axis.

    Parameters
    ----------
    tensor_trial_indices : np.ndarray
        Integer source-table indices with shape ``(n_tensor_trials,)``.
    full_trial_mask : np.ndarray
        Boolean condition mask with shape ``(n_table_trials,)``.

    Returns
    -------
    np.ndarray
        Boolean selector with shape ``(n_tensor_trials,)`` in tensor order.
    """

    indices = np.asarray(tensor_trial_indices, dtype=int).reshape(-1)
    mask = np.asarray(full_trial_mask, dtype=bool).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= mask.size):
        raise ValueError("tensor_trial_indices must index full_trial_mask.")
    return mask[indices]


def compute_ispc(
    phase_tensor: np.ndarray,
    site_a: int,
    site_b: int,
    trial_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> PhaseClusteringResult:
    """
    Calculate inter-site phase-difference clustering across trials.

    Parameters
    ----------
    phase_tensor : np.ndarray
        Complex unit phase with shape ``(site, frequency, trial, time)``.
    site_a, site_b : int
        Zero-based site-axis indices forming the ordered phase difference
        ``phase_a * conjugate(phase_b)``.
    trial_mask : np.ndarray | None, optional
        Boolean trial selector with shape ``(n_trials,)``.
    valid_mask : np.ndarray | None, optional
        Boolean validity array with the same shape as ``phase_tensor``.

    Returns
    -------
    PhaseClusteringResult
        Values and effective counts have shape ``(frequency, time)``. Values
        are dimensionless in ``[0, 1]`` and counts use the validity
        intersection of both sites.
    """

    phase, selected, valid = _validate_phase_clustering_inputs(phase_tensor, trial_mask, valid_mask)
    if int(site_a) < 0 or int(site_a) >= phase.shape[0] or int(site_b) < 0 or int(site_b) >= phase.shape[0]:
        raise ValueError("site_a and site_b must index the phase tensor site axis.")
    if int(site_a) == int(site_b):
        raise ValueError("ISPC requires two distinct site indices.")
    selected_indices = np.flatnonzero(selected)
    phase_a = np.take(phase[int(site_a)], selected_indices, axis=1)
    phase_b = np.take(phase[int(site_b)], selected_indices, axis=1)
    pair_valid = np.take(valid[int(site_a)], selected_indices, axis=1) & np.take(
        valid[int(site_b)], selected_indices, axis=1
    )
    phase_difference = phase_a * np.conjugate(phase_b)
    phase_difference_magnitude = np.abs(phase_difference)
    normalized_difference = np.zeros(phase_difference.shape, dtype=np.complex128)
    np.divide(
        phase_difference,
        phase_difference_magnitude,
        out=normalized_difference,
        where=pair_valid,
    )
    vector_sum = np.sum(np.where(pair_valid, normalized_difference, 0.0j), axis=1)
    effective_count = np.sum(pair_valid, axis=1, dtype=int)
    values = np.zeros(effective_count.shape, dtype=float)
    np.divide(np.abs(vector_sum), effective_count, out=values, where=effective_count > 0)
    return PhaseClusteringResult(
        values=np.clip(values, 0.0, 1.0),
        effective_trial_count=effective_count,
        n_trials=int(np.sum(selected)),
    )


def make_phase_condition_masks(trial_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Build phase-analysis conditions using established project trial semantics.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)`` and columns required
        by ``spike_behavior_pynapple.make_trial_type_masks``.

    Returns
    -------
    dict[str, np.ndarray]
        Boolean arrays with shape ``(n_trials,)`` for ``all``,
        ``correct_rewarded``, ``incorrect``, ``omission``, ``switch``, and
        ``stay``. Switch/stay directly retain the established current
        unrewarded-trial restriction and next-row choice comparison.
    """

    project_masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)
    return {
        "all": np.asarray(project_masks["valid"], dtype=bool),
        "correct_rewarded": np.asarray(project_masks["correct_rewarded"], dtype=bool),
        "incorrect": np.asarray(project_masks["incorrect"], dtype=bool),
        "omission": np.asarray(project_masks["omission"], dtype=bool),
        "switch": np.asarray(project_masks["switch"], dtype=bool),
        "stay": np.asarray(project_masks["stay"], dtype=bool),
    }


def save_phase_clustering_result(
    output_path: Path | str,
    values: np.ndarray,
    effective_trial_count: np.ndarray,
    frequencies_hz: np.ndarray,
    relative_time_s: np.ndarray,
    trial_indices: np.ndarray,
    metadata: dict,
) -> Path:
    """
    Save one phase-clustering matrix and reproducibility metadata to NPZ.

    Parameters
    ----------
    output_path : Path | str
        New output ``.npz`` path. Existing files are never overwritten.
    values : np.ndarray
        ITPC or ISPC matrix with shape ``(n_frequencies, n_times)``.
    effective_trial_count : np.ndarray
        Integer matrix matching ``values`` with per-pixel trial counts.
    frequencies_hz : np.ndarray
        Frequency axis with shape ``(n_frequencies,)`` in Hz.
    relative_time_s : np.ndarray
        Relative time axis with shape ``(n_times,)`` in seconds.
    trial_indices : np.ndarray
        Included trial-table indices with shape ``(n_trials,)``.
    metadata : dict
        Serializable metadata including analysis version, parameters, site
        definitions, units, and axis conventions.

    Returns
    -------
    Path
        Created filesystem path.
    """

    destination = Path(output_path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis result: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        values=np.asarray(values, dtype=float),
        effective_trial_count=np.asarray(effective_trial_count, dtype=int),
        frequencies_hz=np.asarray(frequencies_hz, dtype=float).reshape(-1),
        relative_time_s=np.asarray(relative_time_s, dtype=float).reshape(-1),
        trial_indices=np.asarray(trial_indices, dtype=int).reshape(-1),
        meta=np.asarray(dict(metadata), dtype=object),
    )
    return destination


def save_single_trial_relative_phase_result(
    output_path: Path | str,
    result: SingleTrialRelativePhaseResult,
    display_valid_mask: np.ndarray,
    metadata: dict,
) -> Path:
    """
    Save one relative-phase result and its current display mask to NPZ.

    Parameters
    ----------
    output_path : Path | str
        New ``.npz`` destination. Existing files are never overwritten.
    result : SingleTrialRelativePhaseResult
        Frequency-by-time phase/amplitude arrays, axes, source-rate traces,
        and trial/site metadata.
    display_valid_mask : np.ndarray
        Boolean array matching ``result.phase_angle_rad`` for the selected
        optional amplitude-mask settings.
    metadata : dict
        Serializable analysis parameters, units, source paths, and version.

    Returns
    -------
    Path
        Created filesystem path.
    """

    destination = Path(output_path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis result: {destination}")
    display_valid = np.asarray(display_valid_mask, dtype=bool)
    if display_valid.shape != result.phase_angle_rad.shape:
        raise ValueError("display_valid_mask must match phase_angle_rad.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        relative_phase_complex=result.relative_phase_complex,
        phase_angle_rad=result.phase_angle_rad,
        amplitude_a=result.amplitude_a,
        amplitude_b=result.amplitude_b,
        numerical_valid=result.numerical_valid,
        display_valid=display_valid,
        amplitude_percentiles_a=result.amplitude_percentiles_a,
        amplitude_percentiles_b=result.amplitude_percentiles_b,
        frequencies_hz=result.frequencies_hz,
        relative_time_s=result.relative_time_s,
        source_time_a_s=result.source_time_a_s,
        source_lfp_a=result.source_lfp_a,
        source_time_b_s=result.source_time_b_s,
        source_lfp_b=result.source_lfp_b,
        trial_index=np.asarray(result.trial_index, dtype=int),
        event_time_s=np.asarray(result.event_time_s, dtype=float),
        site_a_label=np.asarray(result.site_a_label),
        site_b_label=np.asarray(result.site_b_label),
        meta=np.asarray(dict(metadata), dtype=object),
    )
    return destination


def save_single_trial_phase_analysis_result(
    output_path: Path | str,
    phase_result: SingleTrialRelativePhaseResult,
    plv_result: WithinTrialPLVResult,
    phase_display_valid_mask: np.ndarray,
    metadata: dict,
) -> Path:
    """
    Save combined single-trial relative-phase and within-trial PLV arrays.

    Parameters
    ----------
    output_path : Path | str
        New ``.npz`` destination. Existing files are never overwritten.
    phase_result : SingleTrialRelativePhaseResult
        Visible frequency-by-time phase and amplitude arrays, source traces,
        and ordered site/trial metadata.
    plv_result : WithinTrialPLVResult
        Visible frequency-by-time dimensionless PLV plus per-frequency window
        sizes and frequency-by-time valid sample counts.
    phase_display_valid_mask : np.ndarray
        Boolean array matching ``phase_result.phase_angle_rad``.
    metadata : dict
        Serializable parameters, units, source identifiers, and version.

    Returns
    -------
    Path
        Created filesystem path.
    """

    destination = Path(output_path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis result: {destination}")
    display_valid = np.asarray(phase_display_valid_mask, dtype=bool)
    if display_valid.shape != phase_result.phase_angle_rad.shape:
        raise ValueError("phase_display_valid_mask must match phase_angle_rad.")
    if not np.array_equal(phase_result.frequencies_hz, plv_result.frequencies_hz):
        raise ValueError("Phase and PLV frequency axes must match.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        relative_phase_complex=phase_result.relative_phase_complex,
        phase_angle_rad=phase_result.phase_angle_rad,
        amplitude_a=phase_result.amplitude_a,
        amplitude_b=phase_result.amplitude_b,
        numerical_valid=phase_result.numerical_valid,
        phase_display_valid=display_valid,
        frequencies_hz=phase_result.frequencies_hz,
        relative_time_s=phase_result.relative_time_s,
        source_time_a_s=phase_result.source_time_a_s,
        source_lfp_a=phase_result.source_lfp_a,
        source_time_b_s=phase_result.source_time_b_s,
        source_lfp_b=phase_result.source_lfp_b,
        plv=plv_result.plv,
        plv_relative_time_s=plv_result.relative_time_s,
        plv_effective_window_s=plv_result.effective_window_s,
        plv_window_sample_count=plv_result.window_sample_count,
        plv_valid_sample_count=plv_result.valid_sample_count,
        plv_valid_fraction=plv_result.valid_fraction,
        plv_window_cycles=np.asarray(plv_result.window_cycles, dtype=float),
        trial_index=np.asarray(phase_result.trial_index, dtype=int),
        event_time_s=np.asarray(phase_result.event_time_s, dtype=float),
        site_a_label=np.asarray(phase_result.site_a_label),
        site_b_label=np.asarray(phase_result.site_b_label),
        meta=np.asarray(dict(metadata), dtype=object),
    )
    return destination


def compute_site_phase_trial_tensor(
    event_times_s: np.ndarray,
    trial_indices: np.ndarray,
    block_loader: Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    frequencies_hz: np.ndarray,
    window: tuple[float, float],
    output_sample_rate_hz: float = 500.0,
    gaussian_width: float = 1.5,
    window_length: float = 1.0,
    precision: int = 16,
    norm: str | None = "l1",
    notch_60_hz: bool = False,
    notch_quality_factor: float = 30.0,
    minimum_relative_magnitude: float = 1e-12,
    maximum_core_duration_s: float = 120.0,
) -> PhaseTrialTensor:
    """
    Transform continuous LFP blocks and then extract all event-aligned trials.

    Parameters
    ----------
    event_times_s : np.ndarray
        Chronological absolute event times with shape ``(n_trials,)`` in seconds.
    trial_indices : np.ndarray
        Trial-table indices with shape ``(n_trials,)``.
    block_loader : Callable
        Function receiving absolute ``(start_s, end_s)`` and returning
        ``(time_s, lfp_values, sample_rate_hz)``. Arrays have shape
        ``(n_samples,)``; times are absolute seconds and sample rate is Hz.
    frequencies_hz : np.ndarray
        Wavelet frequencies with shape ``(n_frequencies,)`` in Hz.
    window : tuple[float, float]
        Event-relative output bounds in seconds.
    output_sample_rate_hz : float, default=500.0
        Exact common event-relative output rate in Hz.
    gaussian_width, window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int, default=16
        Base-2 Pynapple wavelet evaluation precision.
    norm : str | None, default="l1"
        Pynapple wavelet normalization.
    notch_60_hz : bool, default=False
        Whether to apply a 60 Hz notch.
    notch_quality_factor : float, default=30.0
        Dimensionless notch quality factor.
    minimum_relative_magnitude : float, default=1e-12
        Relative coefficient magnitude validity threshold.
    maximum_core_duration_s : float, default=120.0
        Maximum unpadded continuous transform span in seconds.

    Returns
    -------
    PhaseTrialTensor
        One-site tensor with shape ``(1, frequency, trial, time)``. Every
        retained trial is extracted only after its continuous block transform.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    padding_s = lfp_spectrogram.compute_wavelet_padding_s(
        minimum_frequency_hz=float(np.min(frequencies)),
        window_length=float(window_length),
    )
    blocks = plan_continuous_processing_blocks(
        event_times_s=event_times_s,
        trial_indices=trial_indices,
        window=window,
        wavelet_padding_s=padding_s,
        maximum_core_duration_s=float(maximum_core_duration_s),
    )
    block_tensors = []
    excluded_trial_indices = []
    for block in blocks:
        try:
            block_time_s, block_lfp_values, block_sample_rate_hz = block_loader(
                float(block.load_start_s),
                float(block.load_end_s),
            )
            phase_result = compute_wavelet_phase(
                time_s=block_time_s,
                lfp_values=block_lfp_values,
                sample_rate_hz=float(block_sample_rate_hz),
                frequencies_hz=frequencies,
                gaussian_width=float(gaussian_width),
                window_length=float(window_length),
                precision=int(precision),
                norm=norm,
                target_sample_rate_hz=float(output_sample_rate_hz),
                notch_60_hz=bool(notch_60_hz),
                notch_quality_factor=float(notch_quality_factor),
                minimum_relative_magnitude=float(minimum_relative_magnitude),
            )
            block_tensor = make_phase_trial_tensor(
                coefficient_times_s=phase_result.time_s,
                unit_phase=phase_result.phase[np.newaxis, :, :],
                event_times_s=block.event_times_s,
                trial_indices=block.trial_indices,
                window=window,
                output_sample_rate_hz=float(output_sample_rate_hz),
                edge_buffer_s=padding_s,
            )
        except (FileNotFoundError, OSError, ValueError):
            excluded_trial_indices.extend(block.trial_indices.tolist())
            continue
        block_tensors.append(block_tensor)
        excluded_trial_indices.extend(block_tensor.excluded_trial_indices.tolist())
    if not block_tensors:
        raise ValueError("No trials had sufficient continuous LFP support for phase analysis.")
    return PhaseTrialTensor(
        phase=np.concatenate([tensor.phase for tensor in block_tensors], axis=2),
        valid=np.concatenate([tensor.valid for tensor in block_tensors], axis=2),
        relative_time_s=block_tensors[0].relative_time_s.copy(),
        trial_indices=np.concatenate([tensor.trial_indices for tensor in block_tensors]),
        excluded_trial_indices=np.asarray(sorted(set(excluded_trial_indices)), dtype=int),
    )


def _validate_phase_clustering_inputs(
    phase_tensor: np.ndarray,
    trial_mask: np.ndarray | None,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate shared NumPy contracts for ITPC and ISPC calculations."""

    phase = np.asarray(phase_tensor)
    if phase.ndim != 4:
        raise ValueError("phase_tensor must have shape (site, frequency, trial, time).")
    if trial_mask is None:
        selected = np.ones(phase.shape[2], dtype=bool)
    else:
        selected = np.asarray(trial_mask, dtype=bool).reshape(-1)
        if selected.size != phase.shape[2]:
            raise ValueError("trial_mask length must match the phase tensor trial axis.")
    if valid_mask is None:
        valid = np.isfinite(phase.real) & np.isfinite(phase.imag) & (np.abs(phase) > 0.0)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != phase.shape:
            raise ValueError("valid_mask must have the same shape as phase_tensor.")
    return phase, selected, valid
