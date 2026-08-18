"""Continuous-wavelet phase tensors and trial-wise phase-clustering metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis import lfp_spectrogram, spike_behavior_pynapple


ANALYSIS_VERSION = "0.1.0"
PHASE_TENSOR_AXIS_ORDER = ("site", "frequency", "trial", "time")


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
    minimum_relative_magnitude : float, default=1e-12
        Relative coefficient magnitude threshold for valid phase.

    Returns
    -------
    WaveletPhaseResult
        ``phase`` and ``valid`` have shape ``(n_frequencies, n_output_times)``;
        ``time_s`` has shape ``(n_output_times,)`` in seconds and
        ``sample_rate_hz`` is the post-decimation rate in Hz.
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
    coefficient_values = np.asarray(coefficients.values).T
    unit_phase, valid = normalize_wavelet_phase(
        coefficient_values,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
    )
    return WaveletPhaseResult(
        time_s=np.asarray(continuous_tsd.index, dtype=float).reshape(-1),
        frequencies_hz=frequencies.copy(),
        phase=unit_phase,
        valid=valid,
        sample_rate_hz=output_sample_rate_hz,
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
