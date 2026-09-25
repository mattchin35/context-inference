"""Frequency-resolved spike-LFP phase locking from continuous Morlet phase."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from src.neural_analysis.lfp import phase as lfp_phase_clustering


ANALYSIS_VERSION = "0.2.0"


@dataclass(frozen=True)
class SpikePhaseSamples:
    """Complex LFP phase and magnitude sampled at one unit's spike times."""

    phase_vectors: np.ndarray
    amplitude: np.ndarray
    valid: np.ndarray
    spike_times_s: np.ndarray
    frequencies_hz: np.ndarray


@dataclass(frozen=True)
class SpikePhaseLockingResult:
    """Frequency metrics, phase tuning arrays, sampled observations, and sources.

    ``phase_bin_edges_rad`` has shape ``(n_phase_bins + 1,)`` in radians.
    The three phase tuning arrays have shape ``(n_frequencies, n_phase_bins)``:
    ``phase_spike_counts`` is an integer count, ``phase_occupancy_s`` is
    exposure in seconds, and ``phase_firing_rate_hz`` is occupancy-normalized
    rate in Hz. A rate is NaN when its occupancy is zero.
    """

    frequencies_hz: np.ndarray
    ppc: np.ndarray
    resultant_length: np.ndarray
    preferred_phase_rad: np.ndarray
    n_spikes: np.ndarray
    spike_phase_vectors: np.ndarray
    spike_phase_valid: np.ndarray
    amplitude_at_spikes: np.ndarray
    spike_times_s: np.ndarray
    selected_trial_indices: np.ndarray
    merged_intervals_s: np.ndarray
    unit_id: int
    lfp_site_label: str
    phase_bin_edges_rad: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    phase_spike_counts: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    phase_occupancy_s: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    phase_firing_rate_hz: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))


@dataclass(frozen=True)
class PhaseRateSparsityAssessment:
    """Descriptive sampling assessment for one frequency's phase-rate curve.

    Counts refer to valid spikes across all phase bins. Occupancy refers to the
    analyzed exposure in each phase bin; zero occupancy means that bin's firing
    rate cannot be estimated.
    """

    valid_spike_count: int
    phase_bin_count: int
    mean_spikes_per_bin: float
    minimum_mean_spikes_per_bin: float
    minimum_recommended_spikes: int
    zero_occupancy_bin_count: int

    @property
    def low_spike_count(self) -> bool:
        """Return whether mean spike sampling is below the descriptive threshold."""

        return self.mean_spikes_per_bin < self.minimum_mean_spikes_per_bin

    @property
    def has_zero_occupancy(self) -> bool:
        """Return whether at least one phase bin has no valid LFP exposure."""

        return self.zero_occupancy_bin_count > 0

    @property
    def is_sparse(self) -> bool:
        """Return whether either descriptive sparsity criterion is met."""

        return self.low_spike_count or self.has_zero_occupancy


def assess_phase_rate_sparsity(
    phase_spike_counts: np.ndarray,
    phase_occupancy_s: np.ndarray,
    minimum_mean_spikes_per_bin: float = 5.0,
) -> PhaseRateSparsityAssessment:
    """
    Assess sampling sparsity for one frequency's phase firing-rate estimate.

    Parameters
    ----------
    phase_spike_counts : np.ndarray
        One-dimensional array with shape ``(n_phase_bins,)`` containing
        nonnegative valid-spike counts per phase bin.
    phase_occupancy_s : np.ndarray
        One-dimensional array with shape ``(n_phase_bins,)`` containing finite,
        nonnegative valid LFP exposure in seconds per phase bin.
    minimum_mean_spikes_per_bin : float, default=5.0
        Positive descriptive warning threshold in spikes per phase bin. This is
        a visualization heuristic, not a statistical significance criterion.

    Returns
    -------
    PhaseRateSparsityAssessment
        Counts, sampling summaries, and warning criteria for the supplied
        frequency. ``minimum_recommended_spikes`` is the ceiling of the mean
        threshold multiplied by ``n_phase_bins``.
    """

    spike_counts = np.asarray(phase_spike_counts)
    occupancy_s = np.asarray(phase_occupancy_s, dtype=float)
    if spike_counts.ndim != 1 or occupancy_s.ndim != 1:
        raise ValueError("phase_spike_counts and phase_occupancy_s must be one-dimensional.")
    if spike_counts.shape != occupancy_s.shape or spike_counts.size == 0:
        raise ValueError("Counts and occupancy must have the same nonempty shape.")
    if not np.issubdtype(spike_counts.dtype, np.integer):
        raise ValueError("phase_spike_counts must contain integers.")
    if np.any(spike_counts < 0):
        raise ValueError("phase_spike_counts must be nonnegative.")
    if np.any(~np.isfinite(occupancy_s)) or np.any(occupancy_s < 0.0):
        raise ValueError("phase_occupancy_s must contain finite nonnegative values.")
    threshold = float(minimum_mean_spikes_per_bin)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_mean_spikes_per_bin must be finite and positive.")

    phase_bin_count = int(spike_counts.size)
    valid_spike_count = int(np.sum(spike_counts, dtype=np.int64))
    return PhaseRateSparsityAssessment(
        valid_spike_count=valid_spike_count,
        phase_bin_count=phase_bin_count,
        mean_spikes_per_bin=valid_spike_count / phase_bin_count,
        minimum_mean_spikes_per_bin=threshold,
        minimum_recommended_spikes=int(np.ceil(threshold * phase_bin_count)),
        zero_occupancy_bin_count=int(np.count_nonzero(occupancy_s == 0.0)),
    )


def merge_trial_windows(
    event_times_s: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """
    Merge trial-aligned windows into disjoint half-open absolute intervals.

    Parameters
    ----------
    event_times_s : np.ndarray
        Finite behavioral event times with shape ``(n_trials,)`` in absolute
        synchronized seconds.
    window : tuple[float, float]
        Increasing event-relative bounds in seconds.

    Returns
    -------
    np.ndarray
        Sorted interval bounds with shape ``(n_intervals, 2)`` in absolute
        seconds. Touching and overlapping intervals are merged.
    """

    events = np.asarray(event_times_s, dtype=float).reshape(-1)
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must contain increasing bounds.")
    if not np.isfinite(events).all():
        raise ValueError("event_times_s must contain only finite values.")
    if events.size == 0:
        return np.empty((0, 2), dtype=float)
    starts = events + float(window[0])
    stops = events + float(window[1])
    order = np.argsort(starts, kind="stable")
    intervals = np.column_stack((starts[order], stops[order]))
    merged: list[list[float]] = [[float(intervals[0, 0]), float(intervals[0, 1])]]
    for start_s, stop_s in intervals[1:]:
        if float(start_s) <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(stop_s))
        else:
            merged.append([float(start_s), float(stop_s)])
    return np.asarray(merged, dtype=float)


def sample_wavelet_phase_at_spikes(
    wavelet_result: lfp_phase_clustering.WaveletCoefficientResult,
    spike_times_s: np.ndarray,
    absolute_amplitude_threshold: float = 0.0,
    minimum_relative_magnitude: float = 1e-12,
) -> SpikePhaseSamples:
    """
    Interpolate complex wavelet phase and amplitude at exact spike timestamps.

    Parameters
    ----------
    wavelet_result : lfp_phase_clustering.WaveletCoefficientResult
        Continuous coefficients with shape ``(frequency, wavelet_time)`` and
        absolute synchronized times in seconds.
    spike_times_s : np.ndarray
        One-dimensional absolute spike timestamps in seconds. Values must lie
        within the wavelet time range.
    absolute_amplitude_threshold : float, default=0.0
        Nonnegative threshold in source-dependent wavelet-magnitude units. It
        gates phase validity but never weights a phase vector.
    minimum_relative_magnitude : float, default=1e-12
        Nonnegative numerical threshold relative to the largest coefficient
        magnitude in the continuous block.

    Returns
    -------
    SpikePhaseSamples
        ``phase_vectors``, ``amplitude``, and ``valid`` use shape
        ``(frequency, spike)``. Valid phase vectors are unit complex64;
        amplitudes are float32 in source wavelet units.
    """

    times = np.asarray(wavelet_result.time_s, dtype=float).reshape(-1)
    coefficients = np.asarray(wavelet_result.coefficients)
    spikes = np.asarray(spike_times_s, dtype=float).reshape(-1)
    frequencies = np.asarray(wavelet_result.frequencies_hz, dtype=float).reshape(-1)
    if coefficients.shape != (frequencies.size, times.size):
        raise ValueError("coefficients must have shape (frequency, wavelet_time).")
    if times.size < 2 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Wavelet times must be finite and strictly increasing.")
    if np.any(~np.isfinite(spikes)):
        raise ValueError("spike_times_s must contain only finite values.")
    if float(absolute_amplitude_threshold) < 0.0:
        raise ValueError("absolute_amplitude_threshold must be nonnegative.")
    if spikes.size and (float(np.min(spikes)) < times[0] or float(np.max(spikes)) > times[-1]):
        raise ValueError("Spike timestamps must lie within the wavelet coefficient interval.")

    output_shape = (frequencies.size, spikes.size)
    if spikes.size == 0:
        return SpikePhaseSamples(
            phase_vectors=np.zeros(output_shape, dtype=np.complex64),
            amplitude=np.zeros(output_shape, dtype=np.float32),
            valid=np.zeros(output_shape, dtype=bool),
            spike_times_s=spikes.copy(),
            frequencies_hz=frequencies.copy(),
        )

    source_phase, source_valid = lfp_phase_clustering.normalize_wavelet_phase(
        coefficients,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
    )
    right_indices = np.searchsorted(times, spikes, side="left")
    right_indices = np.clip(right_indices, 1, times.size - 1)
    left_indices = right_indices - 1
    left_times = times[left_indices]
    right_times = times[right_indices]
    interpolation_weight = (spikes - left_times) / (right_times - left_times)

    phase_vectors = np.zeros(output_shape, dtype=np.complex64)
    amplitude = np.zeros(output_shape, dtype=np.float32)
    valid = np.zeros(output_shape, dtype=bool)
    coefficient_amplitude = np.abs(coefficients)
    for frequency_index in range(frequencies.size):
        adjacent_valid = source_valid[frequency_index, left_indices] & source_valid[frequency_index, right_indices]
        interpolated_phase = source_phase[frequency_index, left_indices] * (1.0 - interpolation_weight)
        interpolated_phase += source_phase[frequency_index, right_indices] * interpolation_weight
        interpolated_amplitude = coefficient_amplitude[frequency_index, left_indices] * (1.0 - interpolation_weight)
        interpolated_amplitude += coefficient_amplitude[frequency_index, right_indices] * interpolation_weight
        interpolated_magnitude = np.abs(interpolated_phase)
        frequency_valid = adjacent_valid & np.isfinite(interpolated_magnitude) & (interpolated_magnitude > 0.0)
        frequency_valid &= np.isfinite(interpolated_amplitude)
        frequency_valid &= interpolated_amplitude >= float(absolute_amplitude_threshold)
        phase_vectors[frequency_index, frequency_valid] = (
            interpolated_phase[frequency_valid] / interpolated_magnitude[frequency_valid]
        ).astype(np.complex64)
        amplitude[frequency_index] = interpolated_amplitude.astype(np.float32)
        valid[frequency_index] = frequency_valid
    return SpikePhaseSamples(
        phase_vectors=phase_vectors,
        amplitude=amplitude,
        valid=valid,
        spike_times_s=spikes.copy(),
        frequencies_hz=frequencies.copy(),
    )


def _validate_phase_bin_edges(phase_bin_edges_rad: np.ndarray) -> np.ndarray:
    """Validate phase-bin edges and return a float copy.

    Parameters
    ----------
    phase_bin_edges_rad : np.ndarray
        One-dimensional finite, strictly increasing bin edges in radians.
        All edges must lie within ``[-pi, pi]``.

    Returns
    -------
    np.ndarray
        Flattened floating-point edge array with shape ``(n_bins + 1,)`` and
        units of radians.
    """

    edges = np.asarray(phase_bin_edges_rad, dtype=float).reshape(-1)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("phase_bin_edges_rad must contain finite, increasing edges.")
    if edges[0] < -np.pi or edges[-1] > np.pi:
        raise ValueError("phase_bin_edges_rad must lie within [-pi, pi].")
    return edges


def _merge_absolute_intervals(intervals_s: np.ndarray) -> np.ndarray:
    """Merge overlapping absolute half-open intervals.

    Parameters
    ----------
    intervals_s : np.ndarray
        Finite interval bounds with shape ``(n_intervals, 2)`` in seconds,
        where each row is ``[start_s, stop_s)`` and ``start_s < stop_s``.

    Returns
    -------
    np.ndarray
        Sorted disjoint union with shape ``(n_merged_intervals, 2)`` in
        seconds. Touching intervals are merged.
    """

    intervals = np.asarray(intervals_s, dtype=float)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals_s must have shape (n_intervals, 2).")
    if np.any(~np.isfinite(intervals)) or np.any(intervals[:, 0] >= intervals[:, 1]):
        raise ValueError("intervals_s must contain finite increasing bounds.")
    ordered = intervals[np.argsort(intervals[:, 0], kind="stable")]
    merged: list[list[float]] = [[float(ordered[0, 0]), float(ordered[0, 1])]]
    for start_s, stop_s in ordered[1:]:
        if float(start_s) <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(stop_s))
        else:
            merged.append([float(start_s), float(stop_s)])
    return np.asarray(merged, dtype=float)


def _subtract_absolute_intervals(
    intervals_s: np.ndarray,
    covered_intervals_s: np.ndarray,
) -> np.ndarray:
    """Return the half-open portions of intervals not already covered.

    Parameters
    ----------
    intervals_s : np.ndarray
        Finite source intervals with shape ``(n_intervals, 2)`` in seconds.
        Rows are interpreted as half-open ``[start_s, stop_s)`` intervals.
    covered_intervals_s : np.ndarray
        Previously owned intervals with shape ``(n_covered, 2)`` in seconds.
        Overlapping rows are treated as their union before subtraction.

    Returns
    -------
    np.ndarray
        Sorted disjoint half-open intervals from ``intervals_s`` that are not
        covered, with shape ``(n_remaining, 2)`` in seconds.
    """

    source = _merge_absolute_intervals(intervals_s)
    covered = _merge_absolute_intervals(covered_intervals_s)
    if not source.size:
        return np.empty((0, 2), dtype=float)
    if not covered.size:
        return source
    remaining: list[list[float]] = []
    for source_start_s, source_stop_s in source:
        cursor_s = float(source_start_s)
        stop_s = float(source_stop_s)
        for covered_start_s, covered_stop_s in covered:
            covered_start_s = float(covered_start_s)
            covered_stop_s = float(covered_stop_s)
            if covered_stop_s <= cursor_s:
                continue
            if covered_start_s >= stop_s:
                break
            if covered_start_s > cursor_s:
                remaining.append([cursor_s, min(covered_start_s, stop_s)])
            cursor_s = max(cursor_s, covered_stop_s)
            if cursor_s >= stop_s:
                break
        if cursor_s < stop_s:
            remaining.append([cursor_s, stop_s])
    return np.asarray(remaining, dtype=float).reshape(-1, 2)


def bin_phase_spike_counts(
    phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
) -> np.ndarray:
    """Bin valid spike phases into a frequency-by-phase count matrix.

    Parameters
    ----------
    phase_vectors : np.ndarray
        Complex phase observations with shape ``(frequency, spike)``.
    valid_mask : np.ndarray
        Boolean validity mask with the same shape as ``phase_vectors``.
    phase_bin_edges_rad : np.ndarray
        Strictly increasing phase-bin edges with shape ``(n_bins + 1,)`` in
        radians, spanning a subset of ``[-pi, pi]``.

    Returns
    -------
    np.ndarray
        Integer spike counts with shape ``(frequency, n_bins)``.
    """

    phase = np.asarray(phase_vectors)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    edges = _validate_phase_bin_edges(phase_bin_edges_rad)
    if phase.ndim != 2 or phase.shape != valid.shape:
        raise ValueError("phase_vectors and valid_mask must share shape (frequency, spike).")
    finite = np.isfinite(phase.real) & np.isfinite(phase.imag) & (np.abs(phase) > 0.0)
    valid &= finite
    counts = np.zeros((phase.shape[0], edges.size - 1), dtype=int)
    for frequency_index in range(phase.shape[0]):
        angles = np.angle(phase[frequency_index, valid[frequency_index]])
        counts[frequency_index], _ = np.histogram(angles, bins=edges)
    return counts


def compute_phase_occupancy(
    phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    time_s: np.ndarray,
    intervals_s: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
    sample_rate_hz: float,
    amplitude: np.ndarray | None = None,
    absolute_amplitude_threshold: float = 0.0,
) -> np.ndarray:
    """Accumulate continuous phase exposure in seconds by frequency and phase.

    Parameters
    ----------
    phase_vectors : np.ndarray
        Complex unit-phase vectors with shape ``(frequency, time)``.
    valid_mask : np.ndarray
        Numerical validity mask matching ``phase_vectors``.
    time_s : np.ndarray
        Strictly increasing sample-center times with shape ``(time,)`` in
        seconds.
    intervals_s : np.ndarray
        Absolute half-open analysis intervals with shape ``(n_intervals, 2)``;
        overlapping intervals are treated as their union.
    phase_bin_edges_rad : np.ndarray
        Phase-bin edges with shape ``(n_bins + 1,)`` in radians.
    sample_rate_hz : float
        Uniform continuous sample rate in Hz. Each sample contributes a
        centered support of width ``1 / sample_rate_hz`` seconds, clipped to
        the analysis interval union.
    amplitude : np.ndarray | None, default=None
        Optional wavelet magnitude array matching ``phase_vectors``.
    absolute_amplitude_threshold : float, default=0.0
        Minimum amplitude in source wavelet units for exposure validity.

    Returns
    -------
    np.ndarray
        Float exposure in seconds with shape ``(frequency, n_bins)``.
    """

    phase = np.asarray(phase_vectors)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    times = np.asarray(time_s, dtype=float).reshape(-1)
    edges = _validate_phase_bin_edges(phase_bin_edges_rad)
    intervals = _merge_absolute_intervals(intervals_s)
    sample_rate = float(sample_rate_hz)
    if phase.ndim != 2 or phase.shape != valid.shape or phase.shape[1] != times.size:
        raise ValueError("phase_vectors, valid_mask, and time_s have incompatible shapes.")
    if times.size and (np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0)):
        raise ValueError("time_s must be finite and strictly increasing.")
    if sample_rate <= 0.0 or not np.isfinite(sample_rate):
        raise ValueError("sample_rate_hz must be a positive finite value.")
    threshold = float(absolute_amplitude_threshold)
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError("absolute_amplitude_threshold must be finite and nonnegative.")
    if amplitude is not None:
        magnitudes = np.asarray(amplitude, dtype=float)
        if magnitudes.shape != phase.shape:
            raise ValueError("amplitude must match phase_vectors.")
        valid &= np.isfinite(magnitudes) & (magnitudes >= threshold)
    finite_phase = np.isfinite(phase.real) & np.isfinite(phase.imag) & (np.abs(phase) > 0.0)
    valid &= finite_phase
    occupancy = np.zeros((phase.shape[0], edges.size - 1), dtype=float)
    if not intervals.size or not times.size:
        return occupancy

    sample_half_width_s = 0.5 / sample_rate
    support_duration_s = np.zeros(times.size, dtype=float)
    sample_start_s = times - sample_half_width_s
    sample_end_s = times + sample_half_width_s
    for start_s, stop_s in intervals:
        overlap = np.minimum(sample_end_s, stop_s) - np.maximum(sample_start_s, start_s)
        support_duration_s += np.maximum(overlap, 0.0)
    for frequency_index in range(phase.shape[0]):
        supported = valid[frequency_index] & (support_duration_s > 0.0)
        angles = np.angle(phase[frequency_index, supported])
        weights = support_duration_s[supported]
        occupancy[frequency_index], _ = np.histogram(angles, bins=edges, weights=weights)
    return occupancy


def compute_phase_firing_rate_hz(
    spike_counts: np.ndarray,
    occupancy_s: np.ndarray,
) -> np.ndarray:
    """Normalize phase spike counts by phase occupancy in seconds.

    Parameters
    ----------
    spike_counts : np.ndarray
        Nonnegative frequency-by-phase spike counts.
    occupancy_s : np.ndarray
        Nonnegative frequency-by-phase exposure in seconds, matching
        ``spike_counts``.

    Returns
    -------
    np.ndarray
        Floating-point firing rates in Hz. Entries with zero occupancy are
        NaN; infinity is never returned.
    """

    counts = np.asarray(spike_counts)
    occupancy = np.asarray(occupancy_s, dtype=float)
    if counts.ndim != 2 or occupancy.shape != counts.shape:
        raise ValueError("spike_counts and occupancy_s must share a 2-D shape.")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("spike_counts must be finite and nonnegative.")
    if np.any(~np.isfinite(occupancy)) or np.any(occupancy < 0.0):
        raise ValueError("occupancy_s must be finite and nonnegative.")
    rates = np.full(counts.shape, np.nan, dtype=float)
    exposed = occupancy > 0.0
    rates[exposed] = counts[exposed] / occupancy[exposed]
    return rates


def compute_frequency_phase_metrics(
    spike_phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    frequencies_hz: np.ndarray,
    spike_times_s: np.ndarray,
    unit_id: int,
    lfp_site_label: str,
    amplitude_at_spikes: np.ndarray | None = None,
    selected_trial_indices: np.ndarray | None = None,
    merged_intervals_s: np.ndarray | None = None,
    phase_bin_edges_rad: np.ndarray | None = None,
    phase_occupancy_s: np.ndarray | None = None,
    phase_bin_count: int = 24,
) -> SpikePhaseLockingResult:
    """
    Calculate PPC, resultant length, and preferred phase across spikes.

    Parameters
    ----------
    spike_phase_vectors : np.ndarray
        Complex phase observations with shape ``(frequency, spike)``.
    valid_mask : np.ndarray
        Boolean mask matching ``spike_phase_vectors``. Invalid observations are
        excluded without amplitude weighting.
    frequencies_hz : np.ndarray
        Positive frequency vector with shape ``(frequency,)`` in Hz.
    spike_times_s : np.ndarray
        Absolute synchronized timestamps with shape ``(spike,)`` in seconds.
    unit_id : int
        Source sorter cluster identifier.
    lfp_site_label : str
        Human-readable independent LFP probe/channel identifier.
    amplitude_at_spikes : np.ndarray | None, default=None
        Optional wavelet magnitudes matching ``spike_phase_vectors`` in
        source-dependent units. Missing values are represented as NaN.
    selected_trial_indices : np.ndarray | None, default=None
        Integer source trial rows contributing analysis windows.
    merged_intervals_s : np.ndarray | None, default=None
        Disjoint absolute interval bounds with shape ``(n_intervals, 2)`` in
        seconds.
    phase_bin_edges_rad : np.ndarray | None, default=None
        Phase-bin edges with shape ``(n_bins + 1,)`` in radians. If omitted,
        24 equal-width bins spanning ``[-pi, pi]`` are used.
    phase_occupancy_s : np.ndarray | None, default=None
        Optional exposure array with shape ``(frequency, n_bins)`` in seconds.
        Missing exposure is represented as zero for legacy callers.
    phase_bin_count : int, default=24
        Number of equal-width phase bins used when edges are omitted.

    Returns
    -------
    SpikePhaseLockingResult
        Frequency metrics have shape ``(frequency,)``. PPC is dimensionless and
        unclipped; resultant length is in ``[0, 1]``; preferred phase is in
        radians ``[-pi, pi]`` or NaN for a zero mean vector. Phase tuning
        arrays have shape ``(frequency, phase)`` and sampled arrays retain
        shape ``(frequency, spike)``.
    """

    phase = np.asarray(spike_phase_vectors)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    spike_times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    if phase.ndim != 2 or phase.shape != valid.shape:
        raise ValueError("spike_phase_vectors and valid_mask must share shape (frequency, spike).")
    if phase.shape != (frequencies.size, spike_times.size):
        raise ValueError("Phase arrays must match the frequency and spike-time axes.")
    if frequencies.size == 0 or np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain positive finite values.")
    if np.any(~np.isfinite(spike_times)):
        raise ValueError("spike_times_s must contain finite values.")
    if phase_bin_edges_rad is None:
        if int(phase_bin_count) != phase_bin_count or int(phase_bin_count) < 1:
            raise ValueError("phase_bin_count must be a positive integer.")
        phase_edges = np.linspace(-np.pi, np.pi, int(phase_bin_count) + 1)
    else:
        phase_edges = _validate_phase_bin_edges(phase_bin_edges_rad)
    phase_counts = bin_phase_spike_counts(
        phase_vectors=phase,
        valid_mask=valid,
        phase_bin_edges_rad=phase_edges,
    )
    if phase_occupancy_s is None:
        occupancy = np.zeros(phase_counts.shape, dtype=float)
    else:
        occupancy = np.asarray(phase_occupancy_s, dtype=float)
        if occupancy.shape != phase_counts.shape:
            raise ValueError("phase_occupancy_s must have shape (frequency, phase_bin).")
        if np.any(~np.isfinite(occupancy)) or np.any(occupancy < 0.0):
            raise ValueError("phase_occupancy_s must be finite and nonnegative.")
    firing_rate_hz = compute_phase_firing_rate_hz(phase_counts, occupancy)
    finite_phase = np.isfinite(phase.real) & np.isfinite(phase.imag) & (np.abs(phase) > 0.0)
    valid &= finite_phase
    unit_phase = np.zeros(phase.shape, dtype=np.complex64)
    unit_phase[valid] = (phase[valid] / np.abs(phase[valid])).astype(np.complex64)
    vector_sum = np.sum(np.where(valid, unit_phase, 0.0j), axis=1, dtype=np.complex128)
    counts = np.sum(valid, axis=1, dtype=int)

    ppc = np.full(frequencies.size, np.nan, dtype=float)
    enough_spikes = counts >= 2
    ppc[enough_spikes] = (
        np.abs(vector_sum[enough_spikes]) ** 2 - counts[enough_spikes]
    ) / (counts[enough_spikes] * (counts[enough_spikes] - 1))
    resultant_length = np.full(frequencies.size, np.nan, dtype=float)
    has_spikes = counts > 0
    resultant_length[has_spikes] = np.abs(vector_sum[has_spikes]) / counts[has_spikes]
    preferred_phase = np.full(frequencies.size, np.nan, dtype=float)
    has_direction = has_spikes & (np.abs(vector_sum) > np.finfo(float).eps * np.maximum(counts, 1))
    preferred_phase[has_direction] = np.angle(vector_sum[has_direction])

    amplitudes = (
        np.full(phase.shape, np.nan, dtype=np.float32)
        if amplitude_at_spikes is None
        else np.asarray(amplitude_at_spikes, dtype=np.float32)
    )
    if amplitudes.shape != phase.shape:
        raise ValueError("amplitude_at_spikes must match spike_phase_vectors.")
    intervals = (
        np.empty((0, 2), dtype=float)
        if merged_intervals_s is None
        else np.asarray(merged_intervals_s, dtype=float)
    )
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("merged_intervals_s must have shape (n_intervals, 2).")
    trials = (
        np.array([], dtype=int)
        if selected_trial_indices is None
        else np.asarray(selected_trial_indices, dtype=int).reshape(-1)
    )
    return SpikePhaseLockingResult(
        frequencies_hz=frequencies.copy(),
        ppc=ppc,
        resultant_length=resultant_length,
        preferred_phase_rad=preferred_phase,
        n_spikes=counts,
        spike_phase_vectors=unit_phase,
        spike_phase_valid=valid,
        amplitude_at_spikes=amplitudes.copy(),
        spike_times_s=spike_times.copy(),
        selected_trial_indices=trials.copy(),
        merged_intervals_s=intervals.copy(),
        unit_id=int(unit_id),
        lfp_site_label=str(lfp_site_label),
        phase_bin_edges_rad=phase_edges.copy(),
        phase_spike_counts=phase_counts,
        phase_occupancy_s=occupancy.copy(),
        phase_firing_rate_hz=firing_rate_hz,
    )


def _select_spikes_in_intervals(spike_times_s: np.ndarray, intervals_s: np.ndarray) -> np.ndarray:
    """Return sorted spikes inside the union of half-open absolute intervals."""

    spikes = np.sort(np.asarray(spike_times_s, dtype=float).reshape(-1))
    intervals = np.asarray(intervals_s, dtype=float)
    selected_parts = [spikes[(spikes >= start_s) & (spikes < stop_s)] for start_s, stop_s in intervals]
    if not selected_parts:
        return np.array([], dtype=float)
    return np.concatenate(selected_parts)


def compute_trial_aligned_spike_phase_locking(
    unit_spike_times_s: np.ndarray,
    event_times_s: np.ndarray,
    trial_indices: np.ndarray,
    window: tuple[float, float],
    frequencies_hz: np.ndarray,
    wavelet_padding_s: float,
    maximum_core_duration_s: float,
    block_loader: Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    unit_id: int,
    lfp_site_label: str,
    gaussian_width: float = 1.5,
    wavelet_window_length: float = 1.0,
    precision: int = 16,
    norm: str | None = "l1",
    target_sample_rate_hz: float = 500.0,
    notch_60_hz: bool = False,
    notch_quality_factor: float = 30.0,
    minimum_relative_magnitude: float = 1e-12,
    absolute_amplitude_threshold: float = 0.0,
    phase_bin_count: int = 24,
) -> SpikePhaseLockingResult:
    """
    Pool spike phases from selected trial windows using bounded LFP transforms.

    Parameters
    ----------
    unit_spike_times_s : np.ndarray
        One-dimensional absolute synchronized spike times in seconds.
    event_times_s : np.ndarray
        Chronological finite alignment times with shape ``(n_trials,)`` in
        absolute seconds.
    trial_indices : np.ndarray
        Integer source trial rows matching ``event_times_s``.
    window : tuple[float, float]
        Trial-relative half-open analysis bounds in seconds.
    frequencies_hz : np.ndarray
        Positive Morlet frequencies with shape ``(frequency,)`` in Hz.
    wavelet_padding_s : float
        Nonnegative signal support loaded on both sides of every block in
        seconds.
    maximum_core_duration_s : float
        Positive maximum unpadded continuous block span in seconds.
    block_loader : Callable[[float, float], tuple[np.ndarray, np.ndarray, float]]
        Loader receiving absolute start/end seconds and returning absolute
        sample times, one-dimensional LFP values, and sample rate in Hz.
    unit_id : int
        Source sorter cluster id.
    lfp_site_label : str
        Independent LFP probe/channel label.
    gaussian_width, wavelet_window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int, default=16
        Base-2 Morlet precision.
    norm : str | None, default="l1"
        Pynapple wavelet normalization.
    target_sample_rate_hz : float, default=500.0
        Preferred post-decimation wavelet sample rate in Hz.
    notch_60_hz : bool, default=False
        Whether to apply the existing 60 Hz notch before decimation.
    notch_quality_factor : float, default=30.0
        Dimensionless notch quality factor.
    minimum_relative_magnitude : float, default=1e-12
        Relative numerical phase threshold.
    absolute_amplitude_threshold : float, default=0.0
        Source-unit wavelet magnitude threshold applied to both continuous
        phase occupancy and interpolated spike samples.
    phase_bin_count : int, default=24
        Number of equal-width phase bins spanning ``[-pi, pi]`` for the
        occupancy-normalized firing-rate tuning curve.

    Returns
    -------
    SpikePhaseLockingResult
        Frequency metrics and retained ``frequency x spike`` observations.
        Each physical spike in overlapping selected windows is included once.
    """

    events = np.asarray(event_times_s, dtype=float).reshape(-1)
    trials = np.asarray(trial_indices, dtype=int).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if events.shape != trials.shape:
        raise ValueError("event_times_s and trial_indices must have matching shapes.")
    if events.size == 0:
        raise ValueError("At least one selected trial is required.")
    if np.any(~np.isfinite(events)) or np.any(np.diff(events) < 0.0):
        raise ValueError("event_times_s must be finite and chronological.")
    if np.any(~np.isfinite(unit_spike_times_s)):
        raise ValueError("unit_spike_times_s must contain finite values.")

    merged_intervals = merge_trial_windows(events, window)
    if int(phase_bin_count) != phase_bin_count or int(phase_bin_count) < 1:
        raise ValueError("phase_bin_count must be a positive integer.")
    phase_bin_edges = np.linspace(-np.pi, np.pi, int(phase_bin_count) + 1)
    phase_occupancy = np.zeros((frequencies.size, int(phase_bin_count)), dtype=float)
    blocks = lfp_phase_clustering.plan_continuous_processing_blocks(
        event_times_s=events,
        trial_indices=trials,
        window=window,
        wavelet_padding_s=float(wavelet_padding_s),
        maximum_core_duration_s=float(maximum_core_duration_s),
    )
    sampled_parts: list[SpikePhaseSamples] = []
    covered_intervals = np.empty((0, 2), dtype=float)
    for block in blocks:
        block_intervals = merge_trial_windows(block.event_times_s, window)
        owned_intervals = _subtract_absolute_intervals(block_intervals, covered_intervals)
        covered_intervals = _merge_absolute_intervals(
            np.vstack((covered_intervals, owned_intervals))
            if owned_intervals.size
            else covered_intervals
        )
        block_spikes = _select_spikes_in_intervals(unit_spike_times_s, owned_intervals)
        time_s, lfp_values, sample_rate_hz = block_loader(block.load_start_s, block.load_end_s)
        wavelet_result = lfp_phase_clustering.compute_wavelet_coefficients(
            time_s=np.asarray(time_s, dtype=float),
            lfp_values=np.asarray(lfp_values, dtype=float),
            sample_rate_hz=float(sample_rate_hz),
            frequencies_hz=frequencies,
            gaussian_width=float(gaussian_width),
            window_length=float(wavelet_window_length),
            precision=int(precision),
            norm=norm,
            target_sample_rate_hz=float(target_sample_rate_hz),
            notch_60_hz=bool(notch_60_hz),
            notch_quality_factor=float(notch_quality_factor),
        )
        continuous_phase, numerical_valid = lfp_phase_clustering.normalize_wavelet_phase(
            wavelet_result.coefficients,
            minimum_relative_magnitude=float(minimum_relative_magnitude),
        )
        continuous_amplitude = np.abs(np.asarray(wavelet_result.coefficients))
        exposure_valid = numerical_valid & np.isfinite(continuous_amplitude)
        exposure_valid &= continuous_amplitude >= float(absolute_amplitude_threshold)
        coefficient_times = np.asarray(wavelet_result.time_s, dtype=float).reshape(-1)
        sample_half_width_s = 0.5 / float(wavelet_result.sample_rate_hz)
        sample_start_s = coefficient_times - sample_half_width_s
        sample_end_s = coefficient_times + sample_half_width_s
        support_overlap_s = np.zeros(coefficient_times.size, dtype=float)
        for start_s, stop_s in owned_intervals:
            support_overlap_s += np.maximum(
                np.minimum(sample_end_s, stop_s) - np.maximum(sample_start_s, start_s),
                0.0,
            )
        new_time_mask = support_overlap_s > 0.0
        if np.any(new_time_mask):
            phase_occupancy += compute_phase_occupancy(
                phase_vectors=continuous_phase[:, new_time_mask],
                valid_mask=exposure_valid[:, new_time_mask],
                time_s=coefficient_times[new_time_mask],
                intervals_s=owned_intervals,
                phase_bin_edges_rad=phase_bin_edges,
                sample_rate_hz=float(wavelet_result.sample_rate_hz),
            )
        if block_spikes.size:
            sampled_parts.append(
                sample_wavelet_phase_at_spikes(
                    wavelet_result=wavelet_result,
                    spike_times_s=block_spikes,
                    absolute_amplitude_threshold=float(absolute_amplitude_threshold),
                    minimum_relative_magnitude=float(minimum_relative_magnitude),
                )
            )

    if sampled_parts:
        all_spike_times = np.concatenate([part.spike_times_s for part in sampled_parts])
        phase_vectors = np.concatenate([part.phase_vectors for part in sampled_parts], axis=1)
        amplitudes = np.concatenate([part.amplitude for part in sampled_parts], axis=1)
        valid = np.concatenate([part.valid for part in sampled_parts], axis=1)
        order = np.argsort(all_spike_times, kind="stable")
        all_spike_times = all_spike_times[order]
        phase_vectors = phase_vectors[:, order]
        amplitudes = amplitudes[:, order]
        valid = valid[:, order]
    else:
        all_spike_times = np.array([], dtype=float)
        phase_vectors = np.zeros((frequencies.size, 0), dtype=np.complex64)
        amplitudes = np.zeros((frequencies.size, 0), dtype=np.float32)
        valid = np.zeros((frequencies.size, 0), dtype=bool)
    result = compute_frequency_phase_metrics(
        spike_phase_vectors=phase_vectors,
        valid_mask=valid,
        frequencies_hz=frequencies,
        spike_times_s=all_spike_times,
        unit_id=int(unit_id),
        lfp_site_label=lfp_site_label,
        amplitude_at_spikes=amplitudes,
        selected_trial_indices=trials,
        merged_intervals_s=merged_intervals,
        phase_bin_edges_rad=phase_bin_edges,
        phase_occupancy_s=phase_occupancy,
        phase_bin_count=int(phase_bin_count),
    )
    return replace(result, spike_phase_vectors=phase_vectors, spike_phase_valid=valid)


def save_spike_lfp_phase_locking_result(
    output_path: Path | str,
    result: SpikePhaseLockingResult,
    metadata: dict,
) -> Path:
    """
    Save frequency metrics, sampled phase observations, and metadata to NPZ.

    Parameters
    ----------
    output_path : Path | str
        New ``.npz`` destination. Existing files are never overwritten.
    result : SpikePhaseLockingResult
        Frequency metrics and ``frequency x spike`` complex phase, amplitude,
        and validity arrays with frequencies in Hz and timestamps in seconds.
    metadata : dict
        Serializable analysis settings, source paths, units, and version.

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
        frequencies_hz=result.frequencies_hz,
        ppc=result.ppc,
        resultant_length=result.resultant_length,
        preferred_phase_rad=result.preferred_phase_rad,
        n_spikes=result.n_spikes,
        spike_phase_vectors=result.spike_phase_vectors,
        spike_phase_valid=result.spike_phase_valid,
        amplitude_at_spikes=result.amplitude_at_spikes,
        spike_times_s=result.spike_times_s,
        selected_trial_indices=result.selected_trial_indices,
        merged_intervals_s=result.merged_intervals_s,
        phase_bin_edges_rad=result.phase_bin_edges_rad,
        phase_spike_counts=result.phase_spike_counts,
        phase_occupancy_s=result.phase_occupancy_s,
        phase_firing_rate_hz=result.phase_firing_rate_hz,
        unit_id=np.asarray(result.unit_id, dtype=int),
        lfp_site_label=np.asarray(result.lfp_site_label),
        meta=np.asarray(dict(metadata), dtype=object),
    )
    return destination
