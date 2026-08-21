"""Single-trial spike timing relative to a band-limited LFP Hilbert phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import hilbert

from src.neural_analysis import lfp_loading


ANALYSIS_VERSION = "0.1.0"
DEFAULT_PHASE_BAND_HZ = (6.0, 10.0)
DEFAULT_FILTER_PADDING_S = 1.0


@dataclass(frozen=True)
class HilbertPhaseTrace:
    """Band-limited Hilbert phase for one continuous padded LFP trace.

    All arrays have shape ``(n_samples,)``. ``relative_time_s`` is in seconds
    relative to the alignment event. LFP and envelope arrays retain the input
    source voltage units. ``analytic_signal`` is complex source-voltage data;
    ``phase_unit`` is dimensionless complex unit phase; and ``phase_rad`` is
    wrapped radians in ``[-pi, pi]``. ``phase_valid`` marks finite phase
    samples with envelope strictly above the configured threshold.
    """

    relative_time_s: np.ndarray
    raw_lfp: np.ndarray
    bandpassed_lfp: np.ndarray
    analytic_signal: np.ndarray
    envelope: np.ndarray
    phase_unit: np.ndarray
    phase_rad: np.ndarray
    phase_valid: np.ndarray
    sample_rate_hz: float
    frequency_band_hz: tuple[float, float]


@dataclass(frozen=True)
class SpikeHilbertPhaseSamples:
    """Complex Hilbert phase sampled at individual spike timestamps.

    Every array has shape ``(n_spikes,)``. Spike times are seconds relative to
    the alignment event. ``phase_unit`` is dimensionless complex unit phase,
    ``phase_rad`` is wrapped radians, and ``envelope`` retains LFP source
    voltage units. ``valid`` identifies finite phase estimates above the
    requested envelope threshold.
    """

    spike_times_relative_s: np.ndarray
    phase_unit: np.ndarray
    phase_rad: np.ndarray
    envelope: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class SingleTrialSpikeLFPHilbertResult:
    """Visible-trial LFP Hilbert phase and unit spike-phase observations.

    Time-indexed arrays have shape ``(n_visible_samples,)`` and are in seconds
    relative to ``event_time_s``. LFP arrays retain source voltage units.
    ``analytic_signal`` is complex source-voltage data, while ``phase_unit``
    and ``spike_phase_unit`` are dimensionless complex unit vectors.
    Spike-indexed arrays have shape ``(n_visible_spikes,)``; absolute spike
    times are synchronized seconds and relative spike times are seconds from
    the alignment event. Phase angles are wrapped radians in ``[-pi, pi]``.
    """

    relative_time_s: np.ndarray
    raw_lfp: np.ndarray
    bandpassed_lfp: np.ndarray
    analytic_signal: np.ndarray
    envelope: np.ndarray
    phase_unit: np.ndarray
    phase_rad: np.ndarray
    phase_valid: np.ndarray
    spike_times_absolute_s: np.ndarray
    spike_times_relative_s: np.ndarray
    spike_phase_unit: np.ndarray
    spike_phase_rad: np.ndarray
    spike_phase_envelope: np.ndarray
    spike_phase_valid: np.ndarray
    source_sample_rate_hz: float
    frequency_band_hz: tuple[float, float]
    filter_padding_s: float
    trial_index: int
    event_time_s: float
    unit_id: int
    lfp_site_label: str


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return one finite float vector with shape ``(n_samples,)``.

    Parameters
    ----------
    values : np.ndarray
        Input numeric array expected to be one-dimensional and finite.
    name : str
        Human-readable input name used in validation errors.

    Returns
    -------
    np.ndarray
        One-dimensional float array with the input physical units preserved.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_time_axis(relative_time_s: np.ndarray, sample_rate_hz: float) -> None:
    """Validate a source-rate time vector with shape ``(n_samples,)``.

    Parameters
    ----------
    relative_time_s : np.ndarray
        Finite, strictly increasing timestamps in seconds.
    sample_rate_hz : float
        Positive source sampling rate in Hz.

    Returns
    -------
    None
        Raises ``ValueError`` when timestamps are not approximately uniform at
        the specified source sampling rate.
    """

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive.")
    sample_intervals_s = np.diff(relative_time_s)
    if np.any(sample_intervals_s <= 0.0):
        raise ValueError("relative_time_s must be strictly increasing.")
    expected_interval_s = 1.0 / float(sample_rate_hz)
    if not np.allclose(sample_intervals_s, expected_interval_s, rtol=1e-3, atol=expected_interval_s * 1e-6):
        raise ValueError("relative_time_s must be approximately uniform at sample_rate_hz.")


def _validate_frequency_band(
    frequency_band_hz: tuple[float, float], sample_rate_hz: float
) -> tuple[float, float]:
    """Validate and return a bandpass tuple in Hz.

    Parameters
    ----------
    frequency_band_hz : tuple[float, float]
        ``(low_hz, high_hz)`` cutoff frequencies in Hz.
    sample_rate_hz : float
        Positive source sampling rate in Hz.

    Returns
    -------
    tuple[float, float]
        Validated increasing cutoff frequencies in Hz, strictly below Nyquist.
    """

    if len(frequency_band_hz) != 2:
        raise ValueError("frequency_band_hz must contain exactly two values.")
    low_hz, high_hz = (float(frequency_band_hz[0]), float(frequency_band_hz[1]))
    nyquist_hz = float(sample_rate_hz) / 2.0
    if not np.isfinite(low_hz) or not np.isfinite(high_hz) or not (0.0 < low_hz < high_hz < nyquist_hz):
        raise ValueError("frequency_band_hz must satisfy 0 < low_hz < high_hz < Nyquist.")
    return low_hz, high_hz


def _validate_nonnegative_finite(value: float, name: str) -> float:
    """Validate a nonnegative scalar without changing its physical units.

    Parameters
    ----------
    value : float
        Scalar value to validate.
    name : str
        Human-readable input name used in validation errors.

    Returns
    -------
    float
        Finite nonnegative scalar in the input's original physical units.
    """

    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return converted


def compute_hilbert_phase_trace(
    relative_time_s: np.ndarray,
    raw_lfp: np.ndarray,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float] = DEFAULT_PHASE_BAND_HZ,
    minimum_envelope: float = 0.0,
) -> HilbertPhaseTrace:
    """Filter a padded source-rate LFP trace and compute its Hilbert phase.

    Parameters
    ----------
    relative_time_s : np.ndarray
        Finite, increasing source timestamps with shape ``(n_samples,)`` in
        seconds relative to an alignment event.
    raw_lfp : np.ndarray
        Finite source-voltage LFP samples with shape ``(n_samples,)``. No
        resampling or unit conversion is applied.
    sample_rate_hz : float
        Positive source sampling rate in Hz, consistent with the timestamps.
    frequency_band_hz : tuple[float, float], default=(6.0, 10.0)
        Increasing Butterworth bandpass cutoff frequencies in Hz.
    minimum_envelope : float, default=0.0
        Nonnegative source-voltage envelope threshold. Samples at or below
        this threshold are marked invalid; they are not amplitude weighted.

    Returns
    -------
    HilbertPhaseTrace
        Full-length padded arrays with the same ``(n_samples,)`` shape as the
        input. Phase uses ``angle(hilbert(bandpassed_lfp))`` in radians.
    """

    time_s = _as_finite_vector(relative_time_s, "relative_time_s")
    source_lfp = _as_finite_vector(raw_lfp, "raw_lfp")
    if time_s.shape != source_lfp.shape:
        raise ValueError("relative_time_s and raw_lfp must have the same shape.")
    _validate_time_axis(time_s, float(sample_rate_hz))
    band_hz = _validate_frequency_band(frequency_band_hz, float(sample_rate_hz))
    envelope_threshold = _validate_nonnegative_finite(minimum_envelope, "minimum_envelope")

    bandpassed_lfp = np.asarray(
        lfp_loading.filter_lfp_trace(
            relative_time_s=time_s,
            lfp_uv=source_lfp,
            sample_rate_hz=float(sample_rate_hz),
            frequency_band_hz=band_hz,
        ),
        dtype=float,
    )
    if bandpassed_lfp.shape != source_lfp.shape:
        raise ValueError("lfp_loading.filter_lfp_trace must return one value per source sample.")
    if not np.all(np.isfinite(bandpassed_lfp)):
        raise ValueError("lfp_loading.filter_lfp_trace must return only finite values.")

    analytic_signal = np.asarray(hilbert(bandpassed_lfp), dtype=np.complex128)
    envelope = np.abs(analytic_signal)
    numerical_valid = np.isfinite(analytic_signal.real) & np.isfinite(analytic_signal.imag) & np.isfinite(envelope)
    phase_valid = numerical_valid & (envelope > envelope_threshold)
    phase_unit = np.full(analytic_signal.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    phase_unit[phase_valid] = analytic_signal[phase_valid] / envelope[phase_valid]
    phase_rad = np.full(analytic_signal.shape, np.nan, dtype=float)
    phase_rad[phase_valid] = np.angle(phase_unit[phase_valid])

    return HilbertPhaseTrace(
        relative_time_s=time_s,
        raw_lfp=source_lfp,
        bandpassed_lfp=bandpassed_lfp,
        analytic_signal=analytic_signal,
        envelope=envelope,
        phase_unit=phase_unit,
        phase_rad=phase_rad,
        phase_valid=phase_valid,
        sample_rate_hz=float(sample_rate_hz),
        frequency_band_hz=band_hz,
    )


def sample_hilbert_phase_at_spikes(
    relative_time_s: np.ndarray,
    phase_unit: np.ndarray,
    envelope: np.ndarray,
    spike_times_relative_s: np.ndarray,
    minimum_envelope: float = 0.0,
) -> SpikeHilbertPhaseSamples:
    """Sample continuous Hilbert phase at exact event-relative spike times.

    Parameters
    ----------
    relative_time_s : np.ndarray
        Strictly increasing timestamps with shape ``(n_samples,)`` in seconds
        relative to the alignment event.
    phase_unit : np.ndarray
        Complex phase samples with shape ``(n_samples,)``. Values should be
        dimensionless unit vectors; real and imaginary components are
        interpolated separately.
    envelope : np.ndarray
        Source-voltage Hilbert envelope samples with shape ``(n_samples,)``.
    spike_times_relative_s : np.ndarray
        Finite spike timestamps with shape ``(n_spikes,)`` in seconds relative
        to the alignment event. All timestamps must lie in the trace bounds.
    minimum_envelope : float, default=0.0
        Nonnegative source-voltage validity threshold applied after envelope
        interpolation. It does not weight the circular phase mean.

    Returns
    -------
    SpikeHilbertPhaseSamples
        Spike-indexed phase vectors, wrapped phase radians, envelope values,
        and validity with shape ``(n_spikes,)``.
    """

    time_s = _as_finite_vector(relative_time_s, "relative_time_s")
    phase_values = np.asarray(phase_unit, dtype=np.complex128)
    envelope_values = np.asarray(envelope, dtype=float)
    spike_times_s = np.asarray(spike_times_relative_s, dtype=float)
    if phase_values.ndim != 1 or envelope_values.ndim != 1 or spike_times_s.ndim != 1:
        raise ValueError("phase_unit, envelope, and spike_times_relative_s must be one-dimensional.")
    if phase_values.shape != time_s.shape or envelope_values.shape != time_s.shape:
        raise ValueError("phase_unit and envelope must match relative_time_s.")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("relative_time_s must be strictly increasing.")
    if not np.all(np.isfinite(spike_times_s)):
        raise ValueError("spike_times_relative_s must contain only finite values.")
    if np.any(spike_times_s < time_s[0]) or np.any(spike_times_s > time_s[-1]):
        raise ValueError("spike_times_relative_s must lie within the padded trace bounds.")
    envelope_threshold = _validate_nonnegative_finite(minimum_envelope, "minimum_envelope")

    interpolated_real = np.interp(spike_times_s, time_s, phase_values.real)
    interpolated_imag = np.interp(spike_times_s, time_s, phase_values.imag)
    interpolated_envelope = np.interp(spike_times_s, time_s, envelope_values)
    interpolated_phase = interpolated_real + 1j * interpolated_imag
    phase_norm = np.abs(interpolated_phase)
    valid = (
        np.isfinite(interpolated_real)
        & np.isfinite(interpolated_imag)
        & np.isfinite(interpolated_envelope)
        & (phase_norm > 0.0)
        & (interpolated_envelope > envelope_threshold)
    )
    unit_phase = np.full(spike_times_s.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    unit_phase[valid] = interpolated_phase[valid] / phase_norm[valid]
    phase_rad = np.full(spike_times_s.shape, np.nan, dtype=float)
    phase_rad[valid] = np.angle(unit_phase[valid])

    return SpikeHilbertPhaseSamples(
        spike_times_relative_s=spike_times_s,
        phase_unit=unit_phase,
        phase_rad=phase_rad,
        envelope=interpolated_envelope,
        valid=valid,
    )


def compute_single_trial_spike_lfp_hilbert(
    padded_relative_time_s: np.ndarray,
    padded_raw_lfp: np.ndarray,
    source_sample_rate_hz: float,
    unit_spike_times_absolute_s: np.ndarray,
    event_time_s: float,
    visible_window: tuple[float, float],
    frequency_band_hz: tuple[float, float] = DEFAULT_PHASE_BAND_HZ,
    filter_padding_s: float = DEFAULT_FILTER_PADDING_S,
    minimum_envelope: float = 0.0,
    trial_index: int = -1,
    unit_id: int = -1,
    lfp_site_label: str = "",
) -> SingleTrialSpikeLFPHilbertResult:
    """Compute one visible trial's Hilbert phase and sampled unit spike phase.

    Parameters
    ----------
    padded_relative_time_s : np.ndarray
        Finite source-rate timestamps with shape ``(n_padded_samples,)`` in
        event-relative seconds. They must cover ``visible_window`` plus the
        requested filter padding.
    padded_raw_lfp : np.ndarray
        Finite source-voltage LFP values with shape ``(n_padded_samples,)``.
        The source sampling rate and units are preserved.
    source_sample_rate_hz : float
        Positive source sampling rate in Hz, consistent with the timestamps.
    unit_spike_times_absolute_s : np.ndarray
        Finite synchronized unit spike timestamps with shape ``(n_spikes,)``
        in absolute seconds.
    event_time_s : float
        Finite absolute alignment-event time in synchronized seconds.
    visible_window : tuple[float, float]
        Increasing event-relative bounds in seconds. Spikes are selected on
        the half-open interval ``[start, end)``.
    frequency_band_hz : tuple[float, float], default=(6.0, 10.0)
        Increasing phase bandpass cutoffs in Hz.
    filter_padding_s : float, default=1.0
        Nonnegative continuous LFP padding in seconds required on both sides.
    minimum_envelope : float, default=0.0
        Nonnegative source-voltage envelope validity threshold.
    trial_index : int, default=-1
        Trial-table index stored with the result.
    unit_id : int, default=-1
        Selected unit identifier stored with the result.
    lfp_site_label : str, default=""
        Human-readable independent LFP site/channel label.

    Returns
    -------
    SingleTrialSpikeLFPHilbertResult
        Visible-only time arrays with shape ``(n_visible_samples,)`` and
        visible spike arrays with shape ``(n_visible_spikes,)``. LFP retains
        source units, times are seconds, and phase angles are radians.
    """

    time_s = _as_finite_vector(padded_relative_time_s, "padded_relative_time_s")
    raw_lfp = _as_finite_vector(padded_raw_lfp, "padded_raw_lfp")
    if time_s.shape != raw_lfp.shape:
        raise ValueError("padded_relative_time_s and padded_raw_lfp must have the same shape.")
    _validate_time_axis(time_s, float(source_sample_rate_hz))
    if len(visible_window) != 2:
        raise ValueError("visible_window must contain exactly two bounds.")
    visible_start_s, visible_end_s = (float(visible_window[0]), float(visible_window[1]))
    if not np.isfinite(visible_start_s) or not np.isfinite(visible_end_s) or visible_start_s >= visible_end_s:
        raise ValueError("visible_window must contain finite increasing bounds.")
    padding_s = _validate_nonnegative_finite(filter_padding_s, "filter_padding_s")
    sample_interval_s = 1.0 / float(source_sample_rate_hz)
    padded_start_s = visible_start_s - padding_s
    padded_end_s = visible_end_s + padding_s
    if time_s[0] > padded_start_s or time_s[-1] + sample_interval_s < padded_end_s:
        raise ValueError("padded_relative_time_s does not cover visible_window and filter_padding_s.")
    if not np.isfinite(event_time_s):
        raise ValueError("event_time_s must be finite.")
    absolute_spike_times_s = np.asarray(unit_spike_times_absolute_s, dtype=float)
    if absolute_spike_times_s.ndim != 1 or not np.all(np.isfinite(absolute_spike_times_s)):
        raise ValueError("unit_spike_times_absolute_s must be a finite one-dimensional array.")

    trace = compute_hilbert_phase_trace(
        relative_time_s=time_s,
        raw_lfp=raw_lfp,
        sample_rate_hz=float(source_sample_rate_hz),
        frequency_band_hz=frequency_band_hz,
        minimum_envelope=minimum_envelope,
    )
    spike_times_relative_s = absolute_spike_times_s - float(event_time_s)
    visible_spike_mask = (spike_times_relative_s >= visible_start_s) & (spike_times_relative_s < visible_end_s)
    visible_spike_times_absolute_s = absolute_spike_times_s[visible_spike_mask]
    visible_spike_times_relative_s = spike_times_relative_s[visible_spike_mask]
    sampled_spikes = sample_hilbert_phase_at_spikes(
        relative_time_s=trace.relative_time_s,
        phase_unit=trace.phase_unit,
        envelope=trace.envelope,
        spike_times_relative_s=visible_spike_times_relative_s,
        minimum_envelope=minimum_envelope,
    )
    boundary_tolerance_s = sample_interval_s * 1e-6
    visible_sample_mask = (
        (trace.relative_time_s >= visible_start_s - boundary_tolerance_s)
        & (trace.relative_time_s < visible_end_s - boundary_tolerance_s)
    )
    if not np.any(visible_sample_mask):
        raise ValueError("visible_window contains no source LFP samples.")
    visible_time_s = trace.relative_time_s[visible_sample_mask].copy()
    if abs(visible_time_s[0] - visible_start_s) <= boundary_tolerance_s:
        visible_time_s[0] = visible_start_s

    return SingleTrialSpikeLFPHilbertResult(
        relative_time_s=visible_time_s,
        raw_lfp=trace.raw_lfp[visible_sample_mask],
        bandpassed_lfp=trace.bandpassed_lfp[visible_sample_mask],
        analytic_signal=trace.analytic_signal[visible_sample_mask],
        envelope=trace.envelope[visible_sample_mask],
        phase_unit=trace.phase_unit[visible_sample_mask],
        phase_rad=trace.phase_rad[visible_sample_mask],
        phase_valid=trace.phase_valid[visible_sample_mask],
        spike_times_absolute_s=visible_spike_times_absolute_s,
        spike_times_relative_s=sampled_spikes.spike_times_relative_s,
        spike_phase_unit=sampled_spikes.phase_unit,
        spike_phase_rad=sampled_spikes.phase_rad,
        spike_phase_envelope=sampled_spikes.envelope,
        spike_phase_valid=sampled_spikes.valid,
        source_sample_rate_hz=float(source_sample_rate_hz),
        frequency_band_hz=trace.frequency_band_hz,
        filter_padding_s=padding_s,
        trial_index=int(trial_index),
        event_time_s=float(event_time_s),
        unit_id=int(unit_id),
        lfp_site_label=str(lfp_site_label),
    )


def save_single_trial_spike_lfp_hilbert_result(
    output_path: Path | str,
    result: SingleTrialSpikeLFPHilbertResult,
    metadata: Mapping[str, object],
) -> Path:
    """Save a single-trial Hilbert-phase result as a non-overwriting NPZ file.

    Parameters
    ----------
    output_path : Path | str
        New destination ending in ``.npz``. Existing files are never replaced.
    result : SingleTrialSpikeLFPHilbertResult
        Visible-trial arrays and metadata. LFP arrays retain source units,
        times are seconds, and phase arrays are radians or unit complex phase.
    metadata : Mapping[str, object]
        Serializable analysis metadata stored as the named object array
        ``meta``. Callers should include units, axis conventions, and sources.

    Returns
    -------
    Path
        Created ``.npz`` destination containing named result arrays and meta.
    """

    destination = Path(output_path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis result: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        relative_time_s=np.asarray(result.relative_time_s, dtype=float),
        raw_lfp=np.asarray(result.raw_lfp, dtype=float),
        bandpassed_lfp=np.asarray(result.bandpassed_lfp, dtype=float),
        analytic_signal=np.asarray(result.analytic_signal, dtype=np.complex128),
        envelope=np.asarray(result.envelope, dtype=float),
        phase_unit=np.asarray(result.phase_unit, dtype=np.complex128),
        phase_rad=np.asarray(result.phase_rad, dtype=float),
        phase_valid=np.asarray(result.phase_valid, dtype=bool),
        spike_times_absolute_s=np.asarray(result.spike_times_absolute_s, dtype=float),
        spike_times_relative_s=np.asarray(result.spike_times_relative_s, dtype=float),
        spike_phase_unit=np.asarray(result.spike_phase_unit, dtype=np.complex128),
        spike_phase_rad=np.asarray(result.spike_phase_rad, dtype=float),
        spike_phase_envelope=np.asarray(result.spike_phase_envelope, dtype=float),
        spike_phase_valid=np.asarray(result.spike_phase_valid, dtype=bool),
        source_sample_rate_hz=np.asarray(result.source_sample_rate_hz, dtype=float),
        frequency_band_hz=np.asarray(result.frequency_band_hz, dtype=float),
        filter_padding_s=np.asarray(result.filter_padding_s, dtype=float),
        trial_index=np.asarray(result.trial_index, dtype=int),
        event_time_s=np.asarray(result.event_time_s, dtype=float),
        unit_id=np.asarray(result.unit_id, dtype=int),
        lfp_site_label=np.asarray(result.lfp_site_label),
        meta=np.asarray(dict(metadata), dtype=object),
    )
    return destination
