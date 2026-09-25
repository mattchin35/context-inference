"""Welch LFP power summaries on documented trial, epoch, and frequency axes.

Input LFP samples retain their source voltage unit. Power spectral densities
therefore have source-unit-squared/Hz units until callers explicitly convert
valid ratios to decibels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from src.neural_analysis.lfp.config import (
    AnalysisWindowConfig,
    FrequencyBandConfig,
    PowerAnalysisConfig,
)


@dataclass(frozen=True)
class TrialEpochPSDResult:
    """Trial-resolved Welch PSDs on one canonical frequency axis.

    Attributes
    ----------
    frequency_hz : numpy.ndarray
        Float64 shape ``(frequency,)`` canonical one-sided frequencies in Hz.
    psd_linear : numpy.ndarray
        Float64 shape ``(trial, epoch, frequency)`` in source-unit-squared/Hz.
        Unavailable epochs contain NaN across their frequency axis.
    psd_valid : numpy.ndarray
        Boolean shape ``(trial, epoch)``. ``False`` denotes an unavailable
        epoch, including one shorter than the configured Welch segment.
    epoch_names : tuple[str, str, str]
        Ordered names ``("whole", "before", "after")``.
    epoch_sample_counts : numpy.ndarray
        Int64 shape ``(epoch,)`` counts selected from the half-open windows.
    """

    frequency_hz: np.ndarray
    psd_linear: np.ndarray
    psd_valid: np.ndarray
    epoch_names: tuple[str, str, str]
    epoch_sample_counts: np.ndarray


def compute_welch_psd(
    values: np.ndarray,
    sample_rate_hz: float,
    config: PowerAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one one-sided Welch density PSD without implicit segment changes.

    Parameters
    ----------
    values : numpy.ndarray
        Finite one-dimensional shape ``(sample,)`` LFP values in one source
        voltage unit.
    sample_rate_hz : float
        Finite positive sampling rate in samples/second.
    config : PowerAnalysisConfig
        Validated power configuration. Its 0.5-s window determines
        ``nperseg=round(sample_rate_hz * welch_window_s)`` and its overlap
        fraction determines ``noverlap=floor(nperseg * overlap_fraction)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Frequency coordinates and linear density PSD, each shape ``(frequency,)``.
        Frequencies are Hz and PSD is source-unit-squared/Hz.

    Raises
    ------
    ValueError
        If inputs are invalid or contain fewer samples than one configured
        segment. Call :func:`compute_trial_epoch_psds` to retain such epochs as
        unavailable NaN rows rather than raising.
    """

    samples = _finite_vector(values, "values")
    rate_hz = _positive_rate(sample_rate_hz)
    nperseg, noverlap = _welch_lengths(rate_hz, config)
    if samples.size < nperseg:
        raise ValueError("values are shorter than the configured Welch segment")
    filtered = _apply_configured_notch(samples, rate_hz, config)
    frequency_hz, psd_linear = signal.welch(
        filtered,
        fs=rate_hz,
        window="hann_periodic",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    return np.asarray(frequency_hz, dtype=float), np.asarray(psd_linear, dtype=float)


def compute_trial_epoch_psds(
    trial_traces: np.ndarray,
    relative_time_s: np.ndarray,
    sample_rate_hz: float,
    windows: AnalysisWindowConfig,
    config: PowerAnalysisConfig,
) -> TrialEpochPSDResult:
    """Compute whole, before, and after PSDs on a shared canonical grid.

    Parameters
    ----------
    trial_traces : numpy.ndarray
        Float shape ``(trial, time)`` of event-aligned LFP values in one source
        voltage unit. Nonfinite values make only their affected epoch invalid.
    relative_time_s : numpy.ndarray
        Finite strictly increasing shape ``(time,)`` event-relative seconds.
        It must have spacing ``1 / sample_rate_hz`` within floating tolerance.
    sample_rate_hz : float
        Finite positive native sampling rate in samples/second.
    windows : AnalysisWindowConfig
        Validated whole, before, and after seconds intervals. Each is selected
        as ``[start_s, stop_s)``.
    config : PowerAnalysisConfig
        Validated Welch and canonical-frequency settings.

    Returns
    -------
    TrialEpochPSDResult
        PSDs with axes ``(trial, epoch, frequency)``. Epoch order is whole,
        before, after; unavailable epochs are all NaN and have ``False`` flags.
    """

    traces = np.asarray(trial_traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError("trial_traces must have shape (trial, time)")
    time_s = _time_vector(relative_time_s)
    if traces.shape[1] != time_s.size:
        raise ValueError("trial_traces time axis must match relative_time_s")
    rate_hz = _positive_rate(sample_rate_hz)
    _validate_native_spacing(time_s, rate_hz)
    epoch_names = ("whole", "before", "after")
    epoch_bounds = (
        (windows.whole_start_s, windows.whole_stop_s),
        (windows.before_start_s, windows.before_stop_s),
        (windows.after_start_s, windows.after_stop_s),
    )
    masks = tuple((time_s >= start) & (time_s < stop) for start, stop in epoch_bounds)
    counts = np.array([np.count_nonzero(mask) for mask in masks], dtype=np.int64)
    frequency_hz = _canonical_frequency_grid(rate_hz, config)
    psd_linear = np.full(
        (traces.shape[0], len(epoch_names), frequency_hz.size),
        np.nan,
        dtype=float,
    )
    psd_valid = np.zeros((traces.shape[0], len(epoch_names)), dtype=bool)
    nperseg, _ = _welch_lengths(rate_hz, config)

    for trial_index, trace in enumerate(traces):
        for epoch_index, mask in enumerate(masks):
            epoch_values = trace[mask]
            if epoch_values.size < nperseg or not np.all(np.isfinite(epoch_values)):
                continue
            native_hz, native_psd = compute_welch_psd(epoch_values, rate_hz, config)
            psd_linear[trial_index, epoch_index] = interpolate_linear_psd_to_canonical_grid(
                native_hz,
                native_psd,
                frequency_hz,
            )
            psd_valid[trial_index, epoch_index] = np.all(
                np.isfinite(psd_linear[trial_index, epoch_index])
            )
    return TrialEpochPSDResult(
        frequency_hz=frequency_hz,
        psd_linear=psd_linear,
        psd_valid=psd_valid,
        epoch_names=epoch_names,
        epoch_sample_counts=counts,
    )


def interpolate_linear_psd_to_canonical_grid(
    native_frequency_hz: np.ndarray,
    native_psd_linear: np.ndarray,
    canonical_frequency_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate finite linear PSDs over their final frequency axis.

    Parameters
    ----------
    native_frequency_hz : numpy.ndarray
        Finite strictly increasing shape ``(native_frequency,)`` in Hz.
    native_psd_linear : numpy.ndarray
        Float array with final shape ``(native_frequency,)`` in
        source-unit-squared/Hz. Leading axes are preserved.
    canonical_frequency_hz : numpy.ndarray
        Finite strictly increasing shape ``(frequency,)`` target grid in Hz.

    Returns
    -------
    numpy.ndarray
        Float array with shape ``native_psd_linear.shape[:-1] + (frequency,)``
        in unchanged linear units. Values outside native support or without two
        finite supporting points are NaN; no dB interpolation occurs.
    """

    native_hz = _frequency_vector(native_frequency_hz, "native_frequency_hz")
    canonical_hz = _frequency_vector(canonical_frequency_hz, "canonical_frequency_hz")
    native_psd = np.asarray(native_psd_linear, dtype=float)
    if native_psd.ndim < 1 or native_psd.shape[-1] != native_hz.size:
        raise ValueError("native_psd_linear final axis must match native_frequency_hz")
    flattened = native_psd.reshape((-1, native_hz.size))
    interpolated = np.full((flattened.shape[0], canonical_hz.size), np.nan, dtype=float)
    for row_index, row in enumerate(flattened):
        for run_start, run_stop in _finite_runs(np.isfinite(row)):
            run_hz = native_hz[run_start:run_stop]
            run_psd = row[run_start:run_stop]
            if run_hz.size == 1:
                exact_target = np.isclose(canonical_hz, run_hz[0], rtol=0.0, atol=1e-12)
                interpolated[row_index, exact_target] = run_psd[0]
                continue
            target_in_run = (canonical_hz >= run_hz[0]) & (canonical_hz <= run_hz[-1])
            interpolated[row_index, target_in_run] = np.interp(
                canonical_hz[target_in_run],
                run_hz,
                run_psd,
            )
    return interpolated.reshape(native_psd.shape[:-1] + (canonical_hz.size,))


def compute_session_reference_psd(
    whole_psd_linear: np.ndarray,
    valid_trial_mask: np.ndarray,
) -> np.ndarray:
    """Return the session-wide median whole-window PSD across valid trials.

    Parameters
    ----------
    whole_psd_linear : numpy.ndarray
        Float shape ``(trial, frequency)`` PSD values in source-unit-squared/Hz.
    valid_trial_mask : numpy.ndarray
        Boolean shape ``(trial,)`` marking objective, non-user-excluded whole
        trials. Condition and choice/context membership are intentionally absent.

    Returns
    -------
    numpy.ndarray
        Float shape ``(frequency,)`` frequency-wise median in unchanged linear
        units. Frequencies without finite valid trials are NaN.
    """

    psd = np.asarray(whole_psd_linear, dtype=float)
    valid = np.asarray(valid_trial_mask, dtype=bool)
    if psd.ndim != 2 or valid.ndim != 1 or psd.shape[0] != valid.size:
        raise ValueError(
            "whole PSD and valid-trial axes must have shapes "
            "(trial, frequency), (trial,)"
        )
    if not np.any(valid):
        return np.full(psd.shape[1], np.nan, dtype=float)
    return np.nanmedian(psd[valid], axis=0)


def compute_presession_reference_psd(
    time_s: np.ndarray,
    values: np.ndarray,
    first_start_time_s: float,
    sample_rate_hz: float,
    config: PowerAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compute a full-duration pre-session PSD or report it unavailable.

    Parameters
    ----------
    time_s, values : numpy.ndarray
        Matching one-dimensional continuous coordinates in absolute seconds and
        source voltage units. Sampling must exactly cover the requested interval.
    first_start_time_s : float
        Absolute first trial start time in seconds.
    sample_rate_hz : float
        Finite positive native sample rate in samples/second.
    config : PowerAnalysisConfig
        Validated reference duration and Welch settings.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, bool]
        Canonical frequency coordinates in Hz, a same-shape linear PSD in
        source-unit-squared/Hz, and availability. Missing/incomplete/nonfinite
        baselines return an all-NaN PSD and ``False`` without shortening time.
    """

    rate_hz = _positive_rate(sample_rate_hz)
    frequency_hz = _canonical_frequency_grid(rate_hz, config)
    times = _time_vector(time_s)
    samples = np.asarray(values, dtype=float)
    unavailable = (frequency_hz, np.full(frequency_hz.size, np.nan, dtype=float), False)
    if samples.ndim != 1 or samples.shape != times.shape:
        raise ValueError("time_s and values must have matching shape (sample,)")
    if not np.isfinite(first_start_time_s):
        raise ValueError("first_start_time_s must be finite")
    _validate_native_spacing(times, rate_hz)
    duration_s = float(config.presession_reference_duration_s)
    expected_count = _integral_sample_count(duration_s, rate_hz)
    expected_start_s = float(first_start_time_s) - duration_s
    expected_time_s = expected_start_s + np.arange(expected_count, dtype=float) / rate_hz
    mask = (times >= expected_start_s) & (times < float(first_start_time_s))
    if np.count_nonzero(mask) != expected_count:
        return unavailable
    baseline_times = times[mask]
    baseline_values = samples[mask]
    coordinate_tolerance_s = _time_coordinate_tolerance_s(expected_time_s)
    if not np.allclose(
        baseline_times,
        expected_time_s,
        rtol=0.0,
        atol=coordinate_tolerance_s,
    ):
        return unavailable
    if not np.all(np.isfinite(baseline_values)):
        return unavailable
    try:
        native_hz, native_psd = compute_welch_psd(baseline_values, rate_hz, config)
    except ValueError:
        return unavailable
    reference = interpolate_linear_psd_to_canonical_grid(native_hz, native_psd, frequency_hz)
    return frequency_hz, np.asarray(reference, dtype=float), bool(np.all(np.isfinite(reference)))


def normalize_psd_db(psd_linear: np.ndarray, reference_psd_linear: np.ndarray) -> np.ndarray:
    """Normalize linear PSD values to dB only where numerator and reference are positive.

    Parameters
    ----------
    psd_linear : numpy.ndarray
        Float array with final frequency axis in source-unit-squared/Hz.
    reference_psd_linear : numpy.ndarray
        Float shape ``(frequency,)`` positive reference in the same units.

    Returns
    -------
    numpy.ndarray
        Float shape matching ``psd_linear`` in dB. Invalid/nonpositive numerator
        or reference entries are NaN; no epsilon is introduced.
    """

    psd = np.asarray(psd_linear, dtype=float)
    reference = np.asarray(reference_psd_linear, dtype=float)
    if psd.ndim < 1 or reference.ndim != 1 or psd.shape[-1] != reference.size:
        raise ValueError("PSD final axis and reference must share shape (frequency,)")
    valid = np.isfinite(psd) & (psd > 0.0) & np.isfinite(reference) & (reference > 0.0)
    ratio = np.full(psd.shape, np.nan, dtype=float)
    np.divide(psd, reference, out=ratio, where=valid)
    normalized = np.full(psd.shape, np.nan, dtype=float)
    np.log10(ratio, out=normalized, where=valid)
    return 10.0 * normalized


def mean_band_power_linear(
    psd_linear: np.ndarray,
    frequency_hz: np.ndarray,
    band: FrequencyBandConfig,
) -> tuple[np.ndarray | float, float]:
    """Integrate mean linear power over disjoint retained frequency intervals.

    Parameters
    ----------
    psd_linear : numpy.ndarray
        Float array with final shape ``(frequency,)`` in source-unit-squared/Hz.
    frequency_hz : numpy.ndarray
        Finite strictly increasing shape ``(frequency,)`` in Hz.
    band : FrequencyBandConfig
        Outer band bounds and excluded open intervals in Hz. Boundary values are
        interpolated in linear units; excluded gaps are never bridged.

    Returns
    -------
    tuple[numpy.ndarray | float, float]
        Mean power over the final frequency axis in source-unit-squared/Hz and
        exact retained bandwidth in Hz. A row lacking finite support is NaN.
    """

    frequencies = _frequency_vector(frequency_hz, "frequency_hz")
    psd = np.asarray(psd_linear, dtype=float)
    if psd.ndim < 1 or psd.shape[-1] != frequencies.size:
        raise ValueError("psd_linear final axis must match frequency_hz")
    intervals = _retained_intervals(band)
    bandwidth_hz = float(sum(stop - start for start, stop, _, _ in intervals))
    flattened = psd.reshape((-1, frequencies.size))
    mean_values = np.full(flattened.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(flattened):
        integrals = [
            _integrate_linear_segment(
                row,
                frequencies,
                start,
                stop,
                start_is_excluded=start_is_excluded,
                stop_is_excluded=stop_is_excluded,
            )
            for start, stop, start_is_excluded, stop_is_excluded in intervals
        ]
        if all(np.isfinite(integral) for integral in integrals):
            mean_values[row_index] = float(np.sum(integrals) / bandwidth_hz)
    result = mean_values.reshape(psd.shape[:-1])
    return (float(result) if result.ndim == 0 else result), bandwidth_hz


def summarize_trial_median_iqr(
    values: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize trial-level values without changing their physical units.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric trial-resolved array. Its units, such as source-unit-squared/Hz,
        are retained. NaN observations are omitted per output cell.
    axis : int, default=0
        Axis over which trial observations are summarized.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Median, 25th percentile, and 75th percentile arrays with ``axis``
        removed and unchanged input units. All-NaN slices return NaN.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("values must have at least one axis")
    return (
        np.nanmedian(array, axis=axis),
        np.nanpercentile(array, 25.0, axis=axis),
        np.nanpercentile(array, 75.0, axis=axis),
    )


def _apply_configured_notch(
    values: np.ndarray,
    sample_rate_hz: float,
    config: PowerAnalysisConfig,
) -> np.ndarray:
    """Apply the configured zero-phase line-noise notch in unchanged voltage units."""

    if not config.notch_enabled:
        return values.copy()
    if sample_rate_hz <= 2.0 * config.notch_hz:
        raise ValueError("notch frequency must be below the native Nyquist frequency")
    numerator, denominator = signal.iirnotch(
        config.notch_hz,
        config.notch_quality_factor,
        fs=sample_rate_hz,
    )
    return np.asarray(signal.filtfilt(numerator, denominator, values), dtype=float)


def _canonical_frequency_grid(sample_rate_hz: float, config: PowerAnalysisConfig) -> np.ndarray:
    """Return the native-Nyquist 2-Hz-compatible cache grid in Hz."""

    step_hz = float(config.canonical_frequency_step_hz)
    if not np.isfinite(step_hz) or step_hz <= 0.0:
        raise ValueError("canonical_frequency_step_hz must be finite and positive")
    maximum_hz = float(sample_rate_hz) / 2.0
    count = int(np.floor(maximum_hz / step_hz))
    return step_hz * np.arange(count + 1, dtype=float)


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a nonempty finite one-dimensional numeric vector."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return vector


def _frequency_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Validate finite, strictly increasing nonnegative frequency coordinates in Hz."""

    vector = _finite_vector(values, name)
    if np.any(vector < 0.0) or np.any(np.diff(vector) <= 0.0):
        raise ValueError(f"{name} must be nonnegative and strictly increasing")
    return vector


def _time_vector(values: np.ndarray) -> np.ndarray:
    """Validate finite, strictly increasing time coordinates in seconds."""

    vector = _finite_vector(values, "relative_time_s")
    if np.any(np.diff(vector) <= 0.0):
        raise ValueError("relative_time_s must be strictly increasing")
    return vector


def _positive_rate(sample_rate_hz: float) -> float:
    """Validate a finite positive sampling rate in samples/second."""

    rate_hz = float(sample_rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    return rate_hz


def _welch_lengths(sample_rate_hz: float, config: PowerAnalysisConfig) -> tuple[int, int]:
    """Return configured integer Welch segment and overlap sample counts."""

    nperseg = int(round(sample_rate_hz * float(config.welch_window_s)))
    noverlap = int(np.floor(nperseg * float(config.welch_overlap_fraction)))
    if nperseg < 2 or noverlap < 0 or noverlap >= nperseg:
        raise ValueError("configured Welch lengths are invalid")
    return nperseg, noverlap


def _validate_native_spacing(time_s: np.ndarray, sample_rate_hz: float) -> None:
    """Require native-rate support within float precision at the coordinate scale."""

    if time_s.size < 2:
        raise ValueError("time coordinates require at least two samples")
    expected_step_s = 1.0 / sample_rate_hz
    coordinate_tolerance_s = _time_coordinate_tolerance_s(time_s)
    if not np.allclose(
        np.diff(time_s),
        expected_step_s,
        rtol=0.0,
        atol=coordinate_tolerance_s,
    ):
        raise ValueError("time coordinates do not match sample_rate_hz")


def _time_coordinate_tolerance_s(time_s: np.ndarray) -> float:
    """Return seconds tolerance for float64 coordinates without relaxing sample rate.

    Parameters
    ----------
    time_s : numpy.ndarray
        Finite time coordinates in seconds. Values may be event-relative or UTC
        Unix timestamps whose float64 spacing is coarser than one nanosecond.

    Returns
    -------
    float
        Maximum of one nanosecond and two float64 units in the last place at
        the largest coordinate magnitude. This covers unavoidable subtraction
        roundoff but remains far below one 2.5-kHz sample interval.
    """
    maximum_magnitude_s = float(np.max(np.abs(time_s)))
    coordinate_ulp_s = float(np.spacing(maximum_magnitude_s))
    return max(1e-9, 2.0 * coordinate_ulp_s)


def _retained_intervals(
    band: FrequencyBandConfig,
) -> tuple[tuple[float, float, bool, bool], ...]:
    """Split one band into retained intervals with open excluded-gap boundaries."""

    lower_hz = float(band.lower_hz)
    upper_hz = float(band.upper_hz)
    if not np.isfinite(lower_hz) or not np.isfinite(upper_hz) or lower_hz >= upper_hz:
        raise ValueError("band bounds must be finite and increasing")
    start_hz = lower_hz
    intervals: list[tuple[float, float, bool, bool]] = []
    previous_stop_hz = lower_hz
    for excluded_start_hz, excluded_stop_hz in sorted(band.excluded_intervals_hz):
        excluded_start_hz = float(excluded_start_hz)
        excluded_stop_hz = float(excluded_stop_hz)
        if (
            not np.isfinite(excluded_start_hz)
            or not np.isfinite(excluded_stop_hz)
            or excluded_start_hz < lower_hz
            or excluded_stop_hz > upper_hz
            or excluded_start_hz >= excluded_stop_hz
            or excluded_start_hz < previous_stop_hz
        ):
            raise ValueError("band exclusions must be finite, ordered, and within band bounds")
        if excluded_start_hz > start_hz:
            intervals.append((start_hz, excluded_start_hz, start_hz != lower_hz, True))
        start_hz = excluded_stop_hz
        previous_stop_hz = excluded_stop_hz
    if start_hz < upper_hz:
        intervals.append((start_hz, upper_hz, start_hz != lower_hz, False))
    if not intervals:
        raise ValueError("band has no retained frequency bandwidth")
    return tuple(intervals)


def _integrate_linear_segment(
    psd_linear: np.ndarray,
    frequency_hz: np.ndarray,
    start_hz: float,
    stop_hz: float,
    *,
    start_is_excluded: bool,
    stop_is_excluded: bool,
) -> float:
    """Integrate one retained interval with explicit outer/open endpoint rules."""

    interior = (frequency_hz > start_hz) & (frequency_hz < stop_hz)
    if np.any(~np.isfinite(psd_linear[interior])):
        return np.nan
    start_value = _segment_endpoint_value(
        psd_linear,
        frequency_hz,
        start_hz,
        excluded_boundary=start_is_excluded,
        retained_side="right",
    )
    stop_value = _segment_endpoint_value(
        psd_linear,
        frequency_hz,
        stop_hz,
        excluded_boundary=stop_is_excluded,
        retained_side="left",
    )
    if not np.isfinite(start_value) or not np.isfinite(stop_value):
        return np.nan
    retained_interior = interior & np.isfinite(psd_linear)
    coordinates = np.concatenate(
        ([start_hz], frequency_hz[retained_interior], [stop_hz])
    )
    values = np.concatenate(([start_value], psd_linear[retained_interior], [stop_value]))
    return float(np.trapezoid(values, coordinates))


def _segment_endpoint_value(
    psd_linear: np.ndarray,
    frequency_hz: np.ndarray,
    boundary_hz: float,
    *,
    excluded_boundary: bool,
    retained_side: str,
) -> float:
    """Return an included endpoint or one-sided estimate at an excluded gap boundary."""

    exact = np.isclose(frequency_hz, boundary_hz, rtol=0.0, atol=1e-12)
    if not excluded_boundary:
        if np.any(exact):
            exact_values = psd_linear[exact]
            return float(exact_values[0]) if np.all(np.isfinite(exact_values)) else np.nan
        finite = np.isfinite(psd_linear)
        if np.count_nonzero(finite) < 2:
            return np.nan
        finite_frequency_hz = frequency_hz[finite]
        if boundary_hz < finite_frequency_hz[0] or boundary_hz > finite_frequency_hz[-1]:
            return np.nan
        return float(np.interp(boundary_hz, finite_frequency_hz, psd_linear[finite]))

    if retained_side == "left":
        side = (frequency_hz < boundary_hz) & np.isfinite(psd_linear)
        side_frequency_hz = frequency_hz[side][-2:]
        side_psd = psd_linear[side][-2:]
    elif retained_side == "right":
        side = (frequency_hz > boundary_hz) & np.isfinite(psd_linear)
        side_frequency_hz = frequency_hz[side][:2]
        side_psd = psd_linear[side][:2]
    else:
        raise ValueError("retained_side must be left or right")
    if side_frequency_hz.size < 2:
        return np.nan
    slope = (side_psd[1] - side_psd[0]) / (side_frequency_hz[1] - side_frequency_hz[0])
    return float(side_psd[0] + slope * (boundary_hz - side_frequency_hz[0]))


def _finite_runs(finite_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return half-open index runs of contiguous finite samples on one axis."""

    mask = np.asarray(finite_mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    stops = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))


def _integral_sample_count(duration_s: float, sample_rate_hz: float) -> int:
    """Return an exact duration-by-rate sample count or reject fractional support."""

    sample_count_float = float(duration_s) * float(sample_rate_hz)
    sample_count = int(round(sample_count_float))
    if sample_count < 1 or not np.isclose(sample_count_float, sample_count, rtol=0.0, atol=1e-9):
        raise ValueError("duration and sample rate must define an integral positive sample count")
    return sample_count
