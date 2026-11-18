"""Offline ITPC, ISPC, and trial PLV summaries for cached LFP synchrony views.

This module consumes complex unit-phase tensors prepared by
``lfp_phase_clustering``. It deliberately does not compute wavelets or
interpolate phase angles; callers must use that module's real/imaginary
coefficient interpolation onto the shared event-relative grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.neural_analysis.lfp.config import FrequencyBandConfig


@dataclass(frozen=True)
class PhaseClusteringSummary:
    """Condition-resolved ITPC and ISPC arrays on phase-tensor axes.

    ``itpc`` and its count have axes ``(condition, site, frequency, time)``.
    ``ispc``, signed ``ispc_phase_offset_rad``, and its count have axes
    ``(condition, pair, frequency, time)``. Magnitudes are dimensionless;
    offsets are radians for ordered site A minus site B differences.
    """

    itpc: np.ndarray
    itpc_effective_trial_count: np.ndarray
    ispc: np.ndarray
    ispc_phase_offset_rad: np.ndarray
    ispc_effective_trial_count: np.ndarray
    condition_names: tuple[str, ...]
    site_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TrialPLVSummary:
    """Per-trial, per-epoch PLV with explicit coverage diagnostics.

    Every array has axes ``(pair, trial, epoch, frequency)``. PLV is
    dimensionless and NaN when ``computable`` is false. Offsets are radians;
    counts are paired valid samples; fractions are dimensionless coverage.
    """

    plv_by_frequency: np.ndarray
    plv_phase_offset_rad: np.ndarray
    valid_sample_count: np.ndarray
    valid_sample_fraction: np.ndarray
    computable: np.ndarray
    epoch_names: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapBandMean:
    """One deterministic bootstrap scalar estimate and its uncertainty.

    ``estimate``, ``ci_low``, and ``ci_high`` are dimensionless means in the
    input metric's units. ``selected_trial_count`` is finite masked trials;
    ``unstable`` is true exactly when that count is below ten.
    """

    estimate: float
    ci_low: float
    ci_high: float
    selected_trial_count: int
    bootstrap_count: int
    unstable: bool


@dataclass(frozen=True)
class PhaseBandBootstrapSummary:
    """Seeded phase-clustering estimates and trial-resampling uncertainty.

    ``estimate``, ``ci_low``, ``q25``, ``median``, ``q75``, ``ci_high``,
    ``selected_trial_count``, and ``unstable`` have axes ``(epoch, band)``.
    ``bootstrap_values`` has axes ``(bootstrap, epoch, band)``. Values are
    dimensionless phase-clustering magnitudes; trial counts and bootstrap
    counts are categorical integers.
    """

    estimate: np.ndarray
    ci_low: np.ndarray
    q25: np.ndarray
    median: np.ndarray
    q75: np.ndarray
    ci_high: np.ndarray
    selected_trial_count: np.ndarray
    unstable: np.ndarray
    bootstrap_values: np.ndarray


@dataclass(frozen=True)
class PLVExemplarSelection:
    """Deterministic low/high illustrative trial identifiers or missing values."""

    low_trial_index: int | None
    high_trial_index: int | None


def _unit_phase_and_valid(phase: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize finite nonzero complex vectors without changing their axes."""

    array = np.asarray(phase)
    mask = np.asarray(valid, dtype=bool).copy()
    finite_nonzero = np.isfinite(array.real) & np.isfinite(array.imag) & (np.abs(array) > 0.0)
    mask &= finite_nonzero
    normalized = np.zeros(array.shape, dtype=np.complex64)
    normalized[mask] = array[mask] / np.abs(array[mask])
    return normalized, mask


def _validate_phase_tensor(
    phase_tensor: np.ndarray,
    valid_mask: np.ndarray,
    condition_names: Sequence[str],
    condition_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Validate shared phase tensor and documented trial-by-condition membership."""

    phase = np.asarray(phase_tensor)
    valid = np.asarray(valid_mask, dtype=bool)
    names = tuple(str(name) for name in condition_names)
    membership = np.asarray(condition_masks, dtype=bool)
    if phase.ndim != 4:
        raise ValueError("phase_tensor must have shape (site, frequency, trial, time).")
    if valid.shape != phase.shape:
        raise ValueError("valid_mask must match phase_tensor shape.")
    if membership.shape != (phase.shape[2], len(names)):
        raise ValueError("condition_masks must have shape (trial, condition).")
    if not names or len(set(names)) != len(names):
        raise ValueError("condition_names must be nonempty and unique.")
    return phase, valid, names, membership


def compute_phase_clustering_summary(
    phase_tensor: np.ndarray,
    valid_mask: np.ndarray,
    condition_names: Sequence[str],
    condition_masks: np.ndarray,
    site_pairs: Sequence[tuple[int, int]],
) -> PhaseClusteringSummary:
    """Compute condition ITPC and ordered-pair ISPC from unit complex phase.

    Parameters
    ----------
    phase_tensor : np.ndarray
        Complex phase with shape ``(site, frequency, trial, time)`` and no
        physical units. It must originate from shared coefficient interpolation.
    valid_mask : np.ndarray
        Boolean numerical validity with the same axes as ``phase_tensor``.
    condition_names : Sequence[str]
        Unique condition labels with shape ``(condition,)``.
    condition_masks : np.ndarray
        Boolean membership with shape ``(trial, condition)``.
    site_pairs : Sequence[tuple[int, int]]
        Ordered distinct site indices, defining phase A minus phase B.

    Returns
    -------
    PhaseClusteringSummary
        Dimensionless ITPC/ISPC and counts; ISPC phase offsets are radians.
    """

    phase, valid, names, membership = _validate_phase_tensor(
        phase_tensor,
        valid_mask,
        condition_names,
        condition_masks,
    )
    pairs_list: list[tuple[int, int]] = []
    for pair in site_pairs:
        if len(pair) != 2:
            raise ValueError("Each site pair must contain exactly two indices.")
        first, second = pair
        if not isinstance(first, (int, np.integer)) or not isinstance(second, (int, np.integer)):
            raise ValueError("site_pairs must contain integer site indices.")
        pairs_list.append((int(first), int(second)))
    pairs = tuple(pairs_list)
    for first, second in pairs:
        invalid_pair = first == second or first < 0 or second < 0
        invalid_pair |= first >= phase.shape[0] or second >= phase.shape[0]
        if invalid_pair:
            raise ValueError("site_pairs must contain distinct valid site indices.")

    normalized, numerical_valid = _unit_phase_and_valid(phase, valid)
    shape = (len(names), phase.shape[0], phase.shape[1], phase.shape[3])
    itpc = np.full(shape, np.nan, dtype=float)
    itpc_count = np.zeros(shape, dtype=np.int32)
    pair_shape = (len(names), len(pairs), phase.shape[1], phase.shape[3])
    ispc = np.full(pair_shape, np.nan, dtype=float)
    offset = np.full(pair_shape, np.nan, dtype=float)
    ispc_count = np.zeros(pair_shape, dtype=np.int32)

    for condition_index in range(len(names)):
        selected = membership[:, condition_index]
        selected_indices = np.flatnonzero(selected)
        for site_index in range(phase.shape[0]):
            values = np.take(normalized[site_index], selected_indices, axis=1)
            mask = np.take(numerical_valid[site_index], selected_indices, axis=1)
            vector_sum = np.sum(
                np.where(mask, values, 0.0j),
                axis=1,
                dtype=np.complex128,
            )
            count = np.sum(mask, axis=1, dtype=np.int32)
            itpc_count[condition_index, site_index] = count
            np.divide(
                np.abs(vector_sum),
                count,
                out=itpc[condition_index, site_index],
                where=count > 0,
            )
        for pair_index, (site_a, site_b) in enumerate(pairs):
            valid_a = np.take(numerical_valid[site_a], selected_indices, axis=1)
            valid_b = np.take(numerical_valid[site_b], selected_indices, axis=1)
            pair_valid = valid_a & valid_b
            phase_a = np.take(normalized[site_a], selected_indices, axis=1)
            phase_b = np.take(normalized[site_b], selected_indices, axis=1)
            difference = phase_a * np.conjugate(phase_b)
            vector_sum = np.sum(
                np.where(pair_valid, difference, 0.0j),
                axis=1,
                dtype=np.complex128,
            )
            count = np.sum(pair_valid, axis=1, dtype=np.int32)
            ispc_count[condition_index, pair_index] = count
            magnitude = np.abs(vector_sum)
            np.divide(magnitude, count, out=ispc[condition_index, pair_index], where=count > 0)
            offset[condition_index, pair_index][count > 0] = np.angle(vector_sum[count > 0])
    return PhaseClusteringSummary(itpc, itpc_count, ispc, offset, ispc_count, names, pairs)


def compute_trial_plv_by_frequency(
    relative_phase_complex: np.ndarray,
    valid_mask: np.ndarray,
    relative_time_s: np.ndarray,
    epoch_windows: Mapping[str, tuple[float, float]],
    minimum_valid_fraction: float = 0.8,
) -> TrialPLVSummary:
    """Compute direct per-frequency PLV over half-open epochs for every trial.

    Parameters use pair/trial/frequency/time axes for complex phase and its
    matching validity. ``relative_time_s`` has shape ``(time,)`` in seconds;
    epoch bounds are seconds and retain samples ``start <= time < stop``.

    Returns arrays with shape ``(pair, trial, epoch, frequency)``. PLV is
    dimensionless, offsets are radians, and invalid estimates are NaN.
    """

    phase = np.asarray(relative_phase_complex)
    valid = np.asarray(valid_mask, dtype=bool)
    times = np.asarray(relative_time_s, dtype=float)
    if phase.ndim != 4 or valid.shape != phase.shape:
        raise ValueError("phase and valid_mask must share shape (pair, trial, frequency, time).")
    if times.ndim != 1 or phase.shape[3] != times.size or times.size < 2:
        raise ValueError("relative_time_s must be a one-dimensional grid matching the time axis.")
    if not np.isfinite(times).all():
        raise ValueError("relative_time_s must contain finite seconds.")
    time_steps = np.diff(times)
    sample_step_s = float(time_steps[0])
    if sample_step_s <= 0.0 or not np.allclose(time_steps, sample_step_s, rtol=1e-9, atol=1e-12):
        raise ValueError("relative_time_s must be a strictly increasing uniform grid.")
    if not 0.0 <= float(minimum_valid_fraction) <= 1.0:
        raise ValueError("minimum_valid_fraction must be in [0, 1].")
    names = tuple(str(name) for name in epoch_windows)
    windows = tuple(epoch_windows[name] for name in names)
    if not names:
        raise ValueError("epoch_windows must be nonempty.")
    normalized, numerical_valid = _unit_phase_and_valid(phase, valid)
    shape = (phase.shape[0], phase.shape[1], len(names), phase.shape[2])
    plv = np.full(shape, np.nan, dtype=float)
    offset = np.full(shape, np.nan, dtype=float)
    count = np.zeros(shape, dtype=np.int32)
    fraction = np.zeros(shape, dtype=float)
    computable = np.zeros(shape, dtype=bool)
    for epoch_index, (start_s, stop_s) in enumerate(windows):
        if not np.isfinite(start_s) or not np.isfinite(stop_s) or float(start_s) >= float(stop_s):
            raise ValueError("epoch_windows must contain finite increasing second bounds.")
        expected_count_float = (float(stop_s) - float(start_s)) / sample_step_s
        expected_count = int(np.rint(expected_count_float))
        if expected_count < 1 or not np.isclose(expected_count_float, expected_count, atol=1e-9):
            raise ValueError("Each epoch duration must contain an integral number of grid samples.")
        positions = (times >= float(start_s)) & (times < float(stop_s))
        epoch_phase = normalized[..., positions]
        epoch_valid = numerical_valid[..., positions]
        vector_sum = np.sum(
            np.where(epoch_valid, epoch_phase, 0.0j),
            axis=3,
            dtype=np.complex128,
        )
        epoch_count = np.sum(epoch_valid, axis=3, dtype=np.int32)
        count[:, :, epoch_index] = epoch_count
        fraction[:, :, epoch_index] = epoch_count / float(expected_count)
        accepted = (epoch_count >= 2) & (
            fraction[:, :, epoch_index] >= float(minimum_valid_fraction)
        )
        computable[:, :, epoch_index] = accepted
        accepted_sum = vector_sum[accepted]
        plv[:, :, epoch_index][accepted] = np.abs(accepted_sum) / epoch_count[accepted]
        offset[:, :, epoch_index][accepted] = np.angle(accepted_sum)
    return TrialPLVSummary(plv, offset, count, fraction, computable, names)


def aggregate_trial_plv_bands(
    plv_by_frequency: np.ndarray,
    frequencies_hz: np.ndarray,
    bands: Sequence[FrequencyBandConfig],
) -> np.ndarray:
    """Average retained sampled PLVs across each named frequency band.

    ``plv_by_frequency`` has shape ``(pair, trial, epoch, frequency)`` and
    dimensionless values. Frequencies are Hz. Returns dimensionless values with
    shape ``(pair, trial, epoch, band)``; excluded interval endpoints are not
    retained.
    """

    values = np.asarray(plv_by_frequency, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a nonempty one-dimensional vector.")
    if not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_hz must be finite and strictly increasing.")
    if values.ndim != 4 or values.shape[3] != frequencies.size:
        raise ValueError("plv_by_frequency must have shape (pair, trial, epoch, frequency).")
    output = np.full(values.shape[:3] + (len(bands),), np.nan, dtype=float)
    for band_index, band in enumerate(bands):
        retained = (frequencies >= float(band.lower_hz)) & (frequencies <= float(band.upper_hz))
        for lower_hz, upper_hz in band.excluded_intervals_hz:
            retained &= ~((frequencies >= float(lower_hz)) & (frequencies <= float(upper_hz)))
        if np.any(retained):
            retained_values = values[..., retained]
            finite = np.isfinite(retained_values)
            total = np.sum(np.where(finite, retained_values, 0.0), axis=3)
            count = np.sum(finite, axis=3, dtype=np.int32)
            np.divide(total, count, out=output[..., band_index], where=count > 0)
    return output


def bootstrap_phase_clustering_bands(
    phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    trial_mask: np.ndarray,
    frequencies_hz: np.ndarray,
    relative_time_s: np.ndarray,
    epoch_windows: Mapping[str, tuple[float, float]],
    bands: Sequence[FrequencyBandConfig],
    bootstrap_count: int,
    seed: int,
) -> PhaseBandBootstrapSummary:
    """Bootstrap nonlinear phase clustering over trial positions.

    Parameters
    ----------
    phase_vectors : numpy.ndarray
        Complex unit vectors with axes ``(trial, frequency, time)``.
    valid_mask : numpy.ndarray
        Boolean numerical validity with the same axes as ``phase_vectors``.
    trial_mask : numpy.ndarray
        Boolean shape ``(trial,)`` condition/filter/entity selector.
    frequencies_hz : numpy.ndarray
        Finite increasing shape ``(frequency,)`` coordinates in Hz.
    relative_time_s : numpy.ndarray
        Finite increasing shape ``(time,)`` coordinates in seconds.
    epoch_windows : Mapping[str, tuple[float, float]]
        Ordered half-open epoch bounds in seconds.
    bands : Sequence[FrequencyBandConfig]
        Named inclusive outer Hz bounds with open-gap sample exclusions.
    bootstrap_count : int
        Positive number of with-replacement trial resamples.
    seed : int
        Deterministic NumPy random seed.

    Returns
    -------
    PhaseBandBootstrapSummary
        Dimensionless estimates and five bootstrap quantiles on ``(epoch,
        band)`` axes, plus seeded bootstrap scalar values on ``(bootstrap,
        epoch, band)`` axes.

    Raises
    ------
    ValueError
        If any axis, coordinate, window, band, count, or seed is invalid.
    """
    phase = np.asarray(phase_vectors)
    valid = np.asarray(valid_mask, dtype=bool)
    selected = np.asarray(trial_mask, dtype=bool)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if phase.ndim != 3 or valid.shape != phase.shape:
        raise ValueError("phase_vectors and valid_mask must share trial/frequency/time axes")
    if selected.shape != (phase.shape[0],):
        raise ValueError("trial_mask must match the phase trial axis")
    if frequencies.shape != (phase.shape[1],) or times.shape != (phase.shape[2],):
        raise ValueError("phase coordinates do not match frequency/time axes")
    if (
        not np.isfinite(frequencies).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(frequencies) <= 0.0)
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("phase coordinates must be finite and strictly increasing")
    if int(bootstrap_count) != bootstrap_count or int(bootstrap_count) < 1:
        raise ValueError("bootstrap_count must be a positive integer")
    if int(seed) != seed:
        raise ValueError("seed must be an integer")
    epoch_names = tuple(str(name) for name in epoch_windows)
    if not epoch_names or not bands:
        raise ValueError("epoch_windows and bands must be nonempty")
    point_masks = _phase_band_point_masks(
        frequencies,
        times,
        epoch_windows,
        bands,
    )
    trial_positions = np.flatnonzero(selected)
    shape = (len(epoch_names), len(bands))
    estimate = np.full(shape, np.nan, dtype=float)
    ci_low = np.full(shape, np.nan, dtype=float)
    q25 = np.full(shape, np.nan, dtype=float)
    median = np.full(shape, np.nan, dtype=float)
    q75 = np.full(shape, np.nan, dtype=float)
    ci_high = np.full(shape, np.nan, dtype=float)
    selected_count = np.zeros(shape, dtype=np.int32)
    unstable = np.ones(shape, dtype=bool)
    bootstrap_values = np.full((int(bootstrap_count),) + shape, np.nan, dtype=float)
    if not trial_positions.size:
        return PhaseBandBootstrapSummary(
            estimate,
            ci_low,
            q25,
            median,
            q75,
            ci_high,
            selected_count,
            unstable,
            bootstrap_values,
        )

    normalized, numerical_valid = _unit_phase_and_valid(phase, valid)
    selected_phase = np.take(normalized, trial_positions, axis=0)
    selected_valid = np.take(numerical_valid, trial_positions, axis=0)
    observed = _weighted_phase_clustering(
        selected_phase,
        selected_valid,
        np.ones((1, trial_positions.size), dtype=float),
    )[0]
    for epoch_index in range(shape[0]):
        for band_index in range(shape[1]):
            point_mask = point_masks[epoch_index][band_index]
            estimate[epoch_index, band_index] = _finite_point_mean(
                observed,
                point_mask,
            )
            contributing = np.any(selected_valid[:, point_mask], axis=1)
            selected_count[epoch_index, band_index] = np.count_nonzero(contributing)
            unstable[epoch_index, band_index] = (
                selected_count[epoch_index, band_index] < 10
            )

    generator = np.random.default_rng(int(seed))
    chunk_size = min(8, int(bootstrap_count))
    for start in range(0, int(bootstrap_count), chunk_size):
        stop = min(start + chunk_size, int(bootstrap_count))
        draw_positions = generator.integers(
            0,
            trial_positions.size,
            size=(stop - start, trial_positions.size),
        )
        weights = np.zeros((stop - start, trial_positions.size), dtype=float)
        for draw_index, draw in enumerate(draw_positions):
            weights[draw_index] = np.bincount(
                draw,
                minlength=trial_positions.size,
            )
        clustering = _weighted_phase_clustering(
            selected_phase,
            selected_valid,
            weights,
        )
        for epoch_index in range(shape[0]):
            for band_index in range(shape[1]):
                point_mask = point_masks[epoch_index][band_index]
                for chunk_index in range(stop - start):
                    bootstrap_values[start + chunk_index, epoch_index, band_index] = (
                        _finite_point_mean(clustering[chunk_index], point_mask)
                    )
    for epoch_index in range(shape[0]):
        for band_index in range(shape[1]):
            values = bootstrap_values[:, epoch_index, band_index]
            finite = values[np.isfinite(values)]
            if finite.size:
                quantiles = np.percentile(
                    finite,
                    q=(2.5, 25.0, 50.0, 75.0, 97.5),
                    method="linear",
                )
                (
                    ci_low[epoch_index, band_index],
                    q25[epoch_index, band_index],
                    median[epoch_index, band_index],
                    q75[epoch_index, band_index],
                    ci_high[epoch_index, band_index],
                ) = quantiles
    return PhaseBandBootstrapSummary(
        estimate,
        ci_low,
        q25,
        median,
        q75,
        ci_high,
        selected_count,
        unstable,
        bootstrap_values,
    )


def _phase_band_point_masks(
    frequencies_hz: np.ndarray,
    relative_time_s: np.ndarray,
    epoch_windows: Mapping[str, tuple[float, float]],
    bands: Sequence[FrequencyBandConfig],
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Return ``(epoch, band)`` boolean frequency-by-time point masks."""
    masks = []
    for start_s, stop_s in epoch_windows.values():
        if not np.isfinite(start_s) or not np.isfinite(stop_s) or start_s >= stop_s:
            raise ValueError("epoch windows must contain finite increasing seconds")
        time_mask = (relative_time_s >= start_s) & (relative_time_s < stop_s)
        if not np.any(time_mask):
            raise ValueError("epoch window contains no phase samples")
        epoch_masks = []
        for band in bands:
            frequency_mask = (frequencies_hz >= band.lower_hz) & (
                frequencies_hz <= band.upper_hz
            )
            for lower_hz, upper_hz in band.excluded_intervals_hz:
                if lower_hz >= upper_hz:
                    raise ValueError("band exclusions must contain increasing Hz bounds")
                frequency_mask &= ~(
                    (frequencies_hz >= lower_hz) & (frequencies_hz <= upper_hz)
                )
            if not np.any(frequency_mask):
                raise ValueError(f"band {band.name!r} contains no retained frequency")
            epoch_masks.append(frequency_mask[:, None] & time_mask[None, :])
        masks.append(tuple(epoch_masks))
    return tuple(masks)


def _weighted_phase_clustering(
    phase: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted trial clustering maps with float64/complex128 sums."""
    valid_phase = np.where(valid, phase, 0.0j)
    vector_sum = np.tensordot(weights, valid_phase, axes=((1,), (0,)))
    valid_count = np.tensordot(weights, valid, axes=((1,), (0,)))
    output = np.full(vector_sum.shape, np.nan, dtype=float)
    np.divide(
        np.abs(vector_sum),
        valid_count,
        out=output,
        where=valid_count > 0.0,
    )
    return output


def _finite_point_mean(values: np.ndarray, point_mask: np.ndarray) -> float:
    """Return one finite selected-point mean or NaN when no point contributes."""
    selected = np.asarray(values, dtype=float)[point_mask]
    finite = selected[np.isfinite(selected)]
    return float(np.mean(finite)) if finite.size else np.nan


def bootstrap_band_mean(
    trial_values: np.ndarray,
    trial_mask: np.ndarray,
    bootstrap_count: int,
    seed: int,
) -> BootstrapBandMean:
    """Bootstrap a scalar trial mean using only finite selected observations.

    Inputs have shape ``(trial,)`` and dimensionless or common metric units.
    The result preserves those units, uses exactly ``bootstrap_count`` resamples,
    and reports an instability flag when fewer than ten finite trials remain.
    """

    values = np.asarray(trial_values, dtype=float)
    selected = np.asarray(trial_mask, dtype=bool)
    if values.ndim != 1 or selected.ndim != 1 or values.shape != selected.shape:
        raise ValueError("trial_values and trial_mask must share shape (trial,).")
    if int(bootstrap_count) != bootstrap_count or int(bootstrap_count) < 1:
        raise ValueError("trial_values, trial_mask, and bootstrap_count are invalid.")
    if int(seed) != seed:
        raise ValueError("seed must be an integer.")
    observations = values[selected & np.isfinite(values)]
    count = int(observations.size)
    if count == 0:
        return BootstrapBandMean(np.nan, np.nan, np.nan, 0, int(bootstrap_count), True)
    generator = np.random.default_rng(int(seed))
    draw_positions = generator.integers(0, count, size=(int(bootstrap_count), count))
    draw_means = observations[draw_positions].mean(axis=1)
    return BootstrapBandMean(
        float(np.round(observations.mean(), decimals=15)),
        float(np.percentile(draw_means, 2.5)),
        float(np.percentile(draw_means, 97.5)),
        count,
        int(bootstrap_count),
        count < 10,
    )


def select_plv_exemplars(
    trial_indices: np.ndarray,
    values: np.ndarray,
    eligible: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> PLVExemplarSelection:
    """Select nearest-percentile PLV exemplars with earliest-index tie breaks.

    Inputs have shape ``(trial,)``; indices are integer trial-table rows and
    values are dimensionless PLV. Fewer than two finite eligible observations
    produces unavailable ``None`` selections.
    """

    raw_indices = np.asarray(trial_indices)
    if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("trial_indices must be a one-dimensional integer vector.")
    indices = raw_indices.astype(np.int64, copy=True)
    metric = np.asarray(values, dtype=float)
    eligibility = np.asarray(eligible, dtype=bool)
    if metric.ndim != 1 or eligibility.ndim != 1:
        raise ValueError("values and eligible must be one-dimensional.")
    if indices.shape != metric.shape or eligibility.shape != metric.shape:
        raise ValueError("trial_indices, values, and eligible must share shape (trial,).")
    mask = eligibility & np.isfinite(metric)
    percentile_values = (float(low_percentile), float(high_percentile))
    valid_percentiles = all(
        np.isfinite(percentile) and 0.0 <= percentile <= 100.0
        for percentile in percentile_values
    )
    if not valid_percentiles:
        raise ValueError("Percentiles must be finite values in [0, 100].")
    if percentile_values[0] > percentile_values[1]:
        raise ValueError("low_percentile must not exceed high_percentile.")
    if int(np.count_nonzero(mask)) < 2:
        return PLVExemplarSelection(None, None)

    def choose(percentile: float) -> int:
        """Return the earliest trial index nearest one requested percentile."""

        target = float(np.percentile(metric[mask], percentile))
        distance = np.abs(metric - target)
        minimum = np.min(distance[mask])
        tied = mask & np.isclose(distance, minimum, rtol=0.0, atol=1e-12)
        return int(np.min(indices[tied]))

    return PLVExemplarSelection(choose(low_percentile), choose(high_percentile))


def make_synchrony_cache_arrays(
    relative_time_s: np.ndarray,
    source_trace: np.ndarray,
    band_filtered_trace: np.ndarray,
    hilbert_phase_rad: np.ndarray,
    source_voltage_unit: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Validate cacheable exemplar traces and return arrays plus axis/unit schema.

    Source traces have shape ``(site, trial, time)`` in ``source_voltage_unit``.
    Filtered traces share axes ``(site, trial, band, time)`` and those units;
    Hilbert phase has those axes in radians. Time is seconds. No full wavelet
    coefficient tensor is accepted or returned.
    """

    times = np.asarray(relative_time_s, dtype=float)
    source = np.asarray(source_trace, dtype=float)
    filtered = np.asarray(band_filtered_trace, dtype=float)
    phase = np.asarray(hilbert_phase_rad, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("relative_time_s must be a nonempty one-dimensional time axis.")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("relative_time_s must be finite, nonempty, and strictly increasing.")
    if source.ndim != 3 or source.shape[2] != times.size:
        raise ValueError("source_trace must have shape (site, trial, time).")
    if filtered.ndim != 4 or phase.ndim != 4:
        raise ValueError("filtered and phase traces must have shape (site, trial, band, time).")
    expected = (source.shape[0], source.shape[1], filtered.shape[2], times.size)
    if filtered.shape != expected or phase.shape != expected:
        raise ValueError("filtered and phase traces must have shape (site, trial, band, time).")
    if not isinstance(source_voltage_unit, str) or not source_voltage_unit:
        raise ValueError("source_voltage_unit must be a nonempty string.")
    arrays = {
        "relative_time_s": times.copy(),
        "source_trace": source.copy(),
        "band_filtered_trace": filtered.copy(),
        "hilbert_phase_rad": phase.copy(),
    }
    schema = {
        "relative_time_s": {"axes": ["time"], "units": "s"},
        "source_trace": {"axes": ["site", "trial", "time"], "units": source_voltage_unit},
        "band_filtered_trace": {
            "axes": ["site", "trial", "band", "time"],
            "units": source_voltage_unit,
        },
        "hilbert_phase_rad": {"axes": ["site", "trial", "band", "time"], "units": "rad"},
    }
    return arrays, schema
