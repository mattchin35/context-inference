"""Uniform-grid interpolation and segmented PPC edge sufficient statistics.

This module contains the numerical S1 kernel only.  It does not perform file
I/O, checkpointing, multiprocessing, or pipeline orchestration.  Geometry is
built once from source-trial spikes and reused for each bounded target-edge
reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_INT64_INFO = np.iinfo(np.int64)
_INT64_MIN = int(_INT64_INFO.min)
_INT64_MAX = int(_INT64_INFO.max)

# Keep planned allocation accounting tied to the arrays the kernel actually owns.
_BOOL_BYTES = np.dtype(bool).itemsize
_INT64_BYTES = np.dtype(np.int64).itemsize
_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_COMPLEX64_BYTES = np.dtype(np.complex64).itemsize
_COMPLEX128_BYTES = np.dtype(np.complex128).itemsize
_GEOMETRY_BYTES_PER_SPIKE = 2 * _INT64_BYTES + _FLOAT64_BYTES + 2 * _BOOL_BYTES
_GATHER_BYTES_PER_CELL = (
    2 * _COMPLEX64_BYTES
    + _COMPLEX128_BYTES
    + _FLOAT64_BYTES
    + _COMPLEX64_BYTES
    + 3 * _BOOL_BYTES
)
_SEGMENTED_EDGE_BYTES_PER_CELL = 2 * (_COMPLEX128_BYTES + _INT64_BYTES)


@dataclass(frozen=True)
class SourceTrialSpikeGeometry:
    """Owned interpolation geometry for one source trial and one unit block.

    Parameters
    ----------
    source_trial_index : numpy.ndarray
        Int64 scalar stable physical trial-table identity.  It is categorical
        and has no physical units.
    unit_ids : tuple[str, ...]
        Unique stable unit identities defining the unit axis and the order of
        contiguous groups in ``group_offsets``.
    group_offsets : numpy.ndarray
        Owned int64 ``(unit * 2 + 1,)`` offsets.  Groups are unit-major and
        then segment-major in ``before, after`` order.  Offsets index the
        flattened spike axis and have no physical units.
    left_index, right_index : numpy.ndarray
        Owned int64 ``(spike,)`` safe, clamped indices into the canonical phase
        time axis.  Outside-support entries are ignored by ``inside_support``.
    right_weight : numpy.ndarray
        Owned float64 ``(spike,)`` dimensionless right-neighbor interpolation
        weights.  Exact and outside-support entries have weight zero.
    inside_support, exact_sample : numpy.ndarray
        Owned Boolean ``(spike,)`` masks.  Exact means equality to the stored
        canonical time coordinate, not numerical closeness.

    Raises
    ------
    ValueError
        If fields do not satisfy the documented axes, dtypes, or group layout.
    """

    source_trial_index: np.ndarray
    unit_ids: tuple[str, ...]
    group_offsets: np.ndarray
    left_index: np.ndarray
    right_index: np.ndarray
    right_weight: np.ndarray
    inside_support: np.ndarray
    exact_sample: np.ndarray

    def __post_init__(self) -> None:
        """Validate, copy, and freeze geometry supplied through the public constructor."""
        self._set_validated_fields(
            self._validated_fields(
                source_trial_index=self.source_trial_index,
                unit_ids=self.unit_ids,
                group_offsets=self.group_offsets,
                left_index=self.left_index,
                right_index=self.right_index,
                right_weight=self.right_weight,
                inside_support=self.inside_support,
                exact_sample=self.exact_sample,
            )
        )

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        source_trial_index: np.ndarray,
        unit_ids: tuple[str, ...],
        group_offsets: np.ndarray,
        left_index: np.ndarray,
        right_index: np.ndarray,
        right_weight: np.ndarray,
        inside_support: np.ndarray,
        exact_sample: np.ndarray,
    ) -> "SourceTrialSpikeGeometry":
        """Freeze trusted arrays allocated privately by the geometry builder.

        This internal ownership-transfer path avoids copying builder-local arrays
        or building full-size comparison masks at return.  Public construction
        remains defensive; the builder already guarantees content invariants.
        """
        geometry = object.__new__(cls)
        geometry._set_validated_fields(
            cls._trusted_owned_fields(
                source_trial_index=source_trial_index,
                unit_ids=unit_ids,
                group_offsets=group_offsets,
                left_index=left_index,
                right_index=right_index,
                right_weight=right_weight,
                inside_support=inside_support,
                exact_sample=exact_sample,
            )
        )
        return geometry

    @staticmethod
    def _validated_fields(
        *,
        source_trial_index: np.ndarray,
        unit_ids: tuple[str, ...],
        group_offsets: np.ndarray,
        left_index: np.ndarray,
        right_index: np.ndarray,
        right_weight: np.ndarray,
        inside_support: np.ndarray,
        exact_sample: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Defensively validate and copy geometry fields for public construction."""
        source = _owned_int64_scalar(source_trial_index, "source_trial_index")
        if int(source) < 0:
            raise ValueError("source_trial_index must be nonnegative")
        canonical_unit_ids = _unit_id_tuple(unit_ids)
        offsets = _owned_int64_vector(group_offsets, "group_offsets")
        expected_offset_size = len(canonical_unit_ids) * 2 + 1
        if (
            offsets.size != expected_offset_size
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError("group_offsets must define unit-major before/after groups")
        left = _owned_int64_vector(left_index, "left_index")
        right = _owned_int64_vector(right_index, "right_index")
        weight = _owned_float64_vector(right_weight, "right_weight")
        inside = _owned_bool_vector(inside_support, "inside_support")
        exact = _owned_bool_vector(exact_sample, "exact_sample")
        spike_count = left.size
        if (
            right.size != spike_count
            or weight.size != spike_count
            or inside.size != spike_count
            or exact.size != spike_count
            or offsets[-1] != spike_count
            or np.any(left < 0)
            or np.any(right < 0)
            or not np.isfinite(weight).all()
            or np.any(weight < 0.0)
            or np.any(weight > 1.0)
            or np.any(exact & ~inside)
            or np.any(exact & ((left != right) | (weight != 0.0)))
            or np.any(~inside & (weight != 0.0))
        ):
            raise ValueError("source geometry arrays are inconsistent")
        return source, canonical_unit_ids, offsets, left, right, weight, inside, exact

    @staticmethod
    def _trusted_owned_fields(
        *,
        source_trial_index: np.ndarray,
        unit_ids: tuple[str, ...],
        group_offsets: np.ndarray,
        left_index: np.ndarray,
        right_index: np.ndarray,
        right_weight: np.ndarray,
        inside_support: np.ndarray,
        exact_sample: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Check private builder array structure without full-size comparisons."""
        source = _trusted_int64_scalar(source_trial_index, "source_trial_index")
        canonical_unit_ids = _unit_id_tuple(unit_ids)
        offsets = _trusted_int64_vector(group_offsets, "group_offsets")
        left = _trusted_int64_vector(left_index, "left_index")
        right = _trusted_int64_vector(right_index, "right_index")
        weight = _trusted_float64_vector(right_weight, "right_weight")
        inside = _trusted_bool_vector(inside_support, "inside_support")
        exact = _trusted_bool_vector(exact_sample, "exact_sample")
        if (
            offsets.size != len(canonical_unit_ids) * 2 + 1
            or offsets.size == 0
            or offsets[0] != 0
            or offsets[-1] != left.size
            or right.size != left.size
            or weight.size != left.size
            or inside.size != left.size
            or exact.size != left.size
        ):
            raise ValueError("trusted source geometry arrays have inconsistent structure")
        return source, canonical_unit_ids, offsets, left, right, weight, inside, exact

    def _set_validated_fields(
        self,
        values: tuple[
            np.ndarray,
            tuple[str, ...],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> None:
        """Store owned geometry arrays after making element-level mutation impossible."""
        source, unit_ids, offsets, left, right, weight, inside, exact = values
        object.__setattr__(self, "source_trial_index", _freeze_array(source))
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "group_offsets", _freeze_array(offsets))
        object.__setattr__(self, "left_index", _freeze_array(left))
        object.__setattr__(self, "right_index", _freeze_array(right))
        object.__setattr__(self, "right_weight", _freeze_array(weight))
        object.__setattr__(self, "inside_support", _freeze_array(inside))
        object.__setattr__(self, "exact_sample", _freeze_array(exact))


@dataclass(frozen=True)
class SegmentedEdgeStatistics:
    """Owned before/after PPC sufficient statistics for bounded physical edges.

    Parameters
    ----------
    source_trial_index, target_trial_index : numpy.ndarray
        Owned int64 ``(edge,)`` stable physical trial-table identities.  Edge
        order is preserved and a physical source-target pair appears at most
        once.
    phase_vector_sum : numpy.ndarray
        Owned complex128 ``(edge, unit, segment=2, frequency)`` sums of unit
        phase vectors.  Vectors are dimensionless.
    valid_spike_count : numpy.ndarray
        Owned int64 array on the same axes.  Values are nonnegative counts of
        valid spike-phase observations and have no physical units.

    Raises
    ------
    ValueError
        If identities, axes, dtypes, finiteness, or zero-count sums are invalid.
    """

    source_trial_index: np.ndarray
    target_trial_index: np.ndarray
    phase_vector_sum: np.ndarray
    valid_spike_count: np.ndarray

    def __post_init__(self) -> None:
        """Validate, copy, and freeze statistics supplied through the public constructor."""
        self._set_validated_fields(
            self._validated_fields(
                source_trial_index=self.source_trial_index,
                target_trial_index=self.target_trial_index,
                phase_vector_sum=self.phase_vector_sum,
                valid_spike_count=self.valid_spike_count,
            )
        )

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        source_trial_index: np.ndarray,
        target_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
    ) -> "SegmentedEdgeStatistics":
        """Freeze trusted reducer outputs without copies or content comparisons.

        ``compute_segmented_edge_statistics`` has already validated edge
        identities and constructs these arrays with the documented shapes,
        dtypes, and zero-initialized count/sum invariant.  Public construction
        still performs full defensive content validation in ``__post_init__``.
        """
        statistics = object.__new__(cls)
        statistics._set_validated_fields(
            cls._trusted_owned_fields(
                source_trial_index=source_trial_index,
                target_trial_index=target_trial_index,
                phase_vector_sum=phase_vector_sum,
                valid_spike_count=valid_spike_count,
            )
        )
        return statistics

    @staticmethod
    def _validated_fields(
        *,
        source_trial_index: np.ndarray,
        target_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Defensively validate and copy output fields for public construction."""
        source = _owned_int64_vector(source_trial_index, "source_trial_index")
        target = _owned_int64_vector(target_trial_index, "target_trial_index")
        if (
            source.size < 1
            or target.shape != source.shape
            or np.any(source < 0)
            or np.any(target < 0)
            or len(set(zip(source.tolist(), target.tolist(), strict=True))) != source.size
        ):
            raise ValueError("edge identities must be nonnegative unique int64 pairs")
        sums = np.asarray(phase_vector_sum)
        counts = np.asarray(valid_spike_count)
        if sums.dtype != np.dtype(np.complex128) or counts.dtype != np.dtype(np.int64):
            raise ValueError("edge statistics require complex128 sums and int64 counts")
        if (
            sums.ndim != 4
            or sums.shape[0] != source.size
            or sums.shape[1] < 1
            or sums.shape[2] != 2
            or sums.shape[3] < 1
            or counts.shape != sums.shape
            or np.any(counts < 0)
            or not np.isfinite(sums.real).all()
            or not np.isfinite(sums.imag).all()
            or np.any((counts == 0) & (sums != 0.0j))
        ):
            raise ValueError("segmented edge statistic arrays are inconsistent")
        return source, target, sums.copy(), counts.copy()

    @staticmethod
    def _trusted_owned_fields(
        *,
        source_trial_index: np.ndarray,
        target_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check private reducer output structure without full-size comparisons."""
        source = _trusted_int64_vector(source_trial_index, "source_trial_index")
        target = _trusted_int64_vector(target_trial_index, "target_trial_index")
        sums = np.asarray(phase_vector_sum)
        counts = np.asarray(valid_spike_count)
        if (
            source.size < 1
            or target.shape != source.shape
            or sums.dtype != np.dtype(np.complex128)
            or counts.dtype != np.dtype(np.int64)
            or sums.ndim != 4
            or sums.shape[0] != source.size
            or sums.shape[1] < 1
            or sums.shape[2] != 2
            or sums.shape[3] < 1
            or counts.shape != sums.shape
        ):
            raise ValueError("trusted edge statistic arrays have inconsistent structure")
        return source, target, sums, counts

    def _set_validated_fields(
        self,
        values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store owned output arrays after making element-level mutation impossible."""
        source, target, sums, counts = values
        object.__setattr__(self, "source_trial_index", _freeze_array(source))
        object.__setattr__(self, "target_trial_index", _freeze_array(target))
        object.__setattr__(self, "phase_vector_sum", _freeze_array(sums))
        object.__setattr__(self, "valid_spike_count", _freeze_array(counts))


@dataclass(frozen=True)
class KernelAllocationEstimate:
    """Pure checked byte accounting for one bounded segmented-kernel call.

    Parameters
    ----------
    geometry_bytes : int
        Nonnegative bytes for retained source geometry: 26 bytes per spike,
        one int64 source identity, and int64 offsets for every source trial.
    segmented_edge_statistics_bytes : int
        Nonnegative bytes for complex128 sums and int64 counts with two
        segments (48 bytes per ``(edge, unit, frequency)`` cell), plus two
        int64 source/target identities per edge.
    gather_temporary_bytes : int
        Nonnegative conservative bytes for concurrently live gather,
        interpolation, magnitude, normalized-vector, and validity arrays:
        51 bytes per selected ``(edge, spike, frequency)`` cell.
    planned_kernel_peak_bytes : int
        Nonnegative checked sum of the three documented concurrently live
        categories.

    Raises
    ------
    ValueError
        If a field is negative, Boolean, nonintegral, or exceeds int64 range.
    """

    geometry_bytes: int
    segmented_edge_statistics_bytes: int
    gather_temporary_bytes: int
    planned_kernel_peak_bytes: int

    def __post_init__(self) -> None:
        """Validate and canonicalize the four public byte-count fields."""
        for field_name in (
            "geometry_bytes",
            "segmented_edge_statistics_bytes",
            "gather_temporary_bytes",
            "planned_kernel_peak_bytes",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                or int(value) > _INT64_MAX
            ):
                raise ValueError(f"{field_name} must be a nonnegative int64 byte count")
            object.__setattr__(self, field_name, int(value))


def build_source_trial_spike_geometry(
    *,
    phase_time_s: np.ndarray,
    phase_sampling_rate_hz: float,
    source_trial_index: int,
    unit_ids: tuple[str, ...],
    unit_trial_spike_times_s: tuple[np.ndarray, ...],
    segment_bounds_s: tuple[tuple[float, float], tuple[float, float]],
) -> SourceTrialSpikeGeometry:
    """Build reusable source-spike interpolation geometry on one uniform time grid.

    Parameters
    ----------
    phase_time_s : numpy.ndarray
        Float64 ``(time,)`` finite, strictly increasing canonical relative-time
        coordinate in seconds.  It must be uniformly spaced at
        ``1 / phase_sampling_rate_hz`` seconds.
    phase_sampling_rate_hz : float
        Positive finite configured phase sampling rate in Hz.
    source_trial_index : int
        Nonnegative stable physical source trial-table identity with no units.
    unit_ids : tuple[str, ...]
        Unique stable unit identities with shape ``(unit,)``.
    unit_trial_spike_times_s : tuple[numpy.ndarray, ...]
        One finite float ``(spike_in_unit_trial,)`` relative-seconds array per
        unit.  Input spike order is preserved independently within each
        before/after group.
    segment_bounds_s : tuple[tuple[float, float], tuple[float, float]]
        Finite contiguous ``((before_start, boundary), (boundary, after_end))``
        relative-second bounds.  Both segments are lower-inclusive and
        upper-exclusive, so the shared boundary belongs only to after.

    Returns
    -------
    SourceTrialSpikeGeometry
        Frozen owned geometry with safe gather indices.  Outside phase support
        remains represented only when it lies in a configured segment; its
        support mask is false and its gather indices are clamped.

    Raises
    ------
    ValueError
        If the grid/rate, identities, spikes, or segment bounds are malformed.
    """
    phase_time = _validated_uniform_phase_time(phase_time_s, phase_sampling_rate_hz)
    stable_source = _owned_int64_scalar(source_trial_index, "source_trial_index")
    if int(stable_source) < 0:
        raise ValueError("source_trial_index must be nonnegative")
    canonical_unit_ids = _unit_id_tuple(unit_ids)
    if len(unit_trial_spike_times_s) != len(canonical_unit_ids):
        raise ValueError("unit ids and source-trial spike arrays must have matching lengths")
    bounds = _validated_segment_bounds(segment_bounds_s)

    grouped_spikes: list[np.ndarray] = []
    offsets = [0]
    for spike_values in unit_trial_spike_times_s:
        spikes = _finite_float64_vector(spike_values, "unit_trial_spike_times_s")
        before = spikes[(spikes >= bounds[0, 0]) & (spikes < bounds[0, 1])]
        grouped_spikes.append(before)
        offsets.append(offsets[-1] + before.size)
        after = spikes[(spikes >= bounds[1, 0]) & (spikes < bounds[1, 1])]
        grouped_spikes.append(after)
        offsets.append(offsets[-1] + after.size)
    if grouped_spikes:
        flattened_spikes = np.concatenate(grouped_spikes)
    else:
        flattened_spikes = np.empty(0, dtype=np.float64)

    spike_count = flattened_spikes.size
    # This is the only neighbor search.  Reduction consumes these stored indices.
    raw_right_index = np.searchsorted(phase_time, flattened_spikes, side="left")
    clamped_index = np.clip(raw_right_index, 0, phase_time.size - 1).astype(
        np.int64,
        copy=False,
    )
    left_index = clamped_index.copy()
    right_index = clamped_index.copy()
    exact_sample = np.zeros(spike_count, dtype=bool)
    within_right_axis = raw_right_index < phase_time.size
    if np.any(within_right_axis):
        exact_sample[within_right_axis] = (
            phase_time[raw_right_index[within_right_axis]]
            == flattened_spikes[within_right_axis]
        )
    between = (
        ~exact_sample
        & (raw_right_index > 0)
        & (raw_right_index < phase_time.size)
    )
    if np.any(between):
        left_index[between] = raw_right_index[between] - 1
        right_index[between] = raw_right_index[between]
    right_weight = np.zeros(spike_count, dtype=np.float64)
    if np.any(between):
        left_time = phase_time[left_index[between]]
        right_time = phase_time[right_index[between]]
        right_weight[between] = (
            (flattened_spikes[between] - left_time) / (right_time - left_time)
        )
    inside_support = exact_sample | between
    return SourceTrialSpikeGeometry._from_owned_arrays(
        source_trial_index=stable_source,
        unit_ids=canonical_unit_ids,
        group_offsets=np.asarray(offsets, dtype=np.int64),
        left_index=left_index,
        right_index=right_index,
        right_weight=right_weight,
        inside_support=inside_support,
        exact_sample=exact_sample,
    )


def compute_segmented_edge_statistics(
    *,
    source_trial_geometries: tuple[SourceTrialSpikeGeometry, ...],
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
    source_trial_index: np.ndarray,
    target_trial_index: np.ndarray,
) -> SegmentedEdgeStatistics:
    """Reduce bounded stable edges into before/after complex sums and counts.

    Parameters
    ----------
    source_trial_geometries : tuple[SourceTrialSpikeGeometry, ...]
        One geometry per available physical source trial, all with the same
        ordered unit ids.  Their source identities need not be sorted.
    trial_phase_vectors : numpy.ndarray
        Complex64 ``(trial, frequency, time)`` phase coefficients.  Values may
        be nonfinite or zero; such samples are unavailable even when their mask
        is true.
    phase_valid_mask : numpy.ndarray
        Boolean array with exactly the same axes as ``trial_phase_vectors``.
        It is explicit prepared-phase availability, not an inferred replacement
        for finite/nonzero coefficient validation.
    phase_trial_index : numpy.ndarray
        Owned-int64-compatible ``(trial,)`` unique nonnegative stable physical
        trial identities in the unsorted phase-axis order.
    frequencies_hz : numpy.ndarray
        Float64 ``(frequency,)`` finite, positive, strictly increasing Hz
        coordinates matching the phase frequency axis.
    source_trial_index, target_trial_index : numpy.ndarray
        Int64 matching ``(edge,)`` nonnegative stable physical source/target
        trial identities.  Each physical pair must be unique; their supplied
        order becomes the output edge axis.

    Returns
    -------
    SegmentedEdgeStatistics
        Frozen owned complex128 sums and int64 counts with axes
        ``(edge, unit, segment=2, frequency)``.  Segment zero is before and
        segment one is after.  No per-spike phase array survives the call.

    Raises
    ------
    ValueError
        If axes, dtypes, identities, geometries, frequencies, or edge pairs are
        invalid.  No partial statistic is returned.
    """
    phase, phase_valid, phase_ids, frequencies = _validated_reduction_phase_inputs(
        trial_phase_vectors=trial_phase_vectors,
        phase_valid_mask=phase_valid_mask,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
    )
    source_edges, target_edges = _validated_edge_identities(
        source_trial_index,
        target_trial_index,
    )
    geometries, geometry_by_source = _validated_geometries(
        source_trial_geometries,
        phase_time_count=phase.shape[2],
    )
    phase_position_by_id = {int(identity): position for position, identity in enumerate(phase_ids)}
    if any(int(geometry.source_trial_index) not in phase_position_by_id for geometry in geometries):
        raise ValueError("every source geometry must identify a prepared phase trial")
    if any(int(identity) not in geometry_by_source for identity in source_edges):
        raise ValueError("every source edge must have source-trial geometry")
    if any(int(identity) not in phase_position_by_id for identity in target_edges):
        raise ValueError("every target edge must identify a prepared phase trial")

    unit_count = len(geometries[0].unit_ids)
    statistic_shape = (source_edges.size, unit_count, 2, frequencies.size)
    phase_vector_sum = np.zeros(statistic_shape, dtype=np.complex128)
    valid_spike_count = np.zeros(statistic_shape, dtype=np.int64)
    for edge_position, (source_id, target_id) in enumerate(
        zip(source_edges, target_edges, strict=True)
    ):
        geometry = geometry_by_source[int(source_id)]
        target_position = phase_position_by_id[int(target_id)]
        coefficients = phase[target_position]
        target_valid = phase_valid[target_position]
        for unit_position in range(unit_count):
            for segment_position in range(2):
                group_position = unit_position * 2 + segment_position
                start = int(geometry.group_offsets[group_position])
                stop = int(geometry.group_offsets[group_position + 1])
                if start == stop:
                    continue
                vector_sum, count = _reduce_one_spike_group(
                    coefficients=coefficients,
                    explicit_valid_mask=target_valid,
                    left_index=geometry.left_index[start:stop],
                    right_index=geometry.right_index[start:stop],
                    right_weight=geometry.right_weight[start:stop],
                    inside_support=geometry.inside_support[start:stop],
                    exact_sample=geometry.exact_sample[start:stop],
                )
                phase_vector_sum[edge_position, unit_position, segment_position] = vector_sum
                valid_spike_count[edge_position, unit_position, segment_position] = count
    return SegmentedEdgeStatistics._from_owned_arrays(
        source_trial_index=source_edges,
        target_trial_index=target_edges,
        phase_vector_sum=phase_vector_sum,
        valid_spike_count=valid_spike_count,
    )


def estimate_segmented_kernel_allocation(
    *,
    source_trial_spike_count: np.ndarray,
    edge_source_trial_position: np.ndarray,
    frequency_count: int,
) -> KernelAllocationEstimate:
    """Return pure checked int64 byte accounting for a bounded kernel reduction.

    Parameters
    ----------
    source_trial_spike_count : numpy.ndarray
        Int64 nonnegative ``(source_trial, unit, segment=2)`` counts in
        before/after order.  Counts have no physical units.
    edge_source_trial_position : numpy.ndarray
        Int64 ``(edge_in_block,)`` zero-based positions into the source-trial
        axis.  Repeated positions are permitted because different target edges
        can share source geometry.
    frequency_count : int
        Positive non-Boolean frequency-axis length with no physical units.

    Returns
    -------
    KernelAllocationEstimate
        Frozen nonnegative Python-int byte counts.  The helper allocates no
        work arrays, samples no phase, and performs no I/O.  Geometry includes
        one owned int64 source identity per source trial; edge statistics
        include owned int64 source and target identity vectors.

    Raises
    ------
    ValueError
        If axes/counts are invalid or any int64 addition/multiplication would
        overflow, including selected-spike aggregation and the final peak sum.
    """
    counts = np.asarray(source_trial_spike_count)
    positions = np.asarray(edge_source_trial_position)
    if (
        counts.dtype != np.dtype(np.int64)
        or counts.ndim != 3
        or counts.shape[0] < 1
        or counts.shape[1] < 1
        or counts.shape[2] != 2
        or np.any(counts < 0)
    ):
        raise ValueError("source_trial_spike_count must be nonnegative int64 (source, unit, 2)")
    if (
        positions.dtype != np.dtype(np.int64)
        or positions.ndim != 1
        or positions.size < 1
        or np.any(positions < 0)
        or np.any(positions >= counts.shape[0])
    ):
        raise ValueError("edge_source_trial_position must be int64 positions in source axis")
    if (
        isinstance(frequency_count, (bool, np.bool_))
        or not isinstance(frequency_count, (int, np.integer))
        or int(frequency_count) < 1
        or int(frequency_count) > _INT64_MAX
    ):
        raise ValueError("frequency_count must be a positive non-Boolean int64")
    frequencies = int(frequency_count)

    total_spikes = 0
    for spike_count in counts.flat:
        total_spikes = _checked_add(total_spikes, int(spike_count))
    geometry_spike_bytes = _checked_multiply(_GEOMETRY_BYTES_PER_SPIKE, total_spikes)
    offsets_per_source = _checked_add(_checked_multiply(counts.shape[1], 2), 1)
    offset_bytes_per_source = _checked_multiply(offsets_per_source, _INT64_BYTES)
    all_offset_bytes = _checked_multiply(counts.shape[0], offset_bytes_per_source)
    geometry_bytes = _checked_add(geometry_spike_bytes, all_offset_bytes)
    # Each retained geometry owns its scalar stable source-trial identity.
    source_identity_bytes = _checked_multiply(counts.shape[0], _INT64_BYTES)
    geometry_bytes = _checked_add(geometry_bytes, source_identity_bytes)

    selected_spikes = 0
    for source_position in positions:
        source_spike_count = 0
        for spike_count in counts[int(source_position)].flat:
            source_spike_count = _checked_add(source_spike_count, int(spike_count))
        selected_spikes = _checked_add(selected_spikes, source_spike_count)
    gathered_cells = _checked_multiply(selected_spikes, frequencies)
    gather_temporary_bytes = _checked_multiply(
        gathered_cells,
        _GATHER_BYTES_PER_CELL,
    )
    edge_frequency_cells = _checked_multiply(positions.size, counts.shape[1])
    edge_frequency_cells = _checked_multiply(edge_frequency_cells, frequencies)
    segmented_edge_statistics_bytes = _checked_multiply(
        edge_frequency_cells,
        _SEGMENTED_EDGE_BYTES_PER_CELL,
    )
    # The segmented result also owns one source and one target identity per edge.
    edge_identity_count = _checked_multiply(positions.size, 2)
    edge_identity_bytes = _checked_multiply(edge_identity_count, _INT64_BYTES)
    segmented_edge_statistics_bytes = _checked_add(
        segmented_edge_statistics_bytes,
        edge_identity_bytes,
    )
    planned_kernel_peak_bytes = _checked_add(geometry_bytes, gather_temporary_bytes)
    planned_kernel_peak_bytes = _checked_add(
        planned_kernel_peak_bytes,
        segmented_edge_statistics_bytes,
    )
    return KernelAllocationEstimate(
        geometry_bytes=geometry_bytes,
        segmented_edge_statistics_bytes=segmented_edge_statistics_bytes,
        gather_temporary_bytes=gather_temporary_bytes,
        planned_kernel_peak_bytes=planned_kernel_peak_bytes,
    )


def _reduce_one_spike_group(
    *,
    coefficients: np.ndarray,
    explicit_valid_mask: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    right_weight: np.ndarray,
    inside_support: np.ndarray,
    exact_sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one unit/segment's stored geometry across all phase frequencies.

    Parameters are bounded local arrays: complex64/Boolean ``(frequency, time)``
    coefficients/mask and one-dimensional geometry arrays on the group spike
    axis.  The returned complex128/int64 arrays have shape ``(frequency,)``;
    values are dimensionless sums and valid observation counts.  This helper
    creates only temporary per-group phase arrays and returns no per-spike data.

    Per ``(frequency, spike)`` cell, the full-size temporaries are two
    complex64 gathers, one complex128 interpolation buffer, one float64
    magnitude/work buffer, one complex64 normalized buffer, and at most three
    Boolean masks.  This is the 51-byte conservative accounting used by
    ``estimate_segmented_kernel_allocation``.  One-dimensional geometry views
    and frequency-length reductions are not per-cell gather temporaries.
    """
    left_coefficients = coefficients[:, left_index]
    right_coefficients = coefficients[:, right_index]
    usable = explicit_valid_mask[:, left_index]
    right_available = explicit_valid_mask[:, right_index]
    magnitude = np.empty(left_coefficients.shape, dtype=np.float64)

    # Reuse each explicit-mask gather as its own availability buffer.  ``where``
    # preserves entries already marked false instead of allocating comparisons.
    np.isfinite(left_coefficients.real, out=usable, where=usable)
    np.isfinite(left_coefficients.imag, out=usable, where=usable)
    np.abs(left_coefficients, out=magnitude)
    np.not_equal(magnitude, 0.0, out=usable, where=usable)
    np.isfinite(right_coefficients.real, out=right_available, where=right_available)
    np.isfinite(right_coefficients.imag, out=right_available, where=right_available)
    np.abs(right_coefficients, out=magnitude)
    np.not_equal(magnitude, 0.0, out=right_available, where=right_available)

    # Exact samples need only their left neighbor; between-sample entries need
    # both.  The broadcast geometry masks do not create full gather arrays.
    np.logical_or(right_available, exact_sample[np.newaxis, :], out=right_available)
    np.logical_and(usable, right_available, out=usable)
    np.logical_and(usable, inside_support[np.newaxis, :], out=usable)
    del right_available

    interpolated = np.empty(left_coefficients.shape, dtype=np.complex128)
    interpolation_weights = right_weight[np.newaxis, :]
    # Compute both components in place for all spikes.  This avoids Boolean
    # advanced-index temporary arrays and a second complex128 work buffer.
    interpolated.real = left_coefficients.real
    np.subtract(1.0, interpolation_weights, out=magnitude)
    np.multiply(interpolated.real, magnitude, out=interpolated.real)
    np.multiply(right_coefficients.real, interpolation_weights, out=magnitude)
    np.add(interpolated.real, magnitude, out=interpolated.real)
    interpolated.imag = left_coefficients.imag
    np.subtract(1.0, interpolation_weights, out=magnitude)
    np.multiply(interpolated.imag, magnitude, out=interpolated.imag)
    np.multiply(right_coefficients.imag, interpolation_weights, out=magnitude)
    np.add(interpolated.imag, magnitude, out=interpolated.imag)
    del left_coefficients, right_coefficients

    # The gathers are gone before normalization.  Reuse ``usable`` and
    # ``magnitude`` for interpolation validity, then divide complex128 values
    # directly into the required complex64 reduction buffer.
    np.isfinite(interpolated.real, out=usable, where=usable)
    np.isfinite(interpolated.imag, out=usable, where=usable)
    np.abs(interpolated, out=magnitude)
    np.not_equal(magnitude, 0.0, out=usable, where=usable)
    normalized_complex64 = np.zeros(interpolated.shape, dtype=np.complex64)
    np.divide(interpolated, magnitude, out=normalized_complex64, where=usable)
    del interpolated, magnitude
    vector_sum = np.sum(normalized_complex64, axis=1, dtype=np.complex128)
    del normalized_complex64
    valid_count = np.sum(usable, axis=1, dtype=np.int64)
    return vector_sum, valid_count


def _validated_uniform_phase_time(
    phase_time_s: np.ndarray,
    phase_sampling_rate_hz: float,
) -> np.ndarray:
    """Return an owned uniform float64 relative-seconds grid after rate validation."""
    if (
        isinstance(phase_sampling_rate_hz, (bool, np.bool_))
        or not isinstance(phase_sampling_rate_hz, (int, float, np.integer, np.floating))
        or not np.isfinite(phase_sampling_rate_hz)
        or float(phase_sampling_rate_hz) <= 0.0
    ):
        raise ValueError("phase_sampling_rate_hz must be a positive finite Hz value")
    phase_time = np.asarray(phase_time_s)
    if (
        phase_time.dtype != np.dtype(np.float64)
        or phase_time.ndim != 1
        or phase_time.size < 2
        or not np.isfinite(phase_time).all()
        or np.any(np.diff(phase_time) <= 0.0)
    ):
        raise ValueError("phase_time_s must be a finite increasing float64 seconds axis")
    interval_s = 1.0 / float(phase_sampling_rate_hz)
    grid_tolerance_s = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(phase_time)))) * 16.0
    if not np.all(np.abs(np.diff(phase_time) - interval_s) <= grid_tolerance_s):
        raise ValueError("phase_time_s must match the configured uniform sampling rate")
    return phase_time.copy()


def _validated_segment_bounds(
    segment_bounds_s: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Validate contiguous lower-inclusive/upper-exclusive before/after seconds bounds."""
    bounds = np.asarray(segment_bounds_s, dtype=np.float64)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or bounds[0, 1] != bounds[1, 0]
    ):
        raise ValueError("segment_bounds_s must be finite contiguous before/after intervals")
    return bounds.copy()


def _validated_reduction_phase_inputs(
    *,
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate phase tensors, explicit masks, stable trial rows, and frequency Hz axis."""
    phase = np.asarray(trial_phase_vectors)
    valid = np.asarray(phase_valid_mask)
    phase_ids = np.asarray(phase_trial_index)
    frequencies = np.asarray(frequencies_hz)
    if (
        phase.dtype != np.dtype(np.complex64)
        or phase.ndim != 3
        or phase.shape[0] < 1
        or phase.shape[1] < 1
        or phase.shape[2] < 1
        or valid.dtype != np.dtype(bool)
        or valid.shape != phase.shape
    ):
        raise ValueError("phase values must be complex64 with matching Boolean validity")
    if (
        phase_ids.dtype != np.dtype(np.int64)
        or phase_ids.ndim != 1
        or phase_ids.size != phase.shape[0]
        or np.any(phase_ids < 0)
        or np.unique(phase_ids).size != phase_ids.size
    ):
        raise ValueError("phase_trial_index must be unique nonnegative int64 trial rows")
    if (
        frequencies.dtype != np.dtype(np.float64)
        or frequencies.ndim != 1
        or frequencies.size != phase.shape[1]
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise ValueError("frequencies_hz must be positive increasing float64 phase coordinates")
    return phase, valid, phase_ids.copy(), frequencies.copy()


def _validated_edge_identities(
    source_trial_index: np.ndarray,
    target_trial_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return owned unique int64 stable source/target physical edge identities."""
    source = np.asarray(source_trial_index)
    target = np.asarray(target_trial_index)
    if (
        source.dtype != np.dtype(np.int64)
        or target.dtype != np.dtype(np.int64)
        or source.ndim != 1
        or target.shape != source.shape
        or source.size < 1
        or np.any(source < 0)
        or np.any(target < 0)
        or len(set(zip(source.tolist(), target.tolist(), strict=True))) != source.size
    ):
        raise ValueError("source/target edges must be nonnegative unique int64 pairs")
    return source.copy(), target.copy()


def _validated_geometries(
    geometries: Sequence[SourceTrialSpikeGeometry],
    *,
    phase_time_count: int,
) -> tuple[tuple[SourceTrialSpikeGeometry, ...], dict[int, SourceTrialSpikeGeometry]]:
    """Validate source geometry identities, shared unit axis, and safe phase indices."""
    geometry_tuple = tuple(geometries)
    if not geometry_tuple or any(
        not isinstance(geometry, SourceTrialSpikeGeometry) for geometry in geometry_tuple
    ):
        raise ValueError("source_trial_geometries must contain source geometry objects")
    reference_unit_ids = geometry_tuple[0].unit_ids
    geometry_by_source: dict[int, SourceTrialSpikeGeometry] = {}
    for geometry in geometry_tuple:
        source_id = int(geometry.source_trial_index)
        if (
            geometry.unit_ids != reference_unit_ids
            or source_id in geometry_by_source
            or np.any(geometry.left_index >= phase_time_count)
            or np.any(geometry.right_index >= phase_time_count)
        ):
            raise ValueError("source geometries must have unique ids, shared units, and safe time indices")
        geometry_by_source[source_id] = geometry
    return geometry_tuple, geometry_by_source


def _unit_id_tuple(values: Sequence[str]) -> tuple[str, ...]:
    """Return a nonempty unique stable unit-id tuple after categorical validation."""
    unit_ids = tuple(values)
    if (
        not unit_ids
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids)
        or len(set(unit_ids)) != len(unit_ids)
    ):
        raise ValueError("unit_ids must be unique nonempty strings")
    return unit_ids


def _finite_float64_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return an owned finite float64 one-dimensional relative-seconds vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite one-dimensional seconds arrays")
    return vector.copy()


def _owned_int64_scalar(value: object, name: str) -> np.ndarray:
    """Return an owned int64 scalar while rejecting Boolean/nonintegral identities."""
    array = np.asarray(value)
    if array.shape == () and array.dtype == np.dtype(np.int64):
        return array.copy()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int64 scalar")
    integer_value = int(value)
    if integer_value < _INT64_MIN or integer_value > _INT64_MAX:
        raise ValueError(f"{name} must be an int64 scalar")
    return np.asarray(integer_value, dtype=np.int64).copy()


def _trusted_int64_scalar(value: object, name: str) -> np.ndarray:
    """Validate an internally owned int64 scalar without duplicating it."""
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must be an int64 scalar")
    return array


def _owned_int64_vector(values: object, name: str) -> np.ndarray:
    """Return an owned int64 vector while rejecting casts from non-int64 arrays."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.int64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 array")
    return array.copy()


def _trusted_int64_vector(values: object, name: str) -> np.ndarray:
    """Validate an internally owned int64 vector without duplicating it."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.int64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 array")
    return array


def _owned_float64_vector(values: object, name: str) -> np.ndarray:
    """Return an owned float64 vector while rejecting non-float64 geometry arrays."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    return array.copy()


def _trusted_float64_vector(values: object, name: str) -> np.ndarray:
    """Validate an internally owned float64 vector without duplicating it."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    return array


def _owned_bool_vector(values: object, name: str) -> np.ndarray:
    """Return an owned Boolean vector while rejecting casts from integer masks."""
    array = np.asarray(values)
    if array.dtype != np.dtype(bool) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional Boolean array")
    return array.copy()


def _trusted_bool_vector(values: object, name: str) -> np.ndarray:
    """Validate an internally owned Boolean vector without duplicating it."""
    array = np.asarray(values)
    if array.dtype != np.dtype(bool) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional Boolean array")
    return array


def _freeze_array(array: np.ndarray) -> np.ndarray:
    """Return an owned kernel result array after disabling element-level writes."""
    array.setflags(write=False)
    return array


def _checked_add(left: int, right: int) -> int:
    """Return a nonnegative checked int64 sum, raising ValueError instead of wrapping."""
    if left < 0 or right < 0 or left > _INT64_MAX - right:
        raise ValueError("kernel allocation arithmetic overflow")
    return left + right


def _checked_multiply(left: int, right: int) -> int:
    """Return a nonnegative checked int64 product, raising ValueError instead of wrapping."""
    if left < 0 or right < 0 or (left and right > _INT64_MAX // left):
        raise ValueError("kernel allocation arithmetic overflow")
    return left * right
