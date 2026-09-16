"""Uniform-grid interpolation and segmented PPC sufficient statistics.

This numerical kernel builds reusable S1 spike geometry and edge statistics,
then derives S2 same-trial observed statistics that can be pooled repeatedly by
stable physical trial membership.  It does not perform file I/O, checkpointing,
multiprocessing, or pipeline orchestration.
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
class ObservedTrialSegmentedPPCStatistics:
    """Owned same-trial PPC sufficient statistics before condition aggregation.

    Parameters
    ----------
    phase_trial_index : numpy.ndarray
        Int64 ``(trial,)`` unique nonnegative stable physical trial identities
        in the stored trial axis order.  Identities have no physical units and
        need not be monotonic.
    phase_vector_sum : numpy.ndarray
        Complex128 ``(trial, unit, segment=2, frequency)`` dimensionless sums
        of accepted normalized phase vectors.  Segment zero is before and
        segment one is after.
    valid_spike_count : numpy.ndarray
        Int64 array on the same axes as ``phase_vector_sum`` containing
        nonnegative dimensionless valid spike-phase observation counts.
    representative_frequency_index : numpy.ndarray
        Int64 ``(band=2,)`` nondecreasing local frequency-axis positions for
        the representative theta then gamma histograms.  Equal positions
        are valid exactly when both bands use the same Hz coordinate.
    representative_frequency_hz : numpy.ndarray
        Float64 ``(band=2,)`` positive finite nondecreasing representative
        frequency coordinates in Hz corresponding one-to-one to the two
        positions.  Equal coordinates are valid exactly when both bands use
        the same frequency-axis position.
    phase_bin_edges_rad : numpy.ndarray
        Float64 ``(phase_bin + 1,)`` finite strictly increasing histogram edges
        in radians.
    representative_phase_histogram_count : numpy.ndarray
        Int64 ``(trial, unit, segment=2, band=2, phase_bin)`` accepted-phase
        histogram counts.  Each band/segment total does not exceed the
        matching ``valid_spike_count`` selected by
        ``representative_frequency_index``.  A valid complex64 phase at the
        nominal ``-pi`` or ``pi`` boundary may fall outside float64 histogram
        edges used by the legacy histogram contract.

    Raises
    ------
    ValueError
        If public fields violate their documented dtypes, axes, coordinate
        units, finiteness, count coherence, or zero-count sum invariant.
        Public construction copies every numerical array before freezing it.
    """

    phase_trial_index: np.ndarray
    phase_vector_sum: np.ndarray
    valid_spike_count: np.ndarray
    representative_frequency_index: np.ndarray
    representative_frequency_hz: np.ndarray
    phase_bin_edges_rad: np.ndarray
    representative_phase_histogram_count: np.ndarray

    def __post_init__(self) -> None:
        """Defensively validate, copy, and freeze public trial statistic arrays."""
        self._set_validated_fields(
            self._validated_fields(
                phase_trial_index=self.phase_trial_index,
                phase_vector_sum=self.phase_vector_sum,
                valid_spike_count=self.valid_spike_count,
                representative_frequency_index=self.representative_frequency_index,
                representative_frequency_hz=self.representative_frequency_hz,
                phase_bin_edges_rad=self.phase_bin_edges_rad,
                representative_phase_histogram_count=self.representative_phase_histogram_count,
            )
        )

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        phase_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        representative_frequency_index: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> "ObservedTrialSegmentedPPCStatistics":
        """Freeze private reducer buffers with the public axes and units.

        Parameters are newly allocated kernel arrays with the documented trial,
        unit, segment, frequency, band, and radians axes.  This internal path
        transfers ownership without copies after structural validation.

        Returns
        -------
        ObservedTrialSegmentedPPCStatistics
            Frozen owned same-trial sufficient statistics.

        Raises
        ------
        ValueError
            If an internal reducer violates the result-array structure.
        """
        statistics = object.__new__(cls)
        statistics._set_validated_fields(
            cls._trusted_owned_fields(
                phase_trial_index=phase_trial_index,
                phase_vector_sum=phase_vector_sum,
                valid_spike_count=valid_spike_count,
                representative_frequency_index=representative_frequency_index,
                representative_frequency_hz=representative_frequency_hz,
                phase_bin_edges_rad=phase_bin_edges_rad,
                representative_phase_histogram_count=representative_phase_histogram_count,
            )
        )
        return statistics

    @staticmethod
    def _validated_fields(
        *,
        phase_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        representative_frequency_index: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return defensive copies after full public field and coherence checks."""
        values = ObservedTrialSegmentedPPCStatistics._checked_fields(
            phase_trial_index=phase_trial_index,
            phase_vector_sum=phase_vector_sum,
            valid_spike_count=valid_spike_count,
            representative_frequency_index=representative_frequency_index,
            representative_frequency_hz=representative_frequency_hz,
            phase_bin_edges_rad=phase_bin_edges_rad,
            representative_phase_histogram_count=representative_phase_histogram_count,
            check_contents=True,
        )
        return tuple(value.copy() for value in values)  # type: ignore[return-value]

    @staticmethod
    def _trusted_owned_fields(
        *,
        phase_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        representative_frequency_index: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check private trial buffers without duplicating reducer-owned arrays."""
        return ObservedTrialSegmentedPPCStatistics._checked_fields(
            phase_trial_index=phase_trial_index,
            phase_vector_sum=phase_vector_sum,
            valid_spike_count=valid_spike_count,
            representative_frequency_index=representative_frequency_index,
            representative_frequency_hz=representative_frequency_hz,
            phase_bin_edges_rad=phase_bin_edges_rad,
            representative_phase_histogram_count=representative_phase_histogram_count,
            check_contents=False,
        )

    @staticmethod
    def _checked_fields(
        *,
        phase_trial_index: np.ndarray,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        representative_frequency_index: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
        check_contents: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate per-trial statistic dtypes, axes, units, and public values."""
        trial_ids = np.asarray(phase_trial_index)
        sums = np.asarray(phase_vector_sum)
        counts = np.asarray(valid_spike_count)
        representative_index = np.asarray(representative_frequency_index)
        representative_hz = np.asarray(representative_frequency_hz)
        bins_rad = np.asarray(phase_bin_edges_rad)
        histogram = np.asarray(representative_phase_histogram_count)
        if (
            trial_ids.dtype != np.dtype(np.int64)
            or trial_ids.ndim != 1
            or trial_ids.size < 1
            or sums.dtype != np.dtype(np.complex128)
            or sums.ndim != 4
            or sums.shape[0] != trial_ids.size
            or sums.shape[1] < 1
            or sums.shape[2] != 2
            or sums.shape[3] < 1
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != sums.shape
            or representative_index.dtype != np.dtype(np.int64)
            or representative_index.shape != (2,)
            or representative_hz.dtype != np.dtype(np.float64)
            or representative_hz.shape != (2,)
            or bins_rad.dtype != np.dtype(np.float64)
            or bins_rad.ndim != 1
            or bins_rad.size < 2
            or histogram.dtype != np.dtype(np.int64)
            or histogram.shape
            != (sums.shape[0], sums.shape[1], 2, 2, bins_rad.size - 1)
        ):
            raise ValueError("observed trial statistic arrays have incompatible axes or dtypes")
        if check_contents and (
            np.any(trial_ids < 0)
            or np.unique(trial_ids).size != trial_ids.size
            or not np.isfinite(sums.real).all()
            or not np.isfinite(sums.imag).all()
            or np.any(counts < 0)
            or np.any(representative_index < 0)
            or np.any(representative_index >= sums.shape[3])
            or np.any(np.diff(representative_index) < 0)
            or not np.isfinite(representative_hz).all()
            or np.any(representative_hz <= 0.0)
            or np.any(np.diff(representative_hz) < 0.0)
            or (
                (representative_index[0] == representative_index[1])
                != (representative_hz[0] == representative_hz[1])
            )
            or not np.isfinite(bins_rad).all()
            or np.any(np.diff(bins_rad) <= 0.0)
            or np.any(histogram < 0)
            or np.any((counts == 0) & (sums != 0.0j))
            or np.any(
                histogram.sum(axis=-1, dtype=np.int64)
                > counts[..., representative_index]
            )
        ):
            raise ValueError("observed trial statistic values are invalid")
        return (
            trial_ids,
            sums,
            counts,
            representative_index,
            representative_hz,
            bins_rad,
            histogram,
        )

    def _set_validated_fields(
        self,
        values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store owned per-trial arrays after disabling element-level writes."""
        (
            trial_ids,
            sums,
            counts,
            representative_index,
            representative_hz,
            bins_rad,
            histogram,
        ) = values
        object.__setattr__(self, "phase_trial_index", _freeze_array(trial_ids))
        object.__setattr__(self, "phase_vector_sum", _freeze_array(sums))
        object.__setattr__(self, "valid_spike_count", _freeze_array(counts))
        object.__setattr__(
            self,
            "representative_frequency_index",
            _freeze_array(representative_index),
        )
        object.__setattr__(self, "representative_frequency_hz", _freeze_array(representative_hz))
        object.__setattr__(self, "phase_bin_edges_rad", _freeze_array(bins_rad))
        object.__setattr__(self, "representative_phase_histogram_count", _freeze_array(histogram))


@dataclass(frozen=True)
class ObservedSegmentedPPCStatistics:
    """Owned same-trial before/after PPC statistics and representative bins.

    Parameters
    ----------
    phase_vector_sum : numpy.ndarray
        Complex128 ``(unit, segment=2, frequency)`` dimensionless pooled unit
        phase-vector sums.  Segment zero is before and segment one is after.
    valid_spike_count : numpy.ndarray
        Int64 array on the same axes as ``phase_vector_sum``.  Entries are
        nonnegative dimensionless valid spike-phase observation counts.
    contributing_trial_count : numpy.ndarray
        Int64 ``(unit, epoch=3, frequency)`` count of physical trials with at
        least one valid observation.  Epoch order is before, after, whole;
        whole is a per-trial union of the two halves.
    representative_frequency_hz : numpy.ndarray
        Float64 ``(band=2,)`` positive finite nondecreasing representative
        frequency coordinates in Hz, ordered nearest 8 then nearest 40 Hz.
        Equal coordinates are valid when one input frequency supplies both.
    phase_bin_edges_rad : numpy.ndarray
        Float64 ``(phase_bin + 1,)`` finite strictly increasing phase-bin edges
        in radians.
    representative_phase_histogram_count : numpy.ndarray
        Int64 ``(unit, epoch=3, band=2, phase_bin)`` nonnegative counts of
        accepted normalized phases.  The whole epoch is before plus after.

    Raises
    ------
    ValueError
        If public fields have incompatible axes, dtypes, coordinate units, or
        nonfinite/negative values.  Public construction copies every array.
    """

    phase_vector_sum: np.ndarray
    valid_spike_count: np.ndarray
    contributing_trial_count: np.ndarray
    representative_frequency_hz: np.ndarray
    phase_bin_edges_rad: np.ndarray
    representative_phase_histogram_count: np.ndarray

    def __post_init__(self) -> None:
        """Defensively validate, copy, and freeze public statistic arrays."""
        self._set_validated_fields(
            self._validated_fields(
                phase_vector_sum=self.phase_vector_sum,
                valid_spike_count=self.valid_spike_count,
                contributing_trial_count=self.contributing_trial_count,
                representative_frequency_hz=self.representative_frequency_hz,
                phase_bin_edges_rad=self.phase_bin_edges_rad,
                representative_phase_histogram_count=self.representative_phase_histogram_count,
            )
        )

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        contributing_trial_count: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> "ObservedSegmentedPPCStatistics":
        """Freeze privately owned reducer outputs without defensive copies.

        Parameters have the public constructor's documented axes and units,
        but are newly allocated inside this module.  Callers must use the
        public constructor, which copies all arrays before freezing them.

        Returns
        -------
        ObservedSegmentedPPCStatistics
            Frozen owned statistic arrays.

        Raises
        ------
        ValueError
            If a private reducer violates the documented structural contract.
        """
        statistics = object.__new__(cls)
        statistics._set_validated_fields(
            cls._trusted_owned_fields(
                phase_vector_sum=phase_vector_sum,
                valid_spike_count=valid_spike_count,
                contributing_trial_count=contributing_trial_count,
                representative_frequency_hz=representative_frequency_hz,
                phase_bin_edges_rad=phase_bin_edges_rad,
                representative_phase_histogram_count=representative_phase_histogram_count,
            )
        )
        return statistics

    @staticmethod
    def _validated_fields(
        *,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        contributing_trial_count: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return defensive owned copies after complete public validation."""
        values = ObservedSegmentedPPCStatistics._checked_fields(
            phase_vector_sum=phase_vector_sum,
            valid_spike_count=valid_spike_count,
            contributing_trial_count=contributing_trial_count,
            representative_frequency_hz=representative_frequency_hz,
            phase_bin_edges_rad=phase_bin_edges_rad,
            representative_phase_histogram_count=representative_phase_histogram_count,
            check_contents=True,
        )
        return tuple(value.copy() for value in values)  # type: ignore[return-value]

    @staticmethod
    def _trusted_owned_fields(
        *,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        contributing_trial_count: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check privately allocated arrays without copying the reducer output."""
        return ObservedSegmentedPPCStatistics._checked_fields(
            phase_vector_sum=phase_vector_sum,
            valid_spike_count=valid_spike_count,
            contributing_trial_count=contributing_trial_count,
            representative_frequency_hz=representative_frequency_hz,
            phase_bin_edges_rad=phase_bin_edges_rad,
            representative_phase_histogram_count=representative_phase_histogram_count,
            check_contents=False,
        )

    @staticmethod
    def _checked_fields(
        *,
        phase_vector_sum: np.ndarray,
        valid_spike_count: np.ndarray,
        contributing_trial_count: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
        check_contents: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate shared S2 statistic axes, dtypes, units, and contents."""
        sums = np.asarray(phase_vector_sum)
        counts = np.asarray(valid_spike_count)
        contributors = np.asarray(contributing_trial_count)
        representative_hz = np.asarray(representative_frequency_hz)
        bins_rad = np.asarray(phase_bin_edges_rad)
        histogram = np.asarray(representative_phase_histogram_count)
        if (
            sums.dtype != np.dtype(np.complex128)
            or sums.ndim != 3
            or sums.shape[0] < 1
            or sums.shape[1] != 2
            or sums.shape[2] < 1
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != sums.shape
            or contributors.dtype != np.dtype(np.int64)
            or contributors.shape != (sums.shape[0], 3, sums.shape[2])
            or representative_hz.dtype != np.dtype(np.float64)
            or representative_hz.shape != (2,)
            or bins_rad.dtype != np.dtype(np.float64)
            or bins_rad.ndim != 1
            or bins_rad.size < 2
            or histogram.dtype != np.dtype(np.int64)
            or histogram.shape != (sums.shape[0], 3, 2, bins_rad.size - 1)
        ):
            raise ValueError("observed segmented statistic arrays have incompatible axes or dtypes")
        if check_contents and (
            not np.isfinite(sums.real).all()
            or not np.isfinite(sums.imag).all()
            or np.any(counts < 0)
            or np.any(contributors < 0)
            or not np.isfinite(representative_hz).all()
            or np.any(representative_hz <= 0.0)
            or np.any(np.diff(representative_hz) < 0.0)
            or not np.isfinite(bins_rad).all()
            or np.any(np.diff(bins_rad) <= 0.0)
            or np.any(histogram < 0)
            or np.any((counts == 0) & (sums != 0.0j))
        ):
            raise ValueError("observed segmented statistic values are invalid")
        return sums, counts, contributors, representative_hz, bins_rad, histogram

    def _set_validated_fields(
        self,
        values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store owned S2 statistic arrays after disabling element writes."""
        (
            sums,
            counts,
            contributors,
            representative_hz,
            bins_rad,
            histogram,
        ) = values
        object.__setattr__(self, "phase_vector_sum", _freeze_array(sums))
        object.__setattr__(self, "valid_spike_count", _freeze_array(counts))
        object.__setattr__(self, "contributing_trial_count", _freeze_array(contributors))
        object.__setattr__(self, "representative_frequency_hz", _freeze_array(representative_hz))
        object.__setattr__(self, "phase_bin_edges_rad", _freeze_array(bins_rad))
        object.__setattr__(self, "representative_phase_histogram_count", _freeze_array(histogram))


@dataclass(frozen=True)
class ObservedSegmentedPPCMetrics:
    """Owned observed PPC metrics composed from segmented sufficient statistics.

    Parameters
    ----------
    ppc, resultant_length, preferred_phase_rad : numpy.ndarray
        Float64 ``(unit, epoch=3, frequency)`` dimensionless PPC, mean-vector
        magnitude, and preferred phase in radians.  Undefined metrics are NaN.
    spike_count : numpy.ndarray
        Int64 ``(unit, epoch=3, frequency)`` nonnegative valid observation
        counts without physical units.
    computable, reliable, shuffle_eligible : numpy.ndarray
        Boolean arrays on the metric axes.  Computable requires at least two
        spikes, reliable at least fifty, and shuffle eligibility additionally
        requires two physical contributor trials and finite PPC.
    contributing_trial_count : numpy.ndarray
        Int64 array on the metric axes containing physical contributing-trial
        counts.  Whole remains the union of before and after contributors.
    representative_frequency_hz, phase_bin_edges_rad,
    representative_phase_histogram_count : numpy.ndarray
        The owned Hz, radians, and histogram coordinates/counts documented by
        ``ObservedSegmentedPPCStatistics``.

    Raises
    ------
    ValueError
        If fields fail the documented dtypes, axes, units, or value bounds.
        Public construction defensively copies every array.
    """

    ppc: np.ndarray
    resultant_length: np.ndarray
    preferred_phase_rad: np.ndarray
    spike_count: np.ndarray
    computable: np.ndarray
    reliable: np.ndarray
    contributing_trial_count: np.ndarray
    shuffle_eligible: np.ndarray
    representative_frequency_hz: np.ndarray
    phase_bin_edges_rad: np.ndarray
    representative_phase_histogram_count: np.ndarray

    def __post_init__(self) -> None:
        """Defensively validate, copy, and freeze public metric arrays."""
        self._set_validated_fields(
            self._validated_fields(
                ppc=self.ppc,
                resultant_length=self.resultant_length,
                preferred_phase_rad=self.preferred_phase_rad,
                spike_count=self.spike_count,
                computable=self.computable,
                reliable=self.reliable,
                contributing_trial_count=self.contributing_trial_count,
                shuffle_eligible=self.shuffle_eligible,
                representative_frequency_hz=self.representative_frequency_hz,
                phase_bin_edges_rad=self.phase_bin_edges_rad,
                representative_phase_histogram_count=self.representative_phase_histogram_count,
            )
        )

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        ppc: np.ndarray,
        resultant_length: np.ndarray,
        preferred_phase_rad: np.ndarray,
        spike_count: np.ndarray,
        computable: np.ndarray,
        reliable: np.ndarray,
        contributing_trial_count: np.ndarray,
        shuffle_eligible: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
    ) -> "ObservedSegmentedPPCMetrics":
        """Freeze private composition arrays without copying their owned buffers.

        Returns
        -------
        ObservedSegmentedPPCMetrics
            Frozen owned metrics with axes and physical units documented by the
            public constructor.

        Raises
        ------
        ValueError
            If an internal composition buffer has an incompatible structure.
        """
        metrics = object.__new__(cls)
        metrics._set_validated_fields(
            cls._trusted_owned_fields(
                ppc=ppc,
                resultant_length=resultant_length,
                preferred_phase_rad=preferred_phase_rad,
                spike_count=spike_count,
                computable=computable,
                reliable=reliable,
                contributing_trial_count=contributing_trial_count,
                shuffle_eligible=shuffle_eligible,
                representative_frequency_hz=representative_frequency_hz,
                phase_bin_edges_rad=phase_bin_edges_rad,
                representative_phase_histogram_count=representative_phase_histogram_count,
            )
        )
        return metrics

    @staticmethod
    def _validated_fields(
        **values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return complete defensive copies after public metric validation."""
        checked = ObservedSegmentedPPCMetrics._checked_fields(check_contents=True, **values)
        return tuple(value.copy() for value in checked)  # type: ignore[return-value]

    @staticmethod
    def _trusted_owned_fields(
        **values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check private metric buffers without duplicating them at return."""
        return ObservedSegmentedPPCMetrics._checked_fields(check_contents=False, **values)

    @staticmethod
    def _checked_fields(
        *,
        ppc: np.ndarray,
        resultant_length: np.ndarray,
        preferred_phase_rad: np.ndarray,
        spike_count: np.ndarray,
        computable: np.ndarray,
        reliable: np.ndarray,
        contributing_trial_count: np.ndarray,
        shuffle_eligible: np.ndarray,
        representative_frequency_hz: np.ndarray,
        phase_bin_edges_rad: np.ndarray,
        representative_phase_histogram_count: np.ndarray,
        check_contents: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate shared S2 metric axes, dtypes, physical units, and values."""
        ppc_values = np.asarray(ppc)
        resultant = np.asarray(resultant_length)
        preferred = np.asarray(preferred_phase_rad)
        counts = np.asarray(spike_count)
        computable_values = np.asarray(computable)
        reliable_values = np.asarray(reliable)
        contributors = np.asarray(contributing_trial_count)
        eligible = np.asarray(shuffle_eligible)
        representative_hz = np.asarray(representative_frequency_hz)
        bins_rad = np.asarray(phase_bin_edges_rad)
        histogram = np.asarray(representative_phase_histogram_count)
        metric_shape = ppc_values.shape
        if (
            ppc_values.dtype != np.dtype(np.float64)
            or ppc_values.ndim != 3
            or metric_shape[0] < 1
            or metric_shape[1] != 3
            or metric_shape[2] < 1
            or resultant.dtype != np.dtype(np.float64)
            or resultant.shape != metric_shape
            or preferred.dtype != np.dtype(np.float64)
            or preferred.shape != metric_shape
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != metric_shape
            or computable_values.dtype != np.dtype(bool)
            or computable_values.shape != metric_shape
            or reliable_values.dtype != np.dtype(bool)
            or reliable_values.shape != metric_shape
            or contributors.dtype != np.dtype(np.int64)
            or contributors.shape != metric_shape
            or eligible.dtype != np.dtype(bool)
            or eligible.shape != metric_shape
            or representative_hz.dtype != np.dtype(np.float64)
            or representative_hz.shape != (2,)
            or bins_rad.dtype != np.dtype(np.float64)
            or bins_rad.ndim != 1
            or bins_rad.size < 2
            or histogram.dtype != np.dtype(np.int64)
            or histogram.shape != (metric_shape[0], 3, 2, bins_rad.size - 1)
        ):
            raise ValueError("observed segmented metric arrays have incompatible axes or dtypes")
        if check_contents and (
            np.any(np.isinf(ppc_values))
            or np.any(np.isinf(resultant))
            or np.any(np.isinf(preferred))
            or np.any(counts < 0)
            or np.any(contributors < 0)
            or not np.isfinite(representative_hz).all()
            or np.any(representative_hz <= 0.0)
            or np.any(np.diff(representative_hz) < 0.0)
            or not np.isfinite(bins_rad).all()
            or np.any(np.diff(bins_rad) <= 0.0)
            or np.any(histogram < 0)
        ):
            raise ValueError("observed segmented metric values are invalid")
        return (
            ppc_values,
            resultant,
            preferred,
            counts,
            computable_values,
            reliable_values,
            contributors,
            eligible,
            representative_hz,
            bins_rad,
            histogram,
        )

    def _set_validated_fields(
        self,
        values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store owned S2 metric arrays after disabling element-level writes."""
        for field_name, value in zip(
            (
                "ppc",
                "resultant_length",
                "preferred_phase_rad",
                "spike_count",
                "computable",
                "reliable",
                "contributing_trial_count",
                "shuffle_eligible",
                "representative_frequency_hz",
                "phase_bin_edges_rad",
                "representative_phase_histogram_count",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, field_name, _freeze_array(value))


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

    spike_vectors = tuple(
        _finite_float64_vector(spike_values, "unit_trial_spike_times_s")
        for spike_values in unit_trial_spike_times_s
    )
    offsets = [0]
    for spikes in spike_vectors:
        for segment_index in range(2):
            lower = bounds[segment_index, 0]
            upper = bounds[segment_index, 1]
            count = 0
            for raw_spike in spikes:
                spike = float(raw_spike)
                if lower <= spike < upper:
                    count += 1
            offsets.append(offsets[-1] + count)

    spike_count = offsets[-1]
    # Retain one flat scalar-filled staging vector only long enough for the
    # single vectorized neighbor search. Final geometry arrays are filled
    # directly below, avoiding full Boolean/fancy-index selection staging.
    flattened_spikes = np.empty(spike_count, dtype=np.float64)
    write_index = 0
    for spikes in spike_vectors:
        for segment_index in range(2):
            lower = bounds[segment_index, 0]
            upper = bounds[segment_index, 1]
            for raw_spike in spikes:
                spike = float(raw_spike)
                if lower <= spike < upper:
                    flattened_spikes[write_index] = spike
                    write_index += 1
    raw_right_index = np.searchsorted(phase_time, flattened_spikes, side="left")
    left_index = np.empty(spike_count, dtype=np.int64)
    right_index = np.empty(spike_count, dtype=np.int64)
    right_weight = np.empty(spike_count, dtype=np.float64)
    inside_support = np.empty(spike_count, dtype=bool)
    exact_sample = np.empty(spike_count, dtype=bool)
    for spike_index in range(spike_count):
        spike = flattened_spikes[spike_index]
        raw_right = int(raw_right_index[spike_index])
        if raw_right >= phase_time.size:
            clamped = phase_time.size - 1
            left_index[spike_index] = clamped
            right_index[spike_index] = clamped
            right_weight[spike_index] = 0.0
            inside_support[spike_index] = False
            exact_sample[spike_index] = False
        elif phase_time[raw_right] == spike:
            left_index[spike_index] = raw_right
            right_index[spike_index] = raw_right
            right_weight[spike_index] = 0.0
            inside_support[spike_index] = True
            exact_sample[spike_index] = True
        elif raw_right == 0:
            left_index[spike_index] = 0
            right_index[spike_index] = 0
            right_weight[spike_index] = 0.0
            inside_support[spike_index] = False
            exact_sample[spike_index] = False
        else:
            left = raw_right - 1
            left_time = phase_time[left]
            right_time = phase_time[raw_right]
            left_index[spike_index] = left
            right_index[spike_index] = raw_right
            right_weight[spike_index] = (spike - left_time) / (right_time - left_time)
            inside_support[spike_index] = True
            exact_sample[spike_index] = False
    del flattened_spikes, raw_right_index, spike_vectors
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


def compute_observed_trial_segmented_ppc_statistics(
    *,
    source_trial_geometries: tuple[SourceTrialSpikeGeometry, ...],
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
) -> ObservedTrialSegmentedPPCStatistics:
    """Sample each physical trial once and retain condition-reusable statistics.

    Parameters
    ----------
    source_trial_geometries : tuple[SourceTrialSpikeGeometry, ...]
        Exactly one geometry for every prepared physical phase trial.  All
        geometries share one ordered unit axis; each geometry's stable source
        identity selects its same-trial row independent of tuple order.
    trial_phase_vectors : numpy.ndarray
        Complex64 ``(trial, frequency, time)`` phase coefficients.  Values
        with nonfinite or zero magnitude are unavailable even when explicitly
        marked valid.
    phase_valid_mask : numpy.ndarray
        Boolean ``(trial, frequency, time)`` prepared-phase availability mask
        with exactly the phase-vector axes.
    phase_trial_index : numpy.ndarray
        Int64 ``(trial,)`` unique nonnegative stable physical trial identities
        in phase-axis order.  Identities have no physical units.
    frequencies_hz : numpy.ndarray
        Float64 ``(frequency,)`` finite positive strictly increasing Morlet
        coordinates in Hz.
    phase_bin_edges_rad : numpy.ndarray
        Float64 ``(phase_bin + 1,)`` finite strictly increasing histogram edges
        in radians.

    Returns
    -------
    ObservedTrialSegmentedPPCStatistics
        Frozen owned same-trial vector sums/counts on
        ``(trial, unit, segment=2, frequency)`` and representative histograms
        on ``(trial, unit, segment=2, band=2, phase_bin)``.  Stable output
        trial order equals ``phase_trial_index``.  Each nonempty physical
        source-trial/unit/segment group is normalized exactly once; no
        per-spike phase array survives the call.

    Raises
    ------
    ValueError
        If phase axes, stable identities, geometries, frequency coordinates,
        or radians histogram coordinates are invalid.  Every prepared trial
        must have exactly one source geometry.
    """
    phase, phase_valid, phase_ids, frequencies = _validated_reduction_phase_inputs(
        trial_phase_vectors=trial_phase_vectors,
        phase_valid_mask=phase_valid_mask,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
    )
    phase_bins = _validated_phase_bin_edges_rad(phase_bin_edges_rad)
    geometries, geometry_by_source = _validated_geometries(
        source_trial_geometries,
        phase_time_count=phase.shape[2],
    )
    phase_position_by_id = {
        int(identity): position for position, identity in enumerate(phase_ids)
    }
    if (
        len(geometries) != phase_ids.size
        or set(geometry_by_source) != set(phase_position_by_id)
    ):
        raise ValueError("observed reduction requires one geometry for every prepared phase trial")

    unit_count = len(geometries[0].unit_ids)
    frequency_count = frequencies.size
    phase_vector_sum = np.zeros(
        (phase_ids.size, unit_count, 2, frequency_count),
        dtype=np.complex128,
    )
    valid_spike_count = np.zeros(
        (phase_ids.size, unit_count, 2, frequency_count),
        dtype=np.int64,
    )
    representative_frequency_position = np.array(
        [
            int(np.argmin(np.abs(frequencies - 8.0))),
            int(np.argmin(np.abs(frequencies - 40.0))),
        ],
        dtype=np.int64,
    )
    representative_frequency_hz = frequencies[representative_frequency_position].copy()
    representative_phase_histogram_count = np.zeros(
        (phase_ids.size, unit_count, 2, 2, phase_bins.size - 1),
        dtype=np.int64,
    )

    for trial_position, stable_trial_id in enumerate(phase_ids):
        geometry = geometry_by_source[int(stable_trial_id)]
        coefficients = phase[trial_position]
        explicit_valid_mask = phase_valid[trial_position]
        for unit_position in range(unit_count):
            for segment_position in range(2):
                group_position = unit_position * 2 + segment_position
                start = int(geometry.group_offsets[group_position])
                stop = int(geometry.group_offsets[group_position + 1])
                if start == stop:
                    continue
                normalized_complex64, usable = _sample_normalized_spike_group(
                    coefficients=coefficients,
                    explicit_valid_mask=explicit_valid_mask,
                    left_index=geometry.left_index[start:stop],
                    right_index=geometry.right_index[start:stop],
                    right_weight=geometry.right_weight[start:stop],
                    inside_support=geometry.inside_support[start:stop],
                    exact_sample=geometry.exact_sample[start:stop],
                )
                phase_vector_sum[trial_position, unit_position, segment_position] = np.sum(
                    normalized_complex64,
                    axis=1,
                    dtype=np.complex128,
                )
                group_valid_count = np.sum(usable, axis=1, dtype=np.int64)
                valid_spike_count[trial_position, unit_position, segment_position] = (
                    group_valid_count
                )
                for band_position, frequency_position in enumerate(
                    representative_frequency_position
                ):
                    accepted_phase_rad = np.angle(
                        normalized_complex64[
                            frequency_position,
                            usable[frequency_position],
                        ]
                    )
                    representative_phase_histogram_count[
                        trial_position,
                        unit_position,
                        segment_position,
                        band_position,
                    ] += np.histogram(accepted_phase_rad, bins=phase_bins)[0]

    return ObservedTrialSegmentedPPCStatistics._from_owned_arrays(
        # Validation retains the caller's full axis as a view; this public
        # result is the ownership boundary and therefore needs its own ID copy.
        phase_trial_index=phase_ids.copy(),
        phase_vector_sum=phase_vector_sum,
        valid_spike_count=valid_spike_count,
        representative_frequency_index=representative_frequency_position,
        representative_frequency_hz=representative_frequency_hz,
        phase_bin_edges_rad=phase_bins,
        representative_phase_histogram_count=representative_phase_histogram_count,
    )


def compute_selected_observed_trial_segmented_ppc_statistics(
    *,
    source_trial_geometries: tuple[SourceTrialSpikeGeometry, ...],
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
) -> ObservedTrialSegmentedPPCStatistics:
    """Reduce selected source geometries against complete prepared phase views.

    Parameters
    ----------
    source_trial_geometries : tuple[SourceTrialSpikeGeometry, ...]
        Ordered selected source-trial geometries. Each stable geometry identity
        must occur exactly once and must be present on the complete
        ``phase_trial_index`` axis. All geometries share one ordered unit axis.
    trial_phase_vectors : numpy.ndarray
        Complex64 ``(full_trial, frequency, time)`` analytic phase
        coefficients for the complete site-local prepared trial axis. Values
        have no physical units.
    phase_valid_mask : numpy.ndarray
        Boolean array with the same ``(full_trial, frequency, time)`` axes as
        ``trial_phase_vectors``.
    phase_trial_index : numpy.ndarray
        Unique nonnegative int64 ``(full_trial,)`` stable trial identities in
        the complete phase-array order.
    frequencies_hz : numpy.ndarray
        Float64 strictly increasing ``(frequency,)`` phase coordinates in Hz.
    phase_bin_edges_rad : numpy.ndarray
        Float64 strictly increasing ``(phase_bin + 1,)`` histogram edges in
        radians.

    Returns
    -------
    ObservedTrialSegmentedPPCStatistics
        Frozen owned selected-trial statistics in the exact geometry tuple
        order. Sums/counts have axes ``(selected_trial, unit, segment=2,
        frequency)``; histogram counts have axes ``(selected_trial, unit,
        segment=2, band=2, phase_bin)``.

    Raises
    ------
    ValueError
        If complete phase axes/identities or geometry identities are invalid,
        a selected geometry is duplicated or unknown, or the selected
        geometries disagree on their unit/time axes.

    Notes
    -----
    This selection-aware entry point intentionally indexes one full-axis phase
    row at a time. It does not gather/copy selected phase or validity rows, so
    the sampler receives views sharing the caller's prepared arrays. The older
    all-trial function above retains its exact public API and semantics.
    """
    phase, phase_valid, phase_ids, frequencies = _validated_reduction_phase_inputs(
        trial_phase_vectors=trial_phase_vectors,
        phase_valid_mask=phase_valid_mask,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
    )
    phase_bins = _validated_phase_bin_edges_rad(phase_bin_edges_rad)
    geometries, geometry_by_source = _validated_geometries(
        source_trial_geometries,
        phase_time_count=phase.shape[2],
    )
    phase_position_by_id = {
        int(identity): position for position, identity in enumerate(phase_ids)
    }
    if len(geometry_by_source) != len(geometries):
        raise ValueError("selected observed geometries must have unique source trial identities")
    if any(source_id not in phase_position_by_id for source_id in geometry_by_source):
        raise ValueError("selected observed geometry source trial is unknown to phase_trial_index")

    selected_trial_count = len(geometries)
    unit_count = len(geometries[0].unit_ids) if geometries else 0
    frequency_count = frequencies.size
    phase_vector_sum = np.zeros(
        (selected_trial_count, unit_count, 2, frequency_count), dtype=np.complex128
    )
    valid_spike_count = np.zeros(
        (selected_trial_count, unit_count, 2, frequency_count), dtype=np.int64
    )
    representative_frequency_position = np.array(
        [
            int(np.argmin(np.abs(frequencies - 8.0))),
            int(np.argmin(np.abs(frequencies - 40.0))),
        ],
        dtype=np.int64,
    )
    representative_frequency_hz = frequencies[representative_frequency_position].copy()
    representative_phase_histogram_count = np.zeros(
        (selected_trial_count, unit_count, 2, 2, phase_bins.size - 1),
        dtype=np.int64,
    )
    selected_ids = np.empty(selected_trial_count, dtype=np.int64)

    for selected_position, geometry in enumerate(geometries):
        stable_trial_id = int(geometry.source_trial_index)
        selected_ids[selected_position] = stable_trial_id
        full_position = phase_position_by_id[stable_trial_id]
        # Basic indexing yields row views, preserving the complete-array owner.
        coefficients = phase[full_position]
        explicit_valid_mask = phase_valid[full_position]
        for unit_position in range(unit_count):
            for segment_position in range(2):
                group_position = unit_position * 2 + segment_position
                start = int(geometry.group_offsets[group_position])
                stop = int(geometry.group_offsets[group_position + 1])
                if start == stop:
                    continue
                normalized_complex64, usable = _sample_normalized_spike_group(
                    coefficients=coefficients,
                    explicit_valid_mask=explicit_valid_mask,
                    left_index=geometry.left_index[start:stop],
                    right_index=geometry.right_index[start:stop],
                    right_weight=geometry.right_weight[start:stop],
                    inside_support=geometry.inside_support[start:stop],
                    exact_sample=geometry.exact_sample[start:stop],
                )
                phase_vector_sum[selected_position, unit_position, segment_position] = np.sum(
                    normalized_complex64, axis=1, dtype=np.complex128
                )
                group_valid_count = np.sum(usable, axis=1, dtype=np.int64)
                valid_spike_count[selected_position, unit_position, segment_position] = (
                    group_valid_count
                )
                for band_position, frequency_position in enumerate(
                    representative_frequency_position
                ):
                    accepted_phase_rad = np.angle(
                        normalized_complex64[
                            frequency_position,
                            usable[frequency_position],
                        ]
                    )
                    representative_phase_histogram_count[
                        selected_position,
                        unit_position,
                        segment_position,
                        band_position,
                    ] += np.histogram(accepted_phase_rad, bins=phase_bins)[0]

    return ObservedTrialSegmentedPPCStatistics._from_owned_arrays(
        phase_trial_index=selected_ids,
        phase_vector_sum=phase_vector_sum,
        valid_spike_count=valid_spike_count,
        representative_frequency_index=representative_frequency_position,
        representative_frequency_hz=representative_frequency_hz,
        phase_bin_edges_rad=phase_bins,
        representative_phase_histogram_count=representative_phase_histogram_count,
    )


def aggregate_observed_trial_segmented_ppc_statistics(
    *,
    observed_trial_statistics: ObservedTrialSegmentedPPCStatistics,
    membership_trial_index: np.ndarray,
) -> ObservedSegmentedPPCStatistics:
    """Pool reusable same-trial statistics for one ordered condition membership.

    Parameters
    ----------
    observed_trial_statistics : ObservedTrialSegmentedPPCStatistics
        Frozen stable trial statistics with axes and physical units documented
        by its result contract.  This pure reducer does not access raw phase
        coefficients or sample spikes.
    membership_trial_index : numpy.ndarray
        Int64 one-dimensional ordered unique stable physical trial identities.
        The empty int64 vector is valid and returns zero pooled statistics with
        the original unit/frequency, Hz, radians, and histogram axes retained.

    Returns
    -------
    ObservedSegmentedPPCStatistics
        Frozen owned before/after sums and counts on
        ``(unit, segment=2, frequency)``, contributing physical trial counts
        on ``(unit, epoch=3, frequency)``, and representative histograms on
        ``(unit, epoch=3, band=2, phase_bin)``.  Whole contributors are a
        per-trial before/after union and whole histograms add the two halves.

    Raises
    ------
    ValueError
        If the input result type is wrong, membership is not a one-dimensional
        int64 unique nonnegative stable-ID vector, or a member is unknown.
    """
    if not isinstance(observed_trial_statistics, ObservedTrialSegmentedPPCStatistics):
        raise ValueError("observed_trial_statistics must be ObservedTrialSegmentedPPCStatistics")
    membership = np.asarray(membership_trial_index)
    if (
        membership.dtype != np.dtype(np.int64)
        or membership.ndim != 1
        or np.any(membership < 0)
        or np.unique(membership).size != membership.size
    ):
        raise ValueError("membership_trial_index must be unique nonnegative int64 stable IDs")
    trial_position_by_id = {
        int(stable_id): position
        for position, stable_id in enumerate(observed_trial_statistics.phase_trial_index)
    }
    if any(int(stable_id) not in trial_position_by_id for stable_id in membership):
        raise ValueError("membership_trial_index contains an unknown stable trial ID")

    _, unit_count, _, frequency_count = observed_trial_statistics.phase_vector_sum.shape
    phase_vector_sum = np.zeros((unit_count, 2, frequency_count), dtype=np.complex128)
    valid_spike_count = np.zeros((unit_count, 2, frequency_count), dtype=np.int64)
    contributing_trial_count = np.zeros((unit_count, 3, frequency_count), dtype=np.int64)
    phase_bin_count = observed_trial_statistics.phase_bin_edges_rad.size - 1
    representative_phase_histogram_count = np.zeros(
        (unit_count, 3, 2, phase_bin_count),
        dtype=np.int64,
    )
    for stable_id in membership:
        trial_position = trial_position_by_id[int(stable_id)]
        trial_sums = observed_trial_statistics.phase_vector_sum[trial_position]
        trial_counts = observed_trial_statistics.valid_spike_count[trial_position]
        phase_vector_sum += trial_sums
        valid_spike_count += trial_counts
        contributing_trial_count[:, 0] += trial_counts[:, 0] > 0
        contributing_trial_count[:, 1] += trial_counts[:, 1] > 0
        contributing_trial_count[:, 2] += np.logical_or(
            trial_counts[:, 0] > 0,
            trial_counts[:, 1] > 0,
        )
        representative_phase_histogram_count[:, :2] += (
            observed_trial_statistics.representative_phase_histogram_count[
                trial_position
            ]
        )
    representative_phase_histogram_count[:, 2] = (
        representative_phase_histogram_count[:, 0]
        + representative_phase_histogram_count[:, 1]
    )
    return ObservedSegmentedPPCStatistics._from_owned_arrays(
        phase_vector_sum=phase_vector_sum,
        valid_spike_count=valid_spike_count,
        contributing_trial_count=contributing_trial_count,
        representative_frequency_hz=(
            observed_trial_statistics.representative_frequency_hz.copy()
        ),
        phase_bin_edges_rad=observed_trial_statistics.phase_bin_edges_rad.copy(),
        representative_phase_histogram_count=representative_phase_histogram_count,
    )


def compute_observed_segmented_ppc_statistics(
    *,
    source_trial_geometries: tuple[SourceTrialSpikeGeometry, ...],
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
) -> ObservedSegmentedPPCStatistics:
    """Return all-trial pooled observed statistics through the canonical S2 path.

    Parameters
    ----------
    source_trial_geometries : tuple[SourceTrialSpikeGeometry, ...]
        Exactly one geometry for every prepared stable physical trial, sharing
        one ordered unit axis.  Geometry identities need not match tuple order.
    trial_phase_vectors : numpy.ndarray
        Complex64 ``(trial, frequency, time)`` dimensionless analytic phase
        coefficients.  Nonfinite or zero-magnitude values are unavailable.
    phase_valid_mask : numpy.ndarray
        Boolean ``(trial, frequency, time)`` explicit prepared-phase
        availability mask with axes matching ``trial_phase_vectors``.
    phase_trial_index : numpy.ndarray
        Int64 ``(trial,)`` unique nonnegative stable physical trial identities
        in phase-axis order, without physical units.
    frequencies_hz : numpy.ndarray
        Float64 ``(frequency,)`` finite positive strictly increasing Morlet
        frequency coordinates in Hz.
    phase_bin_edges_rad : numpy.ndarray
        Float64 ``(phase_bin + 1,)`` finite strictly increasing histogram-bin
        edges in radians.

    Returns
    -------
    ObservedSegmentedPPCStatistics
        Frozen all-prepared-trial pooled before/after statistics, contributor
        unions, and representative histograms.  It is exactly the membership
        aggregation of the per-trial result in stable phase-axis order.

    Raises
    ------
    ValueError
        If the delegated per-trial reduction rejects the input axes, geometry,
        stable identities, frequency coordinates, or phase-bin coordinates.
    """
    observed_trial_statistics = compute_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=source_trial_geometries,
        trial_phase_vectors=trial_phase_vectors,
        phase_valid_mask=phase_valid_mask,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=phase_bin_edges_rad,
    )
    return aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trial_statistics,
        membership_trial_index=observed_trial_statistics.phase_trial_index,
    )


def compose_observed_segmented_ppc_metrics(
    *,
    observed_statistics: ObservedSegmentedPPCStatistics,
) -> ObservedSegmentedPPCMetrics:
    """Compose before, after, and whole PPC metrics from observed sufficient statistics.

    Parameters
    ----------
    observed_statistics : ObservedSegmentedPPCStatistics
        Frozen same-trial sums/counts with unit, segment, frequency, Hz, and
        radians axes as documented by its result contract.

    Returns
    -------
    ObservedSegmentedPPCMetrics
        Frozen owned float64 dimensionless PPC/resultant metrics and radians
        preferred phase on ``(unit, epoch=3, frequency)``.  Whole metrics are
        calculated after adding before/after complex sums and counts, never by
        averaging half-window PPC values.

    Raises
    ------
    ValueError
        If ``observed_statistics`` is not the frozen S2 statistic contract.
    """
    if not isinstance(observed_statistics, ObservedSegmentedPPCStatistics):
        raise ValueError("observed_statistics must be ObservedSegmentedPPCStatistics")
    unit_count, _, frequency_count = observed_statistics.phase_vector_sum.shape
    epoch_sums = np.empty((unit_count, 3, frequency_count), dtype=np.complex128)
    epoch_sums[:, :2] = observed_statistics.phase_vector_sum
    epoch_sums[:, 2] = (
        observed_statistics.phase_vector_sum[:, 0]
        + observed_statistics.phase_vector_sum[:, 1]
    )
    spike_count = np.empty((unit_count, 3, frequency_count), dtype=np.int64)
    spike_count[:, :2] = observed_statistics.valid_spike_count
    spike_count[:, 2] = (
        observed_statistics.valid_spike_count[:, 0]
        + observed_statistics.valid_spike_count[:, 1]
    )
    ppc, resultant_length, preferred_phase_rad = _pooled_ppc_metrics(
        phase_vector_sum=epoch_sums,
        spike_count=spike_count,
    )
    computable = spike_count >= 2
    reliable = spike_count >= 50
    contributing_trial_count = observed_statistics.contributing_trial_count.copy()
    shuffle_eligible = (
        reliable
        & (contributing_trial_count >= 2)
        & np.isfinite(ppc)
    )
    return ObservedSegmentedPPCMetrics._from_owned_arrays(
        ppc=ppc,
        resultant_length=resultant_length,
        preferred_phase_rad=preferred_phase_rad,
        spike_count=spike_count,
        computable=computable,
        reliable=reliable,
        contributing_trial_count=contributing_trial_count,
        shuffle_eligible=shuffle_eligible,
        representative_frequency_hz=observed_statistics.representative_frequency_hz.copy(),
        phase_bin_edges_rad=observed_statistics.phase_bin_edges_rad.copy(),
        representative_phase_histogram_count=(
            observed_statistics.representative_phase_histogram_count.copy()
        ),
    )


def reduce_segmented_schedule_to_ppc(
    *,
    edge_statistics: SegmentedEdgeStatistics,
    phase_trial_index: np.ndarray,
    schedule: np.ndarray,
) -> np.ndarray:
    """Reduce a supplied whole-window shuffle schedule from segmented edge statistics.

    Parameters
    ----------
    edge_statistics : SegmentedEdgeStatistics
        Frozen unique physical source-target edge sums/counts with axes
        ``(edge, unit, segment=2, frequency)``.  Segment statistics are added
        before calculating each shuffle PPC value.
    phase_trial_index : numpy.ndarray
        Int64 ``(trial,)`` unique nonnegative stable physical trial identities
        in the local schedule's source/target position order.
    schedule : numpy.ndarray
        Int64 ``(shuffle, trial)`` local target positions.  Every row must be
        a derangement permutation of the local trial positions.  This function
        neither changes rows nor derives random seeds.

    Returns
    -------
    numpy.ndarray
        Float64 dimensionless PPC draws with shape ``(shuffle, unit,
        frequency)``.  Draw values are NaN where fewer than two valid phases
        were pooled.

    Raises
    ------
    ValueError
        If identities or schedule rows are malformed, or a scheduled physical
        source-target pair is absent from ``edge_statistics``.
    """
    if not isinstance(edge_statistics, SegmentedEdgeStatistics):
        raise ValueError("edge_statistics must be SegmentedEdgeStatistics")
    phase_ids = _validated_phase_trial_index(phase_trial_index)
    schedule_array = np.asarray(schedule)
    local_positions = np.arange(phase_ids.size, dtype=np.int64)
    if (
        schedule_array.dtype != np.dtype(np.int64)
        or schedule_array.ndim != 2
        or schedule_array.shape[0] < 1
        or schedule_array.shape[1] != phase_ids.size
        or np.any(schedule_array < 0)
        or np.any(schedule_array >= phase_ids.size)
        or np.any(schedule_array == local_positions[np.newaxis, :])
        or not np.all(np.sort(schedule_array, axis=1) == local_positions[np.newaxis, :])
    ):
        raise ValueError("schedule must be a nonempty int64 local-position derangement")

    edge_position_by_pair = {
        (int(source_id), int(target_id)): edge_position
        for edge_position, (source_id, target_id) in enumerate(
            zip(
                edge_statistics.source_trial_index,
                edge_statistics.target_trial_index,
                strict=True,
            )
        )
    }
    unit_count = edge_statistics.phase_vector_sum.shape[1]
    frequency_count = edge_statistics.phase_vector_sum.shape[3]
    phase_vector_sum = np.zeros(
        (schedule_array.shape[0], unit_count, frequency_count),
        dtype=np.complex128,
    )
    valid_spike_count = np.zeros(
        (schedule_array.shape[0], unit_count, frequency_count),
        dtype=np.int64,
    )
    for shuffle_position, target_positions in enumerate(schedule_array):
        for source_position, target_position in enumerate(target_positions):
            edge_position = edge_position_by_pair.get(
                (int(phase_ids[source_position]), int(phase_ids[target_position]))
            )
            if edge_position is None:
                raise ValueError("schedule references an edge absent from edge_statistics")
            phase_vector_sum[shuffle_position] += np.sum(
                edge_statistics.phase_vector_sum[edge_position],
                axis=1,
                dtype=np.complex128,
            )
            valid_spike_count[shuffle_position] += np.sum(
                edge_statistics.valid_spike_count[edge_position],
                axis=1,
                dtype=np.int64,
            )
    ppc, _, _ = _pooled_ppc_metrics(
        phase_vector_sum=phase_vector_sum,
        spike_count=valid_spike_count,
    )
    return ppc


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
    ):
        raise ValueError("source_trial_spike_count must be nonnegative int64 (source, unit, 2)")
    for raw_count in counts.flat:
        if int(raw_count) < 0:
            raise ValueError(
                "source_trial_spike_count must be nonnegative int64 (source, unit, 2)"
            )
    if (
        positions.dtype != np.dtype(np.int64)
        or positions.ndim != 1
        or positions.size < 1
    ):
        raise ValueError("edge_source_trial_position must be int64 positions in source axis")
    for raw_position in positions:
        position = int(raw_position)
        if position < 0 or position >= counts.shape[0]:
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
    normalized_complex64, usable = _sample_normalized_spike_group(
        coefficients=coefficients,
        explicit_valid_mask=explicit_valid_mask,
        left_index=left_index,
        right_index=right_index,
        right_weight=right_weight,
        inside_support=inside_support,
        exact_sample=exact_sample,
    )
    vector_sum = np.sum(normalized_complex64, axis=1, dtype=np.complex128)
    valid_count = np.sum(usable, axis=1, dtype=np.int64)
    return vector_sum, valid_count


def _sample_normalized_spike_group(
    *,
    coefficients: np.ndarray,
    explicit_valid_mask: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    right_weight: np.ndarray,
    inside_support: np.ndarray,
    exact_sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical normalized vectors and validity for one geometry group.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Complex64 ``(frequency, time)`` phase coefficients for one target
        trial.  Coefficients are dimensionless complex analytic phases.
    explicit_valid_mask : numpy.ndarray
        Boolean ``(frequency, time)`` prepared-phase availability mask.  It is
        combined with finite/nonzero coefficient checks rather than replaced.
    left_index, right_index : numpy.ndarray
        Int64 ``(spike,)`` safe stored canonical phase-time neighbor indices.
    right_weight : numpy.ndarray
        Float64 ``(spike,)`` dimensionless right-neighbor interpolation weights.
    inside_support, exact_sample : numpy.ndarray
        Boolean ``(spike,)`` stored support/exact-grid geometry masks.  Exact
        samples require only their left-neighbor coefficient; interpolation
        samples require both coefficients.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A complex64 ``(frequency, spike)`` dimensionless normalized phase array
        with zero values at unavailable observations, and a Boolean array on
        the same axes selecting accepted samples.  Callers may sum these arrays
        but must not retain per-spike outputs after their bounded reduction.

    Raises
    ------
    ValueError
        This private helper assumes prior geometry/phase validation and does
        not perform public input validation.  Array-index failures therefore
        indicate an internal contract violation.

    Notes
    -----
    Per ``(frequency, spike)`` cell, only two complex64 gathers, one complex128
    interpolation buffer, one float64 magnitude/work buffer, one complex64
    normalized result, and at most three Boolean masks are simultaneously live.
    This preserves the S1 51-byte allocation accounting.
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
    # Preserve exact-grid coefficients (including the signed zero that makes
    # -pi distinct from +pi for NumPy histogram boundary assignment).
    np.copyto(interpolated, left_coefficients, where=exact_sample[np.newaxis, :])
    del left_coefficients, right_coefficients

    # The gathers are gone before normalization.  Reuse ``usable`` and
    # ``magnitude`` for interpolation validity, then divide complex128 values
    # directly into the required complex64 reduction buffer.
    np.isfinite(interpolated.real, out=usable, where=usable)
    np.isfinite(interpolated.imag, out=usable, where=usable)
    np.abs(interpolated, out=magnitude)
    np.not_equal(magnitude, 0.0, out=usable, where=usable)
    normalized_complex64 = np.zeros(interpolated.shape, dtype=np.complex64)
    # Normalize components separately so exact ``-1 - 0j`` retains its signed
    # zero and therefore stays at the canonical -pi histogram boundary.
    np.divide(interpolated.real, magnitude, out=normalized_complex64.real, where=usable)
    np.divide(interpolated.imag, magnitude, out=normalized_complex64.imag, where=usable)
    del interpolated, magnitude
    return normalized_complex64, usable


def _validated_uniform_phase_time(
    phase_time_s: np.ndarray,
    phase_sampling_rate_hz: float,
) -> np.ndarray:
    """Validate and return one non-retained uniform float64 seconds grid."""
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
    return phase_time


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
    """Validate full prepared phase inputs without copying the trial-ID axis.

    Parameters
    ----------
    trial_phase_vectors : numpy.ndarray
        Complex64 ``(trial, frequency, time)`` dimensionless prepared phase
        coefficients.
    phase_valid_mask : numpy.ndarray
        Boolean array with exactly the same axes as ``trial_phase_vectors``.
    phase_trial_index : numpy.ndarray
        Int64 ``(trial,)`` nonnegative, unique stable trial identities without
        physical units. The returned value shares this caller-owned storage.
    frequencies_hz : numpy.ndarray
        Float64 positive, strictly increasing ``(frequency,)`` coordinates in
        Hz.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Validated views of phase coefficients, validity, stable trial IDs, and
        frequency coordinates. No returned full-axis value is copied.

    Raises
    ------
    ValueError
        If phase/mask axes or dtypes, stable-ID dtype/axis/identity rules, or
        frequency coordinates are invalid.

    Notes
    -----
    Stable IDs are checked one scalar at a time. This deliberately avoids a
    full-axis Boolean comparison, ``numpy.unique`` output, or ID copy before
    selected/public reducers construct their required owned result identities.
    """
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
    ):
        raise ValueError("phase_trial_index must be unique nonnegative int64 trial rows")
    stable_identity_set: set[int] = set()
    for stable_identity in phase_ids:
        identity = int(stable_identity)
        if identity < 0 or identity in stable_identity_set:
            raise ValueError("phase_trial_index must be unique nonnegative int64 trial rows")
        stable_identity_set.add(identity)
    if (
        frequencies.dtype != np.dtype(np.float64)
        or frequencies.ndim != 1
        or frequencies.size != phase.shape[1]
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise ValueError("frequencies_hz must be positive increasing float64 phase coordinates")
    return phase, valid, phase_ids, frequencies


def _validated_phase_bin_edges_rad(phase_bin_edges_rad: np.ndarray) -> np.ndarray:
    """Return owned finite increasing float64 phase-bin edges in radians.

    Parameters
    ----------
    phase_bin_edges_rad : numpy.ndarray
        Float64 one-dimensional phase-coordinate edges in radians.  At least
        two finite strictly increasing values are required.

    Returns
    -------
    numpy.ndarray
        Owned float64 ``(phase_bin + 1,)`` radians coordinate array.

    Raises
    ------
    ValueError
        If dtype, axis, finiteness, or strict ordering is invalid.
    """
    bins_rad = np.asarray(phase_bin_edges_rad)
    if (
        bins_rad.dtype != np.dtype(np.float64)
        or bins_rad.ndim != 1
        or bins_rad.size < 2
        or not np.isfinite(bins_rad).all()
        or np.any(np.diff(bins_rad) <= 0.0)
    ):
        raise ValueError("phase_bin_edges_rad must be finite increasing float64 radians")
    return bins_rad.copy()


def _validated_phase_trial_index(phase_trial_index: np.ndarray) -> np.ndarray:
    """Return owned unique nonnegative int64 stable physical trial identities.

    Parameters
    ----------
    phase_trial_index : numpy.ndarray
        Int64 one-dimensional physical trial identities without physical units.

    Returns
    -------
    numpy.ndarray
        Owned int64 ``(trial,)`` stable identities in their supplied local order.

    Raises
    ------
    ValueError
        If dtype, shape, uniqueness, or nonnegative identity rules fail.
    """
    phase_ids = np.asarray(phase_trial_index)
    if (
        phase_ids.dtype != np.dtype(np.int64)
        or phase_ids.ndim != 1
        or phase_ids.size < 1
        or np.any(phase_ids < 0)
        or np.unique(phase_ids).size != phase_ids.size
    ):
        raise ValueError("phase_trial_index must be unique nonnegative int64 trial rows")
    return phase_ids.copy()


def _pooled_ppc_metrics(
    *,
    phase_vector_sum: np.ndarray,
    spike_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate PPC and circular metrics from already pooled sufficient statistics.

    Parameters
    ----------
    phase_vector_sum : numpy.ndarray
        Complex128 array on arbitrary leading axes containing dimensionless
        pooled normalized phase-vector sums.
    spike_count : numpy.ndarray
        Int64 nonnegative array matching ``phase_vector_sum`` that contains
        dimensionless valid observation counts.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Float64 arrays matching the input axes: unbiased dimensionless PPC,
        dimensionless resultant length, and preferred phase in radians.  PPC
        is NaN below two samples; resultant is NaN with zero samples; preferred
        phase is NaN when the summed vector has no direction at the established
        floating-point direction epsilon.

    Raises
    ------
    ValueError
        If private pooled arrays do not have matching complex128/int64 axes or
        contain negative counts.
    """
    sums = np.asarray(phase_vector_sum)
    counts = np.asarray(spike_count)
    if (
        sums.dtype != np.dtype(np.complex128)
        or counts.dtype != np.dtype(np.int64)
        or sums.shape != counts.shape
        or np.any(counts < 0)
    ):
        raise ValueError("pooled PPC inputs must be matching complex128/int64 arrays")
    magnitude = np.abs(sums)
    ppc = np.full(sums.shape, np.nan, dtype=np.float64)
    computable = counts >= 2
    ppc[computable] = (
        magnitude[computable] ** 2 - counts[computable]
    ) / (counts[computable] * (counts[computable] - 1))
    resultant_length = np.full(sums.shape, np.nan, dtype=np.float64)
    has_spikes = counts > 0
    resultant_length[has_spikes] = magnitude[has_spikes] / counts[has_spikes]
    preferred_phase_rad = np.full(sums.shape, np.nan, dtype=np.float64)
    has_direction = has_spikes & (
        magnitude > np.finfo(float).eps * np.maximum(counts, 1)
    )
    preferred_phase_rad[has_direction] = np.angle(sums[has_direction])
    return ppc, resultant_length, preferred_phase_rad


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
    """Validate one finite float64 relative-seconds vector without copying it."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite one-dimensional seconds arrays")
    return vector


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
    """Return an owned base-ndarray result after disabling element-level writes."""
    frozen = array if type(array) is np.ndarray else array.view(np.ndarray)
    frozen.setflags(write=False)
    return frozen


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
