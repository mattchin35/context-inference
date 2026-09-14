"""RED contracts for the uniform-grid segmented PPC sufficient-statistic kernel."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ppc_kernel as kernel
from src.neural_analysis import spike_lfp_summary


_ZERO_BOUND_SEGMENT_BOUNDS_S = ((-0.5, 0.0), (0.0, 0.5))
_FOUR_HZ_TIME_S = np.array([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=np.float64)


def _build_geometry(
    *,
    source_trial_index: int,
    unit_ids: tuple[str, ...],
    unit_trial_spike_times_s: tuple[np.ndarray, ...],
    phase_time_s: np.ndarray = _FOUR_HZ_TIME_S,
    phase_sampling_rate_hz: float = 4.0,
    segment_bounds_s: tuple[tuple[float, float], tuple[float, float]] = _ZERO_BOUND_SEGMENT_BOUNDS_S,
) -> object:
    """Build one source-trial geometry from unit-local relative-second spikes.

    Parameters have the kernel contract's units and axes: ``phase_time_s`` is
    float64 ``(time,)`` seconds, and every entry in
    ``unit_trial_spike_times_s`` is float64 ``(spike,)`` relative seconds.
    The returned object is the frozen one-source-trial geometry.
    """
    return kernel.build_source_trial_spike_geometry(
        phase_time_s=phase_time_s,
        phase_sampling_rate_hz=phase_sampling_rate_hz,
        source_trial_index=source_trial_index,
        unit_ids=unit_ids,
        unit_trial_spike_times_s=unit_trial_spike_times_s,
        segment_bounds_s=segment_bounds_s,
    )


def _reduce(
    *,
    source_trial_geometries: tuple[object, ...],
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
    source_trial_index: np.ndarray,
    target_trial_index: np.ndarray,
) -> object:
    """Reduce bounded stable physical edges into before/after sufficient statistics.

    ``trial_phase_vectors`` is complex64 ``(trial, frequency, time)`` and
    ``phase_valid_mask`` is Boolean on the identical axes and
    ``phase_trial_index`` is int64 ``(trial,)`` unique stable trial-table rows
    in phase-axis order. Edge identity arrays are int64 ``(edge,)`` stable
    trial-table rows. The result is the frozen segmented edge statistic with
    axes ``(edge, unit, segment, frequency)``.
    """
    return kernel.compute_segmented_edge_statistics(
        source_trial_geometries=source_trial_geometries,
        trial_phase_vectors=trial_phase_vectors,
        phase_valid_mask=phase_valid_mask,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        source_trial_index=source_trial_index,
        target_trial_index=target_trial_index,
    )


def _one_geometry_and_phase(
    *,
    spikes_s: np.ndarray,
    coefficients: np.ndarray,
    valid: np.ndarray,
    phase_time_s: np.ndarray,
    segment_bounds_s: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[object, np.ndarray, np.ndarray]:
    """Return one geometry and one-frequency one-trial complex64 phase fixture.

    ``coefficients`` and ``valid`` are one-dimensional time-axis values.  The
    returned phase/value mask have axes ``(trial=1, frequency=1, time)``.
    """
    geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.asarray(spikes_s, dtype=np.float64),),
        phase_time_s=phase_time_s,
        phase_sampling_rate_hz=1.0 / float(phase_time_s[1] - phase_time_s[0]),
        segment_bounds_s=segment_bounds_s,
    )
    phase = np.asarray(coefficients, dtype=np.complex64)[np.newaxis, np.newaxis, :]
    mask = np.asarray(valid, dtype=bool)[np.newaxis, np.newaxis, :]
    return geometry, phase, mask


def _single_edge_statistics(
    *,
    geometry: object,
    phase: np.ndarray,
    valid: np.ndarray,
) -> object:
    """Reduce the one physical ``0 -> 0`` edge of a one-trial fixture."""
    return _reduce(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([0], dtype=np.int64),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        source_trial_index=np.array([0], dtype=np.int64),
        target_trial_index=np.array([0], dtype=np.int64),
    )


def _assert_ndarray_fields_are_elementwise_immutable(result: object) -> None:
    """Check that every ndarray field is read-only, not merely frozen by its dataclass."""
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, f"{field.name} must be read-only"
            with pytest.raises(ValueError, match="read-only"):
                value.flat[0] = value.flat[0]


def test_kernel_contracts_are_keyword_only_frozen_and_axis_explicit() -> None:
    """The new kernel exposes only the frozen, keyword-only S1 data contracts."""
    expected_geometry_fields = (
        "source_trial_index",
        "unit_ids",
        "group_offsets",
        "left_index",
        "right_index",
        "right_weight",
        "inside_support",
        "exact_sample",
    )
    expected_statistics_fields = (
        "source_trial_index",
        "target_trial_index",
        "phase_vector_sum",
        "valid_spike_count",
    )
    expected_allocation_fields = (
        "geometry_bytes",
        "segmented_edge_statistics_bytes",
        "gather_temporary_bytes",
        "planned_kernel_peak_bytes",
    )
    assert tuple(field.name for field in fields(kernel.SourceTrialSpikeGeometry)) == expected_geometry_fields
    assert tuple(field.name for field in fields(kernel.SegmentedEdgeStatistics)) == expected_statistics_fields
    assert tuple(field.name for field in fields(kernel.KernelAllocationEstimate)) == expected_allocation_fields

    geometry_signature = inspect.signature(kernel.build_source_trial_spike_geometry)
    assert tuple(geometry_signature.parameters) == (
        "phase_time_s",
        "phase_sampling_rate_hz",
        "source_trial_index",
        "unit_ids",
        "unit_trial_spike_times_s",
        "segment_bounds_s",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in geometry_signature.parameters.values()
    )
    reduction_signature = inspect.signature(kernel.compute_segmented_edge_statistics)
    assert tuple(reduction_signature.parameters) == (
        "source_trial_geometries",
        "trial_phase_vectors",
        "phase_valid_mask",
        "phase_trial_index",
        "frequencies_hz",
        "source_trial_index",
        "target_trial_index",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in reduction_signature.parameters.values()
    )
    allocation_signature = inspect.signature(kernel.estimate_segmented_kernel_allocation)
    assert tuple(allocation_signature.parameters) == (
        "source_trial_spike_count",
        "edge_source_trial_position",
        "frequency_count",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in allocation_signature.parameters.values()
    )

    geometry = _build_geometry(
        source_trial_index=7,
        unit_ids=("unit-7",),
        unit_trial_spike_times_s=(np.array([-0.25, 0.0]),),
    )
    assert geometry.source_trial_index.dtype == np.dtype(np.int64)
    assert geometry.source_trial_index.shape == ()
    assert geometry.group_offsets.dtype == np.dtype(np.int64)
    assert geometry.group_offsets.shape == (3,)
    assert geometry.left_index.dtype == np.dtype(np.int64)
    assert geometry.right_index.dtype == np.dtype(np.int64)
    assert geometry.right_weight.dtype == np.dtype(np.float64)
    assert geometry.inside_support.dtype == np.dtype(bool)
    assert geometry.exact_sample.dtype == np.dtype(bool)
    with pytest.raises(FrozenInstanceError):
        geometry.source_trial_index = np.array(8, dtype=np.int64)


@pytest.mark.parametrize(
    ("phase_time_s", "phase_sampling_rate_hz"),
    (
        (np.array([-0.5, -0.2, 0.0, 0.25, 0.5]), 4.0),
        (_FOUR_HZ_TIME_S, 500.0),
    ),
)
def test_geometry_rejects_nonuniform_or_rate_inconsistent_canonical_grid(
    phase_time_s: np.ndarray,
    phase_sampling_rate_hz: float,
) -> None:
    """Geometry rejects time coordinates that cannot be the configured uniform grid."""
    with pytest.raises(ValueError):
        _build_geometry(
            source_trial_index=0,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([0.0]),),
            phase_time_s=phase_time_s,
            phase_sampling_rate_hz=phase_sampling_rate_hz,
        )


def test_geometry_accepts_valid_non_500_hz_grid_and_preserves_spike_group_order() -> None:
    """A valid 4 Hz grid is supported without moving spikes across unit/segment groups."""
    geometry = _build_geometry(
        source_trial_index=13,
        unit_ids=("unit-b", "unit-a"),
        unit_trial_spike_times_s=(
            np.array([0.25, -0.25, 0.0]),
            np.array([-0.5, 0.25]),
        ),
    )

    assert geometry.source_trial_index.item() == 13
    assert geometry.unit_ids == ("unit-b", "unit-a")
    np.testing.assert_array_equal(geometry.group_offsets, [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(geometry.left_index, [1, 3, 2, 0, 3])
    np.testing.assert_array_equal(geometry.right_index, [1, 3, 2, 0, 3])
    np.testing.assert_array_equal(geometry.right_weight, np.zeros(5, dtype=np.float64))
    np.testing.assert_array_equal(geometry.inside_support, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(geometry.exact_sample, np.ones(5, dtype=bool))


def test_geometry_owns_inputs_and_retains_safe_metadata_for_outside_support_spikes() -> None:
    """Caller mutation cannot alter geometry; unsupported spikes retain safe ignored gathers."""
    time_s = _FOUR_HZ_TIME_S.copy()
    spikes_s = np.array([-1.0, -0.25, 1.0], dtype=np.float64)
    geometry = _build_geometry(
        source_trial_index=13,
        unit_ids=("unit-13",),
        unit_trial_spike_times_s=(spikes_s,),
        phase_time_s=time_s,
        segment_bounds_s=((-1.25, 0.0), (0.0, 1.25)),
    )
    expected_left = geometry.left_index.copy()
    expected_offsets = geometry.group_offsets.copy()
    time_s[1] = 99.0
    spikes_s[:] = 99.0

    np.testing.assert_array_equal(geometry.group_offsets, expected_offsets)
    np.testing.assert_array_equal(geometry.left_index, expected_left)
    np.testing.assert_array_equal(geometry.left_index, [0, 1, 4])
    np.testing.assert_array_equal(geometry.right_index, [0, 1, 4])
    np.testing.assert_array_equal(geometry.right_weight, np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(geometry.inside_support, [False, True, False])
    np.testing.assert_array_equal(geometry.exact_sample, [False, True, False])


def test_kernel_output_ndarray_fields_are_elementwise_immutable() -> None:
    """Frozen kernel results also prohibit element-level mutation of every array field."""
    geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([-0.25, 0.0]),),
    )
    phase = np.ones((1, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)
    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    _assert_ndarray_fields_are_elementwise_immutable(geometry)
    _assert_ndarray_fields_are_elementwise_immutable(statistics)


@pytest.mark.parametrize(
    "source_trial_index",
    (
        np.iinfo(np.int64).max + 1,
        np.iinfo(np.int64).min - 1,
        np.uint64(np.iinfo(np.int64).max + 1),
        np.iinfo(np.uint64).max,
    ),
)
def test_geometry_rejects_source_trial_identity_outside_int64_range(
    source_trial_index: object,
) -> None:
    """Oversized signed and unsigned Python/NumPy identities fail with ValueError."""
    with pytest.raises(ValueError):
        _build_geometry(
            source_trial_index=source_trial_index,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([0.0]),),
        )


def test_geometry_rejects_malformed_unit_spikes_and_segment_bounds() -> None:
    """Geometry requires unique matched unit axes, finite one-dimensional spikes, and contiguous halves."""
    common = dict(
        source_trial_index=0,
        phase_time_s=_FOUR_HZ_TIME_S,
        phase_sampling_rate_hz=4.0,
    )
    with pytest.raises(ValueError):
        _build_geometry(
            **common,
            unit_ids=("unit-1", "unit-1"),
            unit_trial_spike_times_s=(np.array([0.0]), np.array([0.25])),
        )
    with pytest.raises(ValueError):
        _build_geometry(
            **common,
            unit_ids=("unit-1", "unit-2"),
            unit_trial_spike_times_s=(np.array([0.0]),),
        )
    with pytest.raises(ValueError):
        _build_geometry(
            **common,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([np.nan]),),
        )
    with pytest.raises(ValueError):
        _build_geometry(
            **common,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([[0.0]]),),
        )
    for bounds in (
        ((-0.5, -0.1), (0.0, 0.5)),
        ((-0.5, 0.1), (0.0, 0.5)),
        ((-0.5, 0.0), (0.0, np.inf)),
        ((-0.5, 0.0), (0.0, -0.25)),
    ):
        with pytest.raises(ValueError):
            _build_geometry(
                **common,
                unit_ids=("unit-1",),
                unit_trial_spike_times_s=(np.array([0.0]),),
                segment_bounds_s=bounds,
            )


def test_segment_outer_upper_bounds_are_excluded_and_lower_bounds_are_included() -> None:
    """Each half is lower-inclusive/upper-exclusive, including the global outer upper bound."""
    geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([-0.5, 0.0, 0.5]),),
    )
    phase = np.ones((1, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    np.testing.assert_array_equal(geometry.group_offsets, [0, 1, 2])
    np.testing.assert_array_equal(geometry.left_index, [0, 2])
    np.testing.assert_array_equal(geometry.right_index, [0, 2])
    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[1], [1]]]])


def test_exact_grid_samples_use_only_their_coefficient_even_when_neighbors_are_invalid() -> None:
    """Exact first/interior/last samples do not inherit invalid adjacent coefficients."""
    time_s = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    geometry, phase, valid = _one_geometry_and_phase(
        spikes_s=np.array([0.0, 0.5, 1.0]),
        coefficients=np.ones(5, dtype=np.complex64),
        valid=np.array([True, False, True, False, True]),
        phase_time_s=time_s,
        segment_bounds_s=((-0.1, 0.0), (0.0, 1.1)),
    )

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[0], [3]]]])
    np.testing.assert_allclose(statistics.phase_vector_sum, [[[[0.0], [3.0]]]])


def test_nearby_spikes_remain_between_samples_under_strict_canonical_equality() -> None:
    """Adjacent floats and 1e-12-near values require both neighbors, never isclose."""
    time_s = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    exact = 0.5
    geometry, phase, valid = _one_geometry_and_phase(
        spikes_s=np.array(
            [
                exact,
                np.nextafter(exact, -np.inf),
                np.nextafter(exact, np.inf),
                exact + 1e-12,
            ],
            dtype=np.float64,
        ),
        coefficients=np.ones(3, dtype=np.complex64),
        valid=np.array([False, True, False]),
        phase_time_s=time_s,
        segment_bounds_s=((-0.1, 0.0), (0.0, 1.1)),
    )

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    np.testing.assert_array_equal(geometry.exact_sample, [True, False, False, False])
    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[0], [1]]]])
    np.testing.assert_allclose(statistics.phase_vector_sum, [[[[0.0], [1.0]]]])


def test_between_sample_interpolation_requires_two_valid_neighbors_per_frequency() -> None:
    """A between-grid spike is accepted independently at only fully valid frequencies."""
    time_s = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([0.25]),),
        phase_time_s=time_s,
        phase_sampling_rate_hz=2.0,
        segment_bounds_s=((-0.1, 0.0), (0.0, 1.1)),
    )
    phase = np.array(
        [[[1.0 + 0.0j, 1.0j, -1.0 + 0.0j], [1.0 + 0.0j, 1.0j, -1.0 + 0.0j]]],
        dtype=np.complex64,
    )
    valid = np.array([[[True, True, True], [True, False, True]]], dtype=bool)

    statistics = _reduce(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([0], dtype=np.int64),
        frequencies_hz=np.array([8.0, 40.0]),
        source_trial_index=np.array([0], dtype=np.int64),
        target_trial_index=np.array([0], dtype=np.int64),
    )

    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[0, 0], [1, 0]]]])
    expected = (1.0 + 1.0j) / np.sqrt(2.0)
    np.testing.assert_allclose(statistics.phase_vector_sum[0, 0, 1, 0], expected, atol=5e-8)
    assert statistics.phase_vector_sum[0, 0, 1, 1] == 0.0j


def test_interpolation_normalizes_in_complex128_then_casts_each_vector_to_complex64_before_sum() -> None:
    """Repeated non-axis-aligned samples preserve the required complex64 reduction order."""
    time_s = np.array([0.0, 0.25, 0.5], dtype=np.float64)
    spikes_s = np.array([0.125, 0.125, 0.375, 0.125, 0.375], dtype=np.float64)
    geometry, phase, valid = _one_geometry_and_phase(
        spikes_s=spikes_s,
        coefficients=np.array([1.0 + 2.0j, 3.0 - 0.5j, -2.0 + 1.0j]),
        valid=np.ones(3, dtype=bool),
        phase_time_s=time_s,
        segment_bounds_s=((-0.1, 0.0), (0.0, 0.6)),
    )

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)
    coefficients = phase[0, 0].astype(np.complex128)
    interpolation_weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    left = np.array([0, 0, 1, 0, 1], dtype=np.int64)
    right = np.array([1, 1, 2, 1, 2], dtype=np.int64)
    interpolated = (
        (1.0 - interpolation_weights) * coefficients[left]
        + interpolation_weights * coefficients[right]
    )
    normalized_complex64 = (interpolated / np.abs(interpolated)).astype(np.complex64)
    expected_sum = np.sum(normalized_complex64, dtype=np.complex128)

    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[0], [5]]]])
    np.testing.assert_allclose(
        statistics.phase_vector_sum[0, 0, 1, 0],
        expected_sum,
        rtol=0.0,
        atol=0.0,
    )


def test_non_midpoint_interpolation_retains_exact_geometry_weight_and_complex64_reduction_order() -> None:
    """A 0.25 right weight uses complex128 interpolation before one complex64 vector is summed."""
    time_s = np.array([0.0, 0.25, 0.5], dtype=np.float64)
    geometry, phase, valid = _one_geometry_and_phase(
        spikes_s=np.array([0.0625]),
        coefficients=np.array([1.0 + 2.0j, 3.0 - 0.5j, -2.0 + 1.0j]),
        valid=np.ones(3, dtype=bool),
        phase_time_s=time_s,
        segment_bounds_s=((-0.1, 0.0), (0.0, 0.6)),
    )

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)
    interpolated = 0.75 * np.complex128(1.0 + 2.0j) + 0.25 * np.complex128(3.0 - 0.5j)
    expected_vector = np.complex64(interpolated / np.abs(interpolated))

    np.testing.assert_array_equal(geometry.left_index, [0])
    np.testing.assert_array_equal(geometry.right_index, [1])
    np.testing.assert_array_equal(geometry.exact_sample, [False])
    np.testing.assert_array_equal(geometry.inside_support, [True])
    np.testing.assert_array_equal(geometry.right_weight, [0.25])
    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[0], [1]]]])
    np.testing.assert_allclose(
        statistics.phase_vector_sum[0, 0, 1, 0],
        np.complex128(expected_vector),
        rtol=0.0,
        atol=0.0,
    )


def test_between_sample_zero_coefficient_is_invalid_even_when_linear_interpolation_is_nonzero() -> None:
    """Between-sample validity rejects a zero endpoint rather than only a zero interpolated value."""
    time_s = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    geometry, phase, valid = _one_geometry_and_phase(
        spikes_s=np.array([0.25]),
        coefficients=np.array([0.0j, 2.0 + 1.0j, -1.0 + 2.0j]),
        valid=np.ones(3, dtype=bool),
        phase_time_s=time_s,
        segment_bounds_s=((-0.1, 0.0), (0.0, 1.1)),
    )
    interpolated = 0.5 * phase[0, 0, 0] + 0.5 * phase[0, 0, 1]

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    assert np.abs(interpolated) > 0.0
    np.testing.assert_array_equal(geometry.exact_sample, [False])
    np.testing.assert_array_equal(geometry.right_weight, [0.5])
    np.testing.assert_array_equal(statistics.valid_spike_count, np.zeros((1, 1, 2, 1), dtype=np.int64))
    np.testing.assert_array_equal(statistics.phase_vector_sum, np.zeros((1, 1, 2, 1)))


def test_mixed_empty_and_nonempty_source_trains_produce_no_phantom_segment_samples() -> None:
    """Empty unit/trial groups retain offsets and zero statistics without changing axes."""
    source_zero = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-a", "unit-b", "unit-c"),
        unit_trial_spike_times_s=(np.empty(0), np.array([0.0]), np.array([-0.25])),
    )
    source_one = _build_geometry(
        source_trial_index=1,
        unit_ids=("unit-a", "unit-b", "unit-c"),
        unit_trial_spike_times_s=(np.array([0.25]), np.empty(0), np.empty(0)),
    )
    phase = np.ones((2, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)

    statistics = _reduce(
        source_trial_geometries=(source_zero, source_one),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([0, 1], dtype=np.int64),
        frequencies_hz=np.array([8.0]),
        source_trial_index=np.array([0, 1], dtype=np.int64),
        target_trial_index=np.array([0, 1], dtype=np.int64),
    )

    np.testing.assert_array_equal(source_zero.group_offsets, [0, 0, 0, 0, 1, 2, 2])
    np.testing.assert_array_equal(source_one.group_offsets, [0, 0, 1, 1, 1, 1, 1])
    expected_counts = np.array(
        [
            [[[0], [0]], [[0], [1]], [[1], [0]]],
            [[[0], [1]], [[0], [0]], [[0], [0]]],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(statistics.valid_spike_count, expected_counts)
    np.testing.assert_array_equal(
        statistics.phase_vector_sum,
        expected_counts.astype(np.complex128),
    )
    assert statistics.phase_vector_sum.shape == (2, 3, 2, 1)
    assert statistics.phase_vector_sum.dtype == np.dtype(np.complex128)
    assert statistics.valid_spike_count.dtype == np.dtype(np.int64)


def test_outside_nonfinite_zero_and_zero_interpolation_samples_are_invalid() -> None:
    """Support and magnitude failures never contribute a normalized phase vector."""
    time_s = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    geometry, phase, valid = _one_geometry_and_phase(
        spikes_s=np.array([-0.1, 0.25, 1.0, 1.5, 1.6]),
        coefficients=np.array([1.0, -1.0, np.nan + 0.0j, 0.0j]),
        valid=np.ones(4, dtype=bool),
        phase_time_s=time_s,
        segment_bounds_s=((-0.2, 0.0), (0.0, 1.7)),
    )

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    np.testing.assert_array_equal(statistics.valid_spike_count, np.zeros((1, 1, 2, 1), dtype=np.int64))
    np.testing.assert_array_equal(statistics.phase_vector_sum, np.zeros((1, 1, 2, 1)))


@pytest.mark.parametrize(
    ("segment_bounds_s", "spikes_s", "expected_offsets"),
    (
        (((-1.0, 0.0), (0.0, 1.0)), np.array([-0.5, 0.0, 0.5]), np.array([0, 1, 3])),
        (((-1.0, 0.25), (0.25, 1.0)), np.array([0.0, 0.25, 0.5]), np.array([0, 1, 3])),
    ),
)
def test_shared_segment_boundary_is_assigned_only_to_after(
    segment_bounds_s: tuple[tuple[float, float], tuple[float, float]],
    spikes_s: np.ndarray,
    expected_offsets: np.ndarray,
) -> None:
    """Zero and nonzero shared boundaries use before/after half-open segment membership."""
    time_s = np.arange(-1.0, 1.25, 0.25, dtype=np.float64)
    geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(spikes_s,),
        phase_time_s=time_s,
        phase_sampling_rate_hz=4.0,
        segment_bounds_s=segment_bounds_s,
    )
    phase = np.ones((1, 1, time_s.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)

    statistics = _single_edge_statistics(geometry=geometry, phase=phase, valid=valid)

    np.testing.assert_array_equal(geometry.group_offsets, expected_offsets)
    np.testing.assert_array_equal(statistics.valid_spike_count, [[[[1], [2]]]])


def test_segmented_statistics_match_current_edge_oracle_and_preserve_physical_identities() -> None:
    """Search-once geometry reproduces the reference sums/counts after halves recombine."""
    time_s = _FOUR_HZ_TIME_S
    frequencies_hz = np.array([8.0, 40.0], dtype=np.float64)
    phase_by_position = np.array(
        [
            [[1.0, 1.0j, -1.0, -1.0j, 1.0], [1.0j, 1.0, -1.0j, -1.0, 1.0j]],
            [[-1.0, -1.0j, 1.0, 1.0j, -1.0], [1.0, -1.0, 1.0j, -1.0j, 1.0]],
            [[1.0j, -1.0, -1.0j, 1.0, 1.0j], [-1.0j, 1.0j, 1.0, -1.0, -1.0j]],
        ],
        dtype=np.complex64,
    )
    valid_by_position = np.ones(phase_by_position.shape, dtype=bool)
    valid_by_position[1, 1, 3] = False
    phase_trial_index = np.array([999, 101, 305], dtype=np.int64)
    # Phase rows are deliberately not sorted by stable trial-table identity.
    phases = phase_by_position[[2, 0, 1]]
    valid = valid_by_position[[2, 0, 1]]
    phase_for_oracle = np.where(
        valid_by_position,
        phase_by_position,
        0.0j,
    ).astype(np.complex64)
    unit_trial_spikes = (
        (
            np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]),
            np.array([-0.25, 0.125, 0.5]),
            np.array([-0.5, 0.0, 0.25]),
        ),
        (
            np.array([-0.125, 0.0, 0.25]),
            np.array([-0.5, -0.25, 0.5]),
            np.array([0.0, 0.25, 0.5]),
        ),
    )
    source_position_by_trial_id = {101: 0, 305: 1, 999: 2}
    geometries = tuple(
        _build_geometry(
            source_trial_index=source_trial_index,
            unit_ids=("unit-1", "unit-2"),
            unit_trial_spike_times_s=tuple(
                unit[source_position_by_trial_id[source_trial_index]]
                for unit in unit_trial_spikes
            ),
            segment_bounds_s=((-0.5, 0.0), (0.0, 0.75)),
        )
        for source_trial_index in (305, 999, 101)
    )
    source_edges = np.array([101, 305, 999, 101], dtype=np.int64)
    target_edges = np.array([305, 999, 101, 101], dtype=np.int64)
    source_positions = np.array([0, 1, 2, 0], dtype=np.int64)
    target_positions = np.array([1, 2, 0, 0], dtype=np.int64)

    segmented = _reduce(
        source_trial_geometries=geometries,
        trial_phase_vectors=phases,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        source_trial_index=source_edges,
        target_trial_index=target_edges,
    )
    oracle = spike_lfp_summary.compute_edge_sufficient_statistics(
        trial_relative_spike_times_s=unit_trial_spikes,
        phase_time_s=time_s,
        trial_phase_vectors=phase_for_oracle,
        frequencies_hz=frequencies_hz,
        source_trial_position=source_positions,
        target_trial_position=target_positions,
    )

    assert segmented.source_trial_index.dtype == np.dtype(np.int64)
    assert segmented.target_trial_index.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(segmented.source_trial_index, source_edges)
    np.testing.assert_array_equal(segmented.target_trial_index, target_edges)
    assert segmented.phase_vector_sum.dtype == np.dtype(np.complex128)
    assert segmented.valid_spike_count.dtype == np.dtype(np.int64)
    assert segmented.phase_vector_sum.shape == (4, 2, 2, 2)
    assert segmented.valid_spike_count.shape == (4, 2, 2, 2)
    np.testing.assert_array_equal(
        segmented.valid_spike_count.sum(axis=2), oracle.valid_spike_count
    )
    absolute_error = np.abs(
        segmented.phase_vector_sum.sum(axis=2) - oracle.phase_vector_sum
    )
    allowed_error = np.maximum(2e-7, 2e-7 * oracle.valid_spike_count)
    assert np.all(absolute_error <= allowed_error), (
        "Segmented phase sums exceed the per-cell approved tolerance: "
        f"maximum absolute error={absolute_error.max():.3e}, "
        f"maximum allowed error={allowed_error.max():.3e}, "
        f"maximum exceedance={(absolute_error - allowed_error).max():.3e}"
    )
    source_edges[:] = -1
    target_edges[:] = -1
    np.testing.assert_array_equal(segmented.source_trial_index, [101, 305, 999, 101])
    np.testing.assert_array_equal(segmented.target_trial_index, [305, 999, 101, 101])
    with pytest.raises(FrozenInstanceError):
        segmented.phase_vector_sum = np.zeros((4, 2, 2, 2), dtype=np.complex128)


def test_geometry_is_reused_without_neighbor_search_for_unique_target_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target-edge reduction reuses geometry without performing a neighbor search."""
    geometry = _build_geometry(
        source_trial_index=101,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([-0.25, 0.0, 0.25]),),
    )
    phase_trial_index = np.array(
        [101, 305, 999, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
        dtype=np.int64,
    )
    phase = np.ones((phase_trial_index.size, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)

    def search_forbidden(*args: object, **kwargs: object) -> object:
        """Fail if reduction repeats source-spike geometry neighbor search."""
        raise AssertionError("reduction must reuse precomputed source geometry")

    monkeypatch.setattr(kernel.np, "searchsorted", search_forbidden)

    _reduce(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=np.array([8.0]),
        source_trial_index=np.array([101], dtype=np.int64),
        target_trial_index=np.array([999], dtype=np.int64),
    )
    _reduce(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=np.array([8.0]),
        source_trial_index=np.full(11, 101, dtype=np.int64),
        target_trial_index=np.array(
            [305, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            dtype=np.int64,
        ),
    )


def test_segmented_output_is_invariant_to_edge_and_unit_block_partitions() -> None:
    """Concatenated bounded edge/unit calls equal one call with the full bounded block."""
    phases = np.array(
        [
            [[1.0, 1.0j, -1.0, -1.0j, 1.0], [1.0j, 1.0, -1.0j, -1.0, 1.0j]],
            [[-1.0, -1.0j, 1.0, 1.0j, -1.0], [1.0, -1.0, 1.0j, -1.0j, 1.0]],
        ],
        dtype=np.complex64,
    )
    valid = np.ones(phases.shape, dtype=bool)
    source_spikes = (
        (np.array([-0.25, 0.0, 0.25]), np.array([-0.5, 0.5])),
        (np.array([-0.5, -0.25, 0.25]), np.array([0.0, 0.25, 0.5])),
    )
    full_geometries = tuple(
        _build_geometry(
            source_trial_index=trial_index,
            unit_ids=("unit-1", "unit-2"),
            unit_trial_spike_times_s=tuple(unit[trial_index] for unit in source_spikes),
        )
        for trial_index in range(2)
    )
    source_edges = np.array([0, 1, 0, 1], dtype=np.int64)
    target_edges = np.array([1, 0, 0, 1], dtype=np.int64)
    keyword_arguments = dict(
        trial_phase_vectors=phases,
        phase_valid_mask=valid,
        phase_trial_index=np.array([0, 1], dtype=np.int64),
        frequencies_hz=np.array([8.0, 40.0]),
    )
    full = _reduce(
        source_trial_geometries=full_geometries,
        source_trial_index=source_edges,
        target_trial_index=target_edges,
        **keyword_arguments,
    )
    edge_chunks = tuple(
        _reduce(
            source_trial_geometries=full_geometries,
            source_trial_index=source_edges[start:stop],
            target_trial_index=target_edges[start:stop],
            **keyword_arguments,
        )
        for start, stop in ((0, 1), (1, 3), (3, 4))
    )
    edge_chunk_sums = np.concatenate(
        [chunk.phase_vector_sum for chunk in edge_chunks], axis=0
    )
    edge_chunk_counts = np.concatenate(
        [chunk.valid_spike_count for chunk in edge_chunks], axis=0
    )
    unit_chunks = []
    for unit_index, unit_id in enumerate(("unit-1", "unit-2")):
        geometries = tuple(
            _build_geometry(
                source_trial_index=trial_index,
                unit_ids=(unit_id,),
                unit_trial_spike_times_s=(source_spikes[unit_index][trial_index],),
            )
            for trial_index in range(2)
        )
        unit_chunks.append(
            _reduce(
                source_trial_geometries=geometries,
                source_trial_index=source_edges,
                target_trial_index=target_edges,
                **keyword_arguments,
            )
        )

    np.testing.assert_allclose(edge_chunk_sums, full.phase_vector_sum, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(edge_chunk_counts, full.valid_spike_count)
    np.testing.assert_allclose(
        np.concatenate([chunk.phase_vector_sum for chunk in unit_chunks], axis=1),
        full.phase_vector_sum,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk.valid_spike_count for chunk in unit_chunks], axis=1),
        full.valid_spike_count,
    )


def test_segmented_reducer_rejects_malformed_phase_axes_and_edge_identities() -> None:
    """The kernel accepts only complex64 phase, Boolean validity, and unique stable edges."""
    geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([0.0]),),
    )
    phase = np.ones((1, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)
    arguments = dict(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([0], dtype=np.int64),
        frequencies_hz=np.array([8.0]),
        source_trial_index=np.array([0], dtype=np.int64),
        target_trial_index=np.array([0], dtype=np.int64),
    )

    with pytest.raises(ValueError):
        _reduce(**(arguments | {"trial_phase_vectors": phase.astype(np.complex128)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"phase_valid_mask": valid.astype(np.int8)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"phase_valid_mask": valid[:, :, :-1]}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"frequencies_hz": np.array([8], dtype=np.int64)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"frequencies_hz": np.array([[8.0]])}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"frequencies_hz": np.array([8.0, 40.0])}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"source_trial_index": np.array([False])}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"target_trial_index": np.array([1], dtype=np.int64)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"source_trial_index": np.array([1], dtype=np.int64)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"phase_trial_index": np.array([0, 1], dtype=np.int64)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"phase_trial_index": np.array([-1], dtype=np.int64)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"phase_trial_index": np.array([0], dtype=np.int32)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"source_trial_index": np.array([0], dtype=np.int32)}))
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"target_trial_index": np.array([0.0])}))
    two_trial_phase = np.ones((2, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    two_trial_valid = np.ones(two_trial_phase.shape, dtype=bool)
    two_frequency_phase = np.ones((1, 2, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    two_frequency_valid = np.ones(two_frequency_phase.shape, dtype=bool)
    with pytest.raises(ValueError):
        _reduce(
            **(
                arguments
                | {
                    "trial_phase_vectors": two_frequency_phase,
                    "phase_valid_mask": two_frequency_valid,
                    "frequencies_hz": np.array([40.0, 8.0]),
                }
            )
        )
    with pytest.raises(ValueError):
        _reduce(
            **(
                arguments
                | {
                    "trial_phase_vectors": two_frequency_phase,
                    "phase_valid_mask": two_frequency_valid,
                    "frequencies_hz": np.array([0.0, 8.0]),
                }
            )
        )
    with pytest.raises(ValueError):
        _reduce(
            **(
                arguments
                | {
                    "trial_phase_vectors": two_trial_phase,
                    "phase_valid_mask": two_trial_valid,
                    "phase_trial_index": np.array([0, 0], dtype=np.int64),
                }
            )
        )
    with pytest.raises(ValueError):
        _reduce(**(arguments | {"source_trial_geometries": (geometry, geometry)}))
    with pytest.raises(ValueError):
        _reduce(
            **(
                arguments
                | {
                    "trial_phase_vectors": two_trial_phase,
                    "phase_valid_mask": two_trial_valid,
                    "phase_trial_index": np.array([0, 1], dtype=np.int64),
                    "source_trial_index": np.array([1], dtype=np.int64),
                }
            )
        )
    reordered_geometry = _build_geometry(
        source_trial_index=1,
        unit_ids=("unit-b", "unit-a"),
        unit_trial_spike_times_s=(np.array([0.0]), np.array([0.0])),
    )
    ordered_geometry = _build_geometry(
        source_trial_index=0,
        unit_ids=("unit-a", "unit-b"),
        unit_trial_spike_times_s=(np.array([0.0]), np.array([0.0])),
    )
    with pytest.raises(ValueError):
        _reduce(
            source_trial_geometries=(ordered_geometry, reordered_geometry),
            trial_phase_vectors=two_trial_phase,
            phase_valid_mask=two_trial_valid,
            phase_trial_index=np.array([0, 1], dtype=np.int64),
            frequencies_hz=np.array([8.0]),
            source_trial_index=np.array([0], dtype=np.int64),
            target_trial_index=np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError):
        _reduce(
            **(
                arguments
                | {
                    "source_trial_index": np.array([0, 0], dtype=np.int64),
                    "target_trial_index": np.array([0, 0], dtype=np.int64),
                }
            )
        )


def test_kernel_allocation_estimate_matches_named_hand_accounting_without_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pure estimator reports each documented concurrently live kernel category.

    Geometry retains two int64 indices, one float64 weight, and two Boolean
    masks per spike, plus one owned int64 source identity and one
    ``unit * 2 + 1`` int64 offset array per source trial.  Each gathered
    coefficient cell concurrently retains left/right
    complex64 coefficients, complex128 interpolation, float64 magnitude,
    complex64 normalized phase, and three Boolean support/validity arrays:
    ``8 + 8 + 16 + 8 + 8 + 3 == 51`` bytes.  Segmented output has two segments
    of one complex128 sum and one int64 count, plus owned source/target int64
    edge identities: ``48 * edge * unit * frequency + 16 * edge`` bytes.
    """
    source_trial_spike_count = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )
    edge_source_trial_position = np.array([0, 1, 1], dtype=np.int64)
    total_spikes = int(source_trial_spike_count.sum())
    source_identity_bytes = source_trial_spike_count.shape[0] * 8
    geometry_bytes = 26 * total_spikes + 2 * (2 * 2 + 1) * 8 + source_identity_bytes
    selected_spikes = int(
        source_trial_spike_count[edge_source_trial_position].sum()
    )
    gathered_cells = selected_spikes * 2
    gather_temporary_bytes = gathered_cells * (8 + 8 + 16 + 8 + 8 + 3)
    edge_identity_bytes = edge_source_trial_position.size * 2 * 8
    segmented_edge_statistics_bytes = 3 * 2 * 2 * 48 + edge_identity_bytes
    expected_peak = (
        geometry_bytes + gather_temporary_bytes + segmented_edge_statistics_bytes
    )

    def allocation_forbidden(*args: object, **kwargs: object) -> object:
        """Fail if the pure estimator constructs a NumPy work or sampling array."""
        raise AssertionError("allocation estimation must not allocate or sample phase")

    for name in ("empty", "zeros", "ones", "full", "searchsorted"):
        monkeypatch.setattr(kernel.np, name, allocation_forbidden)
    estimate = kernel.estimate_segmented_kernel_allocation(
        source_trial_spike_count=source_trial_spike_count,
        edge_source_trial_position=edge_source_trial_position,
        frequency_count=2,
    )

    assert isinstance(estimate, kernel.KernelAllocationEstimate)
    assert source_identity_bytes == 16
    assert edge_identity_bytes == 48
    assert estimate.geometry_bytes == geometry_bytes
    assert estimate.segmented_edge_statistics_bytes == segmented_edge_statistics_bytes
    assert estimate.gather_temporary_bytes == gather_temporary_bytes
    assert estimate.planned_kernel_peak_bytes == expected_peak
    assert all(
        isinstance(getattr(estimate, field.name), int) and getattr(estimate, field.name) >= 0
        for field in fields(estimate)
    )
    with pytest.raises(FrozenInstanceError):
        estimate.geometry_bytes = 0


def test_kernel_allocation_estimate_accepts_zero_spike_counts_and_charges_offsets_identities_and_edge_statistics() -> None:
    """An all-zero bounded block still retains source/edge identities, offsets, and outputs."""
    source_trial_spike_count = np.zeros((2, 2, 2), dtype=np.int64)
    edge_source_trial_position = np.array([0, 1, 1], dtype=np.int64)

    estimate = kernel.estimate_segmented_kernel_allocation(
        source_trial_spike_count=source_trial_spike_count,
        edge_source_trial_position=edge_source_trial_position,
        frequency_count=1,
    )

    expected_geometry_bytes = 2 * (2 * 2 + 1) * 8 + 2 * 8
    expected_edge_statistics_bytes = 3 * 2 * 1 * 48 + 3 * 2 * 8
    assert estimate.geometry_bytes == expected_geometry_bytes
    assert estimate.segmented_edge_statistics_bytes == expected_edge_statistics_bytes
    assert estimate.gather_temporary_bytes == 0
    assert estimate.planned_kernel_peak_bytes == (
        expected_geometry_bytes + expected_edge_statistics_bytes
    )


@pytest.mark.parametrize(
    ("source_trial_spike_count", "edge_source_trial_position", "frequency_count"),
    (
        (np.zeros((2, 2), dtype=np.int64), np.array([0]), 1),
        (np.zeros((2, 1, 3), dtype=np.int64), np.array([0]), 1),
        (np.array([[[1.5, 0]]]), np.array([0]), 1),
        (np.array([[[True, False]]]), np.array([0]), 1),
        (np.array([[[-1, 0]]]), np.array([0]), 1),
        (np.zeros((1, 1, 2), dtype=np.int64), np.array([[0]]), 1),
        (np.zeros((1, 1, 2), dtype=np.int64), np.array([True]), 1),
        (np.zeros((1, 1, 2), dtype=np.int64), np.array([1]), 1),
        (np.zeros((1, 1, 2), dtype=np.int64), np.array([0]), True),
        (np.zeros((1, 1, 2), dtype=np.int64), np.array([0]), 0),
    ),
)
def test_kernel_allocation_estimate_rejects_malformed_counts_axes_and_integer_arguments(
    source_trial_spike_count: np.ndarray,
    edge_source_trial_position: np.ndarray,
    frequency_count: object,
) -> None:
    """Malformed count axes, Boolean integer arguments, and bad edge positions fail early."""
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=source_trial_spike_count,
            edge_source_trial_position=edge_source_trial_position,
            frequency_count=frequency_count,
        )


def test_kernel_allocation_estimate_rejects_geometry_byte_multiplication_overflow() -> None:
    """Geometry byte accounting fails instead of wrapping a valid large spike count."""
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=np.array(
                [[[np.iinfo(np.int64).max, 0]]], dtype=np.int64
            ),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=1,
        )


def test_kernel_allocation_estimate_rejects_total_spike_count_addition_overflow() -> None:
    """Summing before/after source counts is checked before any byte multiplication."""
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=np.array(
                [[[np.iinfo(np.int64).max, 1]]], dtype=np.int64
            ),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=1,
        )


def test_kernel_allocation_estimate_rejects_gathered_cell_multiplication_overflow() -> None:
    """A safe geometry still rejects selected-spike times 51 gathered-cell overflow."""
    maximum = np.iinfo(np.int64).max
    spike_count = maximum // 51 + 1
    geometry_bytes = 26 * spike_count + 3 * 8 + 8
    segmented_edge_statistics_bytes = 48 + 2 * 8
    assert geometry_bytes <= maximum
    assert segmented_edge_statistics_bytes <= maximum
    assert spike_count * 51 > maximum
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=np.array([[[spike_count, 0]]], dtype=np.int64),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=1,
        )


def test_kernel_allocation_estimate_rejects_planned_peak_addition_overflow() -> None:
    """Individually representable geometry, gather, and edge statistics cannot wrap their peak sum."""
    maximum = np.iinfo(np.int64).max
    spike_count = maximum // 60
    geometry_bytes = 26 * spike_count + 3 * 8 + 8
    gather_temporary_bytes = 51 * spike_count
    segmented_edge_statistics_bytes = 48 + 2 * 8
    assert geometry_bytes <= maximum
    assert gather_temporary_bytes <= maximum
    assert segmented_edge_statistics_bytes <= maximum
    assert geometry_bytes + gather_temporary_bytes + segmented_edge_statistics_bytes > maximum
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=np.array([[[spike_count, 0]]], dtype=np.int64),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=1,
        )


def test_kernel_allocation_estimate_rejects_selected_spike_count_addition_overflow() -> None:
    """Repeated source edges check selected-spike addition before gather-byte multiplication."""
    maximum = np.iinfo(np.int64).max
    spike_count = (maximum - 3 * 8 - 8) // 26
    repeated_edges = np.zeros(27, dtype=np.int64)
    geometry_bytes = 26 * spike_count + 3 * 8 + 8
    segmented_edge_statistics_bytes = repeated_edges.size * (48 + 2 * 8)
    assert geometry_bytes <= maximum
    assert segmented_edge_statistics_bytes <= maximum
    assert spike_count * repeated_edges.size > maximum
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=np.array([[[spike_count, 0]]], dtype=np.int64),
            edge_source_trial_position=repeated_edges,
            frequency_count=1,
        )


def test_kernel_allocation_estimate_rejects_edge_statistics_multiplication_overflow() -> None:
    """Zero spikes do not hide overflow in edge-by-unit-by-frequency output statistics."""
    with pytest.raises(ValueError):
        kernel.estimate_segmented_kernel_allocation(
            source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=np.iinfo(np.int64).max,
        )


# S2 contracts: observed single-pass reuse, whole-from-halves composition, and
# representative histograms.  The S1 geometry/edge contracts above remain
# unchanged; these are deliberately separate result types.
_REPRESENTATIVE_PHASE_BIN_EDGES_RAD = np.array(
    [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
    dtype=np.float64,
)


def _observed_segmented_fixture() -> tuple[
    tuple[object, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return two same-trial sources with cancelling before/after phase sums.

    The phase arrays have axes ``(trial=2, frequency=3, time=5)`` and use
    stable, intentionally unsorted physical trial identities. Each source has
    one unit and one exact-grid spike in each half. Before vectors are ``+1``
    and after vectors are ``-1`` at every frequency, so a whole-window PPC
    must retain cross-half cancellation instead of averaging the two half PPCs.
    """
    phase_trial_index = np.array([29, 7], dtype=np.int64)
    frequencies_hz = np.array([8.0, 20.0, 40.0], dtype=np.float64)
    phase = np.ones((2, 3, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    phase[:, :, 1] = np.complex64(1.0 + 0.0j)
    phase[:, :, 3] = np.complex64(-1.0 + 0.0j)
    valid = np.ones(phase.shape, dtype=bool)
    geometry_by_source = {
        29: _build_geometry(
            source_trial_index=29,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
        ),
        7: _build_geometry(
            source_trial_index=7,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
        ),
    }
    # Geometry order is unrelated to the prepared phase-axis order.
    geometries = (geometry_by_source[7], geometry_by_source[29])
    return (
        geometries,
        phase,
        valid,
        phase_trial_index,
        frequencies_hz,
        _REPRESENTATIVE_PHASE_BIN_EDGES_RAD.copy(),
    )


def _compute_observed_segmented_fixture() -> object:
    """Run the proposed S2 observed reducer on the canonical two-trial fixture."""
    geometries, phase, valid, phase_trial_index, frequencies_hz, phase_bins = (
        _observed_segmented_fixture()
    )
    return kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=phase_bins,
    )


def _observed_trial_reuse_fixture() -> tuple[
    tuple[object, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return three nonmonotonic stable trials for overlapping-condition reuse.

    The three trials have distinct before/after spike counts and distinct phase
    values. Geometry order differs from phase order so the frozen per-trial
    output must bind each stable ID to its own data rather than tuple position.
    """
    phase_trial_index = np.array([29, 7, 61], dtype=np.int64)
    frequencies_hz = np.array([8.0, 40.0], dtype=np.float64)
    phase = np.ones((3, 2, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    # These 3-4-5 vectors normalize exactly to non-boundary complex64 phases.
    phase[0, :, 1] = [3.0 + 4.0j, -4.0 + 3.0j]
    phase[0, :, 3] = [-3.0 - 4.0j, 4.0 - 3.0j]
    phase[1, :, 1] = [4.0 - 3.0j, -3.0 - 4.0j]
    phase[1, :, 3] = [3.0 + 4.0j, -4.0 + 3.0j]
    phase[2, :, 1] = [-4.0 + 3.0j, 3.0 + 4.0j]
    phase[2, :, 3] = [4.0 - 3.0j, -3.0 - 4.0j]
    spike_times_by_source = {
        29: np.array([-0.25, 0.25]),
        7: np.array([-0.25, -0.25, 0.25]),
        61: np.array([-0.25, 0.25, 0.25, 0.25]),
    }
    geometry_by_source = {
        source_id: _build_geometry(
            source_trial_index=source_id,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(spike_times_by_source[source_id],),
        )
        for source_id in phase_trial_index
    }
    return (
        (
            geometry_by_source[7],
            geometry_by_source[61],
            geometry_by_source[29],
        ),
        phase,
        np.ones(phase.shape, dtype=bool),
        phase_trial_index,
        frequencies_hz,
        _REPRESENTATIVE_PHASE_BIN_EDGES_RAD.copy(),
    )


def test_observed_trial_statistics_contracts_expose_stable_trial_axes_and_membership_reducer() -> None:
    """S2 first retains each physical trial before overlapping condition pooling.

    The stable trial axis is int64 and may be nonmonotonic.  Per-trial sums and
    counts use ``(trial, unit, segment=2, frequency)`` axes.  The two int64
    representative-frequency positions bind histogram bands to those counts;
    histograms use ``(trial, unit, segment=2, band=2, phase_bin)`` axes.  A
    histogram total cannot exceed its selected-frequency valid count, although
    legacy complex64 angle rounding can place nominal +/-pi outside float64 bin
    edges. Aggregation receives an explicit ordered unique stable-ID membership
    vector.
    """
    module_doc = (kernel.__doc__ or "").lower()
    assert "segmented ppc" in module_doc
    assert "s1 kernel only" not in module_doc
    assert tuple(
        field.name for field in fields(kernel.ObservedTrialSegmentedPPCStatistics)
    ) == (
        "phase_trial_index",
        "phase_vector_sum",
        "valid_spike_count",
        "representative_frequency_index",
        "representative_frequency_hz",
        "phase_bin_edges_rad",
        "representative_phase_histogram_count",
    )
    trial_signature = inspect.signature(
        kernel.compute_observed_trial_segmented_ppc_statistics
    )
    assert tuple(trial_signature.parameters) == (
        "source_trial_geometries",
        "trial_phase_vectors",
        "phase_valid_mask",
        "phase_trial_index",
        "frequencies_hz",
        "phase_bin_edges_rad",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in trial_signature.parameters.values()
    )
    selected_trial_signature = inspect.signature(
        kernel.compute_selected_observed_trial_segmented_ppc_statistics
    )
    assert tuple(selected_trial_signature.parameters) == (
        "source_trial_geometries",
        "trial_phase_vectors",
        "phase_valid_mask",
        "phase_trial_index",
        "frequencies_hz",
        "phase_bin_edges_rad",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in selected_trial_signature.parameters.values()
    )
    aggregate_signature = inspect.signature(
        kernel.aggregate_observed_trial_segmented_ppc_statistics
    )
    assert tuple(aggregate_signature.parameters) == (
        "observed_trial_statistics",
        "membership_trial_index",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in aggregate_signature.parameters.values()
    )


@pytest.mark.parametrize(
    "source_ids",
    (
        (7, 7),
        (7, 999),
    ),
)
def test_selected_observed_trial_reducer_rejects_duplicate_or_unknown_geometry_rows(
    source_ids: tuple[int, ...],
) -> None:
    """Selection-aware S2 derives selected rows from unique known geometry IDs.

    The new entry point is the no-copy bridge for grouped execution: phase and
    validity retain their complete ``(trial, frequency, time)`` axes, while
    geometries and returned statistics cover only the geometry-tuple rows.
    The existing all-trial reducer remains unchanged for legacy callers.
    """
    geometries, phase, valid, phase_trial_index, frequencies_hz, phase_bins = (
        _observed_trial_reuse_fixture()
    )
    by_source = {int(geometry.source_trial_index): geometry for geometry in geometries}
    unknown_geometry = _build_geometry(
        source_trial_index=999,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
    )
    selected_geometries = tuple(
        unknown_geometry if source_id == 999 else by_source[source_id]
        for source_id in source_ids
    )
    with pytest.raises(ValueError, match="duplicate|unknown|geometry|source|trial"):
        kernel.compute_selected_observed_trial_segmented_ppc_statistics(
            source_trial_geometries=selected_geometries,
            trial_phase_vectors=phase,
            phase_valid_mask=valid,
            phase_trial_index=phase_trial_index,
            frequencies_hz=frequencies_hz,
            phase_bin_edges_rad=phase_bins,
        )


def test_selected_observed_trial_reducer_matches_selected_legacy_trial_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2 selected rows preserve order without materializing phase/valid gathers."""
    geometries, phase, valid, phase_trial_index, frequencies_hz, phase_bins = (
        _observed_trial_reuse_fixture()
    )
    by_source = {int(geometry.source_trial_index): geometry for geometry in geometries}
    selected_geometries = (by_source[61], by_source[29])
    positions = np.array([2, 0], dtype=np.int64)
    expected = kernel.compute_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=selected_geometries,
        trial_phase_vectors=phase[positions],
        phase_valid_mask=valid[positions],
        phase_trial_index=phase_trial_index[positions],
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=phase_bins,
    )
    original_sampler = kernel._sample_normalized_spike_group
    sampler_calls = 0

    def recording_sampler(**kwargs: object) -> object:
        """Selection-aware sampling must index the original full phase arrays."""
        nonlocal sampler_calls
        coefficients = np.asarray(kwargs["coefficients"])
        explicit_valid = np.asarray(kwargs["explicit_valid_mask"])
        assert np.shares_memory(coefficients, phase)
        assert np.shares_memory(explicit_valid, valid)
        sampler_calls += 1
        return original_sampler(**kwargs)

    monkeypatch.setattr(kernel, "_sample_normalized_spike_group", recording_sampler)
    actual = kernel.compute_selected_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=selected_geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=phase_bins,
    )
    assert sampler_calls > 0
    np.testing.assert_array_equal(actual.phase_trial_index, np.array([61, 29], dtype=np.int64))
    for field in fields(expected):
        np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))


def test_one_observed_trial_pass_serves_overlapping_condition_memberships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two overlapping conditions aggregate one physical sampled-trial result.

    Each nonempty source-trial/unit/segment group is sampled once for all
    conditions.  The aggregation reducer performs no phase sampling, preserves
    supplied nonmonotonic stable membership order, and computes whole trial
    contributors as a before/after union.
    """
    geometries, phase, valid, phase_trial_index, frequencies_hz, phase_bins = (
        _observed_trial_reuse_fixture()
    )
    original_sampler = kernel._sample_normalized_spike_group
    sampled_group_sizes: list[int] = []

    def counted_sampler(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Record each physical unit/segment group normalized by the shared pass."""
        sampled_group_sizes.append(np.asarray(kwargs["left_index"]).size)
        return original_sampler(**kwargs)

    monkeypatch.setattr(kernel, "_sample_normalized_spike_group", counted_sampler)
    observed_trials = kernel.compute_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=phase_bins,
    )

    assert isinstance(observed_trials, kernel.ObservedTrialSegmentedPPCStatistics)
    assert observed_trials.phase_trial_index.dtype == np.dtype(np.int64)
    assert observed_trials.phase_vector_sum.dtype == np.dtype(np.complex128)
    assert observed_trials.phase_vector_sum.shape == (3, 1, 2, 2)
    assert observed_trials.valid_spike_count.dtype == np.dtype(np.int64)
    assert observed_trials.valid_spike_count.shape == (3, 1, 2, 2)
    assert observed_trials.representative_frequency_index.dtype == np.dtype(np.int64)
    assert observed_trials.representative_frequency_index.shape == (2,)
    assert observed_trials.representative_phase_histogram_count.dtype == np.dtype(np.int64)
    assert observed_trials.representative_phase_histogram_count.shape == (3, 1, 2, 2, 4)
    np.testing.assert_array_equal(observed_trials.phase_trial_index, [29, 7, 61])
    np.testing.assert_array_equal(observed_trials.representative_frequency_index, [0, 1])
    np.testing.assert_array_equal(observed_trials.representative_frequency_hz, [8.0, 40.0])
    normalized_a = np.complex64(0.6 + 0.8j)
    normalized_b = np.complex64(-0.8 + 0.6j)
    normalized_c = np.complex64(-0.6 - 0.8j)
    normalized_d = np.complex64(0.8 - 0.6j)
    normalized_a_sum = np.complex128(normalized_a)
    normalized_b_sum = np.complex128(normalized_b)
    normalized_c_sum = np.complex128(normalized_c)
    normalized_d_sum = np.complex128(normalized_d)
    expected_sums = np.array(
        [
            [
                [
                    [normalized_a_sum, normalized_b_sum],
                    [normalized_c_sum, normalized_d_sum],
                ]
            ],
            [
                [
                    [2 * normalized_d_sum, 2 * normalized_c_sum],
                    [normalized_a_sum, normalized_b_sum],
                ]
            ],
            [
                [
                    [normalized_b_sum, normalized_a_sum],
                    [3 * normalized_d_sum, 3 * normalized_c_sum],
                ]
            ],
        ],
        dtype=np.complex128,
    )
    expected_counts = np.array(
        [
            [[[1, 1], [1, 1]]],
            [[[2, 2], [1, 1]]],
            [[[1, 1], [3, 3]]],
        ],
        dtype=np.int64,
    )
    expected_histogram = np.zeros((3, 1, 2, 2, 4), dtype=np.int64)
    expected_histogram[0, 0, 0, 0, 2] = 1
    expected_histogram[0, 0, 0, 1, 3] = 1
    expected_histogram[0, 0, 1, 0, 0] = 1
    expected_histogram[0, 0, 1, 1, 1] = 1
    expected_histogram[1, 0, 0, 0, 1] = 2
    expected_histogram[1, 0, 0, 1, 0] = 2
    expected_histogram[1, 0, 1, 0, 2] = 1
    expected_histogram[1, 0, 1, 1, 3] = 1
    expected_histogram[2, 0, 0, 0, 3] = 1
    expected_histogram[2, 0, 0, 1, 2] = 1
    expected_histogram[2, 0, 1, 0, 1] = 3
    expected_histogram[2, 0, 1, 1, 0] = 3
    np.testing.assert_allclose(
        observed_trials.phase_vector_sum,
        expected_sums,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(observed_trials.valid_spike_count, expected_counts)
    np.testing.assert_array_equal(
        observed_trials.representative_phase_histogram_count,
        expected_histogram,
    )
    assert len(sampled_group_sizes) == 6
    assert sorted(sampled_group_sizes) == [1, 1, 1, 1, 2, 3]

    first_membership = np.array([61, 7], dtype=np.int64)
    second_membership = np.array([29, 7], dtype=np.int64)
    first = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=first_membership,
    )
    second = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=second_membership,
    )
    assert isinstance(first, kernel.ObservedSegmentedPPCStatistics)
    assert isinstance(second, kernel.ObservedSegmentedPPCStatistics)
    # Stable membership order is explicit and affects floating reduction order.
    first_positions = np.array([2, 1], dtype=np.int64)
    second_positions = np.array([0, 1], dtype=np.int64)
    np.testing.assert_allclose(
        first.phase_vector_sum,
        np.sum(
            observed_trials.phase_vector_sum[first_positions],
            axis=0,
            dtype=np.complex128,
        ),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        second.phase_vector_sum,
        np.sum(
            observed_trials.phase_vector_sum[second_positions],
            axis=0,
            dtype=np.complex128,
        ),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(first.valid_spike_count, [[[3, 3], [4, 4]]])
    np.testing.assert_array_equal(second.valid_spike_count, [[[3, 3], [2, 2]]])
    np.testing.assert_array_equal(first.contributing_trial_count, 2)
    np.testing.assert_array_equal(second.contributing_trial_count, 2)
    np.testing.assert_array_equal(
        first.representative_phase_histogram_count[:, 2],
        first.representative_phase_histogram_count[:, 0]
        + first.representative_phase_histogram_count[:, 1],
    )
    expected_half_histogram = np.sum(
        observed_trials.representative_phase_histogram_count[first_positions],
        axis=0,
        dtype=np.int64,
    )
    expected_histogram = np.empty((1, 3, 2, 4), dtype=np.int64)
    expected_histogram[:, :2] = expected_half_histogram
    expected_histogram[:, 2] = (
        expected_half_histogram[:, 0] + expected_half_histogram[:, 1]
    )
    np.testing.assert_array_equal(
        first.representative_phase_histogram_count,
        expected_histogram,
    )
    # The physical trial with stable ID 7 is shared, but it is not resampled.
    assert len(sampled_group_sizes) == 6
    assert sorted(sampled_group_sizes) == [1, 1, 1, 1, 2, 3]


def _valid_observed_trial_segmented_statistics_arguments() -> dict[str, np.ndarray]:
    """Return coherent public-construction arrays for per-trial S2 statistics."""
    histogram = np.zeros((2, 1, 2, 2, 2), dtype=np.int64)
    histogram[..., 0] = 1
    return {
        "phase_trial_index": np.array([29, 7], dtype=np.int64),
        "phase_vector_sum": np.ones((2, 1, 2, 2), dtype=np.complex128),
        "valid_spike_count": np.ones((2, 1, 2, 2), dtype=np.int64),
        "representative_frequency_index": np.array([0, 1], dtype=np.int64),
        "representative_frequency_hz": np.array([8.0, 40.0], dtype=np.float64),
        "phase_bin_edges_rad": np.array([-np.pi, 0.0, np.pi], dtype=np.float64),
        "representative_phase_histogram_count": histogram,
    }


def test_observed_trial_statistics_public_contract_owns_arrays_and_rejects_invalid_memberships() -> None:
    """Per-trial S2 arrays are immutable owned values before condition pooling."""
    arguments = _valid_observed_trial_segmented_statistics_arguments()
    expected = {name: values.copy() for name, values in arguments.items()}
    observed_trials = kernel.ObservedTrialSegmentedPPCStatistics(**arguments)
    for field in fields(observed_trials):
        input_values = arguments[field.name]
        actual = getattr(observed_trials, field.name)
        assert actual is not input_values
        assert not np.shares_memory(actual, input_values)
    for values in arguments.values():
        values[...] = 0
    for field in fields(observed_trials):
        np.testing.assert_array_equal(getattr(observed_trials, field.name), expected[field.name])
    _assert_result_arrays_are_owned_and_immutable(observed_trials)
    with pytest.raises(FrozenInstanceError):
        observed_trials.phase_trial_index = np.array([0], dtype=np.int64)

    for membership_trial_index in (
        np.array([29, 7], dtype=np.int32),
        np.array([29, 29], dtype=np.int64),
        np.array([29, 123], dtype=np.int64),
        np.array([[29, 7]], dtype=np.int64),
        np.array([-1], dtype=np.int64),
    ):
        with pytest.raises(ValueError):
            kernel.aggregate_observed_trial_segmented_ppc_statistics(
                observed_trial_statistics=observed_trials,
                membership_trial_index=membership_trial_index,
            )

    empty = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=np.empty(0, dtype=np.int64),
    )
    assert empty.phase_vector_sum.shape == (1, 2, 2)
    assert empty.valid_spike_count.shape == (1, 2, 2)
    assert empty.contributing_trial_count.shape == (1, 3, 2)
    assert empty.representative_phase_histogram_count.shape == (1, 3, 2, 2)
    np.testing.assert_array_equal(empty.phase_vector_sum, 0.0j)
    np.testing.assert_array_equal(empty.valid_spike_count, 0)
    np.testing.assert_array_equal(empty.contributing_trial_count, 0)
    np.testing.assert_array_equal(empty.representative_phase_histogram_count, 0)
    np.testing.assert_array_equal(empty.representative_frequency_hz, [8.0, 40.0])
    np.testing.assert_allclose(
        empty.phase_bin_edges_rad,
        [-np.pi, 0.0, np.pi],
        rtol=0.0,
        atol=0.0,
    )


def test_membership_aggregation_counts_whole_contributors_by_per_trial_union() -> None:
    """Per-trial before contributors {29, 7} and after {7, 61} union to three."""
    arguments = _valid_observed_trial_segmented_statistics_arguments()
    arguments["phase_trial_index"] = np.array([29, 7, 61], dtype=np.int64)
    counts = np.array(
        [
            [[[1, 1], [0, 0]]],
            [[[1, 1], [1, 1]]],
            [[[0, 0], [1, 1]]],
        ],
        dtype=np.int64,
    )
    sums = np.zeros(counts.shape, dtype=np.complex128)
    sums[counts > 0] = 1.0 + 0.0j
    arguments["phase_vector_sum"] = sums
    arguments["valid_spike_count"] = counts
    histogram = np.zeros((3, 1, 2, 2, 2), dtype=np.int64)
    histogram[:, :, :, 0, 0] = counts[:, :, :, 0]
    histogram[:, :, :, 1, 0] = counts[:, :, :, 1]
    arguments["representative_phase_histogram_count"] = histogram
    observed_trials = kernel.ObservedTrialSegmentedPPCStatistics(**arguments)
    pooled = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=np.array([29, 7, 61], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        pooled.contributing_trial_count,
        np.array([[[2, 2], [2, 2], [3, 3]]], dtype=np.int64),
    )


def test_membership_aggregation_preserves_explicit_stable_id_order() -> None:
    """Ordered stable IDs preserve a detectable complex128 cancellation order."""
    per_trial_count = np.array([10**16, 1, 10**16], dtype=np.int64)
    counts = np.broadcast_to(
        per_trial_count[:, np.newaxis, np.newaxis, np.newaxis],
        (3, 1, 2, 2),
    ).copy()
    sums = np.broadcast_to(
        np.array([1.0e16 + 0.0j, 1.0 + 0.0j, -1.0e16 + 0.0j])[
            :, np.newaxis, np.newaxis, np.newaxis
        ],
        counts.shape,
    ).astype(np.complex128, copy=True)
    histogram = np.zeros((3, 1, 2, 2, 2), dtype=np.int64)
    histogram[:, :, :, 0, 0] = counts[:, :, :, 0]
    histogram[:, :, :, 1, 0] = counts[:, :, :, 1]
    observed_trials = kernel.ObservedTrialSegmentedPPCStatistics(
        phase_trial_index=np.array([29, 7, 61], dtype=np.int64),
        phase_vector_sum=sums,
        valid_spike_count=counts,
        representative_frequency_index=np.array([0, 1], dtype=np.int64),
        representative_frequency_hz=np.array([8.0, 40.0], dtype=np.float64),
        phase_bin_edges_rad=np.array([-np.pi, 0.0, np.pi], dtype=np.float64),
        representative_phase_histogram_count=histogram,
    )
    cancellation_first = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=np.array([29, 7, 61], dtype=np.int64),
    )
    cancellation_last = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=np.array([29, 61, 7], dtype=np.int64),
    )
    np.testing.assert_array_equal(cancellation_first.phase_vector_sum, 0.0j)
    np.testing.assert_array_equal(cancellation_last.phase_vector_sum, 1.0 + 0.0j)


def test_observed_trial_statistics_allow_one_frequency_for_both_representative_bands() -> None:
    """One frequency supplies both bands at local index zero without ambiguity."""
    constructor_statistics = kernel.ObservedTrialSegmentedPPCStatistics(
        phase_trial_index=np.array([29], dtype=np.int64),
        phase_vector_sum=np.ones((1, 1, 2, 1), dtype=np.complex128),
        valid_spike_count=np.ones((1, 1, 2, 1), dtype=np.int64),
        representative_frequency_index=np.array([0, 0], dtype=np.int64),
        representative_frequency_hz=np.array([8.0, 8.0], dtype=np.float64),
        phase_bin_edges_rad=np.array([-np.pi, 0.0, np.pi], dtype=np.float64),
        representative_phase_histogram_count=np.array(
            [[[[[1, 0], [1, 0]], [[1, 0], [1, 0]]]]],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        constructor_statistics.representative_frequency_index,
        [0, 0],
    )
    np.testing.assert_array_equal(
        constructor_statistics.representative_frequency_hz,
        [8.0, 8.0],
    )
    np.testing.assert_array_equal(
        constructor_statistics.representative_phase_histogram_count[..., 0, :],
        constructor_statistics.representative_phase_histogram_count[..., 1, :],
    )

    geometry = _build_geometry(
        source_trial_index=29,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
    )
    phase = np.ones((1, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    computed_statistics = kernel.compute_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=np.ones(phase.shape, dtype=bool),
        phase_trial_index=np.array([29], dtype=np.int64),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        phase_bin_edges_rad=np.array([-np.pi, 0.0, np.pi], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        computed_statistics.representative_frequency_index,
        [0, 0],
    )
    np.testing.assert_array_equal(
        computed_statistics.representative_frequency_hz,
        [8.0, 8.0],
    )
    np.testing.assert_array_equal(
        computed_statistics.representative_phase_histogram_count[..., 0, :],
        computed_statistics.representative_phase_histogram_count[..., 1, :],
    )
    histogram_total = computed_statistics.representative_phase_histogram_count.sum(
        axis=-1
    )
    np.testing.assert_array_equal(
        histogram_total[:, :, :, 0],
        computed_statistics.valid_spike_count[..., 0],
    )
    np.testing.assert_array_equal(
        histogram_total[:, :, :, 1],
        computed_statistics.valid_spike_count[..., 0],
    )


@pytest.mark.parametrize(
    ("representative_frequency_index", "representative_frequency_hz"),
    (
        (
            np.array([0, 0], dtype=np.int64),
            np.array([8.0, 40.0], dtype=np.float64),
        ),
        (
            np.array([0, 1], dtype=np.int64),
            np.array([8.0, 8.0], dtype=np.float64),
        ),
    ),
)
def test_observed_trial_statistics_reject_incoherent_representative_coordinates(
    representative_frequency_index: np.ndarray,
    representative_frequency_hz: np.ndarray,
) -> None:
    """Representative frequency positions and Hz coordinates have matching equality."""
    arguments = _valid_observed_trial_segmented_statistics_arguments()
    arguments["representative_frequency_index"] = representative_frequency_index
    arguments["representative_frequency_hz"] = representative_frequency_hz
    with pytest.raises(ValueError):
        kernel.ObservedTrialSegmentedPPCStatistics(**arguments)


@pytest.mark.parametrize(
    ("argument_name", "malformed_value"),
    (
        ("phase_trial_index", np.array([29, 7], dtype=np.int32)),
        ("phase_trial_index", np.array([[29, 7]], dtype=np.int64)),
        ("phase_trial_index", np.array([29, 29], dtype=np.int64)),
        ("phase_trial_index", np.array([-1, 7], dtype=np.int64)),
        ("phase_trial_index", np.array([29], dtype=np.int64)),
        ("phase_vector_sum", np.ones((2, 1, 2, 2), dtype=np.complex64)),
        (
            "phase_vector_sum",
            np.full((2, 1, 2, 2), np.nan + 0.0j, dtype=np.complex128),
        ),
        ("valid_spike_count", np.ones((2, 1, 2, 2), dtype=np.int32)),
        ("valid_spike_count", np.ones((2, 1, 2, 1), dtype=np.int64)),
        ("valid_spike_count", -np.ones((2, 1, 2, 2), dtype=np.int64)),
        ("representative_frequency_index", np.array([0, 1], dtype=np.int32)),
        ("representative_frequency_index", np.array([[0, 1]], dtype=np.int64)),
        ("representative_frequency_index", np.array([-1, 0], dtype=np.int64)),
        ("representative_frequency_index", np.array([1, 0], dtype=np.int64)),
        ("representative_frequency_index", np.array([0, 2], dtype=np.int64)),
        ("representative_frequency_hz", np.array([8.0, 40.0], dtype=np.float32)),
        ("representative_frequency_hz", np.array([8.0], dtype=np.float64)),
        ("representative_frequency_hz", np.array([8.0, np.nan], dtype=np.float64)),
        ("representative_frequency_hz", np.array([0.0, 40.0], dtype=np.float64)),
        ("representative_frequency_hz", np.array([40.0, 8.0], dtype=np.float64)),
        ("phase_bin_edges_rad", np.array([-np.pi, np.pi], dtype=np.float32)),
        ("phase_bin_edges_rad", np.array([[-np.pi, np.pi]], dtype=np.float64)),
        ("phase_bin_edges_rad", np.array([-np.pi, np.nan], dtype=np.float64)),
        ("phase_bin_edges_rad", np.array([-np.pi, 0.0, 0.0], dtype=np.float64)),
        (
            "representative_phase_histogram_count",
            np.ones((2, 1, 2, 2, 2), dtype=np.float64),
        ),
        (
            "representative_phase_histogram_count",
            np.ones((2, 1, 2, 2, 1), dtype=np.int64),
        ),
        (
            "representative_phase_histogram_count",
            -np.ones((2, 1, 2, 2, 2), dtype=np.int64),
        ),
        (
            "representative_phase_histogram_count",
            np.full((2, 1, 2, 2, 2), 2, dtype=np.int64),
        ),
    ),
)
def test_observed_trial_statistics_public_constructor_rejects_malformed_axes(
    argument_name: str,
    malformed_value: np.ndarray,
) -> None:
    """Trial construction rejects incompatible IDs, axes, and histogram overcounts."""
    arguments = _valid_observed_trial_segmented_statistics_arguments()
    arguments[argument_name] = malformed_value
    with pytest.raises(ValueError):
        kernel.ObservedTrialSegmentedPPCStatistics(**arguments)


def test_observed_trial_statistics_reject_histograms_for_zero_count_segments() -> None:
    """A per-trial zero count requires zero counts in both representative bands."""
    arguments = _valid_observed_trial_segmented_statistics_arguments()
    arguments["valid_spike_count"][0, 0, 0] = 0
    arguments["phase_vector_sum"][0, 0, 0] = 0.0j
    with pytest.raises(ValueError):
        kernel.ObservedTrialSegmentedPPCStatistics(**arguments)


def test_observed_trial_statistics_reject_nonzero_sums_for_zero_count_segments() -> None:
    """Zero selected-frequency counts also prohibit nonzero retained vector sums."""
    arguments = _valid_observed_trial_segmented_statistics_arguments()
    arguments["valid_spike_count"][0, 0, 0] = 0
    arguments["representative_phase_histogram_count"][0, 0, 0, :, 0] = 0
    with pytest.raises(ValueError):
        kernel.ObservedTrialSegmentedPPCStatistics(**arguments)


def _assert_result_arrays_are_owned_and_immutable(result: object) -> None:
    """Check frozen S2 result arrays are independent of callers and read-only."""
    _assert_ndarray_fields_are_elementwise_immutable(result)


def test_observed_segmented_contracts_are_frozen_keyword_only_and_axis_explicit() -> None:
    """S2 exposes separate immutable observed statistics and composed metric contracts.

    ``phase_vector_sum`` and ``valid_spike_count`` retain only before/after
    sufficient statistics on ``(unit, segment=2, frequency)`` axes. Trial
    contributions and histograms include the explicitly composed whole epoch on
    ``epoch=(before, after, whole)`` axes; all values are dimensionless except
    the stored Hz and radians coordinates. Representative frequencies are
    positive, finite, nondecreasing float64 values: a single input frequency
    legitimately supplies equal 8/40-Hz representatives.
    """
    expected_statistics_fields = (
        "phase_vector_sum",
        "valid_spike_count",
        "contributing_trial_count",
        "representative_frequency_hz",
        "phase_bin_edges_rad",
        "representative_phase_histogram_count",
    )
    expected_metrics_fields = (
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
    )
    assert tuple(
        field.name for field in fields(kernel.ObservedSegmentedPPCStatistics)
    ) == expected_statistics_fields
    assert tuple(
        field.name for field in fields(kernel.ObservedSegmentedPPCMetrics)
    ) == expected_metrics_fields

    observed_signature = inspect.signature(kernel.compute_observed_segmented_ppc_statistics)
    assert tuple(observed_signature.parameters) == (
        "source_trial_geometries",
        "trial_phase_vectors",
        "phase_valid_mask",
        "phase_trial_index",
        "frequencies_hz",
        "phase_bin_edges_rad",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in observed_signature.parameters.values()
    )
    composition_signature = inspect.signature(
        kernel.compose_observed_segmented_ppc_metrics
    )
    assert tuple(composition_signature.parameters) == ("observed_statistics",)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in composition_signature.parameters.values()
    )
    shuffle_signature = inspect.signature(kernel.reduce_segmented_schedule_to_ppc)
    assert tuple(shuffle_signature.parameters) == (
        "edge_statistics",
        "phase_trial_index",
        "schedule",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in shuffle_signature.parameters.values()
    )
    sampler_signature = inspect.signature(kernel._sample_normalized_spike_group)
    assert tuple(sampler_signature.parameters) == (
        "coefficients",
        "explicit_valid_mask",
        "left_index",
        "right_index",
        "right_weight",
        "inside_support",
        "exact_sample",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in sampler_signature.parameters.values()
    )


def test_observed_single_pass_returns_owned_before_after_sums_trial_unions_and_histograms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One observed pass returns all reusable half statistics without mask copies.

    The reducer may not use a full ``numpy.where(valid, phase, 0j)`` phase copy
    and may not call the generic edge reducer and then perform a second
    same-trial sampling pass for contributions or representative histograms.
    """
    geometries, phase, valid, phase_trial_index, frequencies_hz, phase_bins = (
        _observed_segmented_fixture()
    )

    def forbidden_where(*args: object, **kwargs: object) -> object:
        """Fail if observed reduction constructs the prohibited full mask copy."""
        raise AssertionError("observed reduction must not construct np.where phase copies")

    def forbidden_edge_reduction(*args: object, **kwargs: object) -> object:
        """Fail if observed work is delegated to a second generic edge pass."""
        raise AssertionError("observed reduction must sample same-trial edges only once")

    original_sampler = kernel._sample_normalized_spike_group
    sampled_group_spike_counts: list[int] = []

    def counted_sampler(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Record the one shared sampler call needed for each nonempty group."""
        sampled_group_spike_counts.append(np.asarray(kwargs["left_index"]).size)
        return original_sampler(**kwargs)

    monkeypatch.setattr(kernel.np, "where", forbidden_where)
    monkeypatch.setattr(
        kernel,
        "compute_segmented_edge_statistics",
        forbidden_edge_reduction,
    )
    monkeypatch.setattr(kernel, "_sample_normalized_spike_group", counted_sampler)
    observed = kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=phase_bins,
    )

    assert isinstance(observed, kernel.ObservedSegmentedPPCStatistics)
    assert observed.phase_vector_sum.dtype == np.dtype(np.complex128)
    assert observed.phase_vector_sum.shape == (1, 2, 3)
    assert observed.valid_spike_count.dtype == np.dtype(np.int64)
    assert observed.valid_spike_count.shape == (1, 2, 3)
    assert observed.contributing_trial_count.dtype == np.dtype(np.int64)
    assert observed.contributing_trial_count.shape == (1, 3, 3)
    assert observed.representative_frequency_hz.dtype == np.dtype(np.float64)
    assert observed.representative_frequency_hz.shape == (2,)
    assert observed.phase_bin_edges_rad.dtype == np.dtype(np.float64)
    assert observed.phase_bin_edges_rad.shape == (phase_bins.size,)
    assert observed.representative_phase_histogram_count.dtype == np.dtype(np.int64)
    assert observed.representative_phase_histogram_count.shape == (
        1,
        3,
        2,
        phase_bins.size - 1,
    )
    np.testing.assert_allclose(
        observed.phase_vector_sum,
        np.array([[[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]]]),
    )
    np.testing.assert_array_equal(observed.valid_spike_count, 2)
    # Both physical trials contribute in both halves, but only once to whole.
    np.testing.assert_array_equal(observed.contributing_trial_count, 2)
    np.testing.assert_array_equal(observed.representative_frequency_hz, [8.0, 40.0])
    # Two trials x one unit x two nonempty half groups. The same normalized
    # vectors must feed pooled sums, trial contributors, and histogram bins.
    assert sampled_group_spike_counts == [1, 1, 1, 1]

    phase[:, :, :] = 0.0j
    valid[:, :, :] = False
    phase_trial_index[:] = -1
    frequencies_hz[:] = -1.0
    phase_bins[:] = 0.0
    np.testing.assert_allclose(
        observed.phase_vector_sum,
        np.array([[[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]]]),
    )
    np.testing.assert_array_equal(observed.valid_spike_count, 2)
    np.testing.assert_array_equal(observed.contributing_trial_count, 2)
    np.testing.assert_array_equal(observed.representative_frequency_hz, [8.0, 40.0])
    np.testing.assert_allclose(
        observed.phase_bin_edges_rad,
        _REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
        rtol=0.0,
        atol=0.0,
    )
    _assert_result_arrays_are_owned_and_immutable(observed)
    with pytest.raises(FrozenInstanceError):
        observed.phase_bin_edges_rad = np.array([-np.pi, np.pi], dtype=np.float64)


def test_observed_histograms_share_s1_exact_canonical_sample_rule() -> None:
    """A near-grid spike rejected by S1 is also absent from its 8/40-Hz bins.

    The second spike is one representable float above a stored sample and the
    third is within ``1e-12`` but not exactly on the grid. Their right neighbor
    is unavailable, so both must fail two-neighbor interpolation. Only the
    exact canonical-grid spike contributes to PPC and to both histograms.
    """
    time_s = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    exact = 0.5
    geometry = _build_geometry(
        source_trial_index=17,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(
            np.array(
                [
                    exact,
                    np.nextafter(exact, np.inf),
                    exact + 5e-13,
                ],
                dtype=np.float64,
            ),
        ),
        phase_time_s=time_s,
        phase_sampling_rate_hz=2.0,
        segment_bounds_s=((-0.1, 0.0), (0.0, 1.1)),
    )
    frequencies_hz = np.array([8.0, 40.0], dtype=np.float64)
    phase = np.ones((1, 2, time_s.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)
    # The exact sample only needs its left value; both nonexact samples require
    # the invalid right neighbor at 1.0 seconds.
    valid[:, :, 2] = False

    observed_trials = kernel.compute_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([17], dtype=np.int64),
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    observed = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=np.array([17], dtype=np.int64),
    )

    np.testing.assert_array_equal(geometry.exact_sample, [True, False, False])
    np.testing.assert_array_equal(
        observed_trials.valid_spike_count,
        np.array([[[[0, 0], [1, 1]]]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        observed_trials.representative_phase_histogram_count.sum(axis=-1),
        np.array([[[[0, 0], [1, 1]]]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        observed.contributing_trial_count,
        np.array([[[0, 0], [1, 1], [1, 1]]], dtype=np.int64),
    )
    histogram_total = observed.representative_phase_histogram_count.sum(axis=-1)
    np.testing.assert_array_equal(
        histogram_total,
        np.array([[[0, 0], [1, 1], [1, 1]]]),
    )
    # At each selected representative frequency, accepted histogram samples
    # equal accepted PPC samples in each stored half. Whole is their exact sum,
    # not a duplicate of after's count.
    np.testing.assert_array_equal(
        histogram_total[0, :2, 0],
        observed.valid_spike_count[0, :, 0],
    )
    np.testing.assert_array_equal(
        histogram_total[0, :2, 1],
        observed.valid_spike_count[0, :, 1],
    )
    np.testing.assert_array_equal(
        histogram_total[0, 2],
        observed.valid_spike_count[0, 0] + observed.valid_spike_count[0, 1],
    )


def test_observed_histograms_use_only_8_and_40_hz_with_current_boundary_bins() -> None:
    """Representative histograms retain the current complex64-angle bin semantics.

    Exact canonical-grid samples remain accepted by both PPC and histograms,
    but ``np.angle(complex64)`` gives float32 boundary values.  NumPy therefore
    assigns exact-looking boundary phases according to the current payload
    builder's float32 values rather than a promoted complex128 angle. The
    deliberately different 20-Hz row proves the two outputs select the nearest
    configured 8- and 40-Hz rows, not the intervening frequency.
    """
    phase_time_s = np.arange(-1.0, 1.25, 0.25, dtype=np.float64)
    frequencies_hz = np.array([8.0, 20.0, 40.0], dtype=np.float64)
    phase = np.ones((1, 3, phase_time_s.size), dtype=np.complex64)
    negative_pi = np.complex64(complex(-1.0, -0.0))
    positive_pi = np.complex64(complex(-1.0, 0.0))
    phase[0, 0, :4] = [negative_pi, -1.0j, 1.0, 1.0j]
    phase[0, 0, 4:8] = [positive_pi, 1.0j, 1.0, -1.0j]
    # If the implementation accidentally used this 20-Hz row for gamma, the
    # expected 40-Hz bins below would immediately differ.
    phase[0, 1, :8] = 1.0 + 0.0j
    phase[0, 2, :4] = [-1.0j, -1.0j, 1.0, positive_pi]
    phase[0, 2, 4:8] = [1.0, 1.0, 1.0j, positive_pi]
    geometry = _build_geometry(
        source_trial_index=5,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(phase_time_s[:8],),
        phase_time_s=phase_time_s,
        phase_sampling_rate_hz=4.0,
        segment_bounds_s=((-1.0, 0.0), (0.0, 1.0)),
    )
    observed_trials = kernel.compute_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=np.ones(phase.shape, dtype=bool),
        phase_trial_index=np.array([5], dtype=np.int64),
        frequencies_hz=frequencies_hz,
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    observed = kernel.aggregate_observed_trial_segmented_ppc_statistics(
        observed_trial_statistics=observed_trials,
        membership_trial_index=np.array([5], dtype=np.int64),
    )

    expected = np.array(
        [
            [[1, 0, 1, 1], [2, 0, 1, 0]],
            [[1, 0, 1, 1], [0, 0, 2, 1]],
            [[2, 0, 2, 2], [2, 0, 3, 1]],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(observed.representative_frequency_hz, [8.0, 40.0])
    np.testing.assert_array_equal(
        observed_trials.representative_phase_histogram_count[0, 0],
        expected[:2],
    )
    selected_count = observed_trials.valid_spike_count[
        ..., observed_trials.representative_frequency_index
    ]
    assert np.all(
        observed_trials.representative_phase_histogram_count.sum(axis=-1)
        <= selected_count
    )
    reconstructed_trial_statistics = kernel.ObservedTrialSegmentedPPCStatistics(
        phase_trial_index=observed_trials.phase_trial_index.copy(),
        phase_vector_sum=observed_trials.phase_vector_sum.copy(),
        valid_spike_count=observed_trials.valid_spike_count.copy(),
        representative_frequency_index=(
            observed_trials.representative_frequency_index.copy()
        ),
        representative_frequency_hz=observed_trials.representative_frequency_hz.copy(),
        phase_bin_edges_rad=observed_trials.phase_bin_edges_rad.copy(),
        representative_phase_histogram_count=(
            observed_trials.representative_phase_histogram_count.copy()
        ),
    )
    np.testing.assert_array_equal(
        reconstructed_trial_statistics.representative_phase_histogram_count,
        observed_trials.representative_phase_histogram_count,
    )
    np.testing.assert_array_equal(
        observed.representative_phase_histogram_count[0],
        expected,
    )
    np.testing.assert_array_equal(
        observed.representative_phase_histogram_count[0, 2],
        observed.representative_phase_histogram_count[0, 0]
        + observed.representative_phase_histogram_count[0, 1],
    )
    twenty_hz_counts = np.array(
        [[0, 0, 4, 0], [0, 0, 4, 0], [0, 0, 8, 0]],
        dtype=np.int64,
    )
    assert not np.array_equal(
        observed.representative_phase_histogram_count[0, :, 1],
        twenty_hz_counts,
    )

    # All spikes are exact canonical-grid samples in this fixture.  The new
    # reuse path must therefore retain the current payload helper's precise
    # complex64-angle boundary bins; only near-grid acceptance differs by plan.
    for segment_position, (start, stop) in enumerate(((0, 4), (4, 8))):
        direct_histogram = spike_lfp_summary.build_representative_phase_histograms(
            frequencies_hz=frequencies_hz,
            spike_phase_vectors=phase[0, :, start:stop],
            valid_mask=np.ones((3, stop - start), dtype=bool),
            trial_indices=np.zeros(stop - start, dtype=np.int64),
            phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
        )
        np.testing.assert_array_equal(
            observed_trials.representative_phase_histogram_count[
                0,
                0,
                segment_position,
            ],
            direct_histogram.spike_count_by_band,
        )


def test_composed_whole_observed_metrics_add_sufficient_statistics_not_half_ppc() -> None:
    """Whole PPC retains before/after cross terms and histogram additivity.

    Both halves have PPC 1.0, while their equal and opposite vectors produce
    whole PPC ``-1/3``. That direct-whole value proves composition adds complex
    sums and counts before applying the PPC formula rather than averaging half
    PPC values.
    """
    observed = _compute_observed_segmented_fixture()
    metrics = kernel.compose_observed_segmented_ppc_metrics(
        observed_statistics=observed,
    )

    assert isinstance(metrics, kernel.ObservedSegmentedPPCMetrics)
    assert metrics.ppc.dtype == np.dtype(np.float64)
    assert metrics.ppc.shape == (1, 3, 3)
    assert metrics.resultant_length.dtype == np.dtype(np.float64)
    assert metrics.preferred_phase_rad.dtype == np.dtype(np.float64)
    assert metrics.spike_count.dtype == np.dtype(np.int64)
    assert metrics.computable.dtype == np.dtype(bool)
    assert metrics.reliable.dtype == np.dtype(bool)
    assert metrics.contributing_trial_count.dtype == np.dtype(np.int64)
    assert metrics.shuffle_eligible.dtype == np.dtype(bool)
    assert metrics.representative_phase_histogram_count.dtype == np.dtype(np.int64)
    np.testing.assert_allclose(metrics.ppc[0, 0], 1.0)
    np.testing.assert_allclose(metrics.ppc[0, 1], 1.0)
    np.testing.assert_allclose(metrics.ppc[0, 2], -1.0 / 3.0)
    assert metrics.ppc[0, 2, 0] != np.mean(metrics.ppc[0, :2, 0])
    np.testing.assert_allclose(metrics.resultant_length[0, :2], 1.0)
    np.testing.assert_allclose(metrics.resultant_length[0, 2], 0.0)
    assert np.isnan(metrics.preferred_phase_rad[0, 2]).all()
    np.testing.assert_array_equal(metrics.spike_count, [[[2, 2, 2], [2, 2, 2], [4, 4, 4]]])
    np.testing.assert_array_equal(metrics.computable, True)
    np.testing.assert_array_equal(metrics.reliable, False)
    np.testing.assert_array_equal(metrics.shuffle_eligible, False)
    # Whole trial eligibility is the per-trial union, not 2 + 2 = 4.
    np.testing.assert_array_equal(metrics.contributing_trial_count, 2)
    np.testing.assert_array_equal(
        metrics.representative_phase_histogram_count[:, 2],
        metrics.representative_phase_histogram_count[:, 0]
        + metrics.representative_phase_histogram_count[:, 1],
    )

    _, phase, _, _, frequencies_hz, _ = _observed_segmented_fixture()
    direct = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=(
            np.array([-0.25, 0.25], dtype=np.float64),
            np.array([-0.25, 0.25], dtype=np.float64),
        ),
        phase_time_s=_FOUR_HZ_TIME_S,
        trial_phase_vectors=phase,
        frequencies_hz=frequencies_hz,
        schedule=np.array([[1, 0]], dtype=np.int64),
    )
    np.testing.assert_allclose(metrics.ppc[0, 2], direct.observed_ppc, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(metrics.spike_count[0, 2], direct.spike_count)
    direct_metrics = spike_lfp_summary.compute_observed_ppc(
        probe_label="probe",
        cluster_id=1,
        spike_phase_vectors=np.array(
            [[1.0, 1.0, -1.0, -1.0]] * frequencies_hz.size,
            dtype=np.complex64,
        ),
        valid_mask=np.ones((frequencies_hz.size, 4), dtype=bool),
        frequencies_hz=frequencies_hz,
        spike_times_s=np.arange(4, dtype=np.float64),
    )
    np.testing.assert_allclose(
        metrics.ppc[0, 2],
        direct_metrics.ppc,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        metrics.resultant_length[0, 2],
        direct_metrics.resultant_length,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        metrics.preferred_phase_rad[0, 2],
        direct_metrics.preferred_phase_rad,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(metrics.spike_count[0, 2], direct_metrics.spike_count)
    np.testing.assert_array_equal(metrics.computable[0, 2], direct_metrics.computable)
    np.testing.assert_array_equal(metrics.reliable[0, 2], direct_metrics.reliable)
    _assert_result_arrays_are_owned_and_immutable(metrics)
    with pytest.raises(FrozenInstanceError):
        metrics.ppc = np.zeros((1, 3, 3), dtype=np.float64)


def test_whole_observed_eligibility_uses_combined_count_and_trial_union() -> None:
    """Whole becomes shuffle-eligible when 49 spikes per half combine to 98.

    The same two physical trials contribute in both halves. Whole eligibility
    therefore requires the count threshold from the summed sufficient
    statistics and two (not four) trial contributors from a per-trial union.
    """
    phase = np.ones((2, 2, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)
    geometries = (
        _build_geometry(
            source_trial_index=41,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(
                np.concatenate(
                    (
                        np.full(25, -0.25, dtype=np.float64),
                        np.full(25, 0.25, dtype=np.float64),
                    )
                ),
            ),
        ),
        _build_geometry(
            source_trial_index=83,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(
                np.concatenate(
                    (
                        np.full(24, -0.25, dtype=np.float64),
                        np.full(24, 0.25, dtype=np.float64),
                    )
                ),
            ),
        ),
    )
    observed = kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([41, 83], dtype=np.int64),
        frequencies_hz=np.array([8.0, 40.0], dtype=np.float64),
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    metrics = kernel.compose_observed_segmented_ppc_metrics(
        observed_statistics=observed,
    )

    np.testing.assert_array_equal(metrics.spike_count, [[[49, 49], [49, 49], [98, 98]]])
    np.testing.assert_array_equal(metrics.reliable, [[[False, False], [False, False], [True, True]]])
    np.testing.assert_array_equal(
        metrics.contributing_trial_count,
        [[[2, 2], [2, 2], [2, 2]]],
    )
    np.testing.assert_array_equal(
        metrics.shuffle_eligible,
        [[[False, False], [False, False], [True, True]]],
    )


def test_composed_whole_metrics_preserve_nonzero_preferred_phase_and_zero_one_spike_behavior() -> None:
    """Composition matches direct PPC/circular metrics at useful and degenerate counts."""
    phase = np.ones((2, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    phase[:, 0, 1] = 1.0 + 0.0j
    phase[:, 0, 3] = 1.0j
    geometries = tuple(
        _build_geometry(
            source_trial_index=source_id,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
        )
        for source_id in (101, 303)
    )
    observed = kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=np.ones(phase.shape, dtype=bool),
        phase_trial_index=np.array([101, 303], dtype=np.int64),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    metrics = kernel.compose_observed_segmented_ppc_metrics(
        observed_statistics=observed,
    )
    direct = spike_lfp_summary.compute_observed_ppc(
        probe_label="probe",
        cluster_id=1,
        spike_phase_vectors=np.array([[1.0, 1.0j, 1.0, 1.0j]], dtype=np.complex64),
        valid_mask=np.ones((1, 4), dtype=bool),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        spike_times_s=np.arange(4, dtype=np.float64),
    )
    np.testing.assert_allclose(metrics.ppc[0, 2], direct.ppc, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        metrics.resultant_length[0, 2],
        direct.resultant_length,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics.preferred_phase_rad[0, 2],
        direct.preferred_phase_rad,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(metrics.spike_count[0, 2], direct.spike_count)
    np.testing.assert_array_equal(metrics.computable[0, 2], direct.computable)
    np.testing.assert_array_equal(metrics.reliable[0, 2], direct.reliable)
    np.testing.assert_allclose(metrics.preferred_phase_rad[0, 2], np.pi / 4.0)

    # A second multi-unit call binds empty and one-spike semantics without
    # collapsing the unit axis. PPC is NaN below two samples; resultant and
    # preferred phase are only defined for a nonzero count.
    degenerate_geometry = _build_geometry(
        source_trial_index=11,
        unit_ids=("empty-unit", "one-spike-unit"),
        unit_trial_spike_times_s=(np.empty(0, dtype=np.float64), np.array([0.25])),
    )
    degenerate_phase = np.ones((1, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    degenerate_phase[0, 0, 3] = 1.0j
    degenerate_observed = kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=(degenerate_geometry,),
        trial_phase_vectors=degenerate_phase,
        phase_valid_mask=np.ones(degenerate_phase.shape, dtype=bool),
        phase_trial_index=np.array([11], dtype=np.int64),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    degenerate_metrics = kernel.compose_observed_segmented_ppc_metrics(
        observed_statistics=degenerate_observed,
    )
    np.testing.assert_array_equal(
        degenerate_observed.representative_frequency_hz,
        [8.0, 8.0],
    )
    np.testing.assert_array_equal(
        degenerate_observed.representative_phase_histogram_count[:, :, 0],
        degenerate_observed.representative_phase_histogram_count[:, :, 1],
    )
    np.testing.assert_array_equal(
        degenerate_observed.representative_phase_histogram_count[0],
        np.zeros((3, 2, 4), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        degenerate_observed.representative_phase_histogram_count[1],
        np.array(
            [
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 1], [0, 0, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 1]],
            ],
            dtype=np.int64,
        ),
    )
    assert degenerate_metrics.ppc.shape == (2, 3, 1)
    np.testing.assert_array_equal(degenerate_metrics.spike_count[0], [[0], [0], [0]])
    assert np.isnan(degenerate_metrics.ppc[0]).all()
    assert np.isnan(degenerate_metrics.resultant_length[0]).all()
    assert np.isnan(degenerate_metrics.preferred_phase_rad[0]).all()
    np.testing.assert_array_equal(degenerate_metrics.computable[0], False)
    np.testing.assert_array_equal(degenerate_metrics.reliable[0], False)
    np.testing.assert_array_equal(degenerate_metrics.shuffle_eligible[0], False)
    np.testing.assert_array_equal(degenerate_metrics.spike_count[1], [[0], [1], [1]])
    assert np.isnan(degenerate_metrics.ppc[1]).all()
    assert np.isnan(degenerate_metrics.resultant_length[1, 0]).all()
    assert np.isnan(degenerate_metrics.preferred_phase_rad[1, 0]).all()
    np.testing.assert_allclose(degenerate_metrics.resultant_length[1, 1:], 1.0)
    np.testing.assert_allclose(degenerate_metrics.preferred_phase_rad[1, 1:], np.pi / 2.0)
    np.testing.assert_array_equal(degenerate_metrics.computable[1], False)
    np.testing.assert_array_equal(degenerate_metrics.reliable[1], False)
    np.testing.assert_array_equal(degenerate_metrics.shuffle_eligible[1], False)


def test_whole_contributing_trial_count_uses_per_trial_union_for_partial_overlap() -> None:
    """Before contributors {A, B} and after {B, C} compose to whole count 3."""
    phase = np.ones((3, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    geometries = (
        _build_geometry(
            source_trial_index=10,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([-0.25]),),
        ),
        _build_geometry(
            source_trial_index=20,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
        ),
        _build_geometry(
            source_trial_index=30,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([0.25]),),
        ),
    )
    observed = kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=np.ones(phase.shape, dtype=bool),
        phase_trial_index=np.array([10, 20, 30], dtype=np.int64),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    metrics = kernel.compose_observed_segmented_ppc_metrics(
        observed_statistics=observed,
    )
    np.testing.assert_array_equal(metrics.contributing_trial_count, [[[2], [2], [3]]])


def test_reliable_one_trial_whole_remains_shuffle_ineligible() -> None:
    """Fifty valid phases from one source trial are reliable but not shuffleable."""
    geometry = _build_geometry(
        source_trial_index=10,
        unit_ids=("unit-1",),
        unit_trial_spike_times_s=(
            np.concatenate(
                (
                    np.full(49, -0.25, dtype=np.float64),
                    np.full(1, 0.25, dtype=np.float64),
                )
            ),
        ),
    )
    phase = np.ones((1, 1, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    observed = kernel.compute_observed_segmented_ppc_statistics(
        source_trial_geometries=(geometry,),
        trial_phase_vectors=phase,
        phase_valid_mask=np.ones(phase.shape, dtype=bool),
        phase_trial_index=np.array([10], dtype=np.int64),
        frequencies_hz=np.array([8.0], dtype=np.float64),
        phase_bin_edges_rad=_REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
    )
    metrics = kernel.compose_observed_segmented_ppc_metrics(
        observed_statistics=observed,
    )
    np.testing.assert_array_equal(metrics.spike_count, [[[49], [1], [50]]])
    np.testing.assert_array_equal(metrics.reliable, [[[False], [False], [True]]])
    np.testing.assert_array_equal(metrics.contributing_trial_count, [[[1], [1], [1]]])
    np.testing.assert_array_equal(metrics.shuffle_eligible, [[[False], [False], [False]]])


def test_whole_segmented_shuffle_draws_match_wp5b_direct_whole_for_unchanged_schedule() -> None:
    """Adding the two segment statistics matches WP5B whole-window draw values.

    This S2 reducer receives the exact existing schedule directly. It derives no
    seed and makes no schedule transformation, so before/after/whole schedule
    identities remain the responsibility of the existing caller/planner.
    """
    phase_time_s = _FOUR_HZ_TIME_S
    frequencies_hz = np.array([8.0, 40.0], dtype=np.float64)
    phase_trial_index = np.array([61, 17, 88], dtype=np.int64)
    phase = np.array(
        [
            [[1.0, 1.0, 1.0, -1.0, 1.0], [1.0j, 1.0j, 1.0j, -1.0j, 1.0j]],
            [[1.0, 1.0j, 1.0, 1.0j, 1.0], [1.0, -1.0, 1.0, -1.0, 1.0]],
            [[1.0, -1.0, 1.0, 1.0, 1.0], [-1.0j, 1.0j, -1.0j, 1.0j, -1.0j]],
        ],
        dtype=np.complex64,
    )
    valid = np.ones(phase.shape, dtype=bool)
    trial_spikes = (
        np.array([-0.25], dtype=np.float64),
        np.array([0.0, 0.25], dtype=np.float64),
        np.array([-0.5, -0.25, 0.25], dtype=np.float64),
    )
    geometries = tuple(
        _build_geometry(
            source_trial_index=source_id,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(trial_spikes[position],),
        )
        for source_id, position in ((88, 2), (61, 0), (17, 1))
    )
    schedule = np.array([[1, 2, 0], [2, 0, 1]], dtype=np.int64)
    source_positions = np.tile(np.arange(3, dtype=np.int64), schedule.shape[0])
    target_positions = schedule.reshape(-1)
    source_edges = phase_trial_index[source_positions]
    target_edges = phase_trial_index[target_positions]
    segmented = _reduce(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=phase_trial_index,
        frequencies_hz=frequencies_hz,
        source_trial_index=source_edges,
        target_trial_index=target_edges,
    )

    draws = kernel.reduce_segmented_schedule_to_ppc(
        edge_statistics=segmented,
        phase_trial_index=phase_trial_index,
        schedule=schedule,
    )
    direct_draws = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=(trial_spikes,),
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=frequencies_hz,
        schedule=schedule,
        unit_block_size=1,
        trial_edge_block_size=2,
        shuffle_block_size=1,
        complete_pair_table=False,
    )
    assert draws.dtype == np.dtype(np.float64)
    assert draws.shape == (schedule.shape[0], 1, frequencies_hz.size)
    for shuffle_position in range(schedule.shape[0]):
        np.testing.assert_allclose(
            draws[shuffle_position],
            direct_draws[shuffle_position],
            rtol=0.0,
            atol=2e-7,
        )
    assert not np.allclose(draws[0], draws[1], rtol=0.0, atol=2e-7, equal_nan=True)
    np.testing.assert_array_equal(schedule, [[1, 2, 0], [2, 0, 1]])


@pytest.mark.parametrize(
    ("phase", "valid", "phase_trial_index", "frequencies_hz", "phase_bins"),
    (
        (
            np.ones((2, 3, 5), dtype=np.complex128),
            np.ones((2, 3, 5), dtype=bool),
            np.array([29, 7], dtype=np.int64),
            np.array([8.0, 20.0, 40.0], dtype=np.float64),
            _REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
        ),
        (
            np.ones((2, 3, 5), dtype=np.complex64),
            np.ones((2, 3, 4), dtype=bool),
            np.array([29, 7], dtype=np.int64),
            np.array([8.0, 20.0, 40.0], dtype=np.float64),
            _REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
        ),
        (
            np.ones((2, 3, 5), dtype=np.complex64),
            np.ones((2, 3, 5), dtype=bool),
            np.array([29, 29], dtype=np.int64),
            np.array([8.0, 20.0, 40.0], dtype=np.float64),
            _REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
        ),
        (
            np.ones((2, 3, 5), dtype=np.complex64),
            np.ones((2, 3, 5), dtype=bool),
            np.array([29, 7], dtype=np.int64),
            np.array([8.0, 20.0, 40.0], dtype=np.float32),
            _REPRESENTATIVE_PHASE_BIN_EDGES_RAD,
        ),
        (
            np.ones((2, 3, 5), dtype=np.complex64),
            np.ones((2, 3, 5), dtype=bool),
            np.array([29, 7], dtype=np.int64),
            np.array([8.0, 20.0, 40.0], dtype=np.float64),
            np.array([-np.pi, 0.0, 0.0, np.pi], dtype=np.float64),
        ),
    ),
)
def test_observed_segmented_reducer_rejects_malformed_phase_and_histogram_axes(
    phase: np.ndarray,
    valid: np.ndarray,
    phase_trial_index: np.ndarray,
    frequencies_hz: np.ndarray,
    phase_bins: np.ndarray,
) -> None:
    """Observed S2 inputs reject malformed dtypes, axes, stable IDs, and radians bins."""
    geometries, _, _, _, _, _ = _observed_segmented_fixture()
    with pytest.raises(ValueError):
        kernel.compute_observed_segmented_ppc_statistics(
            source_trial_geometries=geometries,
            trial_phase_vectors=phase,
            phase_valid_mask=valid,
            phase_trial_index=phase_trial_index,
            frequencies_hz=frequencies_hz,
            phase_bin_edges_rad=phase_bins,
        )


def test_observed_segmented_reducer_requires_one_geometry_for_each_prepared_trial() -> None:
    """No observed trial may be silently omitted from pooled sums or histograms."""
    geometries, phase, valid, phase_trial_index, frequencies_hz, phase_bins = (
        _observed_segmented_fixture()
    )
    with pytest.raises(ValueError):
        kernel.compute_observed_segmented_ppc_statistics(
            source_trial_geometries=geometries[:1],
            trial_phase_vectors=phase,
            phase_valid_mask=valid,
            phase_trial_index=phase_trial_index,
            frequencies_hz=frequencies_hz,
            phase_bin_edges_rad=phase_bins,
        )


@pytest.mark.parametrize(
    ("phase_trial_index", "schedule"),
    (
        (np.array([61, 17, 88], dtype=np.int64), np.array([[0, 2, 1]], dtype=np.int64)),
        (np.array([61, 17, 88], dtype=np.int64), np.array([[1.0, 2.0, 0.0]])),
        (np.array([61, 17, 88], dtype=np.int64), np.array([[1, 2, 0]], dtype=np.int32)),
        (np.array([61, 61, 88], dtype=np.int64), np.array([[1, 2, 0]], dtype=np.int64)),
    ),
)
def test_whole_segmented_shuffle_rejects_malformed_schedule_and_trial_identity_axes(
    phase_trial_index: np.ndarray,
    schedule: np.ndarray,
) -> None:
    """Whole schedule reduction requires exact int64 derangements and stable rows."""
    phase = np.ones((3, 2, _FOUR_HZ_TIME_S.size), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)
    geometries = tuple(
        _build_geometry(
            source_trial_index=source_id,
            unit_ids=("unit-1",),
            unit_trial_spike_times_s=(np.array([-0.25, 0.25]),),
        )
        for source_id in (61, 17, 88)
    )
    segmented = _reduce(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase,
        phase_valid_mask=valid,
        phase_trial_index=np.array([61, 17, 88], dtype=np.int64),
        frequencies_hz=np.array([8.0, 40.0], dtype=np.float64),
        source_trial_index=np.array([61, 17, 88], dtype=np.int64),
        target_trial_index=np.array([17, 88, 61], dtype=np.int64),
    )
    with pytest.raises(ValueError):
        kernel.reduce_segmented_schedule_to_ppc(
            edge_statistics=segmented,
            phase_trial_index=phase_trial_index,
            schedule=schedule,
        )


def test_whole_segmented_shuffle_rejects_schedule_pairs_missing_from_edge_statistics() -> None:
    """The reducer fails rather than treating an absent scheduled edge as zero."""
    statistics = kernel.SegmentedEdgeStatistics(
        source_trial_index=np.array([61, 17, 88], dtype=np.int64),
        target_trial_index=np.array([17, 88, 61], dtype=np.int64),
        phase_vector_sum=np.ones((3, 1, 2, 2), dtype=np.complex128),
        valid_spike_count=np.ones((3, 1, 2, 2), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="edge|schedule"):
        kernel.reduce_segmented_schedule_to_ppc(
            edge_statistics=statistics,
            phase_trial_index=np.array([61, 17, 88], dtype=np.int64),
            schedule=np.array([[2, 0, 1]], dtype=np.int64),
        )


def _valid_observed_segmented_statistics_arguments() -> dict[str, np.ndarray]:
    """Return coherent direct-construction arrays for the public S2 result contract."""
    phase_vector_sum = np.array(
        [[[2.0 + 0.0j, 2.0 + 0.0j], [0.0j, 0.0j]]],
        dtype=np.complex128,
    )
    valid_spike_count = np.array([[[2, 2], [0, 0]]], dtype=np.int64)
    contributing_trial_count = np.array([[[1, 1], [0, 0], [1, 1]]], dtype=np.int64)
    phase_bin_edges_rad = np.array([-np.pi, 0.0, np.pi], dtype=np.float64)
    histogram = np.array(
        [[[[0, 2], [0, 2]], [[0, 0], [0, 0]], [[0, 2], [0, 2]]]],
        dtype=np.int64,
    )
    return {
        "phase_vector_sum": phase_vector_sum,
        "valid_spike_count": valid_spike_count,
        "contributing_trial_count": contributing_trial_count,
        "representative_frequency_hz": np.array([8.0, 40.0], dtype=np.float64),
        "phase_bin_edges_rad": phase_bin_edges_rad,
        "representative_phase_histogram_count": histogram,
    }


def _valid_observed_segmented_metric_arguments() -> dict[str, np.ndarray]:
    """Return coherent direct-construction arrays for the composed S2 metric contract."""
    statistics = _valid_observed_segmented_statistics_arguments()
    return {
        "ppc": np.array([[[1.0, 1.0], [np.nan, np.nan], [1.0, 1.0]]]),
        "resultant_length": np.array(
            [[[1.0, 1.0], [np.nan, np.nan], [1.0, 1.0]]],
            dtype=np.float64,
        ),
        "preferred_phase_rad": np.array(
            [[[0.0, 0.0], [np.nan, np.nan], [0.0, 0.0]]],
            dtype=np.float64,
        ),
        "spike_count": np.array([[[2, 2], [0, 0], [2, 2]]], dtype=np.int64),
        "computable": np.array([[[True, True], [False, False], [True, True]]]),
        "reliable": np.zeros((1, 3, 2), dtype=bool),
        "contributing_trial_count": statistics["contributing_trial_count"],
        "shuffle_eligible": np.zeros((1, 3, 2), dtype=bool),
        "representative_frequency_hz": statistics["representative_frequency_hz"],
        "phase_bin_edges_rad": statistics["phase_bin_edges_rad"],
        "representative_phase_histogram_count": statistics[
            "representative_phase_histogram_count"
        ],
    }


def test_observed_segmented_public_results_copy_and_freeze_all_array_fields() -> None:
    """Public S2 dataclasses own and freeze every numerical and coordinate field."""
    statistic_arguments = _valid_observed_segmented_statistics_arguments()
    statistic_expected = {
        name: values.copy() for name, values in statistic_arguments.items()
    }
    observed = kernel.ObservedSegmentedPPCStatistics(**statistic_arguments)
    metric_arguments = _valid_observed_segmented_metric_arguments()
    metric_expected = {
        name: values.copy() for name, values in metric_arguments.items()
    }
    metrics = kernel.ObservedSegmentedPPCMetrics(**metric_arguments)

    for result, arguments in (
        (observed, statistic_arguments),
        (metrics, metric_arguments),
    ):
        for field in fields(result):
            input_values = arguments[field.name]
            result_values = getattr(result, field.name)
            assert input_values is not result_values
            assert not np.shares_memory(result_values, input_values)

    for arguments in (statistic_arguments, metric_arguments):
        for values in arguments.values():
            values[...] = 0
    for result, expected in (
        (observed, statistic_expected),
        (metrics, metric_expected),
    ):
        for field in fields(result):
            actual = getattr(result, field.name)
            baseline = expected[field.name]
            if np.issubdtype(baseline.dtype, np.inexact):
                np.testing.assert_allclose(
                    actual,
                    baseline,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            else:
                np.testing.assert_array_equal(actual, baseline)
    np.testing.assert_array_equal(observed.representative_frequency_hz, [8.0, 40.0])
    np.testing.assert_allclose(
        observed.phase_bin_edges_rad,
        [-np.pi, 0.0, np.pi],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(metrics.representative_frequency_hz, [8.0, 40.0])
    np.testing.assert_allclose(
        metrics.phase_bin_edges_rad,
        [-np.pi, 0.0, np.pi],
        rtol=0.0,
        atol=0.0,
    )
    _assert_result_arrays_are_owned_and_immutable(observed)
    _assert_result_arrays_are_owned_and_immutable(metrics)
    with pytest.raises(FrozenInstanceError):
        observed.valid_spike_count = np.zeros((1, 2, 2), dtype=np.int64)
    with pytest.raises(FrozenInstanceError):
        metrics.shuffle_eligible = np.zeros((1, 3, 2), dtype=bool)


def test_observed_segmented_public_results_allow_equal_representative_frequencies() -> None:
    """One available phase frequency legitimately supplies both 8- and 40-Hz bins.

    Representative coordinates therefore require positive finite nondecreasing,
    rather than strictly increasing, float64 values.
    """
    statistic_arguments = _valid_observed_segmented_statistics_arguments()
    statistic_arguments["representative_frequency_hz"] = np.array(
        [8.0, 8.0],
        dtype=np.float64,
    )
    metric_arguments = _valid_observed_segmented_metric_arguments()
    metric_arguments["representative_frequency_hz"] = np.array(
        [8.0, 8.0],
        dtype=np.float64,
    )
    observed = kernel.ObservedSegmentedPPCStatistics(**statistic_arguments)
    metrics = kernel.ObservedSegmentedPPCMetrics(**metric_arguments)
    np.testing.assert_array_equal(observed.representative_frequency_hz, [8.0, 8.0])
    np.testing.assert_array_equal(metrics.representative_frequency_hz, [8.0, 8.0])


@pytest.mark.parametrize(
    ("argument_name", "malformed_value"),
    (
        ("phase_vector_sum", np.ones((1, 2, 2), dtype=np.complex64)),
        ("phase_vector_sum", np.full((1, 2, 2), np.nan + 0.0j, dtype=np.complex128)),
        ("valid_spike_count", np.ones((1, 2, 3), dtype=np.int64)),
        ("valid_spike_count", np.array([[[2, 2], [-1, 0]]], dtype=np.int64)),
        ("contributing_trial_count", np.ones((1, 2, 2), dtype=np.int64)),
        (
            "contributing_trial_count",
            np.array([[[1, 1], [0, 0], [-1, 1]]], dtype=np.int64),
        ),
        ("representative_frequency_hz", np.array([40.0, 8.0], dtype=np.float64)),
        ("representative_frequency_hz", np.array([0.0, 40.0], dtype=np.float64)),
        ("representative_frequency_hz", np.array([8.0, np.nan], dtype=np.float64)),
        ("phase_bin_edges_rad", np.array([-np.pi, 0.0, 0.0], dtype=np.float64)),
        (
            "representative_phase_histogram_count",
            np.ones((1, 3, 2, 2), dtype=np.float64),
        ),
        (
            "representative_phase_histogram_count",
            -np.ones((1, 3, 2, 2), dtype=np.int64),
        ),
    ),
)
def test_observed_segmented_statistics_public_constructor_rejects_malformed_axes_and_coordinates(
    argument_name: str,
    malformed_value: np.ndarray,
) -> None:
    """Public statistics reject malformed axes, counts, and coordinates.

    The valid representative-frequency contract is positive finite
    nondecreasing float64, allowing equal nearest-frequency selections.
    """
    arguments = _valid_observed_segmented_statistics_arguments()
    arguments[argument_name] = malformed_value
    with pytest.raises(ValueError):
        kernel.ObservedSegmentedPPCStatistics(**arguments)


@pytest.mark.parametrize(
    ("argument_name", "malformed_value"),
    (
        ("ppc", np.ones((1, 3, 2), dtype=np.int64)),
        ("resultant_length", np.ones((1, 3, 2), dtype=np.int64)),
        ("preferred_phase_rad", np.ones((1, 2, 2), dtype=np.float64)),
        ("spike_count", np.ones((1, 2, 2), dtype=np.int64)),
        ("spike_count", -np.ones((1, 3, 2), dtype=np.int64)),
        ("computable", np.ones((1, 3, 2), dtype=np.int64)),
        ("reliable", np.ones((1, 3, 2), dtype=np.int64)),
        ("contributing_trial_count", np.ones((1, 2, 2), dtype=np.int64)),
        ("contributing_trial_count", -np.ones((1, 3, 2), dtype=np.int64)),
        ("shuffle_eligible", np.ones((1, 3, 2), dtype=np.int64)),
        ("representative_frequency_hz", np.array([8.0, 40.0], dtype=np.float32)),
        ("phase_bin_edges_rad", np.array([-np.pi, np.pi], dtype=np.float32)),
        (
            "representative_phase_histogram_count",
            np.ones((1, 3, 2, 2), dtype=np.float64),
        ),
        (
            "representative_phase_histogram_count",
            -np.ones((1, 3, 2, 2), dtype=np.int64),
        ),
    ),
)
def test_observed_segmented_metrics_public_constructor_rejects_malformed_axes_and_coordinates(
    argument_name: str,
    malformed_value: np.ndarray,
) -> None:
    """Composed metric public construction enforces dtype/axis contracts too."""
    arguments = _valid_observed_segmented_metric_arguments()
    arguments[argument_name] = malformed_value
    with pytest.raises(ValueError):
        kernel.ObservedSegmentedPPCMetrics(**arguments)
