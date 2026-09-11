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

    np.testing.assert_array_equal(source_zero.group_offsets, [0, 0, 0, 1, 1, 2, 2])
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
    np.testing.assert_allclose(
        segmented.phase_vector_sum.sum(axis=2),
        oracle.phase_vector_sum,
        rtol=0.0,
        atol=np.maximum(2e-7, 2e-7 * oracle.valid_spike_count),
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
    masks per spike plus one ``unit * 2 + 1`` int64 offset array per source
    trial.  Each gathered coefficient cell concurrently retains left/right
    complex64 coefficients, complex128 interpolation, float64 magnitude,
    complex64 normalized phase, and three Boolean support/validity arrays:
    ``8 + 8 + 16 + 8 + 8 + 3 == 51`` bytes.  Segmented output has two segments
    of one complex128 sum and one int64 count: 48 bytes per edge/unit/frequency.
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
    geometry_bytes = 26 * total_spikes + 2 * (2 * 2 + 1) * 8
    selected_spikes = int(
        source_trial_spike_count[edge_source_trial_position].sum()
    )
    gathered_cells = selected_spikes * 2
    gather_temporary_bytes = gathered_cells * (8 + 8 + 16 + 8 + 8 + 3)
    segmented_edge_statistics_bytes = 3 * 2 * 2 * 48
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


def test_kernel_allocation_estimate_accepts_zero_spike_counts_and_charges_offsets_and_edge_statistics() -> None:
    """An all-zero bounded block still retains each source geometry's offsets and edge outputs."""
    source_trial_spike_count = np.zeros((2, 2, 2), dtype=np.int64)
    edge_source_trial_position = np.array([0, 1, 1], dtype=np.int64)

    estimate = kernel.estimate_segmented_kernel_allocation(
        source_trial_spike_count=source_trial_spike_count,
        edge_source_trial_position=edge_source_trial_position,
        frequency_count=1,
    )

    expected_geometry_bytes = 2 * (2 * 2 + 1) * 8
    expected_edge_statistics_bytes = 3 * 2 * 1 * 48
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
    geometry_bytes = 26 * spike_count + 3 * 8
    segmented_edge_statistics_bytes = 48
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
    geometry_bytes = 26 * spike_count + 3 * 8
    gather_temporary_bytes = 51 * spike_count
    segmented_edge_statistics_bytes = 48
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
    spike_count = (maximum - 3 * 8) // 26
    repeated_edges = np.zeros(27, dtype=np.int64)
    geometry_bytes = 26 * spike_count + 3 * 8
    segmented_edge_statistics_bytes = repeated_edges.size * 48
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
