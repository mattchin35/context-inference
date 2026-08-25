"""Contract tests for offline LFP synchrony summaries and cache-ready arrays."""

from __future__ import annotations

import numpy as np

from src.neural_analysis import lfp_phase_clustering, lfp_synchrony_summary
from src.neural_analysis.lfp_summary_models import FrequencyBandConfig


def _unit_phase(angle_rad: np.ndarray) -> np.ndarray:
    """Return unit complex phase vectors for angles in radians."""

    return np.exp(1j * np.asarray(angle_rad, dtype=float)).astype(np.complex64)


def test_phase_clustering_recovers_locked_and_random_itpc_ispc_and_signed_offset() -> None:
    """ITPC/ISPC should retain magnitudes and the ordered A-minus-B offset."""

    generator = np.random.default_rng(18)
    random_angles = generator.uniform(-np.pi, np.pi, size=(1, 128, 3))
    phase = np.empty((2, 1, 256, 3), dtype=np.complex64)
    phase[0, :, :128] = 1.0
    phase[1, :, :128] = _unit_phase(np.full((1, 128, 3), -0.6))
    phase[0, :, 128:] = _unit_phase(random_angles)
    phase[1, :, 128:] = _unit_phase(random_angles - 0.6)
    valid = np.ones_like(phase, dtype=bool)

    summary = lfp_synchrony_summary.compute_phase_clustering_summary(
        phase_tensor=phase,
        valid_mask=valid,
        condition_names=("locked", "random"),
        condition_masks=np.column_stack(
            ([True] * 128 + [False] * 128, [False] * 128 + [True] * 128)
        ),
        site_pairs=((0, 1),),
    )

    np.testing.assert_allclose(summary.itpc[0], 1.0, atol=1e-6)
    assert float(np.max(summary.itpc[1])) < 0.2
    np.testing.assert_allclose(summary.ispc[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(summary.ispc_phase_offset_rad[0, 0], 0.6, atol=1e-6)
    np.testing.assert_allclose(summary.ispc[1, 0], 1.0, atol=1e-6)


def test_phase_clustering_keeps_pair_specific_validity_for_three_sites() -> None:
    """Each pair should use its own site intersection instead of a global trial set."""

    phase = np.ones((3, 1, 4, 2), dtype=np.complex64)
    valid = np.ones_like(phase, dtype=bool)
    # HPC2 alone lacks trial 2, so only pairs containing HPC2 lose one trial.
    valid[2, 0, 2, 0] = False

    summary = lfp_synchrony_summary.compute_phase_clustering_summary(
        phase_tensor=phase,
        valid_mask=valid,
        condition_names=("all",),
        condition_masks=np.array([[True], [True], [True], [True]]),
        site_pairs=((0, 1), (0, 2), (1, 2)),
    )

    np.testing.assert_array_equal(
        summary.ispc_effective_trial_count[0, :, 0, 0],
        np.array([4, 3, 3]),
    )
    np.testing.assert_array_equal(
        summary.ispc_effective_trial_count[0, :, 0, 1],
        np.array([4, 4, 4]),
    )


def test_trial_plv_uses_direct_half_open_epochs_and_gamma_band_mean_excludes_line_noise() -> None:
    """Whole is direct, zero is after-only, and gamma does not average 58/60/62 Hz."""

    relative_time_s = np.array([-0.004, -0.002, 0.0, 0.002])
    frequencies_hz = np.array([30.0, 58.0, 60.0, 62.0, 64.0])
    phase = np.ones((1, 1, frequencies_hz.size, relative_time_s.size), dtype=np.complex64)
    phase[:, :, :, :2] = -1.0
    valid = np.ones_like(phase, dtype=bool)
    epochs = {"whole": (-0.004, 0.004), "before": (-0.004, 0.0), "after": (0.0, 0.004)}
    gamma = FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),))

    trial_summary = lfp_synchrony_summary.compute_trial_plv_by_frequency(
        relative_phase_complex=phase,
        valid_mask=valid,
        relative_time_s=relative_time_s,
        epoch_windows=epochs,
        minimum_valid_fraction=0.8,
    )
    gamma_fixture = np.array([[[[1.0, 0.0, 0.0, 0.0, 1.0]]]])
    band_mean = lfp_synchrony_summary.aggregate_trial_plv_bands(
        plv_by_frequency=gamma_fixture,
        frequencies_hz=frequencies_hz,
        bands=(gamma,),
    )

    assert trial_summary.plv_by_frequency.shape == (1, 1, 3, 5)
    np.testing.assert_allclose(trial_summary.plv_by_frequency[0, 0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(trial_summary.plv_by_frequency[0, 0, 1:], 1.0, atol=1e-6)
    np.testing.assert_array_equal(trial_summary.valid_sample_count[0, 0, :, 0], np.array([4, 2, 2]))
    np.testing.assert_allclose(band_mean[0, 0, 0, 0], 1.0, atol=1e-6)


def test_trial_plv_coverage_is_independent_of_magnitude_and_requires_two_samples() -> None:
    """Exactly 80% coverage is computable even for low PLV; lower coverage is not."""

    relative_time_s = np.arange(5, dtype=float) / 500.0
    phase = _unit_phase(np.array([[[[0.0, 0.0, np.pi, np.pi, 0.0]]]]))
    valid = np.array([[[[True, True, True, True, False]]]])
    epochs = {"whole": (0.0, 0.01)}

    at_boundary = lfp_synchrony_summary.compute_trial_plv_by_frequency(
        relative_phase_complex=phase,
        valid_mask=valid,
        relative_time_s=relative_time_s,
        epoch_windows=epochs,
        minimum_valid_fraction=0.8,
    )
    below_boundary = lfp_synchrony_summary.compute_trial_plv_by_frequency(
        relative_phase_complex=phase,
        valid_mask=valid & np.array([[[[True, True, True, False, False]]]]),
        relative_time_s=relative_time_s,
        epoch_windows=epochs,
        minimum_valid_fraction=0.8,
    )
    one_sample = lfp_synchrony_summary.compute_trial_plv_by_frequency(
        relative_phase_complex=phase,
        valid_mask=valid & np.array([[[[True, False, False, False, False]]]]),
        relative_time_s=relative_time_s,
        epoch_windows=epochs,
        minimum_valid_fraction=0.0,
    )

    assert at_boundary.computable[0, 0, 0, 0]
    assert at_boundary.valid_sample_count[0, 0, 0, 0] == 4
    assert at_boundary.valid_sample_fraction[0, 0, 0, 0] == 0.8
    assert at_boundary.plv_by_frequency[0, 0, 0, 0] < 1.0
    assert not below_boundary.computable[0, 0, 0, 0]
    assert np.isnan(below_boundary.plv_by_frequency[0, 0, 0, 0])
    assert not one_sample.computable[0, 0, 0, 0]


def test_shared_complex_interpolation_recovers_known_lag_on_exact_500_hz_grid() -> None:
    """The established real/imaginary interpolation contract must preserve cross-site lag."""

    source_time_s = np.arange(-0.02, 0.022, 0.004)
    lag_rad = 0.7
    coefficients = np.stack(
        [
            _unit_phase(2.0 * np.pi * 8.0 * source_time_s),
            _unit_phase(2.0 * np.pi * 8.0 * source_time_s - lag_rad),
        ]
    )[:, np.newaxis, :]

    tensor = lfp_phase_clustering.make_phase_trial_tensor(
        coefficient_times_s=source_time_s,
        unit_phase=coefficients,
        event_times_s=np.array([0.0]),
        trial_indices=np.array([7]),
        window=(-0.01, 0.01),
        output_sample_rate_hz=500.0,
    )
    summary = lfp_synchrony_summary.compute_phase_clustering_summary(
        phase_tensor=tensor.phase,
        valid_mask=tensor.valid,
        condition_names=("all",),
        condition_masks=np.array([[True]]),
        site_pairs=((0, 1),),
    )

    assert tensor.relative_time_s.shape == (10,)
    np.testing.assert_allclose(np.diff(tensor.relative_time_s), 1.0 / 500.0)
    np.testing.assert_allclose(summary.ispc_phase_offset_rad[0, 0, 0], lag_rad, atol=2e-2)


def test_bootstrap_uses_1000_deterministic_condition_resamples() -> None:
    """Bootstrap draws must remain within the selected condition and use the approved count."""

    values = np.array([0.2, 0.4, 0.6, 0.8, 9.0, 9.0])
    selected = np.array([True, True, True, True, False, False])

    first = lfp_synchrony_summary.bootstrap_band_mean(
        trial_values=values,
        trial_mask=selected,
        bootstrap_count=1000,
        seed=44,
    )
    second = lfp_synchrony_summary.bootstrap_band_mean(
        trial_values=values,
        trial_mask=selected,
        bootstrap_count=1000,
        seed=44,
    )

    assert first.bootstrap_count == 1000
    assert first.selected_trial_count == 4
    assert first.estimate == 0.5
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high
    assert first.ci_high < 1.0


def test_small_trial_groups_keep_estimates_and_counts_but_are_flagged_unstable() -> None:
    """Under-10 observations are visible estimates, not silently discarded values."""

    result = lfp_synchrony_summary.bootstrap_band_mean(
        trial_values=np.array([0.2, 0.4, 0.6]),
        trial_mask=np.array([True, True, True]),
        bootstrap_count=1000,
        seed=2,
    )

    assert result.selected_trial_count == 3
    assert result.estimate == 0.4
    assert result.unstable
    assert np.isfinite(result.ci_low)
    assert np.isfinite(result.ci_high)


def test_plv_exemplars_use_percentiles_and_deterministic_earliest_trial_ties() -> None:
    """High/low PLV exemplars should choose the nearest percentile then earliest trial."""

    selection = lfp_synchrony_summary.select_plv_exemplars(
        trial_indices=np.array([11, 4, 9, 2, 7]),
        values=np.array([0.1, 0.3, 0.3, 0.9, 0.9]),
        eligible=np.array([True, True, True, True, True]),
        low_percentile=5.0,
        high_percentile=95.0,
    )

    assert selection.low_trial_index == 11
    assert selection.high_trial_index == 2
    unavailable = lfp_synchrony_summary.select_plv_exemplars(
        trial_indices=np.array([1]),
        values=np.array([0.4]),
        eligible=np.array([True]),
    )
    assert unavailable.low_trial_index is None
    assert unavailable.high_trial_index is None


def test_synchrony_cache_arrays_save_trace_contract_not_full_wavelet_tensor() -> None:
    """Cached exemplars need explicit trace axes/units while wavelet tensors stay temporary."""

    arrays, schema = lfp_synchrony_summary.make_synchrony_cache_arrays(
        relative_time_s=np.arange(-0.01, 0.0, 0.002),
        source_trace=np.ones((2, 3, 5)),
        band_filtered_trace=np.ones((2, 3, 2, 5)),
        hilbert_phase_rad=np.zeros((2, 3, 2, 5)),
        source_voltage_unit="uV",
    )

    assert arrays["source_trace"].shape == (2, 3, 5)
    assert arrays["band_filtered_trace"].shape == (2, 3, 2, 5)
    assert arrays["hilbert_phase_rad"].shape == (2, 3, 2, 5)
    assert schema["source_trace"] == {"axes": ["site", "trial", "time"], "units": "uV"}
    assert schema["band_filtered_trace"] == {
        "axes": ["site", "trial", "band", "time"],
        "units": "uV",
    }
    assert schema["hilbert_phase_rad"] == {
        "axes": ["site", "trial", "band", "time"],
        "units": "rad",
    }
    assert "wavelet_tensor" not in arrays
    assert "wavelet_tensor" not in schema
