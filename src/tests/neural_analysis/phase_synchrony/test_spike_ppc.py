"""WP3 UTC spike sampling, within-trial all-pairs PPC and saved inspection oracles."""

from dataclasses import replace

import numpy as np
import pytest

from src.neural_analysis.lfp.phase import normalize_wavelet_phase
from src.neural_analysis.phase_synchrony import computation, preparation
from src.neural_analysis.phase_synchrony.results import PhaseSynchronyConfig, SelectedUnitSpikes
from src.tests.neural_analysis.phase_synchrony.test_phase_metrics import (
    UTC_ORIGIN, drift_inputs, make_phase,
)


def _unit(times_s: np.ndarray, unit_id: str = "ProbeA:17") -> SelectedUnitSpikes:
    """Return one explicit unit with float (spike,) already synchronized UTC seconds."""
    probe, cluster = unit_id.split(":")
    return SelectedUnitSpikes(unit_id, probe, int(cluster), np.asarray(times_s), {})


def test_exact_phase_hit_uses_only_its_own_validity_at_large_utc_epoch():
    """An exact 500-Hz phase timestamp is valid even if both adjacent samples are invalid."""
    phase = make_phase(np.ones((1, 1, 1001)))
    mask = np.zeros(phase.valid.shape, dtype=bool)
    mask[:, :, 1] = True
    timestamp = phase.choice_times_s[0] + phase.relative_time_s[1]
    sampled, valid = preparation.sample_site_phase_at_spikes(replace(phase, valid=mask), 0, np.array([timestamp]))
    np.testing.assert_array_equal(valid, [[True]])
    np.testing.assert_allclose(sampled, [[1]], rtol=0, atol=1e-7)


def test_phase_interpolation_crosses_wrap_using_normalized_vectors_not_amplitudes():
    """Unequal-amplitude coefficients normalize before circular interpolation across +/-pi."""
    phase = make_phase(np.ones((1, 1, 1001)))
    coefficients = np.array([np.exp(1j * np.deg2rad(179)), 9 * np.exp(1j * np.deg2rad(-179))])
    vectors, _ = normalize_wavelet_phase(coefficients, minimum_relative_magnitude=0.0)
    phase.phase[0, 0, :2] = vectors
    midpoint = phase.choice_times_s[0] - 1 + 0.001
    sampled, valid = preparation.sample_site_phase_at_spikes(phase, 0, np.array([midpoint]))
    assert valid[0, 0]
    np.testing.assert_allclose(sampled[0, 0], -1 + 0j, rtol=0, atol=5e-6)


def test_phase_gaps_exact_neighbors_and_no_extrapolation_are_explicit():
    """Only exact valid samples or two adjacent valid neighbors contribute frequency-specific spikes."""
    phase = make_phase(np.ones((1, 1, 1001)))
    mask = phase.valid.copy()
    mask[:, :, 1] = False
    relative = np.array([-1.002, -1, -0.999, -0.998, -0.997, -0.996, 1.002])
    sampled, valid = preparation.sample_site_phase_at_spikes(
        replace(phase, valid=mask), 0, phase.choice_times_s[0] + relative,
    )
    np.testing.assert_array_equal(valid, [[False, True, False, False, False, True, False]])
    assert not np.isfinite(sampled[~valid]).any() or np.all(sampled[~valid] == 0)


def test_real_right_guard_supports_spikes_near_half_open_metric_end():
    """A spike at choice+0.999 s uses 0.998 and real +1.000 support; guard isn't a metric sample."""
    phase = make_phase(np.ones((1, 1, 1001)))
    phase.phase[:, :, -1] = 1j
    spike = phase.choice_times_s[0] + 0.999
    sampled, valid = preparation.sample_site_phase_at_spikes(phase, 0, np.array([spike]))
    assert valid[0, 0]
    np.testing.assert_allclose(sampled[0, 0], (1 + 1j) / np.sqrt(2), rtol=0, atol=8e-5)
    mask = phase.valid.copy()
    mask[:, :, -1] = False
    _, missing_guard_valid = preparation.sample_site_phase_at_spikes(replace(phase, valid=mask), 0, np.array([spike]))
    assert not missing_guard_valid[0, 0]


def test_opposing_vector_interpolant_is_undefined():
    """Exactly opposing unit vectors average to zero at a midpoint, so no phase is invented."""
    phase = make_phase(np.ones((1, 1, 1001)))
    phase.phase[0, 0, 1] = -1
    # Use Unix epoch zero to isolate the midpoint rule from large-UTC rounding
    # while keeping the required 500-Hz grid and matching metric mask.
    phase = replace(phase, choice_times_s=np.zeros(1))
    target = -0.999
    _, valid = preparation.sample_site_phase_at_spikes(phase, 0, np.array([target]))
    assert not valid[0, 0]


def test_drift_corrected_phase_at_off_grid_utc_spikes_matches_analytic_ppc(drift_inputs):
    """Prepared drifting LFP supplies actual UTC spike phases/PPC without a second clock correction."""
    source, _, choices, config = drift_inputs
    phase = preparation.prepare_site_phase(source, choices, config)
    offsets = np.array([-0.812345, -0.456789, 0.123456, 0.789123])
    spikes = np.concatenate([choice + offsets for choice in choices.choice_time])
    unit = _unit(spikes)
    result = computation.compute_selected_unit_ppc(unit, phase, config)
    for trial, choice in enumerate(choices.choice_time):
        utc_spikes = choice + offsets
        actual, valid = preparation.sample_site_phase_at_spikes(phase, trial, utc_spikes)
        analytic = np.exp(1j * (2 * np.pi * 8 * (utc_spikes - UTC_ORIGIN) + 0.6))
        assert valid.all()
        assert np.max(np.abs(np.angle(actual[0] * np.conj(analytic)))) < 0.02
        expected = (abs(np.sum(analytic)) ** 2 - len(analytic)) / (len(analytic) * (len(analytic) - 1))
        assert result.ppc[trial, 0] == pytest.approx(expected, abs=0.01)
    np.testing.assert_array_equal(result.spike_count, [[4], [4]])
    np.testing.assert_array_equal(result.excluded_spike_count, [[0], [0]])


def test_two_spike_ppc_equals_cosine_and_preserves_negative_values():
    """Two selected spike phases 0/pi yield PPC -1 with no 50-spike reliability cutoff."""
    phase = make_phase(np.ones((1, 1, 1001)))
    phase.phase[:, :, 501] = -1
    spikes = phase.choice_times_s[0] + np.array([0, 0.002])
    result = computation.compute_selected_unit_ppc(_unit(spikes), phase, PhaseSynchronyConfig())
    assert result.valid[0, 0]
    assert result.spike_count[0, 0] == 2
    assert result.ppc[0, 0] == pytest.approx(np.cos(np.pi), abs=1e-7)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_spikes_is_unavailable_not_zero(count):
    """Each (trial,frequency) count below two retains its count and an undefined PPC."""
    phase = make_phase(np.ones((1, 1, 1001)))
    spikes = np.array([phase.choice_times_s[0]])[:count]
    result = computation.compute_selected_unit_ppc(_unit(spikes), phase, PhaseSynchronyConfig())
    assert result.spike_count[0, 0] == count
    assert not result.valid[0, 0] and np.isnan(result.ppc[0, 0])


def test_half_open_counts_selected_spike_times_and_polar_histograms_are_saved():
    """Select [choice-1,choice+1) exactly and save UTC spikes plus fixed (frequency,trial,bin) histograms."""
    phase = make_phase(np.ones((1, 1, 1001)))
    offsets = np.array([-1.001, -1, -0.123456, 0, 0.999, 1])
    spikes = phase.choice_times_s[0] + offsets
    phase.phase[0, 0, :] = np.exp(1j * np.linspace(-np.pi, np.pi, 1001))
    result = computation.compute_selected_unit_ppc(_unit(spikes), phase, PhaseSynchronyConfig())
    np.testing.assert_array_equal(result.spike_count, [[4]])
    np.testing.assert_array_equal(result.spike_times_by_trial[0], spikes[1:5])
    assert result.polar_histogram_count.shape == (1, 1, 24)
    np.testing.assert_allclose(result.phase_bin_edges_rad, np.linspace(-np.pi, np.pi, 25), rtol=0, atol=0)
    sampled, valid = preparation.sample_site_phase_at_spikes(phase, 0, spikes[1:5])
    expected, _ = np.histogram(np.angle(sampled[0, valid[0]]), bins=result.phase_bin_edges_rad)
    np.testing.assert_array_equal(result.polar_histogram_count[0, 0], expected)
    np.testing.assert_array_equal(result.polar_histogram_count.sum(axis=2).T, result.spike_count)
    assert result.unit_id == "ProbeA:17" and result.site_id == "SiteA"
    np.testing.assert_array_equal(result.trial_ids, phase.trial_ids)
    np.testing.assert_array_equal(result.row_positions, phase.row_positions)
    np.testing.assert_array_equal(result.choice_times_s, phase.choice_times_s)


def test_frequency_specific_exclusions_and_band_count_range_are_not_pooled():
    """Three frequencies retain 2/3/2 spikes; theta labels count range 2-3 and saves exact counts."""
    phase = make_phase(np.ones((3, 1, 1001)), frequencies=np.array([6., 8., 10.]))
    mask = phase.valid.copy()
    mask[0, 0, 502] = False
    mask[2, 0, 500] = False
    spikes = phase.choice_times_s[0] + np.array([0, 0.002, 0.004])
    result = computation.compute_selected_unit_ppc(_unit(spikes), replace(phase, valid=mask), PhaseSynchronyConfig())
    np.testing.assert_array_equal(result.spike_count, [[2, 3, 2]])
    np.testing.assert_array_equal(result.excluded_spike_count, [[1, 0, 1]])
    assert result.band_spike_count_min[0, 0] == 2
    assert result.band_spike_count_max[0, 0] == 3
    assert result.band_ppc[0, 0] == pytest.approx(1)
    np.testing.assert_array_equal(result.polar_histogram_count.sum(axis=2).T, result.spike_count)


def test_units_and_trials_remain_independent_including_same_cluster_number():
    """Probe-qualified cluster 17 selections and neighboring choices cannot pool spikes to meet N=2."""
    phase = make_phase(np.ones((1, 2, 1001)))
    first, second = phase.choice_times_s
    selected = _unit(np.array([first - 0.5, first + 0.5, second + 0.1]))
    result = computation.compute_selected_unit_ppc(selected, phase, PhaseSynchronyConfig())
    np.testing.assert_array_equal(result.spike_count, [[2], [1]])
    assert result.valid[0, 0] and not result.valid[1, 0]
    other = computation.compute_selected_unit_ppc(_unit(np.array([first]), "ProbeB:17"), phase, PhaseSynchronyConfig())
    assert other.unit_id == "ProbeB:17" and not other.valid.any()
    np.testing.assert_array_equal(other.spike_count, [[1], [0]])
