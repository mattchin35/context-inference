"""WP3 strict-support phase preparation and within-window LFP PLV/PPC oracles."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp import phase as existing_phase
from src.neural_analysis.phase_synchrony import computation, preparation
from src.neural_analysis.phase_synchrony.results import (
    ContinuousSiteLFP, PhaseSynchronyConfig, SitePhase,
)


UTC_ORIGIN = 1_800_000_000.0


def make_phase(phase: np.ndarray, *, frequencies: np.ndarray | None = None, site_id: str = "SiteA") -> SitePhase:
    """Build complex (frequency,trial,time) unit phase, including one real 500-Hz right guard."""
    phase = np.asarray(phase, dtype=np.complex64)
    frequencies = np.array([8.0]) if frequencies is None else frequencies
    trial_count, time_count = phase.shape[1:]
    times = -1.0 + np.arange(time_count) / 500.0
    return SitePhase(
        site_id=site_id, probe_id="ProbeA" if site_id == "SiteA" else "ProbeB",
        saved_channel_index=3, frequency_hz=frequencies, phase=phase,
        valid=np.ones(phase.shape, dtype=bool), relative_time_s=times,
        metric_mask=(times >= -1) & (times < 1), sample_rate_hz=500.0,
        trial_ids=np.arange(trial_count) * 7 + 21,
        row_positions=np.arange(trial_count) * 3 + 2,
        choice_times_s=UTC_ORIGIN + 20.125 + np.arange(trial_count) * 60,
        meta={
            "phase_support_semantics": "strict_support_v1",
            "phase_alignment_semantics": "irig_uniform_utc_pre_morlet_v1",
            "numerical_amplitude_threshold": 0.0, "absolute_amplitude_threshold": 0.0,
            "interval": "primary", "interval_bounds_s": (-1.0, 1.0),
            "phase_difference_convention": "A minus B",
        },
    )


def make_native_source(*, offset_s: float = 0.0, drift: float = 1.0, phase_offset: float = 0.0,
                       duration_s: float = 100.0, site_id: str = "SiteA") -> ContinuousSiteLFP:
    """Return calibrated 1000-Hz cosine samples under an independent affine UTC clock.

    Native index is (sample,), voltage uV is cos(2*pi*8*UTC-relative seconds +
    phase_offset radians). IRIG anchors independently encode offset/drift.
    """
    samples = np.arange(round(duration_s * 1000) + 1)
    utc_relative = offset_s + samples / (1000.0 * drift)
    values = np.cos(2 * np.pi * 8 * utc_relative + phase_offset)
    return ContinuousSiteLFP(
        site_id, "ProbeA" if site_id == "SiteA" else "ProbeB", 3,
        samples, values, np.ones(samples.shape, dtype=bool), 1000.0,
        pd.DataFrame({"sample_ix": samples[[0, -1]], "utc_unix": UTC_ORIGIN + utc_relative[[0, -1]]}),
        {"voltage_unit": "uV", "notch_enabled": False, "synchronization_id": f"{offset_s}:{drift}"},
    )


@pytest.fixture
def drift_inputs():
    """Return two offset/drifting native sources, two far-separated UTC choices and one-frequency config."""
    choices = pd.DataFrame({
        "trial_id": [21, 28], "row_position": [2, 5],
        "choice_time": UTC_ORIGIN + np.array([20.125, 80.12345]),
    })
    config = PhaseSynchronyConfig(notch_enabled=False, phase_frequencies_hz=(8.0,))
    return (
        make_native_source(offset_s=0.019, drift=1.004, phase_offset=0.6),
        make_native_source(offset_s=-0.023, drift=0.997, phase_offset=-0.2, site_id="SiteB"),
        choices, config,
    )


@pytest.mark.parametrize("interval,count", [("primary", 1000), ("secondary", 2000)])
def test_phase_preparation_keeps_exact_metric_grid_real_guard_and_provenance(drift_inputs, interval, count):
    """Prepared complex (frequency,trial,time) has 1000/2000 metric observations plus actual guard support."""
    source, _, choices, config = drift_inputs
    result = preparation.prepare_site_phase(source, choices, config, interval=interval)
    assert result.phase.shape == result.valid.shape == (1, 2, count + 1)
    assert result.phase.dtype == np.complex64
    assert result.valid.dtype == bool
    assert result.sample_rate_hz == 500.0
    assert np.count_nonzero(result.metric_mask) == count
    half_width = 1 if interval == "primary" else 2
    np.testing.assert_array_equal(result.relative_time_s, -half_width + np.arange(count + 1) / 500)
    assert not result.metric_mask[-1] and result.valid[:, :, -1].all()
    np.testing.assert_array_equal(result.trial_ids, choices.trial_id)
    np.testing.assert_array_equal(result.row_positions, choices.row_position)
    np.testing.assert_array_equal(result.choice_times_s, choices.choice_time)
    assert result.meta["phase_support_semantics"] == "strict_support_v1"
    assert result.meta["phase_alignment_semantics"] == "irig_uniform_utc_pre_morlet_v1"
    assert result.meta["numerical_amplitude_threshold"] == 0.0
    assert result.meta["absolute_amplitude_threshold"] == 0.0
    assert result.meta["morlet"] == {"gaussian_width": 1.5, "window_length": 1.0, "precision": 16, "norm": "l1"}
    assert result.meta["padding_s"] >= 1.0
    assert result.meta["blocks"]
    assert all(np.isfinite(block["utc_origin_s"]) and len(block["support_bounds_s"]) == 2 for block in result.meta["blocks"])


def test_full_drift_preparation_recovers_analytic_phases_and_pair_metrics(drift_inputs):
    """Far-from-block-start phases recover a known 0.8-rad offset after actual UTC waveform alignment."""
    source_a, source_b, choices, config = drift_inputs
    phase_a = preparation.prepare_site_phase(source_a, choices, config)
    phase_b = preparation.prepare_site_phase(source_b, choices, config)
    for prepared, offset in ((phase_a, 0.6), (phase_b, -0.2)):
        analytic_time = (choices.choice_time.to_numpy() - UTC_ORIGIN)[:, None] + prepared.relative_time_s
        analytic = np.exp(1j * (2 * np.pi * 8 * analytic_time + offset))
        error = np.angle(prepared.phase[0] * np.conj(analytic))
        # Existing Morlet bias is about 0.00077 rad at these settings; 0.02
        # allows waveform/circular interpolation while rejecting nominal clocks.
        assert prepared.valid.all()
        assert np.max(np.abs(error)) < 0.02
    metrics = computation.compute_lfp_phase_summary(phase_a, phase_b, config)
    np.testing.assert_allclose(metrics.phase_offset_rad, 0.8, rtol=0, atol=0.02)
    assert np.all(metrics.plv > 0.999) and np.all(metrics.ppc > 0.999)
    np.testing.assert_array_equal(metrics.sample_count, [[1000], [1000]])


def test_source_gap_cannot_be_interpolated_into_valid_phase(drift_inputs):
    """A masked native source gap remains unavailable; another trial retains complete support."""
    source, _, _, config = drift_inputs
    source = make_native_source(duration_s=20)
    mask = source.valid.copy()
    mask[10_200] = False
    choices = pd.DataFrame({"trial_id": [21, 28], "row_position": [2, 5], "choice_time": UTC_ORIGIN + np.array([5, 10])})
    phase = preparation.prepare_site_phase(replace(source, valid=mask), choices, config)
    assert phase.valid[:, 0, phase.metric_mask].all()
    near_gap = np.abs(phase.relative_time_s - 0.2) < 0.01
    assert not phase.valid[:, 1, near_gap].any()
    assert phase.valid[:, 1, :].any()


def test_valid_primary_survives_secondary_support_loss_and_unrelated_edge(drift_inputs):
    """Missing padded secondary/edge support does not discard valid primary data from another choice."""
    _, _, _, config = drift_inputs
    source = make_native_source(offset_s=-0.1, duration_s=4.2)
    choices = pd.DataFrame({"trial_id": [21, 28], "row_position": [2, 5], "choice_time": UTC_ORIGIN + np.array([0.05, 2.0])})
    primary = preparation.prepare_site_phase(source, choices, config, interval="primary")
    secondary = preparation.prepare_site_phase(source, choices, config, interval="secondary")
    assert primary.valid[:, 1, primary.metric_mask].all()
    assert not secondary.valid[:, 1, secondary.metric_mask].all()
    assert not primary.valid[:, 0, primary.metric_mask].all()


def _tiny_coefficient_transform(time_s, lfp_values, sample_rate_hz, frequencies_hz, **kwargs):
    """Inject known (frequency,time) complex coefficients; preserve real supplied source axes/units."""
    factor = round(sample_rate_hz / 500)
    times = np.asarray(time_s)[::factor]
    coefficients = np.full((len(frequencies_hz), len(times)), 1e-20 + 0j, dtype=np.complex64)
    coefficients[:, 0] = 1
    middle = len(times) // 2
    coefficients[:, middle] = 0
    coefficients[:, middle + 2] = np.nan + 1j * np.nan
    assert kwargs.get("notch_60_hz", False) is False
    return existing_phase.WaveletCoefficientResult(
        times, np.asarray(frequencies_hz), coefficients, sample_rate_hz / factor,
        np.asarray(time_s), np.asarray(lfp_values), sample_rate_hz,
    )


def test_explicit_zero_numerical_threshold_retains_tiny_positive_coefficients(drift_inputs, monkeypatch):
    """Positive 1e-20 coefficients remain valid at explicit 0.0; zero/NaN remain unavailable."""
    source, _, choices, config = drift_inputs
    choices = choices.iloc[:1]
    monkeypatch.setattr(existing_phase, "compute_wavelet_coefficients", _tiny_coefficient_transform)
    phase = preparation.prepare_site_phase(source, choices, config)
    metric_valid = phase.valid[0, 0, phase.metric_mask]
    assert np.count_nonzero(metric_valid) > 900
    assert np.count_nonzero(~metric_valid) >= 2
    np.testing.assert_allclose(np.abs(phase.phase[phase.valid]), 1, rtol=0, atol=1e-6)
    gated = preparation.prepare_site_phase(source, choices, replace(config, numerical_amplitude_threshold=1e-12))
    assert not gated.valid[:, :, gated.metric_mask].any()


def test_known_nonzero_phase_offset_has_correct_sign_plv_ppc_and_guard_count():
    """Unit phase A minus B at pi/3 has PLV/PPC one; a contradictory guard never enters M."""
    left = make_phase(np.full((1, 1, 1001), np.exp(1j * np.pi / 3)))
    right = make_phase(np.ones((1, 1, 1001)), site_id="SiteB")
    left.phase[:, :, -1] = -1
    result = computation.compute_lfp_phase_summary(left, right, PhaseSynchronyConfig())
    np.testing.assert_allclose(result.plv, [[1]], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.ppc, [[1]], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.phase_offset_rad, [[np.pi / 3]], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.sample_count, [[1000]])


def test_balanced_opposing_phases_preserve_negative_ppc_and_undefined_offset():
    """M=1000 balanced +1/-1 observations give PLV zero and PPC -1/999, no direction."""
    left = make_phase(np.tile([1, -1], 501)[:1001][None, None, :])
    right = make_phase(np.ones((1, 1, 1001)), site_id="SiteB")
    result = computation.compute_lfp_phase_summary(left, right, PhaseSynchronyConfig())
    assert result.plv[0, 0] == 0
    assert result.ppc[0, 0] == pytest.approx(-1 / 999)
    assert np.isnan(result.phase_offset_rad[0, 0])


def test_lfp_ppc_identity_is_applied_per_frequency_before_band_mean():
    """Different frequency PLVs satisfy individual PPC identities; band PPC is their mean."""
    angles = np.zeros((3, 1, 1001))
    angles[1, 0, :500] = np.pi / 2
    angles[2, 0, ::2] = np.pi
    left = make_phase(np.exp(1j * angles), frequencies=np.array([6., 8., 10.]))
    right = make_phase(np.ones(angles.shape), frequencies=left.frequency_hz, site_id="SiteB")
    result = computation.compute_lfp_phase_summary(left, right, PhaseSynchronyConfig())
    expected_ppc = (1000 * result.plv ** 2 - 1) / 999
    np.testing.assert_allclose(result.ppc, expected_ppc, rtol=0, atol=2e-6)
    assert result.band_ppc[0, 0] == pytest.approx(np.mean(result.ppc[0]), abs=1e-6)
    assert abs(result.band_ppc[0, 0] - (1000 * result.band_plv[0, 0] ** 2 - 1) / 999) > 0.05


@pytest.mark.parametrize("supported_count", [0, 1, 999])
def test_incomplete_lfp_phase_support_is_unavailable_without_eighty_percent_rule(supported_count):
    """Any missing metric phase makes LFP PLV/PPC unavailable; retain actual M, including <2."""
    left = make_phase(np.ones((1, 2, 1001)))
    right = make_phase(np.ones((1, 2, 1001)), site_id="SiteB")
    mask = left.valid.copy()
    mask[0, 0, supported_count:1000] = False
    result = computation.compute_lfp_phase_summary(replace(left, valid=mask), right, PhaseSynchronyConfig())
    assert not result.valid[0, 0] and result.valid[1, 0]
    assert result.sample_count[0, 0] == supported_count
    assert np.isnan(result.plv[0, 0]) and np.isnan(result.ppc[0, 0])


def test_band_summaries_keep_fixed_frequency_set_and_exclude_gamma_line_centers():
    """One missing retained frequency invalidates its band; excluded centers do not."""
    frequencies = np.arange(2, 102, 2, dtype=float)
    left = make_phase(np.ones((50, 2, 1001)), frequencies=frequencies)
    right = make_phase(np.ones(left.phase.shape), frequencies=frequencies, site_id="SiteB")
    mask = left.valid.copy()
    mask[np.isin(frequencies, [58, 60, 62])] = False
    mask[frequencies == 56, 1] = False
    result = computation.compute_lfp_phase_summary(replace(left, valid=mask), right, PhaseSynchronyConfig())
    np.testing.assert_array_equal(result.band_valid, [[True, True], [True, False]])
    assert np.isnan(result.band_ppc[1, 1])
