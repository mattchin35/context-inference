"""WP2 Welch, integrated-power, strict-band and shared-spectrum oracles."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

from src.neural_analysis.lfp.config import FrequencyBandConfig
from src.neural_analysis.lfp.power import mean_band_power_linear
from src.neural_analysis.phase_synchrony import computation
from src.neural_analysis.phase_synchrony.results import PhaseSynchronyConfig, SiteWaveforms


def _waveforms(
    values_uv: np.ndarray,
    *,
    rate_hz: float = 2500.0,
    interval: str = "primary",
    site_id: str = "SiteA",
) -> SiteWaveforms:
    """Build already preprocessed float (trial,time) uV with native relative-second grid."""
    values = np.asarray(values_uv, dtype=float)
    half_width = 1.0 if interval == "primary" else 2.0
    trials, count = values.shape
    return SiteWaveforms(
        site_id=site_id, probe_id="ProbeA" if site_id == "SiteA" else "ProbeB",
        saved_channel_index=3 if site_id == "SiteA" else 7,
        values_uv=values, valid=np.ones(values.shape, dtype=bool),
        relative_time_s=-half_width + np.arange(count) / rate_hz,
        choice_times_s=1_800_000_000.125 + np.arange(trials) * 10.0,
        trial_ids=np.arange(trials) * 7 + 21, row_positions=np.arange(trials) * 3 + 2,
        sample_rate_hz=rate_hz,
        meta={
            "voltage_unit": "uV", "notch_enabled": False,
            "waveform_alignment_semantics": "irig_uniform_utc_v1",
            "interval": interval, "interval_bounds_s": (-half_width, half_width),
            "source": {"identity": f"synthetic-{site_id}"},
            "synchronization": {"identity": f"clock-{site_id}"},
        },
    )


@pytest.fixture
def config() -> PhaseSynchronyConfig:
    """Return explicit no-notch settings; waveform fixtures are already preprocessed."""
    return PhaseSynchronyConfig(notch_enabled=False)


@pytest.fixture
def waveform_pair() -> tuple[SiteWaveforms, SiteWaveforms]:
    """Return seeded (2,5000) uV trials: correlated first pair, independent second pair."""
    rng = np.random.default_rng(1203)
    times = np.arange(5000) / 2500.0
    shared = 3 * np.sin(2 * np.pi * 8 * times) + np.sin(2 * np.pi * 40 * times)
    left = rng.normal(size=(2, 5000)) + shared
    right = np.vstack((0.6 * left[0] + rng.normal(scale=0.5, size=5000), rng.normal(size=5000)))
    return _waveforms(left), _waveforms(right, site_id="SiteB")


@pytest.mark.parametrize("interval,count,segments", [("primary", 5000, 7), ("secondary", 10000, 15)])
def test_native_psd_settings_counts_and_actual_density_match_scipy(config, interval, count, segments):
    """(trial,time) uV gives (trial,frequency) uV^2/Hz with exact 0.5-s segments."""
    values = np.random.default_rng(17).normal(size=(2, count))
    waveforms = _waveforms(values, interval=interval)
    result = computation.compute_site_psd(waveforms, config)
    expected_hz, expected = signal.welch(
        values, fs=2500.0, window="hann_periodic", nperseg=1250, noverlap=625,
        nfft=1250, detrend="constant", scaling="density", average="mean", axis=-1,
    )
    np.testing.assert_array_equal(result.frequency_hz, expected_hz)
    np.testing.assert_allclose(result.psd_uv2_per_hz, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(result.sample_count, [count, count])
    np.testing.assert_array_equal(result.segment_count, [segments, segments])
    assert result.valid.tolist() == [True, True]
    assert result.meta["psd_unit"] == "uV^2/Hz"
    assert result.meta["welch"] == {
        "window": "hann_periodic", "detrend": "constant", "nperseg": 1250,
        "noverlap": 625, "nfft": 1250, "scaling": "density", "average": "mean",
    }
    np.testing.assert_array_equal(result.trial_ids, waveforms.trial_ids)
    np.testing.assert_array_equal(result.row_positions, waveforms.row_positions)
    np.testing.assert_array_equal(result.choice_times_s, waveforms.choice_times_s)


def test_noncanonical_native_rate_keeps_actual_welch_coordinates(config):
    """A 1001-Hz source keeps its actual Hz bins; no interpolation to a 2-Hz grid."""
    waveforms = _waveforms(np.random.default_rng(32).normal(size=(1, 2002)), rate_hz=1001.0)
    result = computation.compute_site_psd(waveforms, config)
    expected_hz, _ = signal.welch(waveforms.values_uv[0], fs=1001, nperseg=500, noverlap=250)
    np.testing.assert_array_equal(result.frequency_hz, expected_hz)
    assert result.frequency_hz[1] == pytest.approx(2.002)


def test_doubling_voltage_quadruples_psd_and_integrated_power(config, waveform_pair):
    """Voltage uV scales by two; density uV^2/Hz and integrated uV^2 scale by four."""
    waveforms, _ = waveform_pair
    source = computation.compute_site_psd(waveforms, config)
    doubled = computation.compute_site_psd(replace(waveforms, values_uv=2 * waveforms.values_uv), config)
    np.testing.assert_allclose(doubled.psd_uv2_per_hz, 4 * source.psd_uv2_per_hz, rtol=1e-12)
    power = computation.compute_band_power(source, config)
    doubled_power = computation.compute_band_power(doubled, config)
    np.testing.assert_allclose(doubled_power.band_power_integrated, 4 * power.band_power_integrated, rtol=1e-12)
    assert power.meta["power_unit"] == "uV^2"
    np.testing.assert_array_equal(power.trial_ids, waveforms.trial_ids)
    np.testing.assert_array_equal(power.row_positions, waveforms.row_positions)
    np.testing.assert_array_equal(power.choice_times_s, waveforms.choice_times_s)


def test_constant_density_integrates_exact_retained_bandwidth(config, waveform_pair):
    """Artificial constant density q uV^2/Hz integrates to theta 4q and gamma 46q uV^2."""
    waveforms, _ = waveform_pair
    psd = computation.compute_site_psd(waveforms, config)
    density = np.full(psd.psd_uv2_per_hz.shape, 3.5)
    result = computation.compute_band_power(replace(psd, psd_uv2_per_hz=density), config)
    np.testing.assert_allclose(result.band_power_integrated, np.tile([14.0, 161.0], (2, 1)), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.bandwidth_hz, [4.0, 46.0])
    assert result.band_names == ("theta", "gamma")
    assert result.valid.all()


def test_integrated_power_matches_existing_disjoint_interval_helper(config, waveform_pair):
    """Irregular smooth density is integrated using existing mean-density times bandwidth."""
    source, _ = waveform_pair
    psd = computation.compute_site_psd(source, config)
    frequencies = psd.frequency_hz
    density = np.vstack((1 + frequencies * 0.02, 3 + 0.0003 * frequencies ** 2))
    result = computation.compute_band_power(replace(psd, psd_uv2_per_hz=density), config)
    for band_index, band in enumerate(config.bands):
        mean, width = mean_band_power_linear(density, frequencies, band)
        np.testing.assert_allclose(result.band_power_integrated[:, band_index], mean * width, rtol=1e-12)


@pytest.mark.parametrize("excluded_value", [1e15, np.nan])
def test_excluded_gamma_bins_cannot_contaminate_retained_boundaries(config, waveform_pair, excluded_value):
    """Changing excluded 58/60/62-Hz density values leaves the 46-Hz gamma integral intact."""
    source, _ = waveform_pair
    psd = computation.compute_site_psd(source, config)
    density = np.full_like(psd.psd_uv2_per_hz, 2.0)
    density[:, np.isin(psd.frequency_hz, [58, 60, 62])] = excluded_value
    result = computation.compute_band_power(replace(psd, psd_uv2_per_hz=density), config)
    np.testing.assert_allclose(result.band_power_integrated, np.tile([8, 92], (2, 1)), rtol=0, atol=1e-12)
    assert result.valid.all()


def test_missing_retained_density_invalidates_only_affected_band_and_trial(config, waveform_pair):
    """A required 56-Hz PSD value missing in one trial cannot become zero or finite-only mean."""
    source, _ = waveform_pair
    psd = computation.compute_site_psd(source, config)
    density = np.ones_like(psd.psd_uv2_per_hz)
    density[1, psd.frequency_hz == 56] = np.nan
    result = computation.compute_band_power(replace(psd, psd_uv2_per_hz=density), config)
    np.testing.assert_array_equal(result.valid, [[True, True], [True, False]])
    assert np.isnan(result.band_power_integrated[1, 1])
    assert result.band_power_integrated[0, 1] == pytest.approx(46)


def test_complete_waveform_support_is_required_per_trial(config, waveform_pair):
    """One masked finite voltage makes that trial PSD unavailable; no retained-segment rescue."""
    source, _ = waveform_pair
    mask = source.valid.copy()
    mask[1, 7] = False
    result = computation.compute_site_psd(replace(source, valid=mask), config)
    assert result.valid.tolist() == [True, False]
    assert np.isfinite(result.psd_uv2_per_hz[0]).all()
    assert np.isnan(result.psd_uv2_per_hz[1]).all()
    assert result.segment_count[1] == 0


def test_true_zero_trace_has_valid_zero_power_but_undefined_coherence(config):
    """Physical zero uV is valid zero uV^2 power and has no positive spectral denominator."""
    source = _waveforms(np.zeros((1, 5000)))
    psd = computation.compute_site_psd(source, config)
    power = computation.compute_band_power(psd, config)
    assert psd.valid.all() and power.valid.all()
    np.testing.assert_array_equal(power.band_power_integrated, [[0, 0]])
    coherence = computation.compute_welch_coherence(source, replace(source, site_id="SiteB"), config)
    assert not coherence.valid.any()
    assert np.isnan(coherence.coherence).all()
    assert not coherence.band_valid.any()


def test_identical_segment_psds_give_duration_invariant_average_power(config):
    """Repeated exact 8/40-Hz segment spectra yield equal 2/4-second average power, not energy."""
    products = []
    for interval, count in (("primary", 5000), ("secondary", 10000)):
        time_s = np.arange(count) / 2500.0
        voltage = np.sin(2 * np.pi * 8 * time_s) + 2 * np.sin(2 * np.pi * 40 * time_s)
        psd = computation.compute_site_psd(_waveforms(voltage[None, :], interval=interval), config)
        products.append(computation.compute_band_power(psd, config).band_power_integrated)
    np.testing.assert_allclose(products[0], products[1], rtol=1e-12, atol=1e-12)


def test_psd_does_not_apply_a_second_window_notch(config, waveform_pair):
    """Already processed (trial,time) uV rows must reach Welch unchanged even when notch is enabled."""
    source, _ = waveform_pair
    metadata = dict(source.meta, notch_enabled=True, notch_quality_factor=30.0)
    psd = computation.compute_site_psd(replace(source, meta=metadata), replace(config, notch_enabled=True))
    _, expected = signal.welch(source.values_uv, fs=2500, nperseg=1250, noverlap=625, axis=-1)
    np.testing.assert_allclose(psd.psd_uv2_per_hz, expected, rtol=1e-12, atol=1e-15)


def test_coherence_matches_scipy_within_each_trial(config, waveform_pair):
    """Matched Welch autos/cross spectra produce SciPy magnitude-squared coherence per trial."""
    left, right = waveform_pair
    result = computation.compute_welch_coherence(left, right, config)
    assert result.coherence.shape == result.valid.shape == (2, 626)
    for trial in range(2):
        frequencies, expected = signal.coherence(
            left.values_uv[trial], right.values_uv[trial], fs=2500,
            window="hann_periodic", nperseg=1250, noverlap=625, nfft=1250, detrend="constant",
        )
        np.testing.assert_array_equal(result.frequency_hz, frequencies)
        np.testing.assert_allclose(result.coherence[trial], expected, rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(result.segment_count, [7, 7])
    np.testing.assert_array_equal(result.trial_ids, left.trial_ids)
    np.testing.assert_array_equal(result.row_positions, left.row_positions)
    np.testing.assert_array_equal(result.choice_times_s, left.choice_times_s)


def test_opposite_trial_cross_spectra_are_not_pooled(config, waveform_pair):
    """Two trials with opposite offsets each have coherence one; their cross spectra cannot cancel."""
    source, _ = waveform_pair
    left = replace(source, values_uv=np.tile(source.values_uv[0], (2, 1)))
    right = replace(left, site_id="SiteB", values_uv=np.vstack((left.values_uv[0], -left.values_uv[1])))
    result = computation.compute_welch_coherence(left, right, config)
    np.testing.assert_allclose(result.coherence[result.valid], 1.0, rtol=1e-12, atol=1e-12)


def test_undefined_dc_does_not_invalidate_positive_retained_bands(config):
    """Exact-zero DC denominator remains unavailable while positive theta/gamma bins stay valid."""
    # Periodic alternation has exactly zero Welch DC in the installed SciPy;
    # nonzero retained-bin leakage still has a mathematically defined ratio.
    voltage = np.tile([1.0, -1.0], 2500)[None, :]
    source = _waveforms(voltage)
    result = computation.compute_welch_coherence(source, replace(source, site_id="SiteB"), config)
    dc = result.frequency_hz == 0
    assert not result.valid[:, dc].any()
    assert np.isnan(result.coherence[:, dc]).all()
    assert result.band_valid.all()
    np.testing.assert_allclose(result.band_mean, [[1, 1]], rtol=1e-12, atol=1e-12)


def test_one_segment_configuration_is_rejected_instead_of_shortening_or_padding(config, waveform_pair):
    """A 2-second Welch segment supplies one segment per primary trial and must raise."""
    left, right = waveform_pair
    one_segment = replace(config, welch_segment_s=2.0)
    with pytest.raises(ValueError, match="segment"):
        computation.compute_site_psd(left, one_segment)
    with pytest.raises(ValueError, match="segment"):
        computation.compute_welch_coherence(left, right, one_segment)


def test_shared_auto_psds_are_reused_without_estimator_recomputation(config, waveform_pair, monkeypatch):
    """Supplying matched autos performs only cross-spectrum work and preserves the numerical oracle."""
    left, right = waveform_pair
    psd_a = computation.compute_site_psd(left, config)
    psd_b = computation.compute_site_psd(right, config)
    expected = computation.compute_welch_coherence(left, right, config)
    monkeypatch.setattr(computation, "compute_site_psd", _forbid_auto_recomputation)
    monkeypatch.setattr(signal, "welch", _forbid_auto_recomputation)
    actual = computation.compute_welch_coherence(left, right, config, psd_a=psd_a, psd_b=psd_b)
    np.testing.assert_allclose(actual.coherence, expected.coherence, rtol=1e-12, atol=1e-14)
    assert actual.meta["auto_psd_ids"] == (psd_a.identity, psd_b.identity)


def _forbid_auto_recomputation(*args, **kwargs):
    """Reject any auto-estimator call/arguments; returns nothing and always raises."""
    raise AssertionError("Compatible shared auto-PSDs must be reused")


@pytest.mark.parametrize("changed", ["values", "validity", "choices", "relative_grid", "synchronization", "settings", "frequencies"])
def test_incompatible_supplied_auto_psd_is_rejected(config, waveform_pair, changed):
    """Shared autos must match exact rows, support, clocks, settings and frequency coordinates."""
    left, right = waveform_pair
    psd_a = computation.compute_site_psd(left, config)
    psd_b = computation.compute_site_psd(right, config)
    if changed == "values":
        left = replace(left, values_uv=left.values_uv + 0.001)
    elif changed == "validity":
        mask = left.valid.copy()
        mask[0, 0] = False
        left = replace(left, valid=mask)
    elif changed == "choices":
        left = replace(left, choice_times_s=left.choice_times_s + 0.125)
    elif changed == "relative_grid":
        left = replace(left, relative_time_s=left.relative_time_s + 0.0001)
    elif changed == "synchronization":
        left = replace(left, meta=dict(left.meta, synchronization={"identity": "changed"}))
    elif changed == "settings":
        config = replace(config, welch_segment_s=0.4)
    else:
        psd_a = replace(psd_a, frequency_hz=psd_a.frequency_hz + 0.1)
    with pytest.raises(ValueError):
        computation.compute_welch_coherence(left, right, config, psd_a=psd_a, psd_b=psd_b)


def test_band_changes_do_not_invalidate_shared_site_psd(config, waveform_pair):
    """Changing band integration alone retains identical auto-spectrum identity and arrays."""
    left, right = waveform_pair
    psd = computation.compute_site_psd(left, config)
    new_config = replace(config, bands=(FrequencyBandConfig("narrow", 6, 8),))
    same_psd = computation.compute_site_psd(left, new_config)
    assert same_psd.identity == psd.identity
    np.testing.assert_array_equal(same_psd.psd_uv2_per_hz, psd.psd_uv2_per_hz)
    result = computation.compute_welch_coherence(left, right, new_config, psd_a=psd)
    assert result.band_mean.shape == (2, 1)


def test_strict_band_mean_uses_fixed_centers_and_frequency_specific_validity(config):
    """(trial,frequency) scalars reduce per fixed retained centers, without skipping required NaNs."""
    frequencies = np.arange(2, 102, 2, dtype=float)
    values = np.vstack((frequencies / 100, frequencies / 100, frequencies / 100))
    values[:, np.isin(frequencies, [58, 60, 62])] = np.nan
    values[1, frequencies == 8] = np.nan
    values[2, frequencies == 56] = np.nan
    means, valid, masks = computation.strict_band_mean(values, frequencies, config.bands)
    np.testing.assert_array_equal(masks.sum(axis=1), [3, 23])
    np.testing.assert_array_equal(valid, [[True, True], [False, True], [True, False]])
    assert means[0, 0] == pytest.approx(0.08)
    assert means[0, 1] == pytest.approx(np.mean(frequencies[masks[1]]) / 100)
    assert np.isnan(means[1, 0]) and np.isnan(means[2, 1])
