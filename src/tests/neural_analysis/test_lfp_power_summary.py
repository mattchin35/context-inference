"""RED contracts for offline Welch LFP power summaries.

All signals use source-voltage units. PSD arrays use source-unit-squared/Hz,
with axes documented by the public functions under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.neural_analysis.lfp_power_summary import (
    compute_presession_reference_psd,
    compute_session_reference_psd,
    compute_trial_epoch_psds,
    compute_welch_psd,
    interpolate_linear_psd_to_canonical_grid,
    mean_band_power_linear,
    normalize_psd_db,
    summarize_trial_median_iqr,
)
from src.neural_analysis.lfp_summary_models import (
    AnalysisWindowConfig,
    FrequencyBandConfig,
    PowerAnalysisConfig,
)


def _power_config() -> PowerAnalysisConfig:
    """Return defaults with a 2-Hz canonical frequency grid."""

    return PowerAnalysisConfig(notch_enabled=False)


def _half_open_time_grid(sample_rate_hz: float) -> np.ndarray:
    """Return the exact event-relative grid for [-2, 2) seconds."""

    return -2.0 + np.arange(int(4.0 * sample_rate_hz), dtype=float) / sample_rate_hz


def test_welch_psd_recovers_8_and_40_hz_peaks_on_exact_2_hz_grid() -> None:
    """A 500-Hz signal returns density PSD bins at 0, 2, ..., 250 Hz."""

    sample_rate_hz = 500.0
    time_s = np.arange(int(4.0 * sample_rate_hz), dtype=float) / sample_rate_hz
    signal_uv = 2.0 * np.sin(2.0 * np.pi * 8.0 * time_s) + np.sin(2.0 * np.pi * 40.0 * time_s)

    frequency_hz, psd_linear = compute_welch_psd(signal_uv, sample_rate_hz, _power_config())

    assert np.array_equal(frequency_hz, np.arange(0.0, 252.0, 2.0))
    assert psd_linear.shape == frequency_hz.shape
    assert psd_linear[np.flatnonzero(frequency_hz == 8.0)[0]] > psd_linear[
        np.flatnonzero(frequency_hz == 6.0)[0]
    ]
    assert psd_linear[np.flatnonzero(frequency_hz == 8.0)[0]] > psd_linear[
        np.flatnonzero(frequency_hz == 10.0)[0]
    ]
    assert psd_linear[np.flatnonzero(frequency_hz == 40.0)[0]] > psd_linear[
        np.flatnonzero(frequency_hz == 38.0)[0]
    ]
    assert psd_linear[np.flatnonzero(frequency_hz == 40.0)[0]] > psd_linear[
        np.flatnonzero(frequency_hz == 42.0)[0]
    ]


def test_trial_epoch_psds_have_documented_axes_and_half_open_epoch_counts() -> None:
    """PSD output axes are (trial, whole/before/after epoch, frequency)."""

    sample_rate_hz = 500.0
    relative_time_s = _half_open_time_grid(sample_rate_hz)
    trial_traces_uv = np.vstack(
        [
            np.sin(2.0 * np.pi * 8.0 * relative_time_s),
            np.sin(2.0 * np.pi * 40.0 * relative_time_s),
        ]
    )

    result = compute_trial_epoch_psds(
        trial_traces_uv,
        relative_time_s,
        sample_rate_hz,
        AnalysisWindowConfig(),
        _power_config(),
    )

    assert result.epoch_names == ("whole", "before", "after")
    assert result.psd_linear.shape == (2, 3, 126)
    assert result.psd_valid.shape == (2, 3)
    assert np.array_equal(result.frequency_hz, np.arange(0.0, 252.0, 2.0))
    assert np.array_equal(result.epoch_sample_counts, np.array([2000, 1000, 1000]))
    assert result.psd_valid.all()


def test_welch_settings_use_periodic_hann_constant_detrend_and_half_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public wrapper forwards the approved 2-Hz density-estimator settings."""

    observed: dict[str, object] = {}

    def fake_welch(values: np.ndarray, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Record the public SciPy call and return one frequency-density pair."""

        observed["values"] = values
        observed.update(kwargs)
        return np.array([0.0]), np.array([1.0])

    monkeypatch.setattr("src.neural_analysis.lfp_power_summary.signal.welch", fake_welch)
    frequency_hz, psd_linear = compute_welch_psd(np.arange(1000.0), 1000.0, _power_config())

    assert np.array_equal(frequency_hz, np.array([0.0]))
    assert np.array_equal(psd_linear, np.array([1.0]))
    assert observed["fs"] == 1000.0
    assert observed["window"] == "hann_periodic"
    assert observed["detrend"] == "constant"
    assert observed["nperseg"] == 500
    assert observed["noverlap"] == 250
    assert observed["scaling"] == "density"
    assert observed["return_onesided"] is True


def test_native_psds_interpolate_in_linear_units_to_one_canonical_grid() -> None:
    """Sites with different native bins share exact 2-Hz cache coordinates."""

    canonical_hz = np.arange(0.0, 12.0, 2.0)
    native_hz = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
    interpolated = interpolate_linear_psd_to_canonical_grid(
        native_hz,
        np.array([2.0 * native_hz + 1.0]),
        canonical_hz,
    )

    assert interpolated.shape == (1, canonical_hz.size)
    assert np.array_equal(canonical_hz, np.arange(0.0, 12.0, 2.0))
    assert np.allclose(interpolated[0], 2.0 * canonical_hz + 1.0)


def test_identical_psd_reference_normalizes_to_zero_db_and_nonpositive_reference_is_nan() -> None:
    """Normalization has no undocumented epsilon substitution."""

    psd_linear = np.array([[2.0, 4.0, 8.0]])

    assert np.allclose(normalize_psd_db(psd_linear, np.array([2.0, 4.0, 8.0])), 0.0)
    normalized = normalize_psd_db(psd_linear, np.array([2.0, 0.0, -1.0]))
    assert normalized[0, 0] == pytest.approx(0.0)
    assert np.isnan(normalized[0, 1])
    assert np.isnan(normalized[0, 2])


def test_session_reference_uses_all_valid_whole_trials_not_condition_filter_membership() -> None:
    """The frequency-wise median is invariant to later condition/filter selections."""

    whole_psd_linear = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [100.0, 200.0],
            [np.nan, np.nan],
        ]
    )
    valid_trial_mask = np.array([True, True, True, False])

    reference = compute_session_reference_psd(whole_psd_linear, valid_trial_mask)

    assert np.array_equal(reference, np.array([3.0, 4.0]))
    selected_condition_membership = np.array([True, False, False, False])
    selected_reference = np.nanmedian(
        whole_psd_linear[selected_condition_membership],
        axis=0,
    )
    assert not np.array_equal(reference, selected_reference)


def test_presession_reference_requires_exactly_ten_seconds_before_first_trial() -> None:
    """A shorter baseline is unavailable rather than silently shortened."""

    sample_rate_hz = 500.0
    config = _power_config()
    first_start_time_s = 20.0
    full_time_s = np.arange(10.0, 20.0, 1.0 / sample_rate_hz)
    full_values_uv = np.sin(2.0 * np.pi * 8.0 * full_time_s)
    frequency_hz, full_reference, full_available = compute_presession_reference_psd(
        full_time_s, full_values_uv, first_start_time_s, sample_rate_hz, config
    )
    short_time_s = np.arange(10.002, 20.0, 1.0 / sample_rate_hz)
    _, short_reference, short_available = compute_presession_reference_psd(
        short_time_s,
        np.sin(2.0 * np.pi * 8.0 * short_time_s),
        first_start_time_s,
        sample_rate_hz,
        config,
    )

    assert full_available is True
    assert frequency_hz.shape == full_reference.shape
    assert np.isfinite(full_reference).all()
    assert short_available is False
    assert np.isnan(short_reference).all()


def test_gamma_band_mean_uses_disjoint_intervals_and_exact_retained_bandwidth() -> None:
    """Gamma includes boundary-adjacent retained bandwidth without bridging 58-62 Hz."""

    frequency_hz = np.arange(20.0, 90.0, 2.0)
    psd_linear = frequency_hz.copy()
    gamma = FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),))

    mean_power, retained_bandwidth_hz = mean_band_power_linear(psd_linear, frequency_hz, gamma)

    expected_integral = (
        (58.0**2 - 30.0**2) / 2.0
        + (80.0**2 - 62.0**2) / 2.0
    )
    assert retained_bandwidth_hz == pytest.approx(46.0)
    assert mean_power == pytest.approx(expected_integral / 46.0)


def test_constant_psd_band_means_preserve_theta_and_gamma_power_and_bandwidth() -> None:
    """Constant linear PSD has the same mean in both retained theta and gamma bands."""

    frequency_hz = np.arange(0.0, 102.0, 2.0)
    constant_psd_linear = np.full(frequency_hz.size, 7.5)
    theta = FrequencyBandConfig("theta", 6.0, 10.0)
    gamma = FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),))

    theta_mean, theta_bandwidth_hz = mean_band_power_linear(
        constant_psd_linear,
        frequency_hz,
        theta,
    )
    gamma_mean, gamma_bandwidth_hz = mean_band_power_linear(
        constant_psd_linear,
        frequency_hz,
        gamma,
    )

    assert theta_mean == pytest.approx(7.5)
    assert theta_bandwidth_hz == pytest.approx(4.0)
    assert gamma_mean == pytest.approx(7.5)
    assert gamma_bandwidth_hz == pytest.approx(46.0)


def test_trial_median_iqr_preserves_trial_axis_and_source_units() -> None:
    """Condition plots receive trial-level source-unit summaries, not collapsed inputs."""

    trial_band_power_mv_squared_per_hz = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    median, lower_quartile, upper_quartile = summarize_trial_median_iqr(
        trial_band_power_mv_squared_per_hz,
        axis=0,
    )

    assert np.array_equal(median, np.array([2.5, 25.0]))
    assert np.array_equal(lower_quartile, np.array([1.75, 17.5]))
    assert np.array_equal(upper_quartile, np.array([3.25, 32.5]))
