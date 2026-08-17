from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis import lfp_loading, lfp_spectrogram, unit_spike_plotting


def _make_trial_df(n_trials: int = 4) -> pd.DataFrame:
    """Build valid trial metadata with all behavior events used by the view."""

    starts = 10.0 + np.arange(n_trials, dtype=float) * 10.0
    return pd.DataFrame(
        {
            "start_time": starts,
            "choice_time": starts + 1.0,
            "led_on_time": starts + 0.5,
            "reward_time": starts + 1.3,
            "give_reward": np.zeros(n_trials, dtype=int),
            "correct": np.ones(n_trials, dtype=int),
            "reward": np.ones(n_trials, dtype=int),
            "action": np.arange(n_trials, dtype=int) % 2,
        }
    )


def test_compute_wavelet_padding_uses_lowest_frequency_support():
    """Padding should follow Pynapple's eight-width Morlet cutoff on each side."""
    padding_s = lfp_spectrogram.compute_wavelet_padding_s(
        minimum_frequency_hz=2.0,
        window_length=1.0,
    )

    assert padding_s == 4.0


def test_load_trial_lfp_trace_with_sample_rate_preserves_existing_trace_contract(monkeypatch):
    """The spectrogram loader should add sample rate without changing trace loading."""
    expected_time_s = np.array([-0.1, 0.0, 0.1], dtype=float)
    expected_lfp_uv = np.array([1.0, 2.0, 3.0], dtype=float)

    def fake_load_trial_lfp_trace(**kwargs):
        assert kwargs["sample_rate_hz"] == 1000.0
        return expected_time_s, expected_lfp_uv

    monkeypatch.setattr(lfp_loading, "load_trial_lfp_trace", fake_load_trial_lfp_trace)

    time_s, lfp_uv, sample_rate_hz = lfp_loading.load_trial_lfp_trace_with_sample_rate(
        lfp_path="test.lf.bin",
        saved_channel_index=2,
        alignment_time_s=10.0,
        window=(-0.1, 0.2),
        lfp_irig_df=pd.DataFrame(),
        sample_rate_hz=1000.0,
    )

    np.testing.assert_array_equal(time_s, expected_time_s)
    np.testing.assert_array_equal(lfp_uv, expected_lfp_uv)
    assert sample_rate_hz == 1000.0


def test_compute_decimation_factor_targets_500_hz_without_violating_nyquist():
    """Integer decimation should reduce common LFP rates while retaining 80 Hz."""
    assert (
        lfp_spectrogram.compute_decimation_factor(
            sample_rate_hz=2500.0,
            target_sample_rate_hz=500.0,
            maximum_frequency_hz=80.0,
        )
        == 5
    )
    assert (
        lfp_spectrogram.compute_decimation_factor(
            sample_rate_hz=1250.0,
            target_sample_rate_hz=500.0,
            maximum_frequency_hz=80.0,
        )
        == 2
    )


def test_compute_morlet_log_power_localizes_synthetic_frequency_and_trims_window():
    """A 10 Hz sinusoid should peak near 10 Hz after padded transform and trimming."""
    sample_rate_hz = 200.0
    relative_time_s = np.arange(-1.0, 1.0, 1.0 / sample_rate_hz)
    lfp_values = np.sin(2.0 * np.pi * 10.0 * relative_time_s)
    frequencies_hz = np.array([6.0, 8.0, 10.0, 12.0, 20.0], dtype=float)

    result = lfp_spectrogram.compute_morlet_log_power(
        relative_time_s=relative_time_s,
        lfp_values=lfp_values,
        sample_rate_hz=sample_rate_hz,
        frequencies_hz=frequencies_hz,
        visible_window=(-0.5, 0.5),
        gaussian_width=1.5,
        window_length=0.25,
        precision=12,
        norm="l1",
        target_sample_rate_hz=200.0,
        notch_60_hz=False,
    )

    assert result.log_power_db.shape == (result.time_s.size, frequencies_hz.size)
    assert result.lfp_values.shape == result.time_s.shape
    assert result.time_s[0] >= -0.5
    assert result.time_s[-1] < 0.5
    assert np.isfinite(result.log_power_db).all()
    mean_power_by_frequency = result.log_power_db.mean(axis=0)
    assert frequencies_hz[int(np.argmax(mean_power_by_frequency))] == 10.0


def test_optional_notch_filter_suppresses_60_hz_and_defaults_off():
    """The explicit notch path should attenuate line noise without changing default data."""
    sample_rate_hz = 500.0
    time_s = np.arange(0.0, 2.0, 1.0 / sample_rate_hz)
    signal = np.sin(2.0 * np.pi * 10.0 * time_s) + np.sin(2.0 * np.pi * 60.0 * time_s)

    unchanged = lfp_spectrogram.apply_optional_60_hz_notch(
        signal,
        sample_rate_hz=sample_rate_hz,
        enabled=False,
        quality_factor=30.0,
    )
    filtered = lfp_spectrogram.apply_optional_60_hz_notch(
        signal,
        sample_rate_hz=sample_rate_hz,
        enabled=True,
        quality_factor=30.0,
    )

    np.testing.assert_allclose(unchanged, signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    original_fft = np.abs(np.fft.rfft(signal))
    filtered_fft = np.abs(np.fft.rfft(filtered))
    sixty_hz_index = int(np.argmin(np.abs(frequencies - 60.0)))
    assert filtered_fft[sixty_hz_index] < original_fft[sixty_hz_index] * 0.2


def test_select_reference_trial_indices_is_deterministic_and_capped():
    """Shared scaling should use evenly distributed valid session trials."""
    trial_df = _make_trial_df(n_trials=100)

    first = lfp_spectrogram.select_reference_trial_indices(
        trial_df,
        alignment_event="choice_time",
        maximum_trial_count=24,
    )
    second = lfp_spectrogram.select_reference_trial_indices(
        trial_df,
        alignment_event="choice_time",
        maximum_trial_count=24,
    )

    assert first.size == 24
    np.testing.assert_array_equal(first, second)
    assert first[0] == 0
    assert first[-1] == 99


def test_estimate_shared_log_power_limits_pools_reference_trials():
    """Color limits should use pooled robust percentiles rather than per-trial scaling."""
    arrays = [
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        np.array([[4.0, 5.0], [6.0, 1000.0]], dtype=float),
    ]

    limits = lfp_spectrogram.estimate_shared_log_power_limits(
        arrays,
        lower_percentile=2.0,
        upper_percentile=98.0,
    )

    expected = tuple(np.percentile(np.concatenate([array.ravel() for array in arrays]), [2.0, 98.0]))
    np.testing.assert_allclose(limits, expected)


def test_plot_trial_lfp_spectrogram_and_behavior_builds_three_panel_view():
    """The LFP view should contain only spectrogram, trace, and behavior axes."""
    trial_df = _make_trial_df()
    lick_times = {
        "left_entry": nap.Ts(t=np.array([10.2, 10.8], dtype=float)),
        "right_entry": nap.Ts(t=np.array([10.4, 11.1], dtype=float)),
    }
    time_s = np.linspace(-1.0, 1.0, 101)
    frequencies_hz = np.geomspace(2.0, 80.0, 8)
    log_power_db = np.tile(np.linspace(-10.0, 10.0, time_s.size)[:, None], (1, 8))

    figure, axes = unit_spike_plotting.plot_trial_lfp_spectrogram_and_behavior(
        trial_df=trial_df,
        trial_index=0,
        lick_times=lick_times,
        spectrogram_time_s=time_s,
        frequencies_hz=frequencies_hz,
        log_power_db=log_power_db,
        lfp_time_s=time_s,
        lfp_values=np.sin(2.0 * np.pi * 10.0 * time_s),
        alignment_event="start_time",
        window=(-1.0, 1.0),
        power_limits_db=(-8.0, 8.0),
        lfp_label="HPC channel 4",
        power_unit_label="dB re 1 uV^2",
        reference_trial_count=24,
    )

    assert np.asarray(axes, dtype=object).shape == (3,)
    assert axes[0].get_ylabel() == "Frequency (Hz)"
    assert axes[1].get_ylabel() == "LFP (uV)"
    assert axes[2].get_ylabel() == "Behavior"
    assert axes[0].collections[0].get_clim() == (-8.0, 8.0)
    legend_labels = {text.get_text() for text in axes[2].get_legend().get_texts()}
    assert {"LED", "Choice", "Reward", "Left licks", "Right licks"}.issubset(legend_labels)
    assert all(axis.get_xlim() == (-1.0, 1.0) for axis in axes)
    assert any("24 reference trials" in text.get_text() for text in figure.texts)
    plt.close(figure)
