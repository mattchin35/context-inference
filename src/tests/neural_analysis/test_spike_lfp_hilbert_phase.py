from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.neural_analysis import spike_lfp_hilbert_phase


def _wrapped_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return elementwise wrapped angular difference in radians."""

    return np.angle(np.exp(1j * (np.asarray(left) - np.asarray(right))))


def _padded_sine_trace() -> tuple[np.ndarray, np.ndarray, float]:
    """Build a long source-rate 8 Hz trace with ample samples away from edges."""

    sample_rate_hz = 1_000.0
    time_s = np.arange(-2.0, 3.0, 1.0 / sample_rate_hz)
    return time_s, np.sin(2.0 * np.pi * 8.0 * time_s), sample_rate_hz


def test_module_exposes_approved_public_contract():
    """The standalone Hilbert viewer module must expose its approved analysis API."""

    assert spike_lfp_hilbert_phase.ANALYSIS_VERSION
    for name in (
        "SingleTrialSpikeLFPHilbertResult",
        "compute_hilbert_phase_trace",
        "sample_hilbert_phase_at_spikes",
        "compute_single_trial_spike_lfp_hilbert",
        "save_single_trial_spike_lfp_hilbert_result",
    ):
        assert hasattr(spike_lfp_hilbert_phase, name)


def test_hilbert_phase_trace_recovers_central_phase_of_a_long_8_hz_sinusoid(monkeypatch):
    """The phase convention should be angle(hilbert(bandpassed_lfp)) away from edges."""

    time_s, raw_lfp, sample_rate_hz = _padded_sine_trace()
    monkeypatch.setattr(
        spike_lfp_hilbert_phase.lfp_loading,
        "filter_lfp_trace",
        lambda **kwargs: np.asarray(kwargs["lfp_uv"], dtype=float),
    )

    trace = spike_lfp_hilbert_phase.compute_hilbert_phase_trace(
        relative_time_s=time_s,
        raw_lfp=raw_lfp,
        sample_rate_hz=sample_rate_hz,
        frequency_band_hz=(6.0, 10.0),
    )

    central = np.abs(time_s) < 1.0
    expected_phase = 2.0 * np.pi * 8.0 * time_s - np.pi / 2.0
    assert trace.frequency_band_hz == (6.0, 10.0)
    assert trace.sample_rate_hz == sample_rate_hz
    np.testing.assert_allclose(np.abs(trace.phase_unit[central]), 1.0, atol=1e-6)
    assert np.max(np.abs(_wrapped_difference(trace.phase_rad[central], expected_phase[central]))) < 0.03


def test_sample_hilbert_phase_interpolates_complex_components_across_wrapped_angles():
    """Spike phase interpolation must use real/imaginary components, not wrapped angles."""

    sampled = spike_lfp_hilbert_phase.sample_hilbert_phase_at_spikes(
        relative_time_s=np.array([0.0, 1.0]),
        phase_unit=np.array([np.exp(1j * 3.0 * np.pi / 4.0), np.exp(-1j * 3.0 * np.pi / 4.0)]),
        envelope=np.array([2.0, 2.0]),
        spike_times_relative_s=np.array([0.5]),
    )

    assert sampled.valid.tolist() == [True]
    np.testing.assert_allclose(sampled.phase_rad, [np.pi], atol=1e-6)
    np.testing.assert_allclose(np.abs(sampled.phase_unit), [1.0], atol=1e-6)


def test_single_trial_processing_filters_padded_trace_before_half_open_visible_crop(monkeypatch):
    """Raw and Hilbert arrays should be cropped only after processing the padded trace."""

    time_s, raw_lfp, sample_rate_hz = _padded_sine_trace()
    filter_calls: list[np.ndarray] = []

    def fake_filter_lfp_trace(**kwargs):
        """Record the padded input and return it unchanged for an exact crop test."""

        filter_calls.append(np.asarray(kwargs["relative_time_s"], dtype=float))
        return np.asarray(kwargs["lfp_uv"], dtype=float)

    monkeypatch.setattr(spike_lfp_hilbert_phase.lfp_loading, "filter_lfp_trace", fake_filter_lfp_trace)
    result = spike_lfp_hilbert_phase.compute_single_trial_spike_lfp_hilbert(
        padded_relative_time_s=time_s,
        padded_raw_lfp=raw_lfp,
        source_sample_rate_hz=sample_rate_hz,
        unit_spike_times_absolute_s=np.array([99.0, 100.0, 100.25, 101.0]),
        event_time_s=100.0,
        visible_window=(0.0, 1.0),
        filter_padding_s=1.0,
        trial_index=7,
        unit_id=12,
        lfp_site_label="PFC channel 4",
    )

    assert len(filter_calls) == 1
    np.testing.assert_array_equal(filter_calls[0], time_s)
    assert result.relative_time_s[0] == 0.0
    assert result.relative_time_s[-1] < 1.0
    assert result.relative_time_s.size == 1_000
    np.testing.assert_allclose(result.raw_lfp, raw_lfp[(time_s >= 0.0) & (time_s < 1.0)])
    np.testing.assert_allclose(result.spike_times_absolute_s, [100.0, 100.25])
    np.testing.assert_allclose(result.spike_times_relative_s, [0.0, 0.25])
    assert result.source_sample_rate_hz == sample_rate_hz


def test_single_trial_processing_handles_empty_spikes_and_validates_inputs(monkeypatch):
    """Empty visible spike sets are valid, while invalid scientific inputs fail clearly."""

    time_s, raw_lfp, sample_rate_hz = _padded_sine_trace()
    monkeypatch.setattr(
        spike_lfp_hilbert_phase.lfp_loading,
        "filter_lfp_trace",
        lambda **kwargs: np.asarray(kwargs["lfp_uv"], dtype=float),
    )
    result = spike_lfp_hilbert_phase.compute_single_trial_spike_lfp_hilbert(
        padded_relative_time_s=time_s,
        padded_raw_lfp=raw_lfp,
        source_sample_rate_hz=sample_rate_hz,
        unit_spike_times_absolute_s=np.array([50.0]),
        event_time_s=100.0,
        visible_window=(0.0, 1.0),
    )
    assert result.spike_times_relative_s.shape == (0,)
    assert result.spike_phase_rad.shape == (0,)
    with pytest.raises(ValueError):
        spike_lfp_hilbert_phase.compute_hilbert_phase_trace(
            relative_time_s=time_s,
            raw_lfp=raw_lfp,
            sample_rate_hz=sample_rate_hz,
            frequency_band_hz=(10.0, 6.0),
        )


def test_saver_writes_named_arrays_metadata_and_refuses_overwrite(tmp_path: Path, monkeypatch):
    """Saved trial results must remain auditable and never overwrite an existing NPZ."""

    time_s, raw_lfp, sample_rate_hz = _padded_sine_trace()
    monkeypatch.setattr(
        spike_lfp_hilbert_phase.lfp_loading,
        "filter_lfp_trace",
        lambda **kwargs: np.asarray(kwargs["lfp_uv"], dtype=float),
    )
    result = spike_lfp_hilbert_phase.compute_single_trial_spike_lfp_hilbert(
        padded_relative_time_s=time_s,
        padded_raw_lfp=raw_lfp,
        source_sample_rate_hz=sample_rate_hz,
        unit_spike_times_absolute_s=np.array([100.25]),
        event_time_s=100.0,
        visible_window=(0.0, 1.0),
        trial_index=4,
        unit_id=8,
        lfp_site_label="HPC channel 2",
    )
    output_path = tmp_path / "single_trial_phase.npz"

    spike_lfp_hilbert_phase.save_single_trial_spike_lfp_hilbert_result(
        output_path=output_path,
        result=result,
        metadata={"phase_convention": "angle(hilbert(filtered_lfp))"},
    )

    saved = np.load(output_path, allow_pickle=True)
    for name in ("relative_time_s", "raw_lfp", "bandpassed_lfp", "analytic_signal", "envelope", "phase_unit", "phase_rad", "phase_valid", "spike_times_absolute_s", "spike_times_relative_s", "spike_phase_unit", "spike_phase_rad", "spike_phase_envelope", "spike_phase_valid", "meta"):
        assert name in saved.files
    assert saved["meta"].item()["phase_convention"] == "angle(hilbert(filtered_lfp))"
    with pytest.raises(FileExistsError):
        spike_lfp_hilbert_phase.save_single_trial_spike_lfp_hilbert_result(output_path, result, {})
