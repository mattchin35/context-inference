from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_loading


def make_lfp_meta(n_saved_channels: int = 4, n_samples: int = 10) -> dict[str, str]:
    """Build minimal SpikeGLX metadata for LFP-loading tests."""

    return {
        "typeThis": "imec",
        "nSavedChans": str(n_saved_channels),
        "fileSizeBytes": str(2 * int(n_saved_channels) * int(n_samples)),
    }


def test_validate_lfp_saved_channel_accepts_valid_saved_channel():
    """A saved channel index inside the binary row range should be accepted."""
    lfp_loading.validate_lfp_saved_channel(make_lfp_meta(n_saved_channels=4), saved_channel_index=2)


def test_validate_lfp_saved_channel_rejects_negative_channel():
    """Negative saved channel indices are invalid for binary row indexing."""
    with pytest.raises(ValueError, match="nonnegative"):
        lfp_loading.validate_lfp_saved_channel(make_lfp_meta(n_saved_channels=4), saved_channel_index=-1)


def test_validate_lfp_saved_channel_rejects_channel_at_or_above_n_saved():
    """Saved channel indices should be less than nSavedChans."""
    with pytest.raises(ValueError, match="nSavedChans"):
        lfp_loading.validate_lfp_saved_channel(make_lfp_meta(n_saved_channels=4), saved_channel_index=4)


def test_map_lfp_time_window_to_samples_maps_alignment_and_window():
    """UTC-aligned trial windows should map to LFP sample windows by interpolation."""
    irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0, 1000, 2000], dtype=np.int64),
            "utc_unix": np.array([10.0, 11.0, 12.0], dtype=float),
        }
    )

    relative_time_s, start_sample, stop_sample = lfp_loading.map_lfp_time_window_to_samples(
        alignment_time_s=10.5,
        window=(-0.1, 0.2),
        lfp_irig_df=irig_df,
        sample_rate_hz=1000.0,
    )

    assert start_sample == 400
    assert stop_sample == 700
    assert relative_time_s.shape == (300,)
    np.testing.assert_allclose(relative_time_s[[0, -1]], np.array([-0.1, 0.199]))


def test_read_lfp_saved_channel_window_uses_saved_channel_and_gain_correction(monkeypatch, tmp_path: Path):
    """LFP window reads should use saved channel indexing and return microvolts."""
    lfp_path = tmp_path / "test.lf.bin"
    lfp_path.write_bytes(b"placeholder")
    raw_data = np.arange(40, dtype=np.int16).reshape(4, 10)
    seen = {}

    monkeypatch.setattr(lfp_loading.readSGLX, "readMeta", lambda _: make_lfp_meta(n_saved_channels=4, n_samples=10))
    monkeypatch.setattr(lfp_loading.readSGLX, "SampRate", lambda _: 2500.0)
    monkeypatch.setattr(lfp_loading.readSGLX, "makeMemMapRaw", lambda _, __: raw_data)

    def fake_gain_correct(data_array, channel_list, meta):
        seen["data_array"] = data_array.copy()
        seen["channel_list"] = list(channel_list)
        seen["meta"] = meta
        return np.asarray(data_array, dtype=float) * 1e-6

    monkeypatch.setattr(lfp_loading.readSGLX, "GainCorrectIM", fake_gain_correct)

    lfp_uv, sample_rate_hz = lfp_loading.read_lfp_saved_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=2,
        start_sample=3,
        stop_sample=5,
    )

    np.testing.assert_array_equal(seen["data_array"], raw_data[[2], 3:5])
    assert seen["channel_list"] == [2]
    assert sample_rate_hz == 2500.0
    np.testing.assert_allclose(lfp_uv, raw_data[2, 3:5].astype(float))


def test_load_trial_lfp_trace_returns_relative_time_and_uv(monkeypatch, tmp_path: Path):
    """High-level LFP loading should return aligned time and gain-corrected trace arrays."""
    lfp_path = tmp_path / "test.lf.bin"
    lfp_path.write_bytes(b"placeholder")
    irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0, 1000], dtype=np.int64),
            "utc_unix": np.array([10.0, 11.0], dtype=float),
        }
    )

    def fake_read_window(lfp_path, saved_channel_index, start_sample, stop_sample):
        assert saved_channel_index == 1
        assert start_sample == 100
        assert stop_sample == 300
        return np.arange(200, dtype=float), 1000.0

    monkeypatch.setattr(lfp_loading, "read_lfp_saved_channel_window", fake_read_window)

    relative_time_s, lfp_uv = lfp_loading.load_trial_lfp_trace(
        lfp_path=lfp_path,
        saved_channel_index=1,
        alignment_time_s=10.2,
        window=(-0.1, 0.1),
        lfp_irig_df=irig_df,
        sample_rate_hz=1000.0,
    )

    np.testing.assert_allclose(relative_time_s[[0, -1]], np.array([-0.1, 0.099]))
    np.testing.assert_allclose(lfp_uv[[0, -1]], np.array([0.0, 199.0]))
    assert relative_time_s.shape == lfp_uv.shape


def test_filter_lfp_trace_returns_unfiltered_signal_when_band_is_none():
    """Default LFP display should keep gain-corrected microvolt values unchanged."""
    relative_time_s = np.arange(5, dtype=float) / 1000.0
    lfp_uv = np.array([0.0, 1.0, -2.0, 3.0, -4.0], dtype=float)

    filtered_lfp_uv = lfp_loading.filter_lfp_trace(
        relative_time_s=relative_time_s,
        lfp_uv=lfp_uv,
        sample_rate_hz=1000.0,
        frequency_band_hz=None,
    )

    np.testing.assert_allclose(filtered_lfp_uv, lfp_uv)


def test_filter_lfp_trace_calls_pynapple_bandpass_with_expected_contract(monkeypatch):
    """Band-limited LFP display should use Pynapple's Butterworth bandpass helper."""
    relative_time_s = np.arange(4, dtype=float) / 1000.0
    lfp_uv = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    seen = {}

    class FakeFilteredTrace:
        """Minimal stand-in for a Pynapple time series returned by the filter."""

        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)

    def fake_apply_bandpass_filter(signal, cutoff, fs, mode):
        seen["signal_values"] = np.asarray(signal.values, dtype=float)
        seen["signal_times"] = np.asarray(signal.index.values, dtype=float)
        seen["cutoff"] = cutoff
        seen["fs"] = fs
        seen["mode"] = mode
        return FakeFilteredTrace()

    monkeypatch.setattr(lfp_loading.nap, "apply_bandpass_filter", fake_apply_bandpass_filter)

    filtered_lfp_uv = lfp_loading.filter_lfp_trace(
        relative_time_s=relative_time_s,
        lfp_uv=lfp_uv,
        sample_rate_hz=1000.0,
        frequency_band_hz=(5.0, 10.0),
    )

    np.testing.assert_allclose(seen["signal_values"], lfp_uv)
    np.testing.assert_allclose(seen["signal_times"], relative_time_s)
    assert seen["cutoff"] == (5.0, 10.0)
    assert seen["fs"] == 1000.0
    assert seen["mode"] == "butter"
    np.testing.assert_allclose(filtered_lfp_uv, FakeFilteredTrace.values)


def test_load_trial_lfp_trace_filters_padded_window_and_trims_to_requested_window(monkeypatch, tmp_path: Path):
    """Filtered LFP loading should pad before filtering and return only requested samples."""
    lfp_path = tmp_path / "test.lf.bin"
    lfp_path.write_bytes(b"placeholder")
    irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0, 1000], dtype=np.int64),
            "utc_unix": np.array([10.0, 11.0], dtype=float),
        }
    )
    seen = {}

    def fake_read_window(lfp_path, saved_channel_index, start_sample, stop_sample):
        assert saved_channel_index == 1
        seen["start_sample"] = start_sample
        seen["stop_sample"] = stop_sample
        return np.arange(stop_sample - start_sample, dtype=float), 1000.0

    def fake_filter_lfp_trace(relative_time_s, lfp_uv, sample_rate_hz, frequency_band_hz):
        seen["filter_time_shape"] = relative_time_s.shape
        seen["filter_lfp_shape"] = lfp_uv.shape
        seen["frequency_band_hz"] = frequency_band_hz
        return np.asarray(lfp_uv, dtype=float) + 1000.0

    monkeypatch.setattr(lfp_loading, "read_lfp_saved_channel_window", fake_read_window)
    monkeypatch.setattr(lfp_loading, "filter_lfp_trace", fake_filter_lfp_trace)

    relative_time_s, lfp_uv = lfp_loading.load_trial_lfp_trace(
        lfp_path=lfp_path,
        saved_channel_index=1,
        alignment_time_s=10.5,
        window=(-0.1, 0.1),
        lfp_irig_df=irig_df,
        sample_rate_hz=1000.0,
        frequency_band_hz=(50.0, 70.0),
        filter_padding_s=0.2,
    )

    assert seen["start_sample"] == 200
    assert seen["stop_sample"] == 800
    assert seen["filter_time_shape"] == (600,)
    assert seen["filter_lfp_shape"] == (600,)
    assert seen["frequency_band_hz"] == (50.0, 70.0)
    np.testing.assert_allclose(relative_time_s[[0, -1]], np.array([-0.1, 0.099]))
    np.testing.assert_allclose(lfp_uv[[0, -1]], np.array([1200.0, 1399.0]))
    assert relative_time_s.shape == lfp_uv.shape == (200,)


def test_decode_lfp_sync_calls_existing_sync_decoder(monkeypatch, tmp_path: Path):
    """LFP sync decoding should delegate to the existing IRIG decoder."""
    lfp_path = tmp_path / "test.lf.bin"
    lfp_path.write_bytes(b"placeholder")
    expected_df = pd.DataFrame({"sample_ix": [0], "utc_unix": [10.0]})
    seen = {}

    def fake_decode_binary_file_irig_utc(binary_file, digital_word, irig_line, bit_period_s, utc_offset_hours):
        seen["args"] = (binary_file, digital_word, irig_line, bit_period_s, utc_offset_hours)
        return expected_df, 2500.0

    monkeypatch.setattr(
        lfp_loading.ephys_sync_utils,
        "decode_binary_file_irig_utc",
        fake_decode_binary_file_irig_utc,
    )

    sync_df, sample_rate_hz = lfp_loading.decode_lfp_sync(lfp_path, digital_word=0, irig_line=6)

    assert seen["args"] == (lfp_path, 0, 6, 1.0, 0.0)
    assert sync_df is expected_df
    assert sample_rate_hz == 2500.0


def test_decode_lfp_sync_passes_user_selected_utc_offset(monkeypatch, tmp_path: Path):
    """User-selected whole-hour LFP offsets should reach the sync decoder."""
    lfp_path = tmp_path / "test.lf.bin"
    lfp_path.write_bytes(b"placeholder")
    expected_df = pd.DataFrame({"sample_ix": [0], "utc_unix": [10.0]})
    seen = {}

    def fake_decode_binary_file_irig_utc(binary_file, digital_word, irig_line, bit_period_s, utc_offset_hours):
        seen["utc_offset_hours"] = utc_offset_hours
        return expected_df, 2500.0

    monkeypatch.setattr(
        lfp_loading.ephys_sync_utils,
        "decode_binary_file_irig_utc",
        fake_decode_binary_file_irig_utc,
    )

    sync_df, sample_rate_hz = lfp_loading.decode_lfp_sync(
        lfp_path,
        digital_word=0,
        irig_line=6,
        utc_offset_hours=-1.0,
    )

    assert seen["utc_offset_hours"] == -1.0
    assert sync_df is expected_df
    assert sample_rate_hz == 2500.0
