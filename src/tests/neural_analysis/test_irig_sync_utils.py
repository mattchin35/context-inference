from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.irig_tools import irig_core
from src.neural_analysis import ephys_sync_utils
from src.irig_tools import irig_sync_utils


def build_irig_signal(
    start_time_utc: datetime,
    n_frames: int,
    sample_rate_hz: float,
    bit_period_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic IRIG-H signal and expected per-bit UTC values.

    Args:
        start_time_utc: UTC datetime at the first frame start.
        n_frames: Number of consecutive 60-bit IRIG-H frames.
        sample_rate_hz: Sampling rate in Hz.
        bit_period_s: IRIG-H bit period in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Boolean sync signal with shape ``(n_samples,)``.
            - Expected UTC unix time for each IRIG bit onset with shape
              ``(n_frames * 60,)`` in seconds.
    """
    if start_time_utc.tzinfo is None:
        raise ValueError("start_time_utc must be timezone-aware.")

    all_bits: list[object] = []
    expected_unix_by_bit: list[float] = []
    for frame_ix in range(n_frames):
        frame_start = start_time_utc + timedelta(seconds=60 * frame_ix)
        frame_bits = irig_core.encode_irig_frame(
            frame_start.timestamp(),
            irig_format="standard_decisecond",
        )
        all_bits.extend(frame_bits)
        expected_unix_by_bit.extend(frame_start.timestamp() + np.arange(60, dtype=float))

    samples_per_bit = int(round(sample_rate_hz * bit_period_s))
    signal = np.zeros(len(all_bits) * samples_per_bit, dtype=bool)
    pulse_width_lookup = {
        False: int(round(0.2 * sample_rate_hz)),
        True: int(round(0.5 * sample_rate_hz)),
        "P": int(round(0.8 * sample_rate_hz)),
    }
    for bit_ix, bit in enumerate(all_bits):
        start_sample = bit_ix * samples_per_bit
        signal[start_sample:start_sample + pulse_width_lookup[bit]] = True

    return signal, np.asarray(expected_unix_by_bit, dtype=float)


def test_find_signal_edges_detects_rising_and_falling_indices():
    signal = np.array([0, 0, 1, 1, 0, 1, 0], dtype=bool)

    rising_ix, falling_ix = irig_sync_utils.find_signal_edges(signal)

    assert np.array_equal(rising_ix, np.array([2, 5], dtype=int))
    assert np.array_equal(falling_ix, np.array([4, 6], dtype=int))


def test_pulse_lengths_from_edges_pairs_rising_and_falling_edges():
    rising_ix = np.array([2, 5], dtype=int)
    falling_ix = np.array([4, 6], dtype=int)

    paired_rising_ix, pulse_lengths_samples = irig_sync_utils.pulse_lengths_from_edges(
        rising_ix,
        falling_ix,
    )

    assert np.array_equal(paired_rising_ix, np.array([2, 5], dtype=int))
    assert np.array_equal(pulse_lengths_samples, np.array([2.0, 1.0]))


def test_decode_sync_line_to_irig_utc_recovers_expected_bit_times():
    sample_rate_hz = 10.0
    start_time_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    irig_signal, expected_unix = build_irig_signal(
        start_time_utc=start_time_utc,
        n_frames=4,
        sample_rate_hz=sample_rate_hz,
    )

    irig_df = irig_sync_utils.decode_sync_line_to_irig_utc(
        sync_signal=irig_signal,
        sample_rate_hz=sample_rate_hz,
    )

    assert irig_df.shape[0] == expected_unix.size
    assert np.array_equal(
        irig_df["sample_ix"].to_numpy(dtype=int),
        np.arange(expected_unix.size, dtype=int) * int(sample_rate_hz),
    )
    assert np.allclose(irig_df["utc_unix"].to_numpy(dtype=float), expected_unix)


def test_map_sample_indices_to_utc_interpolates_between_irig_edges():
    sample_rate_hz = 10.0
    start_time_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    irig_signal, _ = build_irig_signal(
        start_time_utc=start_time_utc,
        n_frames=4,
        sample_rate_hz=sample_rate_hz,
    )
    irig_df = irig_sync_utils.decode_sync_line_to_irig_utc(
        sync_signal=irig_signal,
        sample_rate_hz=sample_rate_hz,
    )
    sample_ix = np.array([0, 5, 127], dtype=np.int64)

    mapped_df = irig_sync_utils.map_sample_indices_to_utc(sample_ix, irig_df)

    expected_unix = start_time_utc.timestamp() + sample_ix / sample_rate_hz
    assert np.array_equal(mapped_df["sample_ix"].to_numpy(dtype=np.int64), sample_ix)
    assert np.allclose(mapped_df["utc_unix"].to_numpy(dtype=float), expected_unix)


def test_map_digital_rising_edges_to_utc_maps_only_rising_edges():
    sample_rate_hz = 10.0
    start_time_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    irig_signal, _ = build_irig_signal(
        start_time_utc=start_time_utc,
        n_frames=4,
        sample_rate_hz=sample_rate_hz,
    )
    irig_df = irig_sync_utils.decode_sync_line_to_irig_utc(
        sync_signal=irig_signal,
        sample_rate_hz=sample_rate_hz,
    )
    digital_signal = np.zeros_like(irig_signal, dtype=bool)
    digital_signal[15:18] = True
    digital_signal[42:47] = True

    rising_df = irig_sync_utils.map_digital_rising_edges_to_utc(
        digital_signal=digital_signal,
        sample_rate_hz=sample_rate_hz,
        irig_df=irig_df,
    )

    assert np.array_equal(rising_df["sample_ix"].to_numpy(dtype=int), np.array([15, 42], dtype=int))
    assert np.allclose(
        rising_df["utc_unix"].to_numpy(dtype=float),
        start_time_utc.timestamp() + np.array([1.5, 4.2], dtype=float),
    )


def test_map_sample_indices_to_utc_raises_with_too_few_irig_points():
    irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0], dtype=np.int64),
            "utc_unix": np.array([1.0], dtype=float),
        }
    )

    with pytest.raises(ValueError, match="at least two"):
        irig_sync_utils.map_sample_indices_to_utc(np.array([0, 1], dtype=np.int64), irig_df)


def test_save_stream_sync_npz_writes_named_arrays_and_meta(tmp_path: Path):
    output_path = tmp_path / "aligned_imec0_sync.npz"
    arrays = {
        "sample_ix": np.array([1, 2, 3], dtype=np.int64),
        "utc_unix": np.array([10.0, 11.0, 12.0], dtype=float),
    }
    meta = {
        "generator": "test_save_stream_sync_npz_writes_named_arrays_and_meta",
        "sample_rate_hz": 30000.0,
        "units": {"sample_ix": "samples", "utc_unix": "seconds"},
    }

    saved_path = ephys_sync_utils.save_stream_sync_npz(output_path, arrays=arrays, meta=meta)

    assert saved_path == output_path
    with np.load(output_path, allow_pickle=True) as loaded:
        assert np.array_equal(loaded["sample_ix"], arrays["sample_ix"])
        assert np.allclose(loaded["utc_unix"], arrays["utc_unix"])
        assert loaded["meta"].item() == meta


def test_apply_utc_hour_offset_shifts_unix_and_datetime_consistently():
    irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0, 10], dtype=np.int64),
            "utc_unix": np.array([100.0, 101.0], dtype=float),
            "utc_datetime": [
                datetime.fromtimestamp(100.0, tz=timezone.utc),
                datetime.fromtimestamp(101.0, tz=timezone.utc),
            ],
        }
    )

    shifted_df = ephys_sync_utils.apply_utc_hour_offset(irig_df, utc_offset_hours=1.5)

    expected_unix = np.array([5500.0, 5501.0], dtype=float)
    assert np.allclose(shifted_df["utc_unix"].to_numpy(dtype=float), expected_unix)
    assert shifted_df["utc_datetime"].iloc[0] == datetime.fromtimestamp(expected_unix[0], tz=timezone.utc)
    assert shifted_df["utc_datetime"].iloc[1] == datetime.fromtimestamp(expected_unix[1], tz=timezone.utc)


def test_write_alignment_note_records_offset_and_processing_time(tmp_path: Path):
    output_root = tmp_path / "aligned"

    note_path = ephys_sync_utils.write_alignment_note(output_root=output_root, utc_offset_hours=0.0)

    assert note_path == output_root / "time_adjustment_note.txt"
    note_lines = note_path.read_text().splitlines()
    assert len(note_lines) == 2
    assert note_lines[0] == "UTC hour adjustment: 0.0"
    assert note_lines[1].startswith("Processed at:")


def test_decode_binary_file_irig_utc_applies_requested_hour_offset():
    base_irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0, 10], dtype=np.int64),
            "recording_time_s": np.array([0.0, 1.0], dtype=float),
            "pulse_len_samples": np.array([2.0, 2.0], dtype=float),
            "irig_bit": np.array([False, True], dtype=object),
            "utc_unix": np.array([100.0, 101.0], dtype=float),
            "utc_datetime": [
                datetime.fromtimestamp(100.0, tz=timezone.utc),
                datetime.fromtimestamp(101.0, tz=timezone.utc),
            ],
        }
    )

    def fake_read_digital_line(
        binary_file: Path | str,
        digital_word: int,
        digital_line: int,
    ) -> tuple[np.ndarray, float]:
        return np.zeros(20, dtype=bool), 10.0

    def fake_decode_sync_line_to_irig_utc(
        sync_signal: np.ndarray,
        sample_rate_hz: float,
        bit_period_s: float = 1.0,
    ) -> pd.DataFrame:
        return base_irig_df.copy(deep=True)

    shifted_irig_df, sample_rate_hz = ephys_sync_utils.decode_binary_file_irig_utc(
        binary_file=Path("fake.bin"),
        digital_word=0,
        irig_line=6,
        utc_offset_hours=-1.0,
        read_digital_line_fn=fake_read_digital_line,
        decode_sync_line_to_irig_utc_fn=fake_decode_sync_line_to_irig_utc,
    )

    assert sample_rate_hz == 10.0
    assert np.allclose(shifted_irig_df["utc_unix"].to_numpy(dtype=float), np.array([-3500.0, -3499.0]))


def test_decode_ni_irig_debug_returns_expected_pulse_and_frame_columns(
    tmp_path: Path,
):
    ni_file = tmp_path / "run0_g0_t0.nidq.bin"
    ni_file.touch()
    sample_rate_hz = 10.0
    start_time_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    irig_signal, _ = build_irig_signal(
        start_time_utc=start_time_utc,
        n_frames=4,
        sample_rate_hz=sample_rate_hz,
    )

    def fake_read_digital_line(
        binary_file: Path | str,
        digital_word: int,
        digital_line: int,
    ) -> tuple[np.ndarray, float]:
        return irig_signal, sample_rate_hz

    pulse_df, frame_df = ephys_sync_utils.decode_ni_irig_debug(
        ni_file=ni_file,
        digital_word=0,
        irig_line=0,
        read_digital_line_fn=fake_read_digital_line,
    )

    assert {
        "pulse_ix",
        "rising_sample_ix",
        "falling_sample_ix",
        "recording_time_s",
        "pulse_width_samples",
        "pulse_width_s",
        "irig_bit",
        "utc_unix",
        "utc_datetime",
    }.issubset(pulse_df.columns)
    assert {
        "frame_ix",
        "frame_start_pulse_ix",
        "frame_start_sample_ix",
        "utc_unix",
        "utc_datetime",
    }.issubset(frame_df.columns)
    assert pulse_df.shape[0] == 240
    assert frame_df.shape[0] >= 2
    assert set(pulse_df["irig_bit"].dropna().unique()) == {False, True, "P"}
    assert "local_datetime" in pulse_df.columns
    assert "local_datetime" in frame_df.columns
    assert str(pulse_df["local_datetime"].iloc[0].tzinfo) == "America/New_York"
    assert str(frame_df["local_datetime"].iloc[0].tzinfo) == "America/New_York"
