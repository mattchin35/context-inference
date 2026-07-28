from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.irig_tools import irig_core
from src.irig_tools import open_ephys_irig


def _write_ttl_folder(
    ttl_dir: Path,
    states: np.ndarray,
    sample_numbers: np.ndarray,
    timestamps: np.ndarray | None,
    full_words: np.ndarray | None = None,
) -> None:
    """Write a minimal Open Ephys TTL event folder for tests.

    Args:
        ttl_dir: Destination directory.
        states: Signed Open Ephys channel-state values, shape ``(n_events,)``.
        sample_numbers: Open Ephys sample numbers, shape ``(n_events,)`` in samples.
        timestamps: Event timestamps, shape ``(n_events,)`` in seconds, or
            ``None`` to omit the file.
        full_words: Optional TTL word after each transition, shape
            ``(n_events,)``.

    Returns:
        None.
    """
    ttl_dir.mkdir(parents=True, exist_ok=True)
    np.save(ttl_dir / "states.npy", states.astype(np.int16))
    np.save(ttl_dir / "sample_numbers.npy", sample_numbers.astype(np.int64))
    if timestamps is not None:
        np.save(ttl_dir / "timestamps.npy", timestamps.astype(float))
    if full_words is not None:
        np.save(ttl_dir / "full_words.npy", full_words.astype(np.uint64))


def _transition_df(times_s: list[float], states: list[int], sample_rate_hz: float = 30_000.0) -> pd.DataFrame:
    """Build a transition table in the expected Open Ephys adapter schema."""
    times = np.asarray(times_s, dtype=float)
    return pd.DataFrame(
        {
            "sample_ix": np.rint(times * float(sample_rate_hz)).astype(np.int64),
            "recording_time_s": times,
            "line": np.zeros(times.shape[0], dtype=np.int64),
            "state": np.asarray(states, dtype=np.int64),
        }
    )


def _synthetic_open_ephys_irig_folder(
    ttl_dir: Path,
    start_time_utc: datetime,
    n_frames: int,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic Open Ephys TTL transitions with one-sample high glitches.

    Args:
        ttl_dir: Destination TTL folder.
        start_time_utc: UTC datetime encoded in the first IRIG frame.
        n_frames: Number of consecutive 60-bit IRIG frames.
        sample_rate_hz: Sample rate used to create sample numbers, in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Valid IRIG rising sample numbers, shape ``(n_bits,)`` in samples.
            - Expected per-bit UTC Unix timestamps, shape ``(n_bits,)`` in seconds.
    """
    if start_time_utc.tzinfo is None:
        raise ValueError("start_time_utc must be timezone-aware.")

    pulse_width_lookup = {False: 0.2, True: 0.5, "P": 0.8}
    transition_times: list[float] = []
    transition_states: list[int] = []
    valid_onset_samples: list[int] = []
    expected_unix: list[float] = []

    for frame_ix in range(n_frames):
        frame_start = start_time_utc + timedelta(seconds=60 * frame_ix)
        frame_bits = irig_core.encode_irig_frame(
            frame_start.timestamp(),
            irig_format="neurokairos",
        )
        for bit_ix, bit in enumerate(frame_bits):
            bit_time_s = float(frame_ix * 60 + bit_ix)
            if bit_ix % 17 == 0:
                glitch_time_s = bit_time_s + 0.123
                transition_times.extend([glitch_time_s, glitch_time_s + 1.0 / sample_rate_hz])
                transition_states.extend([1, 0])

            transition_times.extend([bit_time_s, bit_time_s + pulse_width_lookup[bit]])
            transition_states.extend([1, 0])
            valid_onset_samples.append(int(round(bit_time_s * sample_rate_hz)))
            expected_unix.append(frame_start.timestamp() + bit_ix)

    order = np.argsort(np.asarray(transition_times, dtype=float), kind="stable")
    ordered_times = np.asarray(transition_times, dtype=float)[order]
    ordered_states = np.asarray(transition_states, dtype=np.int64)[order]
    signed_states = np.where(ordered_states == 1, 1, -1)
    sample_numbers = np.rint(ordered_times * sample_rate_hz).astype(np.int64)
    full_words = ordered_states.astype(np.uint64)
    _write_ttl_folder(
        ttl_dir,
        states=signed_states,
        sample_numbers=sample_numbers,
        timestamps=ordered_times,
        full_words=full_words,
    )
    return np.asarray(valid_onset_samples, dtype=np.int64), np.asarray(expected_unix, dtype=float)


def test_load_open_ephys_ttl_events_maps_signed_states_to_line_and_state(tmp_path: Path):
    ttl_dir = tmp_path / "TTL"
    _write_ttl_folder(
        ttl_dir,
        states=np.array([1, -1, 2, -2], dtype=np.int16),
        sample_numbers=np.array([10, 20, 30, 40], dtype=np.int64),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        full_words=np.array([1, 0, 2, 0], dtype=np.uint64),
    )

    event_df = open_ephys_irig.load_open_ephys_ttl_events(ttl_dir)

    assert event_df["line"].tolist() == [0, 0, 1, 1]
    assert event_df["state"].tolist() == [1, 0, 1, 0]
    assert event_df["sample_ix"].tolist() == [10, 20, 30, 40]
    assert np.allclose(event_df["recording_time_s"], [0.1, 0.2, 0.3, 0.4])
    assert event_df["full_word"].tolist() == [1, 0, 2, 0]


def test_load_open_ephys_ttl_events_validates_equal_lengths(tmp_path: Path):
    ttl_dir = tmp_path / "TTL"
    _write_ttl_folder(
        ttl_dir,
        states=np.array([1, -1], dtype=np.int16),
        sample_numbers=np.array([10], dtype=np.int64),
        timestamps=np.array([0.1, 0.2], dtype=float),
    )

    with pytest.raises(ValueError, match="same length"):
        open_ephys_irig.load_open_ephys_ttl_events(ttl_dir)


def test_load_open_ephys_ttl_events_can_compute_timestamps_from_sample_rate(tmp_path: Path):
    ttl_dir = tmp_path / "TTL"
    _write_ttl_folder(
        ttl_dir,
        states=np.array([1, -1], dtype=np.int16),
        sample_numbers=np.array([3000, 6000], dtype=np.int64),
        timestamps=None,
    )

    event_df = open_ephys_irig.load_open_ephys_ttl_events(
        ttl_dir,
        sample_rate_hz=1_000.0,
    )

    assert np.allclose(event_df["recording_time_s"], [3.0, 6.0])


def test_extract_line_transitions_selects_requested_line(tmp_path: Path):
    ttl_dir = tmp_path / "TTL"
    _write_ttl_folder(
        ttl_dir,
        states=np.array([1, -1, 2, -2], dtype=np.int16),
        sample_numbers=np.array([10, 20, 30, 40], dtype=np.int64),
        timestamps=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
    )
    event_df = open_ephys_irig.load_open_ephys_ttl_events(ttl_dir)

    line_df = open_ephys_irig.extract_line_transitions(event_df, line=1)

    assert line_df["line"].tolist() == [1, 1]
    assert line_df["state"].tolist() == [1, 0]
    assert line_df["sample_ix"].tolist() == [30, 40]


def test_debounce_line_transitions_removes_short_high_glitches():
    transition_df = _transition_df(
        times_s=[0.100000, 0.100033333333, 1.0, 1.2],
        states=[1, 0, 1, 0],
    )

    debounced_df = open_ephys_irig.debounce_line_transitions(
        transition_df,
        min_high_duration_s=0.01,
        min_low_duration_s=0.01,
    )

    assert np.allclose(debounced_df["recording_time_s"], [1.0, 1.2])
    assert debounced_df["state"].tolist() == [1, 0]


def test_debounce_line_transitions_merges_short_low_gaps():
    transition_df = _transition_df(
        times_s=[1.0, 1.100000, 1.100033333333, 1.2],
        states=[1, 0, 1, 0],
    )

    debounced_df = open_ephys_irig.debounce_line_transitions(
        transition_df,
        min_high_duration_s=0.01,
        min_low_duration_s=0.01,
    )

    assert np.allclose(debounced_df["recording_time_s"], [1.0, 1.2])
    assert debounced_df["state"].tolist() == [1, 0]


def test_extract_irig_pulses_from_transitions_returns_onset_and_width_contract():
    transition_df = _transition_df(
        times_s=[1.0, 1.2, 2.0, 2.5],
        states=[1, 0, 1, 0],
        sample_rate_hz=10_000.0,
    )

    pulse_df = open_ephys_irig.extract_irig_pulses_from_transitions(transition_df)

    assert pulse_df.columns.tolist() == [
        "pulse_ix",
        "source_onset_ix",
        "source_onset_time_s",
        "pulse_width_s",
        "pulse_width_samples",
    ]
    assert pulse_df["source_onset_ix"].tolist() == [10_000, 20_000]
    assert np.allclose(pulse_df["source_onset_time_s"], [1.0, 2.0])
    assert np.allclose(pulse_df["pulse_width_s"], [0.2, 0.5])
    assert pulse_df["pulse_width_samples"].tolist() == [2_000, 5_000]


def test_decode_open_ephys_ttl_irig_recovers_neurokairos_frames_with_glitches(tmp_path: Path):
    sample_rate_hz = 30_000.0
    start_time_utc = datetime(2026, 7, 27, 18, 38, 0, tzinfo=timezone.utc)
    expected_sample_ix, expected_unix = _synthetic_open_ephys_irig_folder(
        ttl_dir=tmp_path / "TTL",
        start_time_utc=start_time_utc,
        n_frames=3,
        sample_rate_hz=sample_rate_hz,
    )

    decoded = open_ephys_irig.decode_open_ephys_ttl_irig(
        tmp_path / "TTL",
        line=0,
        min_high_duration_s=0.01,
        min_low_duration_s=0.01,
        irig_format="neurokairos",
    )

    assert decoded.invalid_pulse_ix.size == 0
    assert decoded.frame_df.shape[0] == 3
    assert np.array_equal(
        decoded.pulse_df["source_onset_ix"].to_numpy(dtype=np.int64),
        expected_sample_ix,
    )
    assert np.allclose(decoded.pulse_df["utc_unix"].to_numpy(dtype=float), expected_unix)


def test_decode_open_ephys_ttl_irig_utc_matches_existing_sync_schema(tmp_path: Path):
    sample_rate_hz = 20_000.0
    _synthetic_open_ephys_irig_folder(
        ttl_dir=tmp_path / "TTL",
        start_time_utc=datetime(2026, 7, 27, 18, 38, 0, tzinfo=timezone.utc),
        n_frames=2,
        sample_rate_hz=sample_rate_hz,
    )

    irig_df = open_ephys_irig.decode_open_ephys_ttl_irig_utc(
        tmp_path / "TTL",
        line=0,
        min_high_duration_s=0.01,
        min_low_duration_s=0.01,
        irig_format="neurokairos",
    )

    assert irig_df.columns.tolist() == [
        "sample_ix",
        "recording_time_s",
        "pulse_len_samples",
        "irig_bit",
        "utc_unix",
        "utc_datetime",
    ]
    assert irig_df.shape[0] == 120
    assert np.allclose(
        irig_df["utc_unix"].to_numpy(dtype=float),
        datetime(2026, 7, 27, 18, 38, 0, tzinfo=timezone.utc).timestamp()
        + np.arange(120, dtype=float),
    )
