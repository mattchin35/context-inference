from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.irig_tools import decode_irig_logic_csv
from src.irig_tools import irig_core


def build_transition_dataframe(
    start_time_utc: datetime,
    n_frames: int,
    bit_period_s: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic NeuroKairos transition table.

    Args:
        start_time_utc: UTC-aware datetime for the first frame start.
        n_frames: Number of consecutive 60-pulse frames.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        tuple[pd.DataFrame, np.ndarray]:
            - Transition dataframe with columns ``utc_seconds``,
              ``local_datetime``, and ``pin_state``.
            - Expected rising-edge UTC times with shape ``(n_frames * 60,)`` in
              Unix seconds.
    """
    if start_time_utc.tzinfo is None:
        raise ValueError("start_time_utc must be timezone-aware.")

    local_zone = ZoneInfo("America/New_York")
    bit_width_lookup = {
        False: 0.2 * bit_period_s,
        True: 0.5 * bit_period_s,
        "P": 0.8 * bit_period_s,
    }

    transition_rows: list[dict[str, object]] = []
    expected_rising_utc: list[float] = []
    for frame_ix in range(n_frames):
        frame_start = start_time_utc + timedelta(seconds=60 * frame_ix)
        frame_bits = irig_core.encode_irig_frame(
            frame_start.timestamp(),
            irig_format="neurokairos",
        )
        for bit_ix, bit in enumerate(frame_bits):
            rising_time = frame_start.timestamp() + bit_ix * bit_period_s
            falling_time = rising_time + bit_width_lookup[bit]
            expected_rising_utc.append(rising_time)
            transition_rows.append(
                {
                    "utc_seconds": rising_time,
                    "local_datetime": datetime.fromtimestamp(rising_time, tz=local_zone).isoformat(),
                    "pin_state": 1,
                }
            )
            transition_rows.append(
                {
                    "utc_seconds": falling_time,
                    "local_datetime": datetime.fromtimestamp(falling_time, tz=local_zone).isoformat(),
                    "pin_state": 0,
                }
            )

    transition_df = pd.DataFrame(transition_rows)
    return transition_df, np.asarray(expected_rising_utc, dtype=float)


def test_load_irig_transition_csv_validates_required_columns(tmp_path):
    csv_path = tmp_path / "transitions.csv"
    pd.DataFrame(
        {
            "utc_seconds": [1.0],
            "local_datetime": ["2026-04-03T14:29:00-04:00"],
            "pin_state": [1],
        }
    ).to_csv(csv_path, index=False)

    transition_df = decode_irig_logic_csv.load_irig_transition_csv(csv_path)

    assert list(transition_df.columns) == ["utc_seconds", "local_datetime", "pin_state"]

    bad_csv_path = tmp_path / "bad_transitions.csv"
    pd.DataFrame({"utc_seconds": [1.0], "pin_state": [1]}).to_csv(bad_csv_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        decode_irig_logic_csv.load_irig_transition_csv(bad_csv_path)


def test_extract_irig_pulses_from_transitions_pairs_rises_and_falls():
    transition_df = pd.DataFrame(
        {
            "utc_seconds": [10.0, 10.8, 11.0, 11.2],
            "local_datetime": [
                "2026-04-03T14:29:00-04:00",
                "2026-04-03T14:29:00.800000-04:00",
                "2026-04-03T14:29:01-04:00",
                "2026-04-03T14:29:01.200000-04:00",
            ],
            "pin_state": [1, 0, 1, 0],
        }
    )

    pulse_df = decode_irig_logic_csv.extract_irig_pulses_from_transitions(transition_df)

    assert list(
        pulse_df.columns
    ) == [
        "rising_edge_ix",
        "rising_edge_utc_seconds",
        "falling_edge_utc_seconds",
        "pulse_width_s",
        "rising_edge_local_datetime",
    ]
    assert np.array_equal(pulse_df["rising_edge_ix"].to_numpy(dtype=int), np.array([0, 1], dtype=int))
    assert np.allclose(pulse_df["pulse_width_s"].to_numpy(dtype=float), np.array([0.8, 0.2], dtype=float))


def test_extract_irig_pulses_from_transitions_rejects_malformed_transition_order():
    malformed_df = pd.DataFrame(
        {
            "utc_seconds": [10.0, 10.8, 11.0, 11.2],
            "local_datetime": [
                "2026-04-03T14:29:00-04:00",
                "2026-04-03T14:29:00.800000-04:00",
                "2026-04-03T14:29:01-04:00",
                "2026-04-03T14:29:01.200000-04:00",
            ],
            "pin_state": [1, 1, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="alternate"):
        decode_irig_logic_csv.extract_irig_pulses_from_transitions(malformed_df)


def test_classify_irig_pulse_widths_seconds_handles_desktop_jitter():
    pulse_widths = np.array([0.19999, 0.50001, 0.79999], dtype=float)

    bit_labels = decode_irig_logic_csv.classify_irig_pulse_widths_seconds(pulse_widths)

    assert np.array_equal(bit_labels, np.array([False, True, "P"], dtype=object))


def test_find_irig_frame_spans_and_decode_neurokairos_frame_match_synthetic_frame():
    frame_start_utc = datetime(2026, 4, 3, 18, 29, 0, tzinfo=timezone.utc)
    transition_df, _ = build_transition_dataframe(start_time_utc=frame_start_utc, n_frames=2)
    pulse_df = decode_irig_logic_csv.extract_irig_pulses_from_transitions(transition_df)
    irig_bits = decode_irig_logic_csv.classify_irig_pulse_widths_seconds(pulse_df["pulse_width_s"])

    frame_spans = decode_irig_logic_csv.find_irig_frame_spans(irig_bits)
    frame_fields = decode_irig_logic_csv.decode_neurokairos_frame_fields(
        irig_bits[frame_spans[0][0]:frame_spans[0][1]],
        century_base=2000,
    )

    assert frame_spans[0] == (0, 60)
    assert frame_fields["seconds"] == 0
    assert frame_fields["minutes"] == 29
    assert frame_fields["hours"] == 18
    assert frame_fields["day_of_year"] == 93
    assert frame_fields["year_two_digit"] == 26


def test_find_irig_frame_spans_skips_partial_and_malformed_initial_frames():
    """Frame detection should tolerate recordings that start mid-frame."""
    valid_frame = np.full(60, False, dtype=object)
    valid_frame[list(irig_core.MARKER_POSITIONS)] = "P"

    partial_initial_frame = valid_frame[21:60]
    mid_frame_start_bits = np.concatenate([partial_initial_frame, valid_frame, valid_frame])

    frame_spans = decode_irig_logic_csv.find_irig_frame_spans(mid_frame_start_bits)

    assert frame_spans[0:2] == [(39, 99), (99, 159)]

    malformed_frame = valid_frame.copy()
    malformed_frame[9] = False
    malformed_then_valid_bits = np.concatenate(
        [partial_initial_frame, malformed_frame, valid_frame, valid_frame]
    )

    frame_spans = decode_irig_logic_csv.find_irig_frame_spans(malformed_then_valid_bits)

    assert frame_spans[0] == (99, 159)


def test_build_frame_debug_table_reports_frame_level_fields():
    frame_start_utc = datetime(2026, 4, 3, 18, 29, 0, tzinfo=timezone.utc)
    transition_df, _ = build_transition_dataframe(start_time_utc=frame_start_utc, n_frames=2)
    pulse_df = decode_irig_logic_csv.extract_irig_pulses_from_transitions(transition_df)

    frame_debug_df = decode_irig_logic_csv.build_frame_debug_table(pulse_df, century_base=2000)

    assert frame_debug_df.shape[0] == 2
    assert {
        "frame_ix",
        "frame_start_pulse_ix",
        "observed_start_utc_seconds",
        "decoded_utc_seconds",
        "year_two_digit",
    }.issubset(frame_debug_df.columns)
    assert np.isclose(frame_debug_df.loc[0, "observed_start_utc_seconds"], frame_start_utc.timestamp())
    assert np.isclose(frame_debug_df.loc[0, "decoded_utc_seconds"], frame_start_utc.timestamp())
