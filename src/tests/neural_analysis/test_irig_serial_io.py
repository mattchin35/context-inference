from __future__ import annotations

import numpy as np
import pytest

from src.irig_tools import irig_serial_io


def test_parse_datetime_timestamp_format_preserves_existing_behavior():
    text = "\n".join(
        [
            "2026-07-07 12:06:54.000\tsyncPinState:HIGH;elapsed:0",
            "2026-07-07 12:06:54.200\tsyncPinState:LOW;elapsed:200000",
        ]
    )

    parsed_df = irig_serial_io.parse_coolterm_serial_text(
        text,
        pc_timestamp_format="datetime",
    )

    assert parsed_df["pc_date"].tolist() == ["2026-07-07", "2026-07-07"]
    assert parsed_df["pc_time"].tolist() == ["12:06:54.000", "12:06:54.200"]
    assert parsed_df["time_label"].tolist() == ["elapsed", "elapsed"]
    assert parsed_df["time_value"].tolist() == [0.0, 0.2]


def test_parse_millisecondtime_uses_hardcoded_local_date_and_trailing_semicolon():
    text = "\n".join(
        [
            "15:08:52.532\tsyncPinState:LOW;syncIntervalUs:499991;",
            "15:08:53.332\tsyncPinState:HIGH;syncIntervalUs:799985;",
        ]
    )

    parsed_df = irig_serial_io.parse_coolterm_serial_text(
        text,
        pc_timestamp_format="millisecondtime",
        pc_reference_date="2026-07-09",
    )

    assert parsed_df["pc_date"].tolist() == ["2026-07-09", "2026-07-09"]
    assert parsed_df["pc_time"].tolist() == ["15:08:52.532", "15:08:53.332"]
    assert parsed_df["label"].tolist() == ["syncPinState", "syncPinState"]
    assert parsed_df["value"].tolist() == ["LOW", "HIGH"]
    assert parsed_df["time_label"].tolist() == ["syncIntervalUs", "syncIntervalUs"]
    assert np.allclose(parsed_df["time_value"].to_numpy(dtype=float), np.array([0.499991, 0.799985]))


def test_parse_millisecondtime_requires_reference_date():
    with pytest.raises(ValueError, match="pc_reference_date"):
        irig_serial_io.parse_coolterm_serial_text(
            "15:08:52.532\tsyncPinState:LOW;syncIntervalUs:499991;",
            pc_timestamp_format="millisecondtime",
        )


def test_parse_millisecondtime_raises_when_times_decrease():
    text = "\n".join(
        [
            "15:08:52.532\tsyncPinState:LOW;syncIntervalUs:499991;",
            "15:08:51.332\tsyncPinState:HIGH;syncIntervalUs:799985;",
        ]
    )

    with pytest.raises(ValueError, match="decreased"):
        irig_serial_io.parse_coolterm_serial_text(
            text,
            pc_timestamp_format="millisecondtime",
            pc_reference_date="2026-07-09",
        )


def test_parse_no_pc_timestamp_with_reference_start_time_synthesizes_pc_times():
    text = "\n".join(
        [
            "syncPinState:HIGH;syncIntervalUs:0;",
            "syncPinState:LOW;syncIntervalUs:200000;",
            "syncPinState:HIGH;syncIntervalUs:800000;",
        ]
    )

    parsed_df = irig_serial_io.parse_coolterm_serial_text(
        text,
        pc_timestamp_format=None,
        pc_reference_date="2026-07-09",
        pc_reference_start_time="15:08:52.532",
    )

    assert parsed_df["pc_date"].tolist() == ["2026-07-09", "2026-07-09", "2026-07-09"]
    assert parsed_df["pc_time"].tolist() == ["15:08:52.532", "15:08:52.732", "15:08:53.532"]


def test_parse_no_pc_timestamp_without_start_time_leaves_empty_pc_time_and_transition_raises():
    parsed_df = irig_serial_io.parse_coolterm_serial_text(
        "syncPinState:HIGH;syncIntervalUs:0;\nsyncPinState:LOW;syncIntervalUs:200000;",
        pc_timestamp_format=None,
        pc_reference_date="2026-07-09",
    )

    assert parsed_df["pc_date"].tolist() == ["2026-07-09", "2026-07-09"]
    assert parsed_df["pc_time"].tolist() == ["", ""]
    with pytest.raises(ValueError, match="pc_time"):
        irig_serial_io.treadmill_log_to_transition_df(parsed_df)


def test_transition_conversion_warns_and_drops_repeated_sync_state():
    text = "\n".join(
        [
            "15:16:00.018\tsyncPinState:HIGH;syncIntervalUs:199998;",
            "15:16:00.769\tsyncPinState:HIGH;syncIntervalUs:743232;",
            "15:16:00.829\tsyncPinState:LOW;syncIntervalUs:56753;",
        ]
    )
    parsed_df = irig_serial_io.parse_coolterm_serial_text(
        text,
        pc_timestamp_format="millisecondtime",
        pc_reference_date="2026-07-09",
    )

    with pytest.warns(UserWarning, match="Repeated syncPinState"):
        transition_df = irig_serial_io.treadmill_log_to_transition_df(parsed_df)

    assert transition_df["pin_state"].tolist() == [1, 0]
    assert np.isclose(transition_df["utc_seconds"].diff().iloc[1], 0.799985)
