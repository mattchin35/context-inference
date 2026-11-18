"""Redundant state reports change pulse pairing, never the elapsed clock or speed."""

import numpy as np
import pandas as pd
import pytest

from src.irig_tools import irig_serial_io
from src.mouse_behavior_preprocessing import treadmill_speed as speed
from src.tests.neural_analysis.test_treadmill_speed import synthetic_speed_messages as messages_fixture


synthetic_speed_messages = messages_fixture


@pytest.mark.parametrize("state,count", [("LOW", 1), ("HIGH", 1), ("LOW", 3), ("HIGH", 3)])
def test_redundant_reports_preserve_all_sample_times_and_frame_anchors(synthetic_speed_messages, state, count):
    """Extra same-level sync reports split one interval; total time and UTC values stay exact."""
    parsed = irig_serial_io.parse_coolterm_serial_text(synthetic_speed_messages["text"], pc_timestamp_format="ignore")
    original = speed.align_treadmill_speed(parsed)
    modified = _insert_redundant_reports(parsed, state, count)
    modified_before = modified.copy(deep=True)
    ignored = np.flatnonzero(modified.inserted.to_numpy())
    sample_rows = np.flatnonzero(modified.label.to_numpy() == "runSpeed")
    original_sample_rows = np.flatnonzero(parsed.label.to_numpy() == "runSpeed")
    original_clock = speed.reconstruct_treadmill_clock(parsed)
    modified_clock = speed.reconstruct_treadmill_clock(modified)
    np.testing.assert_allclose(modified_clock[sample_rows], original_clock[original_sample_rows], rtol=0, atol=1e-10)

    result = speed.align_treadmill_speed(modified)

    np.testing.assert_array_equal(result.speed_mm_s, synthetic_speed_messages["speed_mm_s"])
    np.testing.assert_array_equal(result.source_rows, sample_rows)
    np.testing.assert_allclose(result.utc_s, synthetic_speed_messages["utc_s"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(result.utc_s, original.utc_s, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.frame_utc_s, original.frame_utc_s)
    np.testing.assert_allclose(result.frame_clock_s, original.frame_clock_s, rtol=0, atol=1e-10)
    assert result.meta["redundant_sync_rows"] == ignored.tolist()
    assert result.meta["rejected_speed_rows"] == []
    records = result.meta["redundant_sync_records"]
    assert len(records) == count
    for row, record in zip(ignored, records, strict=True):
        assert record["source_row"] == row and record["state"] == state
        assert record["source_clock_s"] == modified_clock[row]
        assert record["logged_interval_s"] == modified.time_value.iloc[row]
    pd.testing.assert_frame_equal(modified, modified_before)


def _insert_redundant_reports(parsed: pd.DataFrame, state: str, count: int) -> pd.DataFrame:
    """Encode extra same-state reports while preserving original absolute source times.

    parsed uses seconds and mm/s. Insert count records inside one HIGH/LOW
    phase, then re-encode every interval/offset relative to the latest report.
    Return a new parsed table with an inserted marker; no original row is lost.
    """
    clock = speed.reconstruct_treadmill_clock(parsed)
    sync = np.flatnonzero(parsed.label.to_numpy() == "syncPinState")
    index = next(index for index in range(len(sync) - 1) if parsed.value.iloc[sync[index]] == state)
    start, end = clock[sync[index]], clock[sync[index + 1]]
    original = parsed.assign(source_clock_s=clock, inserted=False)
    additions = []
    for fraction in np.linspace(0, 1, count + 2)[1:-1]:
        record = parsed.iloc[sync[index]].to_dict()
        record.update(source_clock_s=start + fraction * (end - start), inserted=True)
        additions.append(record)
    modified = pd.concat([original, pd.DataFrame(additions)], ignore_index=True).sort_values(
        "source_clock_s", kind="stable").reset_index(drop=True)
    last_edge = 0.0
    intervals = []
    for record in modified.itertuples():
        intervals.append(record.source_clock_s - last_edge)
        if record.label == "syncPinState":
            last_edge = record.source_clock_s
    return modified.assign(time_value=intervals).drop(columns="source_clock_s")
