"""Exact source-clock fixtures and seeded IRIG-aligned signed speed contracts."""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.irig_tools import irig_core, irig_serial_io


@pytest.fixture
def synthetic_speed_messages():
    """Return raw text and independent UTC/value oracles with a 20-ppm source clock.

    Three complete NeuroKairos frames have rising onsets 1 UTC second apart.
    Receipt-time prefixes are deliberately wrong and batched. Signed speeds,
    zero, duplicate timestamps, leading data, and a trailing open pulse occur.
    """
    first_utc = 1_699_999_980.0
    scale = 1.00002
    events = []
    for frame in range(3):
        bits = irig_core.encode_irig_frame(first_utc + frame * 60)
        for offset, bit in enumerate(bits):
            start = frame * 60 + offset
            width = .8 if bit == "P" else (.5 if bit else .2)
            events.extend([(start, 0, "syncPinState", "HIGH"), (start + width, 0, "syncPinState", "LOW")])
    times = np.array([-.2, .1, .1, .25, .9, 1.1, 4.5, 17.3, 65.4, 124.123, 171.7])
    speeds = np.array([-3., 12., 13., 0., -4., 21., 0., 15., -12., 0., 7.])
    for time, speed in zip(times, speeds):
        events.append((float(time), 1, "distance", "123"))
        events.append((float(time), 2, "runSpeed", str(speed)))
    events.append((180., 0, "syncPinState", "HIGH"))
    events.sort(key=lambda event: (event[0], event[1]))
    last_edge_us = 0
    lines = []
    source_rows = []
    for row, (time, _, label, value) in enumerate(events):
        clock_us = round((time + 1) * scale * 1e6)
        interval_us = clock_us - last_edge_us
        time_label = "syncIntervalUs" if label == "syncPinState" else "timeSincePinChangeUs"
        if label == "syncPinState":
            last_edge_us = clock_us
        if label == "runSpeed":
            source_rows.append(row)
        lines.append(f"1900-01-01 00:00:00.000\t{label}:{value};{time_label}:{interval_us};")
    return {
        "text": "\n".join(lines), "utc_s": first_utc + times,
        "speed_mm_s": speeds, "source_rows": np.array(source_rows),
        "frame_utc_s": first_utc + np.arange(3) * 60,
        "scale": scale,
    }


@pytest.mark.parametrize("suffix", ["", "Us"])
def test_reconstruct_clock_uses_edge_intervals_and_offsets_not_message_cumsum(suffix):
    """Repeated offsets stay equal; first interval places leading speed correctly."""
    module = importlib.import_module("src.mouse_behavior_preprocessing.treadmill_speed")
    parsed = pd.DataFrame({
        "label": ["runSpeed", "distance", "syncPinState", "runSpeed", "syncPinState", "runSpeed", "syncPinState"],
        "value": ["-1", "9", "HIGH", "0", "LOW", "7", "HIGH"],
        "time_label": [f"timeSincePinChange{suffix}"] * 2 + [f"syncInterval{suffix}", f"timeSincePinChange{suffix}", f"syncInterval{suffix}", f"timeSincePinChange{suffix}", f"syncInterval{suffix}"],
        "time_value": [.2, .2, .8, .1, .2, .25, .8],
    })
    original = parsed.copy(deep=True)
    clock = module.reconstruct_treadmill_clock(parsed)
    np.testing.assert_allclose(clock, [.2, .2, .8, .9, 1., 1.25, 1.8])
    pd.testing.assert_frame_equal(parsed, original)


def test_seeded_irig_alignment_preserves_values_duplicates_and_leading_data(synthetic_speed_messages):
    """Absolute time comes from encoded frames plus measured source timing, never PC dates."""
    module = importlib.import_module("src.mouse_behavior_preprocessing.treadmill_speed")
    fixture = synthetic_speed_messages
    parsed = irig_serial_io.parse_coolterm_serial_text(fixture["text"], pc_timestamp_format="ignore")
    result = module.align_treadmill_speed(parsed)
    np.testing.assert_allclose(result.utc_s, fixture["utc_s"], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(result.speed_mm_s, fixture["speed_mm_s"])
    np.testing.assert_array_equal(result.source_rows, fixture["source_rows"])
    np.testing.assert_array_equal(result.frame_utc_s, fixture["frame_utc_s"])
    np.testing.assert_allclose(np.diff(result.frame_clock_s), 60 * fixture["scale"])
    assert result.meta["incomplete_boundary_pulses"] == 1
    assert result.meta["invalid_pulse_indices"] == []
    assert result.meta["clock_seconds_per_utc_second"] == pytest.approx(fixture["scale"])


@pytest.mark.parametrize("fault", ["negative", "nonfinite", "timing_label", "offset_reset", "beyond_edge", "sync_state", "speed_nan", "no_speed", "too_short"])
def test_invalid_inputs_are_explicit_and_never_fabricate_zero_speed(synthetic_speed_messages, fault):
    """Unsupported timing, resets, insufficient anchors, and missing data remain unavailable."""
    module = importlib.import_module("src.mouse_behavior_preprocessing.treadmill_speed")
    parsed = irig_serial_io.parse_coolterm_serial_text(synthetic_speed_messages["text"], pc_timestamp_format="ignore")
    speed_rows = np.flatnonzero(parsed.label.to_numpy() == "runSpeed")
    sync_rows = np.flatnonzero(parsed.label.to_numpy() == "syncPinState")
    if fault == "negative":
        parsed.loc[speed_rows[0], "time_value"] = -1
    elif fault == "nonfinite":
        parsed.loc[sync_rows[1], "time_value"] = np.inf
    elif fault == "timing_label":
        parsed.loc[speed_rows[0], "time_label"] = "elapsed"
    elif fault == "offset_reset":
        parsed.loc[speed_rows[2], "time_value"] = .01
    elif fault == "beyond_edge":
        parsed.loc[speed_rows[0], "time_value"] = 1.5
    elif fault == "sync_state":
        parsed.loc[sync_rows[1], "value"] = "HIGH"
    elif fault == "speed_nan":
        parsed.loc[speed_rows[0], "value"] = "nan"
    elif fault == "no_speed":
        parsed = parsed.loc[parsed.label != "runSpeed"]
    else:
        parsed = parsed.iloc[:70]
    with pytest.raises(ValueError):
        module.align_treadmill_speed(parsed)


def test_bad_pulse_diagnostics_do_not_remove_speed_with_valid_later_anchors(synthetic_speed_messages):
    """An undecodable initial pulse does not require dropping the remaining valid frames."""
    module = importlib.import_module("src.mouse_behavior_preprocessing.treadmill_speed")
    parsed = irig_serial_io.parse_coolterm_serial_text(synthetic_speed_messages["text"], pc_timestamp_format="ignore")
    # Move the first LOW from .8 to .01 seconds and compensate the next HIGH.
    sync_rows = np.flatnonzero(parsed.label.to_numpy() == "syncPinState")
    transfer = float(parsed.loc[sync_rows[1], "time_value"]) - .01
    parsed.loc[sync_rows[1], "time_value"] = .01
    parsed.loc[sync_rows[2], "time_value"] += transfer
    # First-pulse speed offsets were relative to the original HIGH; remove those
    # records rather than create a contradictory fixture after moving the edge.
    parsed = parsed.drop(index=np.arange(sync_rows[0] + 1, sync_rows[2])).reset_index(drop=True)
    # Reinsert the modified LOW that the contiguous drop removed.
    low = pd.DataFrame({"pc_date": [""], "pc_time": [""], "label": ["syncPinState"], "value": ["LOW"], "time_label": ["syncIntervalUs"], "time_value": [.01]})
    parsed = pd.concat([parsed.iloc[:sync_rows[0]+1], low, parsed.iloc[sync_rows[0]+1:]], ignore_index=True)
    result = module.align_treadmill_speed(parsed)
    assert result.meta["invalid_pulse_indices"] == [0]
    assert len(result.frame_utc_s) >= 2
