"""Regression tests for reading raw semicolon-delimited behavior logs."""

import numpy as np
import pandas as pd

from src.mouse_behavior_preprocessing import process_behavior_log


def test_process_file_reads_mixed_notes_across_parser_chunks(tmp_path):
    """Read a Path log and check (n, 3) output retains epoch seconds and notes."""
    raw_path = tmp_path / "large.log"
    output_path = tmp_path / "processed"
    event_count = 140_000
    start_time = 1_700_000_000
    # Exceed the C parser's chunk size, with numeric notes before text notes.
    with raw_path.open("w") as log:
        log.write(f"INFO;{start_time};context;exit_standby;\n")
        for index in range(1, event_count + 1):
            log.write(f"INFO;{start_time + index};action;left_entry;1\n")
        log.write(f"INFO;{start_time + event_count + 1};action;left_exit;reason_ITI\n")

    events = process_behavior_log.process_file(output_path, raw_path)

    assert list(events.columns) == ["Time", "Event", "Note"]
    assert len(events) == event_count + 1
    np.testing.assert_array_equal(
        events["Time"], np.arange(start_time + 1, start_time + event_count + 2)
    )
    assert events["Event"].tolist() == ["left_entry"] * event_count + ["left_exit"]
    assert events["Note"].tolist() == ["1"] * event_count + ["reason_ITI"]
    pd.testing.assert_frame_equal(
        pd.read_csv(output_path / "large_events.csv"), events.reset_index(drop=True)
    )


def test_process_file_preserves_variable_width_events_and_start_filter(tmp_path):
    """Read str paths; retain ordered event fields and epoch seconds after start."""
    raw_path = tmp_path / "variable.log"
    raw_path.write_text(
        "INFO;1700000000;action;left_entry;before_start\n"
        "INFO;1700000001;context;exit_standby;\n"
        "INFO;1700000001;action;same_time;excluded\n"
        "INFO;1700000002;action;trial_start;\n"
        "Traceback diagnostic without event columns\n"
        "INFO;1700000003;action;correct_choice_left_patch;reward_True;extra;fields\n"
        "INFO;1700000004;action;left_exit;reason_ITI\n"
    )

    events = process_behavior_log.process_file(tmp_path / "processed", str(raw_path))

    assert list(events.columns) == ["Time", "Event", "Note"]
    assert events["Time"].tolist() == [1700000002, 1700000003, 1700000004]
    assert events["Event"].tolist() == [
        "trial_start", "correct_choice_left_patch", "left_exit"
    ]
    assert pd.isna(events["Note"].iloc[0])
    assert events["Note"].iloc[1:].tolist() == ["reward_True", "reason_ITI"]
