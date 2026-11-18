"""Characterize processed context attribution without reconstructing triggers."""

from pathlib import Path

import pandas as pd
import pytest

from src.behavior_analysis import collect_events
from src.mouse_behavior_preprocessing import process_behavior_log


def _context_events(context_time: float, context_first_at_tie: bool = True) -> pd.DataFrame:
    """Return (9,3) raw events; context_time is UTC seconds, tie flag is unitless.

    Rows are chronological; at 4 s the flag controls source event order.
    Three completed trials plus one unfinalized start expose the snapshot.
    """
    rows = [
        (1.0, "enter_left_patch", ""),
        (2.0, "trial_start", ""),
        (3.0, "correct_choice_left_patch", "reward_True"),
        (4.0, "trial_start", ""),
        (5.0, "wrong_choice_right_patch", "reward_False"),
        (6.0, "trial_start", ""),
        (7.0, "correct_choice_right_patch", "reward_True"),
        (8.0, "trial_start", ""),
    ]
    context_row = (context_time, "enter_right_patch", "")
    insert_position = 3 if context_time <= 4.0 and context_first_at_tie else 4
    rows.insert(insert_position, context_row)
    return pd.DataFrame(rows, columns=["Time", "Event", "Note"])


def _parse(events: pd.DataFrame) -> pd.DataFrame:
    """Convert chronological (N,3) raw events (UTC seconds) to (T,C) trial rows."""
    return process_behavior_log.iterate_trials(
        events,
        collect_events.get_context_events(events, event_list=[]),
        collect_events.get_choice_events(events, event_list=[]),
        collect_events.get_reward_events(events, event_list=[]),
        collect_events.get_stimulus_events(events, event_list=[]),
    )


@pytest.mark.parametrize("context_time,expected_states,expected_blocks", [
    (3.5, [1, 0, 0], [0, 1, 1]),
    (4.5, [1, 1, 0], [0, 0, 1]),
])
def test_context_is_snapshotted_at_start_not_at_choice_or_finalization(
    context_time, expected_states, expected_blocks,
):
    """Input: UTC context time/(3,) expected codes. Output: None; assert snapshots."""
    trials = _parse(_context_events(context_time))
    assert trials["state_int"].tolist() == expected_states
    assert trials["cur_block"].tolist() == expected_blocks
    assert trials["cur_trial"].tolist() == [0, 1, 2]
    assert trials["action"].tolist() == [1, 1, 0]
    assert trials["correct"].tolist() == [1, 0, 1]
    assert trials["reward"].tolist() == [1, 0, 1]
    assert trials["choice_time"].tolist() == [3.0, 5.0, 7.0]


@pytest.mark.parametrize("context_first,expected_states,expected_blocks", [
    (True, [1, 0, 0], [0, 1, 1]),
    (False, [1, 1, 0], [0, 0, 1]),
])
def test_equal_timestamp_events_follow_source_row_order_in_parser(
    context_first, expected_states, expected_blocks,
):
    """Input: tie order/(3,) expected codes. Output: None; no choice-time inference."""
    events = _context_events(4.0, context_first_at_tie=context_first)
    trials = _parse(events)
    assert trials["state_int"].tolist() == expected_states
    assert trials["cur_block"].tolist() == expected_blocks


@pytest.mark.parametrize("context_time,context_first,expected_states,expected_blocks", [
    (3.5, True, [1, 0, 0], [0, 1, 1]),
    (4.5, False, [1, 1, 0], [0, 0, 1]),
    (4.0, True, [1, 0, 0], [0, 1, 1]),
    (4.0, False, [1, 1, 0], [0, 0, 1]),
])
def test_public_trial_preparation_retains_context_snapshots(
    tmp_path: Path, context_time, context_first, expected_states, expected_blocks,
):
    """Input: output Path, UTC time, tie order/(3,) codes. Output: None; verify CSV.

    This small chronological log characterizes the public sorting path. The
    separate parser test specifies event-order semantics for timestamp ties.
    """
    events = _context_events(context_time, context_first_at_tie=context_first)
    trials = process_behavior_log.make_trial_df(
        cleaned_data=events, session_id="synthetic-context", save_name="context",
        output_path=tmp_path,
        session_info={
            "correct_reward_probability": 0.8,
            "incorrect_reward_probability": 0.0,
            "switch_probability": 0.1,
        },
    )
    assert trials["state_int"].tolist() == expected_states
    assert trials["cur_block"].tolist() == expected_blocks
    saved = pd.read_csv(tmp_path / "context.csv")
    assert saved["state_int"].tolist() == expected_states
    assert saved["cur_block"].tolist() == expected_blocks


@pytest.mark.parametrize("context_first", [True, False])
def test_public_preparation_preserves_source_order_with_many_timestamp_ties(
    tmp_path: Path, context_first: bool,
):
    """Input: temporary Path and tie-order bool. Output: None; assert (3,) contexts.

    Interleaved trial_stop rows are recognized no-op context events. They make
    tie stability observable beyond the tiny event tables above.
    """
    events = _context_events(4.0, context_first_at_tie=context_first)
    tied_rows = pd.DataFrame({
        "Time": [4.0] * 32, "Event": ["trial_stop"] * 32, "Note": [""] * 32,
    })
    events = pd.concat([events.iloc[:3], tied_rows, events.iloc[3:]], ignore_index=True)
    direct = _parse(events)
    # Public preparation accepts unsorted input. Rotate only the distinct-time
    # tail so that the source order of all 4-second events stays unchanged.
    unsorted_events = pd.concat([events.iloc[37:], events.iloc[:37]], ignore_index=True)
    prepared = process_behavior_log.make_trial_df(
        cleaned_data=unsorted_events, session_id="synthetic-context-ties", save_name="ties",
        output_path=tmp_path,
        session_info={
            "correct_reward_probability": 0.8,
            "incorrect_reward_probability": 0.0,
            "switch_probability": 0.1,
        },
    )
    assert prepared["state_int"].tolist() == direct["state_int"].tolist()
    assert prepared["cur_block"].tolist() == direct["cur_block"].tolist()
