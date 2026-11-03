"""Pure context timing, source identity, and baseline-pool contracts."""

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.encoding.predictors import (
    build_context_values,
    diagnose_context_events,
    encode_context,
    select_baseline_trials,
    validate_trial_tables,
)


@pytest.fixture
def trials() -> pd.DataFrame:
    """Return a (6,7) full chronological table; times are UTC seconds.

    No inputs. Trial IDs begin at 10 and are irregular; index labels are neither
    trial IDs nor row positions. Context codes are 0=right, 1=left, 2=dark.
    """
    return pd.DataFrame({
        "cur_trial": [10, 12, 17, 19, 24, 30],
        "cur_block": [4, 4, 5, 5, 6, 6],
        "start_time": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        "choice_time": [105.0, 115.0, np.nan, 135.0, 145.0, 155.0],
        "state_int": [0, 1, 0, 2, 1, 0],
        "action": [0, 1, "no_choice", 0, 1, 0],
        "experimenter_reward_given": [0, 1, 0, 0, 0, 0],
    }, index=[90, 5, 33, 8, 12, 1])


@pytest.fixture
def bin_edges_s() -> np.ndarray:
    """Return (9,) float choice-relative edges in seconds for eight 500-ms bins."""
    return np.arange(-2.0, 2.5, 0.5)


def test_native_encodings_preserve_values_and_column_order() -> None:
    """No inputs; (3,) states yield native float64 (3,P) X/mask/names; return None."""
    states = np.array([0.0, 1.0, 0.0])
    signed, signed_valid, signed_names = encode_context(states, "signed_context")
    separate, separate_valid, separate_names = encode_context(states, "separate_context")

    np.testing.assert_array_equal(signed, [[-1.0], [1.0], [-1.0]])
    np.testing.assert_array_equal(separate, [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(signed_valid, [True, True, True])
    np.testing.assert_array_equal(separate_valid, signed_valid)
    assert signed.dtype == separate.dtype == np.dtype("float64")
    assert signed_names == ("signed_context",)
    assert separate_names == ("left_context", "right_context")


@pytest.mark.parametrize("encoding", ["signed_context", "separate_context"])
def test_encoding_keeps_invalid_context_unavailable(encoding: str) -> None:
    """Input encoding string; invalid dimensionless states yield NaN rows, return None."""
    states = np.array([0, 1, 2, np.nan, 9, -1, 0.5, np.inf])
    original = states.copy()
    matrix, valid, _ = encode_context(states, encoding)

    np.testing.assert_array_equal(valid, [True, True, False, False, False, False, False, False])
    assert np.isnan(matrix[~valid]).all()
    np.testing.assert_array_equal(states, original)


@pytest.mark.parametrize("states,encoding", [
    (np.array([[0, 1]]), "signed_context"),
    (np.array([0, 1]), "current_action"),
])
def test_encoding_rejects_implicit_reshape_and_unknown_predictors(
    states: np.ndarray, encoding: str,
) -> None:
    """Input state ndarray and encoding string; invalid axis/name raises, return None."""
    with pytest.raises(ValueError, match="one-dimensional|shape|encoding|context"):
        encode_context(states, encoding)


def test_baseline_excludes_manual_no_choice_and_dark_without_renumbering(trials: pd.DataFrame) -> None:
    """Input full (6,7) table in UTC seconds; assert row-position masks, return None."""
    original = trials.copy(deep=True)
    selected = select_baseline_trials(trials)

    np.testing.assert_array_equal(selected.row_position, np.arange(6))
    np.testing.assert_array_equal(selected.trial_id, [10, 12, 17, 19, 24, 30])
    np.testing.assert_array_equal(selected.block_id, [4, 4, 5, 5, 6, 6])
    np.testing.assert_array_equal(selected.baseline_valid, [True, False, False, False, True, True])
    assert "manual_reward" in selected.baseline_invalid_reason.iloc[1]
    assert "no_animal_choice" in selected.baseline_invalid_reason.iloc[2]
    assert "dark_context" in selected.baseline_invalid_reason.iloc[3]
    assert selected.baseline_invalid_reason.iloc[0] == ""
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("action", [None, np.nan, "None", "none", "no_choice", ""])
def test_baseline_uses_existing_no_choice_sentinels(trials: pd.DataFrame, action: object) -> None:
    """Input UTC-second table and scalar action sentinel; mask row 0, return None."""
    trials["action"] = trials.action.astype(object)
    trials.iloc[0, trials.columns.get_loc("action")] = action
    assert not select_baseline_trials(trials).baseline_valid.iloc[0]


@pytest.mark.parametrize("state", [np.nan, "None", 9, 0.5])
def test_missing_or_invalid_entering_context_excludes_baseline(
    trials: pd.DataFrame, state: object,
) -> None:
    """Input full UTC table and dimensionless state; invalid pool row, return None."""
    trials["state_int"] = trials.state_int.astype(object)
    trials.iloc[0, trials.columns.get_loc("state_int")] = state
    assert not select_baseline_trials(trials).baseline_valid.iloc[0]


def test_choice_updated_uses_immediate_unfiltered_next_row(
    trials: pd.DataFrame, bin_edges_s: np.ndarray,
) -> None:
    """Input (6,7) UTC table/(9,) relative-second edges; assert (6,8) states, return None."""
    original = trials.copy(deep=True)
    values, valid = build_context_values(trials, bin_edges_s, "choice_updated")

    assert values.shape == valid.shape == (6, 8)
    np.testing.assert_array_equal(values[0], [0, 0, 0, 0, 1, 1, 1, 1])
    # The next manual trial supplies known left; the next no-choice row supplies right.
    np.testing.assert_array_equal(values[1], [1, 1, 1, 1, 0, 0, 0, 0])
    assert valid[0].all() and valid[1].all()
    assert not valid[2, 4:].any()  # Immediate next context is dark, not later left.
    assert np.isnan(values[2, 4:]).all()
    assert not valid[3, :4].any()
    np.testing.assert_array_equal(values[3, 4:], [1, 1, 1, 1])
    assert valid[5, :4].all()
    assert not valid[5, 4:].any()
    assert np.isnan(values[5, 4:]).all()
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("next_state", [np.nan, "None", 2, 9, 0.5, np.inf])
def test_invalid_next_state_masks_post_bins_without_carry_forward(
    trials: pd.DataFrame, bin_edges_s: np.ndarray, next_state: object,
) -> None:
    """Input UTC table/(9,) second edges/state scalar; assert post mask, return None."""
    trials["state_int"] = trials.state_int.astype(object)
    trials.iloc[1, trials.columns.get_loc("state_int")] = next_state
    values, valid = build_context_values(trials, bin_edges_s, "choice_updated")

    np.testing.assert_array_equal(values[0, :4], [0, 0, 0, 0])
    assert valid[0, :4].all()
    assert not valid[0, 4:].any()
    assert np.isnan(values[0, 4:]).all()


def test_trial_window_keeps_entering_state_including_terminal_post_bins(
    trials: pd.DataFrame, bin_edges_s: np.ndarray,
) -> None:
    """Input UTC table/(9,) second edges; (6,8) states hold entering value, return None."""
    values, valid = build_context_values(trials, bin_edges_s, "trial_window")
    np.testing.assert_array_equal(values[0], np.zeros(8))
    np.testing.assert_array_equal(values[1], np.ones(8))
    assert valid[5].all()
    np.testing.assert_array_equal(values[5], np.zeros(8))
    assert not valid[3].any()
    assert np.isnan(values[3]).all()


@pytest.mark.parametrize("edges,mode", [
    (np.array([-2.0, -0.25, 0.25, 2.0]), "choice_updated"),
    (np.array([-2.0, 0.0, 0.0, 2.0]), "choice_updated"),
    (np.array([-2.0, np.nan, 2.0]), "choice_updated"),
    (np.array([-2.0, 0.0, 2.0]), "reward_updated"),
])
def test_invalid_bin_geometry_or_timing_is_rejected(
    trials: pd.DataFrame, edges: np.ndarray, mode: str,
) -> None:
    """Input UTC table/relative-second edges/mode; ambiguous update fails, return None."""
    with pytest.raises(ValueError, match="bin|zero|edge|timing"):
        build_context_values(trials, edges, mode)


def test_complete_trial_tables_match_irregular_ids_despite_index_labels(trials: pd.DataFrame) -> None:
    """Input full UTC table; equal source identities ignore DataFrame index, return None."""
    source = trials.copy(deep=True).reset_index(drop=True)
    source["irrelevant_source_column"] = "unused"
    validate_trial_tables(trials, source)


@pytest.mark.parametrize("column,new_value", [
    ("cur_trial", 999), ("start_time", 110.01),
    ("choice_time", 115.01), ("state_int", 0),
])
def test_trial_table_identity_time_and_state_conflicts_are_errors(
    trials: pd.DataFrame, column: str, new_value: float,
) -> None:
    """Input UTC table/column/scalar (state and ID unitless); mismatch fails, return None."""
    source = trials.copy(deep=True)
    source.iloc[1, source.columns.get_loc(column)] = new_value
    with pytest.raises(ValueError, match=column):
        validate_trial_tables(trials, source)


@pytest.mark.parametrize("failure", ["duplicate_id", "reversed_time", "missing_row", "missing_column"])
def test_ambiguous_or_nonchronological_sources_cannot_be_joined_by_position(
    trials: pd.DataFrame, failure: str,
) -> None:
    """Input (6,7) UTC table/failure name; reject ambiguous complete mapping, return None."""
    source = trials.copy(deep=True)
    if failure == "duplicate_id":
        source.iloc[1, source.columns.get_loc("cur_trial")] = 10
    elif failure == "reversed_time":
        source = source.iloc[[1, 0, 2, 3, 4, 5]]
    elif failure == "missing_row":
        source = source.iloc[:-1]
    else:
        source = source.drop(columns="state_int")
    with pytest.raises(ValueError, match="trial|chronolog|start_time|state_int"):
        validate_trial_tables(trials, source)


def test_context_event_diagnostics_allow_delayed_logs_and_preserve_times() -> None:
    """No inputs; ordered (4,2) UTC events may lag choices; return None."""
    trial_table = pd.DataFrame({"cur_trial": [10, 11, 12, 13],
        "start_time": [100.0, 110.0, 120.0, 130.0],
        "choice_time": [105.0, 115.0, 125.0, 135.0], "state_int": [0, 0, 1, 1]})
    events = pd.DataFrame({"Time": [99.0, 106.0, 115.2, 136.0],
        "Event": ["enter_right_patch", "reward", "enter_left_patch", "choice"]})
    original = events.copy(deep=True)

    assert diagnose_context_events(trial_table, events) == []
    pd.testing.assert_frame_equal(events, original)


def test_conflicting_event_order_is_advisory_and_never_changes_context_values(
    trials: pd.DataFrame, bin_edges_s: np.ndarray,
) -> None:
    """Input UTC table/relative-second edges; diagnostics list is advisory, return None."""
    before_values, before_valid = build_context_values(trials, bin_edges_s, "choice_updated")
    events = pd.DataFrame({"Time": [99.0, 106.0, 116.0],
        "Event": ["enter_left_patch", "enter_right_patch", "enter_dark_period"]})

    diagnostics = diagnose_context_events(trials, events)
    assert diagnostics and all(isinstance(message, str) for message in diagnostics)
    assert any("state" in message.lower() or "context" in message.lower() for message in diagnostics)
    after_values, after_valid = build_context_values(trials, bin_edges_s, "choice_updated")
    np.testing.assert_array_equal(before_values, after_values)
    np.testing.assert_array_equal(before_valid, after_valid)


def test_absent_events_are_reported_without_terminal_fallback(
    trials: pd.DataFrame, bin_edges_s: np.ndarray,
) -> None:
    """Input UTC table/relative-second edges; absent events advisory, terminal NaN, return None."""
    diagnostics = diagnose_context_events(trials, pd.DataFrame(columns=["Time", "Event"]))
    assert diagnostics
    values, valid = build_context_values(trials, bin_edges_s, "choice_updated")
    assert not valid[-1, 4:].any()
    assert np.isnan(values[-1, 4:]).all()
