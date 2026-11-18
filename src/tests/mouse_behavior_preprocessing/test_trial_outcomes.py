"""Explicit outcomes keep invalid attempts distinct from missing observations."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import collect_events, general_behavior_assessment
from src.behavior_analysis import trial_policy_models, trial_policy_results
from src.behavior_analysis.gather_trial_features import TaskParams
from src.mouse_behavior_preprocessing import process_behavior_log


def parse_events(events, notes=None):
    """Parse N event-name strings and optional N notes into a T-row trial table.

    Synthetic timestamps are seconds; actions retain 0=right, 1=left.
    """
    frame = pd.DataFrame({"Time": np.arange(len(events), dtype=float),
                          "Event": events, "Note": notes or [""] * len(events)})
    return process_behavior_log.iterate_trials(
        frame, collect_events.get_context_events(frame, event_list=[]),
        collect_events.get_choice_events(frame, event_list=[]),
        collect_events.get_reward_events(frame, event_list=[]),
        collect_events.get_stimulus_events(frame, event_list=[]))


def test_explicit_switch_retains_completed_final_attempt():
    """A terminal switch produces one no-choice row, including at end-of-file."""
    table = parse_events(["enter_left_patch", "trial_start", "invalid_switch", "trial_stop"])
    assert len(table) == 1
    row = table.iloc[0]
    assert row.trial_outcome == "invalid_switch"
    assert row.action == "no_choice"
    assert row.correct == row.choice_time == "None"
    assert row.reward == row.experimenter_reward_given == 0
    assert row.cur_trial == 0


@pytest.mark.parametrize("event,outcome", [
    ("correct_choice_left_patch", "choice"),
    ("giving_reward_left_patch", "manual_reward"),
])
def test_completed_final_choice_and_manual_reward_are_retained(event, outcome):
    """Existing terminal outcome events finalize the last row without another start."""
    table = parse_events(["enter_left_patch", "trial_start", event], ["", "", "reward_True"])
    assert len(table) == 1
    assert table.iloc[0].trial_outcome == outcome


def test_unknown_attempt_is_preserved_without_inventing_switch():
    """An earlier empty attempt remains unknown; the final unfinished one is omitted."""
    table = parse_events(["enter_left_patch", "trial_start", "trial_stop", "trial_start"])
    assert len(table) == 1
    assert table.iloc[0].trial_outcome == "unknown"
    assert table.iloc[0].experimenter_reward_given == "None"


@pytest.mark.parametrize("events,notes", [
    (["invalid_switch", "correct_choice_left_patch"], ["", "reward_True"]),
    (["correct_choice_left_patch", "invalid_switch"], ["reward_True", ""]),
    (["invalid_switch", "giving_reward_left_patch"], ["", "reward_True"]),
])
def test_conflicting_terminal_outcomes_raise_with_trial_id(events, notes):
    """Contradictory outcome evidence must not silently overwrite an attempt."""
    with pytest.raises(ValueError, match="trial 0"):
        parse_events(["enter_left_patch", "trial_start", *events], ["", "", *notes])


@pytest.fixture
def outcome_table():
    """Return four chronological dimensionless trials, with one explicit invalid row."""
    return pd.DataFrame({"cur_trial": [0, 1, 2, 3], "cur_block": [0] * 4,
                         "action": [1, "no_choice", 0, 1], "reward": [1, 0, 0, 1],
                         "experimenter_reward_given": [0] * 4,
                         "trial_outcome": ["choice", "invalid_switch", "choice", "choice"]})


def test_invalid_rows_preserve_candidate_history_and_valid_denominators(outcome_table):
    """Removing an explicit invalid attempt leaves every later candidate score unchanged."""
    history = trial_policy_models.prepare_trial_policy_history(outcome_table, session_id="synthetic")
    candidates = trial_policy_models.resolve_trial_policy_candidates()
    scores, _ = trial_policy_models.replay_trial_policy_candidates(history, candidates, TaskParams())
    retained = outcome_table.iloc[[0, 2, 3]]
    compressed = trial_policy_models.prepare_trial_policy_history(retained, session_id="synthetic")
    expected, _ = trial_policy_models.replay_trial_policy_candidates(compressed, candidates, TaskParams())
    np.testing.assert_array_equal(scores[[0, 2, 3]], expected)
    assert not history.comparison_valid[1] and not history.history_update_valid[1]
    assert "invalid_switch" in history.comparison_exclusion_reason[1]
    assert general_behavior_assessment.make_valid_behavior_mask(outcome_table).tolist() == [True, False, True, True]
    messages = trial_policy_results.trial_policy_warnings(history)
    assert len(messages) == 1 and "invalid switch" in messages[0]
    assert "unknown manual" not in messages[0]


def test_outcome_counts_and_cache_identity(outcome_table):
    """Outcome labels affect provenance even when normalized actions are identical."""
    explicit = trial_policy_models.prepare_trial_policy_history(outcome_table, session_id="synthetic")
    legacy = trial_policy_models.prepare_trial_policy_history(outcome_table.drop(columns="trial_outcome"), session_id="synthetic")
    assert explicit.metadata["input_digest"] != legacy.metadata["input_digest"]
    assert explicit.metadata["trial_outcome_counts"] == {
        "recorded_attempts": 4, "valid_animal_choices": 3, "invalid_switch": 1,
        "manual_reward": 0, "unknown": 0}


def test_legacy_unknowns_and_valid_choice_unknown_flags_are_distinct(outcome_table):
    """No-choice unknowns are reported separately without weakening choice exclusions."""
    table = outcome_table.astype({"experimenter_reward_given": object})
    table.loc[1, "trial_outcome"] = "unknown"
    table.loc[[1, 2], "experimenter_reward_given"] = "None"
    history = trial_policy_models.prepare_trial_policy_history(table, session_id="synthetic")
    messages = trial_policy_results.trial_policy_warnings(history)
    assert any("unclassified no-choice" in message and "(1)" in message for message in messages)
    assert any("unknown manual-reward flag" in message and "(2)" in message for message in messages)
    assert not history.comparison_valid[1:3].any()


def test_outcomes_survive_csv_roundtrip(outcome_table, tmp_path):
    """CSV preserves T-row outcomes, action conventions, and invalid exclusion reasons."""
    path = tmp_path / "trials.csv"
    outcome_table.to_csv(path, index=False)
    restored = pd.read_csv(path, na_filter=False)
    assert restored.trial_outcome.tolist() == outcome_table.trial_outcome.tolist()
    history = trial_policy_models.prepare_trial_policy_history(restored, session_id="synthetic")
    assert "invalid_switch" in history.comparison_exclusion_reason[1]


def test_invalid_outcome_cannot_disguise_a_numeric_choice(outcome_table):
    """An inconsistent explicit invalid row fails at the analysis input boundary."""
    table = outcome_table.copy()
    table.loc[1, "action"] = 0
    with pytest.raises(ValueError, match="invalid_switch"):
        trial_policy_models.prepare_trial_policy_history(table, session_id="synthetic")
