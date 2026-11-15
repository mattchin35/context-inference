"""Behavioral matching and original-row extraction contracts, independent of neural data."""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import selection
from src.tests.neural_analysis.event_sequences.example_fixtures import example_trials
from src.tests.neural_analysis.event_sequences.test_additional_selection import _episode


@pytest.mark.parametrize("family", ["probe", "omission_probe"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("repeats", [1, 2, 3, 4])
@pytest.mark.parametrize("reward", [0, 1])
def test_probe_return_policy_accepts_reward_and_omission(family, side, repeats, reward):
    """Input: unitless family/side/x/reward. Require exact (1,S) source positions."""
    trials = _episode((family, side, repeats))
    trials.iloc[-1, trials.columns.get_loc("reward")] = reward
    original = trials.copy(deep=True)
    matches = selection.select_additional_sequences(trials, probe_return_outcome="rewarded_or_omission")
    np.testing.assert_array_equal(matches[(family, side, repeats)], [np.arange(len(trials))])
    assert len(selection.select_additional_sequences(trials)[(family, side, repeats)]) == reward
    pd.testing.assert_frame_equal(trials, original)


@pytest.mark.parametrize("family", ["probe", "omission_probe"])
@pytest.mark.parametrize("column,value", [
    ("correct", 0), ("reward", np.nan), ("reward", "None"),
    ("cur_block", 99), ("state_int", 0), ("experimenter_reward_given", 1),
    ("cur_trial", 999), ("choice_time", np.nan),
])
def test_relaxed_return_policy_preserves_episode_constraints(family, column, value):
    """Input: invalid last-row scalar. Reject its entire (S,C) source episode."""
    trials = _episode((family, "left", 1)).astype(object)
    trials.iloc[-1, trials.columns.get_loc(column)] = value
    matches = selection.select_additional_sequences(trials, probe_return_outcome="rewarded_or_omission")
    assert matches[(family, "left", 1)].shape[0] == 0


def test_invalid_policy_is_rejected_and_other_families_are_unchanged(example_trials):
    """Input: full (N,C) trials. Compare every non-probe array and reject unknown policy."""
    strict = selection.select_additional_sequences(example_trials)
    relaxed = selection.select_additional_sequences(example_trials, probe_return_outcome="rewarded_or_omission")
    for key in strict:
        if key[0] not in ("probe", "omission_probe"):
            np.testing.assert_array_equal(strict[key], relaxed[key])
    with pytest.raises(ValueError, match="probe_return_outcome"):
        selection.select_additional_sequences(example_trials, probe_return_outcome="anything")


def test_catalogue_contains_all_families_and_preserves_source_rows(example_trials):
    """Input: full (N,C) table. Return checks for all 32 (E,S) condition arrays."""
    examples = importlib.import_module("src.neural_analysis.event_sequences.examples")
    original = example_trials.copy(deep=True)
    matches = examples.find_behavior_examples(example_trials)
    assert len(matches) == 32
    assert {key[0] for key, positions in matches.items() if len(positions)} == {
        "probe", "omission_probe", "ideal_transition", "lucky_omission_switch", "indecisive_transition",
    }
    # The indecisive episode's rewarded tail independently qualifies as a probe.
    np.testing.assert_array_equal(matches[("probe", "left", 1)], [[0, 1, 2], [4, 5, 6], [24, 25, 26]])
    rows = examples.extract_example_trials(example_trials, np.array([4, 5, 6]), trials_before=5, trials_after=5)
    assert rows.row_position.tolist() == list(range(12))
    assert rows.trial_id.tolist() == list(range(501, 513))
    assert rows.index_label.tolist() == ["duplicate"] * 12
    assert rows.loc[rows.in_episode, "row_position"].tolist() == [4, 5, 6]
    assert rows.attrs["context_truncated"] is True
    assert pd.isna(rows.loc[3, "action"])
    pd.testing.assert_frame_equal(example_trials, original)


@pytest.mark.parametrize("positions", [np.array([], dtype=int), np.array([0, 2]), np.array([-1, 0]), np.array([1.5])])
def test_example_extraction_rejects_invalid_positions(example_trials, positions):
    """Input: invalid unitless position vector. Raise instead of selecting wrong source rows."""
    examples = importlib.import_module("src.neural_analysis.event_sequences.examples")
    with pytest.raises(ValueError):
        examples.extract_example_trials(example_trials, positions)


def test_intervening_nonchoice_cannot_be_bridged():
    """Input: four chronological rows. No (E,S) probe may skip the inserted non-choice."""
    examples = importlib.import_module("src.neural_analysis.event_sequences.examples")
    trials = _episode(("probe", "left", 1))
    blocker = trials.iloc[[1]].copy()
    blocker["action"] = np.nan
    trials = pd.concat([trials.iloc[:1], blocker, trials.iloc[1:]])
    trials["cur_trial"] = np.arange(4)
    trials["choice_time"] = 1_700_000_000 + np.arange(4)
    assert examples.find_behavior_examples(trials)[("probe", "left", 1)].shape == (0, 3)
