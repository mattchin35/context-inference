"""Chronological labels, broad no-reward outcomes, and shared block folds."""

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population_decoding.targets import (
    TARGET_NAMES, build_population_targets, assign_block_folds, validate_trial_tables,
)


def test_explicit_class_order_and_switch_labels():
    """No inputs; assert (N,5) integer labels, including both directional events."""
    labels = build_population_targets(
        np.array([0, 1, 0, 1, 0]), np.array([1, 1, 0, 0, 1]),
        np.array([1, 0, 1, 0, np.nan]), np.zeros(5),
    )
    assert TARGET_NAMES == ("context", "choice", "outcome", "left_choice_switch", "right_choice_switch")
    np.testing.assert_array_equal(labels.values[:, 2], [0, 2, 1, 3, -1])
    np.testing.assert_array_equal(labels.values[:, 3], [0, 1, 0, 0, -1])
    np.testing.assert_array_equal(labels.values[:, 4], [0, 0, 0, 1, -1])
    assert labels.valid[-1, :2].all()
    assert not labels.valid[-1, 2:].any()
    assert labels.class_names[2] == ("left_reward", "right_reward", "left_no_reward", "right_no_reward")


@pytest.mark.parametrize("bad_action,manual", [("no_choice", 0), (1, 1), (np.nan, 0), (1, np.nan)])
def test_unfiltered_invalid_successor_is_not_skipped(bad_action, manual):
    """Input successor action/manual flag; missing switch labels never become zeros."""
    labels = build_population_targets(
        np.zeros(3), np.array([1, bad_action, 0], dtype=object),
        np.zeros(3), np.array([0, manual, 0]),
    )
    assert not labels.valid[0, 3:].any()
    assert labels.valid[0, :3].all()
    assert all(labels.reasons[0, 3:])


def test_successor_needs_only_known_animal_action():
    """No inputs; unknown successor reward/context do not invalidate current switch."""
    labels = build_population_targets([1, np.nan], [1, 0], [0, np.nan], [0, 0])
    assert labels.values[0, 3] == 1
    assert labels.valid[0, 3]


def test_folds_are_global_and_block_grouped():
    """No inputs; return no data, verify deterministic (N,) map before target masks."""
    blocks = np.repeat(np.arange(10), 3)
    baseline = np.ones(30, bool)
    baseline[0] = False
    first, reason = assign_block_folds(blocks, baseline)
    second, _ = assign_block_folds(blocks, baseline)
    assert not reason
    np.testing.assert_array_equal(first, second)
    assert first[0] == -1
    assert set(first[baseline]) == set(range(5))
    for block in np.unique(blocks):
        assert np.unique(first[baseline & (blocks == block)]).size == 1
    missing, reason = assign_block_folds(np.arange(4), np.ones(4, bool))
    assert (missing == -1).all() and reason


def test_used_source_fields_only(population_config_path):
    """Input source-session config; correctness is unnecessary, used-field mismatch fails."""
    source = pd.read_csv(population_config_path.parent / "trials.csv").drop(columns="correct")
    augmented = source.copy()
    augmented["unrelated_model_value"] = 123
    validate_trial_tables(augmented, source)
    augmented.loc[1, "action"] = 1
    with pytest.raises(ValueError, match="action"):
        validate_trial_tables(augmented, source)


def test_trial_identity_and_chronology_are_not_repaired(population_config_path):
    """Input synthetic sources; reordered/duplicated rows raise rather than join/sort."""
    source = pd.read_csv(population_config_path.parent / "trials.csv")
    with pytest.raises(ValueError):
        validate_trial_tables(source.iloc[::-1], source)
    duplicate = source.copy()
    duplicate.loc[1, "cur_trial"] = duplicate.loc[0, "cur_trial"]
    with pytest.raises(ValueError):
        validate_trial_tables(duplicate, duplicate)
