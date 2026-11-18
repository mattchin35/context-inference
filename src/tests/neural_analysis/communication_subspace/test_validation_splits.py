"""Whole-choice chronological validation and physical-window purging."""

from __future__ import annotations

import numpy as np
import pytest

from src.neural_analysis.communication_subspace.preparation import build_validation_splits


def test_five_chronological_folds_and_latest_inner_block():
    """45 separated choices yield five nine-choice tests and eight-choice validations."""
    times = 1_700_000_000.0 + 10 * np.arange(45)
    splits = build_validation_splits(times)

    assert len(splits) == 5
    for fold, split in enumerate(splits):
        np.testing.assert_array_equal(split.outer_test_indices, np.arange(9 * fold, 9 * (fold + 1)))
        assert len(split.outer_train_indices) == 36
        assert len(split.inner_validation_indices) == 8  # ceil(20% of 36)
        np.testing.assert_array_equal(split.inner_validation_indices, split.outer_train_indices[-8:])
        assert len(split.inner_train_indices) == 28
        assert len(split.outer_purged_indices) == 0
        assert len(split.inner_purged_indices) == 0
        assert not set(split.outer_train_indices) & set(split.outer_test_indices)
        assert not set(split.inner_train_indices) & set(split.inner_validation_indices)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([split.outer_test_indices for split in splits])), np.arange(45),
    )


def test_remainder_goes_to_earliest_outer_blocks():
    """Five chronological folds of 43 choices use sizes 9,9,9,8,8."""
    splits = build_validation_splits(10.0 * np.arange(43))
    assert [len(split.outer_test_indices) for split in splits] == [9, 9, 9, 8, 8]
    assert [int(split.outer_test_indices[0]) for split in splits] == [0, 9, 18, 27, 35]


def test_outer_and_inner_purges_use_physical_half_open_intervals():
    """Overlapping windows are purged; intervals meeting at an endpoint survive."""
    times = 10.0 * np.arange(45)
    # Outer fold 1 tests positions 9..17. Position 8 overlaps 9; position 18
    # meets test 17's right endpoint and must remain eligible.
    times[8] = times[9] - 2.0
    times[18] = times[17] + 4.0
    # Outer purge leaves 35 training choices, so inner validation is 38..44.
    # Position 37 overlaps 38 and must leave inner training.
    times[37] = times[38] - 2.0
    splits = build_validation_splits(times)
    split = splits[1]

    assert 8 in split.outer_purged_indices
    assert 8 not in split.outer_train_indices
    assert 18 in split.outer_train_indices
    assert 37 in split.inner_purged_indices
    assert 37 not in split.inner_train_indices
    assert 38 in split.inner_validation_indices
    for train in split.outer_train_indices:
        for test in split.outer_test_indices:
            assert abs(times[train] - times[test]) >= 4.0
    for train in split.inner_train_indices:
        for valid in split.inner_validation_indices:
            assert abs(times[train] - times[valid]) >= 4.0


def test_insufficient_choices_or_invalid_times_fail_without_fold_reduction():
    """The shared five-fold plan must stay valid and use ordered finite UTC times."""
    for times in (10.0 * np.arange(9), np.array([0.0, 10.0, np.nan] + list(10.0 * np.arange(3, 45))),
                  np.array([0.0, 10.0, 9.0] + list(10.0 * np.arange(3, 45)))):
        with pytest.raises(ValueError, match="choice|fold|time|finite|chronological"):
            build_validation_splits(times)


def test_dense_overlaps_fail_when_purging_exhausts_training():
    """Repeated choice windows cannot create a leakage-free train/validation plan."""
    times = 0.1 * np.arange(45)
    with pytest.raises(ValueError, match="purge|overlap|choice|fold|train"):
        build_validation_splits(times)
