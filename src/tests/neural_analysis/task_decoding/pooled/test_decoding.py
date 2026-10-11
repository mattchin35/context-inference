"""Master-eligibility and fold tests for pooled-window decoding."""

import numpy as np
import pytest

from src.neural_analysis.task_decoding.pooled import decoding


def test_master_eligibility_preserves_target_mask_and_reason_precedence():
    """Independent masks and the first failing full-table reason should be stable."""
    baseline = np.array([False, True, True, True, False, True], dtype=bool)
    target = np.array([False, False, True, True, True, True], dtype=bool)
    bilateral = np.array([False, False, False, True, True, True], dtype=bool)

    result = decoding.build_master_eligibility(baseline, target, bilateral)

    np.testing.assert_array_equal(result.target_eligible, target)
    np.testing.assert_array_equal(
        result.master_eligible,
        np.array([False, False, False, True, False, True]),
    )
    reason_names = np.full(result.reason_index.shape, "", dtype="U32")
    unavailable = result.reason_index >= 0
    reason_names[unavailable] = np.asarray(decoding.POOLED_REASON_CODES)[
        result.reason_index[unavailable]
    ]
    assert reason_names.tolist() == [
        "baseline_ineligible",
        "target_ineligible",
        "neural_window_unavailable",
        "",
        "baseline_ineligible",
        "",
    ]


@pytest.mark.parametrize(
    ("baseline", "target", "bilateral"),
    [
        (np.ones((2, 1), dtype=bool), np.ones(2, dtype=bool), np.ones(2, dtype=bool)),
        (np.ones(2, dtype=int), np.ones(2, dtype=bool), np.ones(2, dtype=bool)),
        (np.ones(2, dtype=bool), np.ones(3, dtype=bool), np.ones(2, dtype=bool)),
    ],
)
def test_master_eligibility_rejects_malformed_axes(baseline, target, bilateral):
    """Eligibility inputs must be aligned one-dimensional Boolean trial masks."""
    with pytest.raises(ValueError, match="Boolean|one-dimensional|shape"):
        decoding.build_master_eligibility(baseline, target, bilateral)


def test_master_outer_split_projects_local_folds_back_to_tensor_rows():
    """Excluded tensor rows should retain -1 while eligible blocks remain intact."""
    target_values = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan])
    block_ids = np.array([0, 0, "excluded", 1, 1, 2, 2, 3, 3, 4, 4, "excluded"], dtype=object)
    master_eligible = np.array(
        [True, True, False, True, True, True, True, True, True, True, True, False],
        dtype=bool,
    )

    plan = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="numerical",
        fold_count=5,
    )

    assert plan.is_available
    np.testing.assert_array_equal(plan.fold_ids[~master_eligible], np.array([-1, -1]))
    assert np.all(plan.fold_ids[master_eligible] >= 0)
    for split in plan.splits:
        assert np.all(master_eligible[split.train_indices])
        assert np.all(master_eligible[split.test_indices])
        train_blocks = set(block_ids[split.train_indices])
        test_blocks = set(block_ids[split.test_indices])
        assert train_blocks.isdisjoint(test_blocks)


def test_master_outer_split_is_deterministic_and_condition_independent():
    """One target fold map should be reused even when a condition lacks classes."""
    block_ids = np.repeat(np.arange(6), 2)
    target_values = np.tile(np.array([0, 1], dtype=int), 6)
    master_eligible = np.ones(target_values.size, dtype=bool)

    first = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="categorical",
        fold_count=3,
    )
    repeated = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="categorical",
        fold_count=3,
    )

    assert first.is_available
    np.testing.assert_array_equal(first.fold_ids, repeated.fold_ids)
    condition_mask = target_values == 0
    assert any(
        np.unique(target_values[split.test_indices[condition_mask[split.test_indices]]]).size < 2
        for split in first.splits
    )
    np.testing.assert_array_equal(first.fold_ids, repeated.fold_ids)


def test_master_outer_split_returns_full_length_unavailable_plan():
    """Insufficient eligible groups should not shrink the tensor-row fold axis."""
    target_values = np.array([0, 1, 0, 1, 0, 1])
    block_ids = np.array([0, 0, 1, 1, 2, 2])
    master_eligible = np.array([True, True, False, False, False, False])

    plan = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="categorical",
        fold_count=3,
    )

    assert not plan.is_available
    assert plan.fold_ids.shape == target_values.shape
    assert np.all(plan.fold_ids == -1)
    assert plan.splits == ()
    assert plan.unavailable_reason
