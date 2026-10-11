"""Eligibility and fold primitives for pooled-window task-variable decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.neural_analysis.task_decoding import modeling


POOLED_REASON_CODES = (
    "baseline_ineligible",
    "target_ineligible",
    "neural_window_unavailable",
    "target_outer_unavailable",
    "empty_training_subset",
    "empty_test_subset",
    "training_class_unavailable",
    "test_class_unavailable",
    "training_target_unavailable",
    "test_target_unavailable",
    "pfc_transform_unavailable",
    "hpc_transform_unavailable",
    "regional_transform_unavailable",
    "no_features",
    "fit_convergence_failure",
    "outer_estimator_failure",
    "invalid_probability",
    "invalid_prediction",
    "invalid_metric",
    "source_score_unavailable",
)


@dataclass(frozen=True)
class MasterEligibility:
    """Full-table target eligibility and fixed-precedence unavailable reasons.

    Parameters
    ----------
    target_eligible : numpy.ndarray
        Read-only Boolean target-valid mask, shape ``(trial,)``.
    master_eligible : numpy.ndarray
        Read-only Boolean baseline/target/bilateral intersection, shape
        ``(trial,)``.
    reason_index : numpy.ndarray
        Read-only int16 indices into :data:`POOLED_REASON_CODES`, shape
        ``(trial,)``; eligible rows contain ``-1``.
    """

    target_eligible: np.ndarray
    master_eligible: np.ndarray
    reason_index: np.ndarray


def _boolean_trial_mask(value: np.ndarray, name: str) -> np.ndarray:
    """Validate one exact one-dimensional Boolean trial mask.

    Parameters
    ----------
    value : numpy.ndarray
        Proposed dimensionless mask.
    name : str
        Field name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Independent Boolean copy with shape ``(trial,)``.
    """
    mask = np.asarray(value)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional Boolean mask.")
    if mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a Boolean mask.")
    return mask.copy()


def build_master_eligibility(
    baseline_eligible: np.ndarray,
    target_eligible: np.ndarray,
    bilateral_full_window_coverage: np.ndarray,
) -> MasterEligibility:
    """Combine full-table eligibility while preserving independent audit facts.

    Parameters
    ----------
    baseline_eligible, target_eligible, bilateral_full_window_coverage : numpy.ndarray
        Aligned one-dimensional Boolean masks with shape ``(trial,)``.
        Baseline and target eligibility are dimensionless; bilateral coverage
        means both neural regions cover the complete choice-relative window.

    Returns
    -------
    MasterEligibility
        Independent target mask, complete intersection, and one reason per
        unavailable row using baseline, target, then neural precedence.
    """
    baseline = _boolean_trial_mask(baseline_eligible, "baseline_eligible")
    target = _boolean_trial_mask(target_eligible, "target_eligible")
    bilateral = _boolean_trial_mask(
        bilateral_full_window_coverage,
        "bilateral_full_window_coverage",
    )
    if baseline.shape != target.shape or baseline.shape != bilateral.shape:
        raise ValueError("Eligibility masks must have the same trial shape.")

    master = baseline & target & bilateral
    reason_index = np.full(master.shape, -1, dtype=np.int16)
    reason_index[~baseline] = POOLED_REASON_CODES.index("baseline_ineligible")
    reason_index[baseline & ~target] = POOLED_REASON_CODES.index("target_ineligible")
    reason_index[baseline & target & ~bilateral] = POOLED_REASON_CODES.index(
        "neural_window_unavailable"
    )
    for array in (target, master, reason_index):
        array.setflags(write=False)
    return MasterEligibility(
        target_eligible=target,
        master_eligible=master,
        reason_index=reason_index,
    )


def make_master_outer_split(
    target_values: np.ndarray,
    block_ids: np.ndarray,
    master_eligible: np.ndarray,
    *,
    target_family: str,
    fold_count: int,
) -> modeling.SplitPlan:
    """Build one target-wide grouped fold map on the common tensor-row axis.

    Parameters
    ----------
    target_values : numpy.ndarray
        Categorical labels or numerical target values, shape ``(tensor_trial,)``.
        Values outside ``master_eligible`` are ignored and may be nonfinite.
    block_ids : numpy.ndarray
        Exact behavioral block identities, shape ``(tensor_trial,)``.
    master_eligible : numpy.ndarray
        Boolean target-specific eligibility on the common bilateral tensor-row
        axis, shape ``(tensor_trial,)``.
    target_family : {"categorical", "numerical"}
        Selects the existing deterministic grouped splitter.
    fold_count : int
        Requested outer grouped-fold count, exactly three or five.

    Returns
    -------
    modeling.SplitPlan
        Fold IDs with shape ``(tensor_trial,)`` and ``-1`` on excluded rows.
        Available split indices are positions on this same tensor-row axis.
        Scientific insufficiency returns a full-length unavailable plan.
    """
    values = np.asarray(target_values)
    groups = np.asarray(block_ids)
    eligible = _boolean_trial_mask(master_eligible, "master_eligible")
    if values.ndim != 1 or groups.ndim != 1:
        raise ValueError("target_values and block_ids must be one-dimensional.")
    if values.shape != groups.shape or values.shape != eligible.shape:
        raise ValueError("Target, block, and eligibility arrays must share one shape.")

    eligible_indices = np.flatnonzero(eligible)
    local_plan = modeling.make_outer_splits(
        values[eligible_indices],
        groups[eligible_indices],
        target_family=target_family,
        fold_count=fold_count,
    )
    full_fold_ids = np.full(values.shape, -1, dtype=int)
    if not local_plan.is_available:
        return modeling.SplitPlan(
            fold_ids=full_fold_ids,
            splits=(),
            is_available=False,
            unavailable_reason=local_plan.unavailable_reason,
        )

    full_fold_ids[eligible_indices] = local_plan.fold_ids
    splits = tuple(
        modeling.GroupedSplit(
            fold_id=split.fold_id,
            train_indices=eligible_indices[split.train_indices],
            test_indices=eligible_indices[split.test_indices],
        )
        for split in local_plan.splits
    )
    return modeling.SplitPlan(
        fold_ids=full_fold_ids,
        splits=splits,
        is_available=True,
        unavailable_reason=None,
    )
