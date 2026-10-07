"""Grouped task-variable decoder construction and auditing helpers.

This module operates on already matched PFC/HPC firing-rate tensors.  It does
not load experimental files or modify trial eligibility. It implements both
fixed and leakage-safe nested tuned regularization with audit-ready records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import numbers
import time
from typing import Any
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.neural_analysis.task_decoding.config import (
    COEFFICIENT_TOLERANCE,
    FROZEN_ESTIMATOR_CONTROLS,
    FROZEN_TUNING_GRID,
)


_REGION_CONFIGURATIONS = ("PFC", "HPC", "PFC+HPC")
_REPRESENTATIONS = ("pca", "units")


def _timed_call(
    function: Callable[..., Any],
    *args: object,
    timing_callback: Callable[[dict[str, object]], None] | None,
    timing_operation: str,
    timing_region: str | None,
    timing_representation: str | None,
    **kwargs: object,
) -> Any:
    """Call one existing modeling operation and optionally report its duration.

    Parameters
    ----------
    function : callable
        Existing split, transform, feature, or estimator function.
    *args, **kwargs : object
        Arguments forwarded unchanged to ``function``.
    timing_callback : callable or None
        Consumer receiving one JSON-safe timing event after successful return.
    timing_operation : str
        Stable operation label.
    timing_region : str or None
        Regional transform/cell identity, or None for split construction.
    timing_representation : str or None
        ``pca``/``units`` for cell operations, otherwise None.

    Returns
    -------
    object
        Exact value returned by ``function``.
    """
    started = time.perf_counter()
    value = function(*args, **kwargs)
    elapsed_seconds = time.perf_counter() - started
    if timing_callback is not None:
        timing_callback(
            {
                "operation": timing_operation,
                "region_configuration": timing_region,
                "representation": timing_representation,
                "elapsed_seconds": elapsed_seconds,
            }
        )
    return value


@dataclass(frozen=True)
class GroupedSplit:
    """One deterministic grouped train/test partition.

    Parameters
    ----------
    fold_id : int
        Zero-based identifier of this grouped fold.
    train_indices, test_indices : np.ndarray
        One-dimensional integer original matched-tensor row positions with
        shapes ``(n_train,)`` and ``(n_test,)``.  They are dimensionless and
        disjoint.
    """

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class SplitPlan:
    """Validated grouped folds or an explicit scientific unavailable state.

    Parameters
    ----------
    fold_ids : np.ndarray
        One-dimensional integer array aligned to all input rows.  Valid rows
        contain zero-based fold IDs; excluded inner-plan rows are ``-1``.
    splits : tuple[GroupedSplit, ...]
        Valid deterministic grouped folds.  It is empty when unavailable.
    is_available : bool
        Whether every requested split was constructible and valid.
    unavailable_reason : str or None
        Human-readable scientific reason when ``is_available`` is false.
    """

    fold_ids: np.ndarray
    splits: tuple[GroupedSplit, ...]
    is_available: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class RegionTransform:
    """Training-only scaling and PCA for one regional firing-rate tensor.

    Parameters
    ----------
    is_available : bool
        False only when no nonconstant finite training-unit feature remains.
    unavailable_reason : str or None
        Scientific missing-feature reason when unavailable.
    unit_means_hz, unit_scales_hz : np.ndarray
        One-dimensional training pooled means and standard deviations in Hz,
        each shape ``(n_input_units,)``.
    usable_unit_indices : np.ndarray
        One-dimensional positions of nonconstant usable input units, shape
        ``(n_usable_units,)``.
    pca_model : sklearn.decomposition.PCA or None
        PCA fit only on pooled standardized training observations.
    component_limit, effective_component_count : int
        Deterministic dimensional limit and fitted retained component count.
    cap_warning : str or None
        Visible message when requested PCs exceed the fitted component count.
    input_unit_count : int
        Width of tensors accepted by transform methods.
    requested_pc_count : int
        Positive region-specific requested PC count.
    """

    is_available: bool
    unavailable_reason: str | None
    unit_means_hz: np.ndarray
    unit_scales_hz: np.ndarray
    usable_unit_indices: np.ndarray
    pca_model: PCA | None
    component_limit: int
    effective_component_count: int
    cap_warning: str | None
    input_unit_count: int
    requested_pc_count: int

    def transform_units(self, rate_tensor_hz: np.ndarray) -> np.ndarray:
        """Apply fixed training scaling and select usable direct units.

        Parameters
        ----------
        rate_tensor_hz : np.ndarray
            Float firing-rate array shaped ``(trial, time_bin, input_unit)`` in
            Hz.  Its final axis must equal ``input_unit_count``.

        Returns
        -------
        np.ndarray
            Standardized direct-unit values shaped
            ``(trial, time_bin, usable_unit)`` in pooled-training standard
            deviation units.

        Raises
        ------
        ValueError
            If this transform is unavailable or input dimensions/values are
            invalid.
        """
        normalized = _validate_rate_tensor(rate_tensor_hz, "rate_tensor_hz")
        if not self.is_available:
            raise ValueError(self.unavailable_reason or "Regional features are unavailable.")
        if normalized.shape[2] != self.input_unit_count:
            raise ValueError("rate_tensor_hz unit axis does not match this regional transform.")
        # Select first so removed constant columns never enter a division.
        usable_means = self.unit_means_hz[self.usable_unit_indices]
        usable_scales = self.unit_scales_hz[self.usable_unit_indices]
        usable_rates = normalized[:, :, self.usable_unit_indices]
        return (usable_rates - usable_means) / usable_scales

    def transform_pca(self, rate_tensor_hz: np.ndarray) -> np.ndarray:
        """Apply training scaling and the fitted regional PCA without rescaling PCs.

        Parameters
        ----------
        rate_tensor_hz : np.ndarray
            Float firing-rate array shaped ``(trial, time_bin, input_unit)`` in
            Hz.  Its final axis must equal ``input_unit_count``.

        Returns
        -------
        np.ndarray
            PCA scores shaped ``(trial, time_bin, effective_component)`` in
            PCA-score units with no second scaler.

        Raises
        ------
        ValueError
            If this transform is unavailable or input dimensions/values are
            invalid.
        """
        if not self.is_available or self.pca_model is None:
            raise ValueError(self.unavailable_reason or "Regional PCA features are unavailable.")
        standardized = self.transform_units(rate_tensor_hz)
        trial_count, time_count, _unit_count = standardized.shape
        scores = self.pca_model.transform(standardized.reshape(-1, standardized.shape[2]))
        return scores.reshape(trial_count, time_count, self.effective_component_count)


@dataclass(frozen=True)
class FeatureMatrix:
    """One time-bin feature matrix with stable inspectable identities.

    Parameters
    ----------
    values : np.ndarray
        Floating feature matrix shaped ``(trial, feature)``.  Direct values
        are pooled-training standardized units; PCA values are PC scores.
    feature_ids : tuple[str, ...]
        Stable feature identifiers in the final matrix-column order.
    """

    values: np.ndarray
    feature_ids: tuple[str, ...]


@dataclass(frozen=True)
class FitResult:
    """One fitted estimator's held-out score and provenance values.

    Parameters
    ----------
    is_valid : bool
        Whether fit, predictions, coefficients, and held-out metrics are
        finite and scientifically valid.
    reason : str or None
        Scientific invalidity reason, never an unexpected programming error.
    coefficients : np.ndarray
        One-dimensional feature coefficients shaped ``(n_features,)`` on the
        input feature scale. Invalid results retain an ``(n_features,)`` NaN
        vector so every feature position remains inspectable.
    intercept : float
        Fitted scalar intercept on the estimator's native scale; NaN when not
        available.
    metrics : dict[str, float]
        Held-out metrics: classification has balanced accuracy and AUC;
        numerical regression has native-target R2.
    positive_class : int or float or None
        Recorded categorical positive class, normally one, or None for
        numerical targets.
    positive_scores : np.ndarray or None
        One-dimensional categorical positive-class held-out probabilities.
    convergence_status : str
        ``"converged"``, ``"warning"``, or ``"invalid"``.
    """

    is_valid: bool
    reason: str | None
    coefficients: np.ndarray
    intercept: float
    metrics: dict[str, float]
    positive_class: int | float | None
    positive_scores: np.ndarray | None
    convergence_status: str


@dataclass(frozen=True)
class FoldRecord:
    """Audit record for one target/time/region/representation outer-fold fit.

    Parameters
    ----------
    record_key : tuple[int, int, str, str]
        Stable ``(outer_fold, time_bin, region, representation)`` identity.
    target_identifier : str
        Original target name retained for result-table provenance.
    inner_fold_count : int
        Configured three-fold inner-CV count. It governs tuned selection and is
        retained as inactive configuration provenance for fixed records.
    outer_fold_id, time_bin_index : int
        Zero-based grouped fold and event-relative time-bin positions.
    outer_test_indices, inner_selection_indices : np.ndarray
        Dimensionless matched-trial row positions. The selection array is an
        empty ``(0,)`` array for fixed records and a defensive global outer-
        training-row copy shaped ``(n_outer_train,)`` for tuned records.
    region_configuration, representation, regularization_mode : str
        Requested region, feature representation, and decoder mode labels.
    is_valid : bool
        Whether the fit and all held-out outputs are scientifically valid.
    status : str
        ``"valid"`` or ``"unavailable"`` audit state for this fold cell.
    reason : str or None
        Scientific invalidity reason when the cell is unavailable.
    train_count, test_count : int
        Numbers of matched trials in the outer training and test partitions.
    train_class_counts, test_class_counts : dict[int, int] or None
        Binary label counts for categorical targets, otherwise None.
    requested_feature_count, effective_feature_count : int
        Requested pre-removal and fitted post-removal feature widths.
    estimator_class : str or None
        Concrete estimator class name, or None when no estimator ran.
    estimator_parameters, selected_parameters : dict[str, object] or None
        Frozen fit controls and optional tuned candidate values, respectively.
    convergence_status : str
        ``"converged"``, ``"warning"``, ``"invalid"``, or ``"not_run"``.
    metrics : dict[str, float]
        Held-out finite metric values; empty when unavailable.
    coefficients : np.ndarray
        Feature coefficients shaped ``(effective_feature_count,)`` on the
        declared coefficient scale.
    intercept : float
        Scalar model intercept on the estimator's native response scale.
    feature_ids : tuple[str, ...]
        Stable feature names in coefficient-column order.
    positive_class : int or float or None
        Saved categorical positive class, or None for numerical targets.
    coefficient_scale_label : str
        Human-readable units for direct-unit coefficient interpretation.
    candidate_inner_scores : tuple[tuple[float or None, ...], ...]
        Tuned-mode primary metric values shaped
        ``(candidate, inner_fold)``. None marks an invalid evaluation; fixed
        records retain the empty tuple.
    candidate_inner_statuses : tuple[tuple[str, ...], ...]
        Tuned-mode ``"valid"``/``"invalid"`` status matrix aligned to scores;
        fixed records retain the empty tuple.
    candidate_inner_reasons : tuple[tuple[str or None, ...], ...]
        Tuned-mode scientific invalidity reason matrix aligned to scores;
        fixed records retain the empty tuple.
    selected_candidate_index : int or None
        Zero-based declared-grid index used for the outer refit, or None for
        fixed records and cells without a valid tuned candidate.
    """

    record_key: tuple[int, int, str, str]
    target_identifier: str
    inner_fold_count: int
    outer_fold_id: int
    outer_test_indices: np.ndarray
    inner_selection_indices: np.ndarray
    time_bin_index: int
    region_configuration: str
    representation: str
    regularization_mode: str
    is_valid: bool
    status: str
    reason: str | None
    train_count: int
    test_count: int
    train_class_counts: dict[int, int] | None
    test_class_counts: dict[int, int] | None
    requested_feature_count: int
    effective_feature_count: int
    estimator_class: str | None
    estimator_parameters: dict[str, object]
    convergence_status: str
    metrics: dict[str, float]
    coefficients: np.ndarray
    intercept: float
    feature_ids: tuple[str, ...]
    positive_class: int | float | None
    coefficient_scale_label: str
    selected_parameters: dict[str, object] | None
    candidate_inner_scores: tuple[tuple[float | None, ...], ...] = ()
    candidate_inner_statuses: tuple[tuple[str, ...], ...] = ()
    candidate_inner_reasons: tuple[tuple[str | None, ...], ...] = ()
    selected_candidate_index: int | None = None


@dataclass(frozen=True)
class CellSummary:
    """Complete-fold aggregation for one time/region/representation cell.

    Parameters
    ----------
    is_available : bool
        True only when every requested outer-fold score is finite.
    primary_mean : float or None
        Unweighted arithmetic mean of all valid requested fold scores, or None
        when a complete primary cell is unavailable.
    valid_fold_count, expected_fold_count : int
        Number of finite valid fold scores and requested grouped folds.
    fold_scores : tuple[float or None, ...]
        Ordered per-fold primary scores, preserving invalid entries as None.
    unavailable_reason : str or None
        Concise reason for an unavailable primary mean.
    """

    is_available: bool
    primary_mean: float | None
    valid_fold_count: int
    expected_fold_count: int
    fold_scores: tuple[float | None, ...]
    unavailable_reason: str | None


@dataclass(frozen=True)
class CoefficientSummary:
    """Cross-fold direct-unit coefficient summary excluding intercepts.

    Arrays are sorted in descending mean absolute coefficient order.  NaN input
    coefficients represent a unit unavailable in that fold, not a penalty zero.
    """

    feature_ids: tuple[str, ...]
    regions: tuple[str, ...]
    feature_statuses: tuple[str, ...]
    mean_absolute_coefficients: np.ndarray
    signed_median_coefficients: np.ndarray
    signed_iqr_coefficients: np.ndarray
    selection_frequencies: np.ndarray
    contributing_fold_counts: np.ndarray
    nonzero_counts_per_fit: np.ndarray
    nonzero_fractions_per_fit: np.ndarray
    intercepts: np.ndarray


@dataclass(frozen=True)
class TargetDecodingResult:
    """All fixed- or tuned-mode fold records and summaries for one target.

    Parameters
    ----------
    target_identifier : str
        Original target name retained unchanged for output provenance.
    inner_fold_count : int
        Configured three-fold inner-CV count. It is inactive in fixed mode and
        governs nested tuning in tuned mode.
    is_available : bool
        Whether the requested outer grouped split plan was scientifically
        available for this target.
    status : str
        ``"available"`` or ``"unavailable"`` target-level audit state.
    unavailable_reason : str or None
        Exact outer-split unavailable reason, or None when available.
    outer_fold_ids : np.ndarray
        One-dimensional grouped outer assignment shaped ``(trial,)``.
    inner_split_plans : dict[int, SplitPlan]
        Empty in fixed mode. Tuned results map each outer-fold ID to its global
        row-position inner split plan, including unavailable plans.
    fold_records : tuple[FoldRecord, ...]
        One record per outer fold, time bin, region, and representation.
    cell_summaries : dict[tuple[int, str, str], CellSummary]
        Complete-fold summaries keyed by ``(time_bin, region, representation)``.
    candidate_inner_scores : dict[tuple[int, int, str, str], tuple]
        Tuned per-outer-cell candidate-by-inner primary score tuples. Fixed
        results keep an empty mapping rather than fabricating inner data.
    candidate_inner_statuses : dict[tuple[int, int, str, str], tuple]
        Tuned candidate-by-inner valid/invalid status tuples, aligned to scores.
    candidate_inner_reasons : dict[tuple[int, int, str, str], tuple]
        Tuned candidate-by-inner scientific invalidity reasons, aligned to scores.
    selected_candidate_indices : dict[tuple[int, int, str, str], int or None]
        Selected declared-grid index for each tuned outer cell, or None when no
        candidate can be selected.
    """

    target_identifier: str
    inner_fold_count: int
    is_available: bool
    status: str
    unavailable_reason: str | None
    outer_fold_ids: np.ndarray
    inner_split_plans: dict[int, SplitPlan]
    fold_records: tuple[FoldRecord, ...]
    cell_summaries: dict[tuple[int, str, str], CellSummary]
    candidate_inner_scores: dict[tuple[int, int, str, str], tuple] = field(
        default_factory=dict
    )
    candidate_inner_statuses: dict[tuple[int, int, str, str], tuple] = field(
        default_factory=dict
    )
    candidate_inner_reasons: dict[tuple[int, int, str, str], tuple] = field(
        default_factory=dict
    )
    selected_candidate_indices: dict[tuple[int, int, str, str], int | None] = field(
        default_factory=dict
    )


def _validate_vector(values: object, name: str) -> np.ndarray:
    """Normalize one nonempty one-dimensional array without coercing its labels.

    Parameters
    ----------
    values : object
        Array-like trial-aligned labels, targets, or blocks.
    name : str
        Name used in validation errors.

    Returns
    -------
    np.ndarray
        One-dimensional nonempty array retaining its original scalar labels.

    Raises
    ------
    ValueError
        If the input is empty or not one-dimensional.
    """
    normalized = np.asarray(values)
    if normalized.ndim != 1 or normalized.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    return normalized


def _validate_fold_count(
    fold_count: object,
    *,
    allowed_counts: tuple[int, ...],
    name: str,
) -> int:
    """Validate one frozen grouped-CV fold count without coercing user input.

    Parameters
    ----------
    fold_count : object
        Candidate scalar configuration value. Booleans, fractional numbers,
        numeric strings, and unsupported integers are invalid.
    allowed_counts : tuple[int, ...]
        Scientifically approved built-in integer choices.
    name : str
        Configuration field name used in validation errors.

    Returns
    -------
    int
        The unchanged approved built-in integer fold count.

    Raises
    ------
    ValueError
        If ``fold_count`` is not an approved nonboolean integer.
    """
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise ValueError(f"{name} must be one of {allowed_counts} as an integer.")
    if fold_count not in allowed_counts:
        raise ValueError(f"{name} must be one of {allowed_counts} as an integer.")
    return fold_count


def _validate_positive_builtin_int(value: object, name: str) -> int:
    """Validate a positive built-in integer without silently coercing input.

    Parameters
    ----------
    value : object
        Candidate dimensionless configuration value. Booleans, NumPy scalar
        types, fractional values, and numeric strings are rejected.
    name : str
        Configuration field name used in validation errors.

    Returns
    -------
    int
        The unchanged positive built-in integer.

    Raises
    ------
    ValueError
        If ``value`` is not a positive built-in integer.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer.")
    return value


def _validate_categorical_target_encoding(target_values: np.ndarray) -> None:
    """Require the project-wide nonboolean numeric binary encoding ``{0, 1}``.

    Parameters
    ----------
    target_values : np.ndarray
        One-dimensional categorical labels shaped ``(trial,)``. Values may be
        integer or floating numeric zeros and ones, and are dimensionless.

    Returns
    -------
    None
        The input is accepted unchanged because callers retain its native
        numeric dtype for split and estimator provenance.

    Raises
    ------
    ValueError
        If labels are boolean, nonnumeric, nonfinite, or outside ``{0, 1}``.
    """
    numeric_dtype = np.issubdtype(target_values.dtype, np.integer) or np.issubdtype(
        target_values.dtype,
        np.floating,
    )
    if np.issubdtype(target_values.dtype, np.bool_) or not numeric_dtype:
        raise ValueError("categorical target_values must use nonboolean numeric labels {0, 1}.")
    if not np.isfinite(target_values).all():
        raise ValueError("categorical target_values must contain finite labels {0, 1}.")
    is_binary = np.logical_or(target_values == 0, target_values == 1)
    if not is_binary.all():
        raise ValueError("categorical target_values must use exactly labels {0, 1}.")


def _validate_rate_tensor(rate_tensor_hz: object, name: str) -> np.ndarray:
    """Validate a finite three-axis float firing-rate tensor in Hz.

    Parameters
    ----------
    rate_tensor_hz : object
        Array-like values expected to have shape ``(trial, time_bin, unit)``
        and values in Hz.
    name : str
        Name used in validation errors.

    Returns
    -------
    np.ndarray
        Float64 finite rate tensor with unchanged shape and Hz units.

    Raises
    ------
    ValueError
        If the tensor is not three-dimensional, has a zero axis, or is
        non-finite.
    """
    try:
        normalized = np.asarray(rate_tensor_hz, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric firing-rate tensor in Hz.") from error
    if normalized.ndim != 3 or any(size < 1 for size in normalized.shape):
        raise ValueError(f"{name} must have shape (trial, time_bin, unit) with nonzero axes.")
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} must contain finite firing rates in Hz.")
    return normalized


def _unavailable_split_plan(length: int, reason: str) -> SplitPlan:
    """Create an unavailable split plan preserving the input-row length.

    Parameters
    ----------
    length : int
        Number of original matched-tensor rows represented by ``fold_ids``.
    reason : str
        Declared scientific unavailable reason.

    Returns
    -------
    SplitPlan
        Empty unavailable plan with all fold IDs set to ``-1``.
    """
    return SplitPlan(
        fold_ids=np.full(length, -1, dtype=int),
        splits=(),
        is_available=False,
        unavailable_reason=reason,
    )


def _normalize_group_labels(block_ids: np.ndarray) -> np.ndarray:
    """Create sortable, type-preserving behavioral-group identities.

    Parameters
    ----------
    block_ids : numpy.ndarray
        One-dimensional scalar behavioral-block labels with shape ``(trial,)``.
        Labels are dimensionless identities and may mix Python scalar types.

    Returns
    -------
    numpy.ndarray
        One-dimensional object array with shape ``(trial,)``.  Each element is
        a sortable tuple containing the scalar type's module/qualname, a sort
        category, a native homogeneous numeric or text sort value, and
        ``repr(value)``.  The final representation preserves identities such
        as ``-0.0`` versus ``0.0`` without collapsing ``1``, ``1.0``, and
        ``True``.
    """
    normalized = np.empty(block_ids.size, dtype=object)
    for index, value in enumerate(block_ids):
        scalar_type = type(value)
        type_identity = f"{scalar_type.__module__}.{scalar_type.__qualname__}"
        representation = repr(value)
        if isinstance(value, bool):
            normalized[index] = (type_identity, "bool", int(value), representation)
        elif isinstance(value, numbers.Real):
            normalized[index] = (type_identity, "real", value, representation)
        elif isinstance(value, str):
            normalized[index] = (type_identity, "text", value, representation)
        else:
            normalized[index] = (type_identity, "repr", representation, representation)
    return normalized


def _build_grouped_split_plan(
    target_values: np.ndarray,
    block_ids: np.ndarray,
    row_indices: np.ndarray,
    *,
    target_family: str,
    fold_count: int,
    full_length: int,
) -> SplitPlan:
    """Construct and validate grouped folds on a selected original-row subset.

    Parameters
    ----------
    target_values, block_ids : np.ndarray
        Full trial-aligned target and group arrays shaped ``(n_rows,)``.
    row_indices : np.ndarray
        One-dimensional selected original row positions used by this splitter.
    target_family : {"categorical", "numerical"}
        Chooses StratifiedGroupKFold or GroupKFold.
    fold_count : int
        Positive requested deterministic fold count.
    full_length : int
        Length of returned full-row fold assignment array.

    Returns
    -------
    SplitPlan
        Valid global-index grouped folds, or a declared unavailable plan.
    """
    selected_values = target_values[row_indices]
    normalized_groups = _normalize_group_labels(block_ids)
    selected_groups = normalized_groups[row_indices]
    if target_family == "categorical":
        if np.unique(selected_values).size < 2:
            return _unavailable_split_plan(full_length, "constant target")
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=False)
        splitter_target = selected_values
    elif target_family == "numerical":
        try:
            numerical_values = np.asarray(selected_values, dtype=float)
        except (TypeError, ValueError):
            return _unavailable_split_plan(full_length, "non-numeric numerical target")
        if not np.isfinite(numerical_values).all():
            return _unavailable_split_plan(full_length, "non-finite numerical target")
        if np.unique(numerical_values).size < 2:
            return _unavailable_split_plan(full_length, "constant target")
        splitter = GroupKFold(n_splits=fold_count)
        splitter_target = None
    else:
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    try:
        local_splits = list(
            splitter.split(
                np.zeros((row_indices.size, 1), dtype=float),
                splitter_target,
                selected_groups,
            )
        )
    except ValueError as error:
        return _unavailable_split_plan(full_length, f"invalid grouped folds: {error}")
    if len(local_splits) != fold_count:
        return _unavailable_split_plan(full_length, "invalid grouped fold count")

    fold_ids = np.full(full_length, -1, dtype=int)
    splits: list[GroupedSplit] = []
    for fold_id, (local_train, local_test) in enumerate(local_splits):
        train_indices = row_indices[np.asarray(local_train, dtype=int)]
        test_indices = row_indices[np.asarray(local_test, dtype=int)]
        train_groups = set(normalized_groups[train_indices])
        test_groups = set(normalized_groups[test_indices])
        if train_groups.intersection(test_groups):
            return _unavailable_split_plan(full_length, "grouped folds split a behavioral block")
        if target_family == "categorical":
            train_classes = np.unique(target_values[train_indices])
            test_classes = np.unique(target_values[test_indices])
            if train_classes.size < 2 or test_classes.size < 2:
                return _unavailable_split_plan(
                    full_length,
                    "categorical grouped fold lacks both classes",
                )
        fold_ids[test_indices] = fold_id
        splits.append(
            GroupedSplit(
                fold_id=fold_id,
                train_indices=np.asarray(train_indices, dtype=int),
                test_indices=np.asarray(test_indices, dtype=int),
            )
        )
    return SplitPlan(
        fold_ids=fold_ids,
        splits=tuple(splits),
        is_available=True,
        unavailable_reason=None,
    )


def make_outer_splits(
    target_values: np.ndarray,
    block_ids: np.ndarray,
    *,
    target_family: str,
    fold_count: int,
) -> SplitPlan:
    """Build exact deterministic grouped outer folds for one target.

    Parameters
    ----------
    target_values : np.ndarray
        One-dimensional categorical labels or native numerical targets with
        shape ``(trial,)``.
    block_ids : np.ndarray
        One-dimensional behavioral group labels with shape ``(trial,)``.
    target_family : {"categorical", "numerical"}
        Categorical targets use StratifiedGroupKFold; numerical targets use
        GroupKFold without target discretization.
    fold_count : int
        Requested grouped outer fold count, normally three or five.

    Returns
    -------
    SplitPlan
        Deterministic validated global-index folds or a declared unavailable
        plan with all fold IDs ``-1``.
    """
    approved_fold_count = _validate_fold_count(
        fold_count,
        allowed_counts=(3, 5),
        name="outer_fold_count",
    )
    values = _validate_vector(target_values, "target_values")
    groups = _validate_vector(block_ids, "block_ids")
    if values.shape != groups.shape:
        raise ValueError("target_values and block_ids must have the same trial shape.")
    if target_family == "categorical":
        _validate_categorical_target_encoding(values)
    elif target_family != "numerical":
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    return _build_grouped_split_plan(
        values,
        groups,
        np.arange(values.size, dtype=int),
        target_family=target_family,
        fold_count=approved_fold_count,
        full_length=values.size,
    )


def make_inner_splits(
    target_values: np.ndarray,
    block_ids: np.ndarray,
    outer_train_indices: np.ndarray,
    *,
    target_family: str,
    fold_count: int,
) -> SplitPlan:
    """Build grouped inner folds only from one outer-training subset.

    Parameters
    ----------
    target_values : np.ndarray
        One-dimensional target values with shape ``(all_matched_trials,)``.
    block_ids : np.ndarray
        One-dimensional behavioral blocks aligned to ``target_values``.
    outer_train_indices : np.ndarray
        One-dimensional distinct original matched-tensor row positions allowed
        in every inner train and validation partition.
    target_family : {"categorical", "numerical"}
        Chooses deterministic grouped splitter family.
    fold_count : int
        Requested inner grouped fold count, currently three.

    Returns
    -------
    SplitPlan
        Valid global-index inner folds or a declared unavailable plan. Rows
        outside ``outer_train_indices`` retain fold ID ``-1``.
    """
    approved_fold_count = _validate_fold_count(
        fold_count,
        allowed_counts=(3,),
        name="inner_fold_count",
    )
    values = _validate_vector(target_values, "target_values")
    groups = _validate_vector(block_ids, "block_ids")
    if values.shape != groups.shape:
        raise ValueError("target_values and block_ids must have the same trial shape.")
    if target_family == "categorical":
        _validate_categorical_target_encoding(values)
    elif target_family != "numerical":
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    raw_indices = np.asarray(outer_train_indices)
    if raw_indices.ndim != 1:
        raise ValueError("outer_train_indices must be a one-dimensional array.")
    if raw_indices.size == 0:
        return _unavailable_split_plan(values.size, "empty outer training subset")
    indices_are_integer = np.issubdtype(raw_indices.dtype, np.integer)
    if np.issubdtype(raw_indices.dtype, np.bool_) or not indices_are_integer:
        raise ValueError("outer_train_indices must be supplied as integer trial positions.")
    indices_are_invalid = (
        (raw_indices < 0).any()
        or (raw_indices >= values.size).any()
        or np.unique(raw_indices).size != raw_indices.size
    )
    if indices_are_invalid:
        raise ValueError("outer_train_indices must be distinct in-range trial positions.")
    indices = np.asarray(raw_indices, dtype=int)
    return _build_grouped_split_plan(
        values,
        groups,
        indices,
        target_family=target_family,
        fold_count=approved_fold_count,
        full_length=values.size,
    )


def fit_region_transform(
    training_rate_tensor_hz: np.ndarray,
    requested_pc_count: int,
) -> RegionTransform:
    """Fit pooled training-only unit scaling and one separate regional PCA.

    Parameters
    ----------
    training_rate_tensor_hz : np.ndarray
        Finite regional rates shaped ``(training_trial, time_bin, unit)`` in
        Hz.  Trials must contain outer-training rows only.
    requested_pc_count : int
        Positive requested retained regional PC count, dimensionless.

    Returns
    -------
    RegionTransform
        Train-only scaling/PCA state.  Zero usable unit columns return an
        unavailable transform; one usable unit remains a valid one-PC fit.

    Raises
    ------
    ValueError
        If input tensor or requested count is malformed.
    """
    approved_pc_count = _validate_positive_builtin_int(
        requested_pc_count,
        "requested_pc_count",
    )
    tensor = _validate_rate_tensor(training_rate_tensor_hz, "training_rate_tensor_hz")
    pooled_hz = tensor.reshape(-1, tensor.shape[2])
    means_hz = pooled_hz.mean(axis=0)
    scales_hz = pooled_hz.std(axis=0)
    usable = np.flatnonzero(np.isfinite(means_hz) & np.isfinite(scales_hz) & (scales_hz > 0.0))
    if usable.size == 0:
        return RegionTransform(
            is_available=False,
            unavailable_reason="no usable training features remain after constant-feature removal",
            unit_means_hz=means_hz,
            unit_scales_hz=scales_hz,
            usable_unit_indices=usable,
            pca_model=None,
            component_limit=0,
            effective_component_count=0,
            cap_warning=None,
            input_unit_count=tensor.shape[2],
            requested_pc_count=approved_pc_count,
        )
    standardized = (pooled_hz[:, usable] - means_hz[usable]) / scales_hz[usable]
    component_limit = min(usable.size, standardized.shape[0])
    effective_count = min(approved_pc_count, component_limit)
    pca_model = PCA(
        n_components=effective_count,
        **dict(FROZEN_ESTIMATOR_CONTROLS["PCA"]),
    ).fit(standardized)
    cap_warning = None
    if effective_count != approved_pc_count:
        cap_warning = (
            f"Requested {approved_pc_count} PCs but fitted {effective_count} "
            "from available training observations and units."
        )
    return RegionTransform(
        is_available=True,
        unavailable_reason=None,
        unit_means_hz=means_hz,
        unit_scales_hz=scales_hz,
        usable_unit_indices=usable,
        pca_model=pca_model,
        component_limit=component_limit,
        effective_component_count=effective_count,
        cap_warning=cap_warning,
        input_unit_count=tensor.shape[2],
        requested_pc_count=approved_pc_count,
    )


def _validate_unit_ids(unit_ids: Sequence[str], expected_count: int, name: str) -> tuple[str, ...]:
    """Validate stable ordered direct-unit identifiers.

    Parameters
    ----------
    unit_ids : sequence[str]
        Stable IDs ordered like the regional tensor unit axis.
    expected_count : int
        Required number of identifiers matching that tensor unit axis.
    name : str
        Name used in validation errors.

    Returns
    -------
    tuple[str, ...]
        Immutable ordered stable identifiers.
    """
    if isinstance(unit_ids, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of stable string IDs, not scalar text.")
    try:
        normalized = tuple(unit_ids)
    except TypeError as error:
        raise ValueError(f"{name} must contain one stable string ID per tensor unit.") from error
    invalid_identifiers = (
        len(normalized) != expected_count
        or any(not isinstance(value, str) or not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    )
    if invalid_identifiers:
        raise ValueError(
            f"{name} must contain one nonempty unique stable string ID per tensor unit."
        )
    return normalized


def build_region_features(
    pfc_transform: RegionTransform,
    hpc_transform: RegionTransform,
    pfc_rate_tensor_hz: np.ndarray,
    hpc_rate_tensor_hz: np.ndarray,
    *,
    time_bin_index: int,
    region_configuration: str,
    representation: str,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
) -> FeatureMatrix:
    """Construct one time-bin PFC/HPC/combined direct or PCA feature matrix.

    Parameters
    ----------
    pfc_transform, hpc_transform : RegionTransform
        Independently fit regional train-only transforms.
    pfc_rate_tensor_hz, hpc_rate_tensor_hz : np.ndarray
        Matched rate tensors shaped ``(trial, time_bin, unit)`` in Hz.
    time_bin_index : int
        Zero-based selected common time-bin position.
    region_configuration : {"PFC", "HPC", "PFC+HPC"}
        Standalone or concatenated requested region configuration.
    representation : {"pca", "units"}
        PCA scores or direct standardized units.
    pfc_unit_ids, hpc_unit_ids : sequence[str]
        Stable IDs ordered along the original regional tensor unit axes.

    Returns
    -------
    FeatureMatrix
        Values shaped ``(trial, feature)`` with stable IDs in the exact column
        order.  Combined PCA concatenates two separate regional projections.

    Raises
    ------
    ValueError
        If requested transforms are unavailable or tensor/selector contracts
        are invalid.
    """
    if region_configuration not in _REGION_CONFIGURATIONS:
        raise ValueError(f"Unknown region configuration {region_configuration!r}.")
    if representation not in _REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}.")
    pfc_tensor = _validate_rate_tensor(pfc_rate_tensor_hz, "pfc_rate_tensor_hz")
    hpc_tensor = _validate_rate_tensor(hpc_rate_tensor_hz, "hpc_rate_tensor_hz")
    if pfc_tensor.shape[:2] != hpc_tensor.shape[:2]:
        raise ValueError("PFC and HPC tensors must share trial and time-bin axes.")
    if not isinstance(time_bin_index, int) or not 0 <= time_bin_index < pfc_tensor.shape[1]:
        raise ValueError("time_bin_index is outside the common time-bin axis.")
    pfc_ids = _validate_unit_ids(pfc_unit_ids, pfc_tensor.shape[2], "pfc_unit_ids")
    hpc_ids = _validate_unit_ids(hpc_unit_ids, hpc_tensor.shape[2], "hpc_unit_ids")

    def regional_features(
        region_name: str,
        transform: RegionTransform,
        tensor: np.ndarray,
        identifiers: tuple[str, ...],
    ) -> FeatureMatrix:
        """Return one requested regional feature matrix without touching the other region.

        Parameters
        ----------
        region_name : str
            Prefix used for PCA feature IDs.
        transform : RegionTransform
            Training-only transform for this region.
        tensor : np.ndarray
            Matched rate tensor in Hz, shaped ``(trial, time_bin, unit)``.
        identifiers : tuple[str, ...]
            Original stable direct-unit IDs aligned to ``tensor``.

        Returns
        -------
        FeatureMatrix
            Selected-time direct standardized units or PCA scores.
        """
        if representation == "units":
            values = transform.transform_units(tensor)[:, time_bin_index, :]
            feature_ids = tuple(identifiers[index] for index in transform.usable_unit_indices)
        else:
            values = transform.transform_pca(tensor)[:, time_bin_index, :]
            feature_ids = tuple(
                f"{region_name}:PC{index + 1}"
                for index in range(transform.effective_component_count)
            )
        return FeatureMatrix(values=values, feature_ids=feature_ids)

    pfc_features: FeatureMatrix | None = None
    hpc_features: FeatureMatrix | None = None
    if region_configuration in {"PFC", "PFC+HPC"}:
        pfc_features = regional_features("PFC", pfc_transform, pfc_tensor, pfc_ids)
    if region_configuration in {"HPC", "PFC+HPC"}:
        hpc_features = regional_features("HPC", hpc_transform, hpc_tensor, hpc_ids)
    if region_configuration == "PFC":
        assert pfc_features is not None
        return pfc_features
    if region_configuration == "HPC":
        assert hpc_features is not None
        return hpc_features
    assert pfc_features is not None and hpc_features is not None
    return FeatureMatrix(
        values=np.column_stack((pfc_features.values, hpc_features.values)),
        feature_ids=(*pfc_features.feature_ids, *hpc_features.feature_ids),
    )


def make_estimator(
    *,
    target_family: str,
    parameters: Mapping[str, float] | None = None,
):
    """Construct one frozen elastic-net decoder without deprecated APIs.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Chooses logistic elastic net or linear ElasticNet.
    parameters : mapping[str, float] or None
        Optional family-specific tuning overrides. Categorical estimators
        permit only ``C`` and ``l1_ratio``; numerical estimators permit only
        ``alpha`` and ``l1_ratio``. None uses only immutable controls owned by
        ``task_decoding.config``.

    Returns
    -------
    sklearn.linear_model.LogisticRegression or ElasticNet
        Unfitted estimator with fixed documented controls.  Logistic
        construction deliberately omits deprecated ``penalty``.
    """
    if target_family == "categorical":
        estimator_name = "LogisticRegression"
        allowed_overrides = {"C", "l1_ratio"}
        estimator_constructor = LogisticRegression
    elif target_family == "numerical":
        estimator_name = "ElasticNet"
        allowed_overrides = {"alpha", "l1_ratio"}
        estimator_constructor = ElasticNet
    else:
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping of approved tuning overrides.")
    override_parameters = {} if parameters is None else dict(parameters)
    disallowed_overrides = set(override_parameters).difference(allowed_overrides)
    if disallowed_overrides:
        raise ValueError(
            "parameters may only override "
            f"{sorted(allowed_overrides)} for {target_family} targets."
        )
    controls = dict(FROZEN_ESTIMATOR_CONTROLS[estimator_name])
    controls.update(override_parameters)
    return estimator_constructor(**controls)


def tuning_candidates(target_family: str) -> tuple[dict[str, float], ...]:
    """Expand the frozen 15-candidate grid in its declared deterministic order.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Chooses logistic ``C`` or ElasticNet ``alpha`` as the first tuning
        dimension. The second dimension is always ``l1_ratio``.

    Returns
    -------
    tuple[dict[str, float], ...]
        Fifteen fresh candidate dictionaries ordered first by the estimator
        strength sequence and then by the l1-ratio sequence. Values are
        dimensionless regularization controls.

    Raises
    ------
    ValueError
        If ``target_family`` is unsupported.
    """
    if target_family == "categorical":
        estimator_name = "LogisticRegression"
        strength_name = "C"
    elif target_family == "numerical":
        estimator_name = "ElasticNet"
        strength_name = "alpha"
    else:
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    grid = FROZEN_TUNING_GRID[estimator_name]
    return tuple(
        {strength_name: strength, "l1_ratio": l1_ratio}
        for strength in grid[strength_name]
        for l1_ratio in grid["l1_ratio"]
    )


def select_tuning_candidate(
    candidates: Sequence[Mapping[str, float]],
    *,
    candidate_scores: Sequence[Sequence[float | None]],
    candidate_valid: Sequence[bool],
) -> int | None:
    """Select the earliest valid candidate with the highest mean inner score.

    Parameters
    ----------
    candidates : sequence[mapping[str, float]]
        Declared-order candidate controls. Values are dimensionless and are
        returned only by index, so mappings remain unmodified.
    candidate_scores : sequence[sequence[float or None]]
        Candidate-by-inner-fold primary metric matrix. Each finite score is
        balanced accuracy or native-target R2; None denotes invalidity.
    candidate_valid : sequence[bool]
        One all-inner-fold validity flag per candidate, aligned to
        ``candidates`` and ``candidate_scores``.

    Returns
    -------
    int or None
        Zero-based declared-order winner, or None when no candidate has only
        finite valid inner scores.

    Raises
    ------
    ValueError
        If score or validity sequences do not align with ``candidates``.
    """
    if len(candidate_scores) != len(candidates) or len(candidate_valid) != len(candidates):
        raise ValueError("Candidate scores and validity flags must align with candidates.")
    selected_index: int | None = None
    selected_mean = float("-inf")
    for candidate_index, (scores, is_valid) in enumerate(
        zip(candidate_scores, candidate_valid, strict=True)
    ):
        score_array = np.asarray(scores, dtype=float)
        if not is_valid or score_array.size == 0 or not np.isfinite(score_array).all():
            continue
        candidate_mean = float(np.mean(score_array))
        if candidate_mean > selected_mean:
            selected_index = candidate_index
            selected_mean = candidate_mean
    return selected_index


def classification_metrics(
    target_values: np.ndarray,
    positive_scores: np.ndarray,
    *,
    positive_class: int | float,
) -> dict[str, float]:
    """Score held-out binary classification from one saved positive-score vector.

    Parameters
    ----------
    target_values : np.ndarray
        One-dimensional held-out binary labels with shape ``(trial,)``.
    positive_scores : np.ndarray
        One-dimensional held-out probability of ``positive_class`` with shape
        ``(trial,)``.
    positive_class : int or float
        Label assigned to scores at least 0.5 for balanced accuracy and used
        as the positive orientation for AUC.

    Returns
    -------
    dict[str, float]
        Finite ``balanced_accuracy`` and ``auc`` calculated from the same
        held-out score vector.

    Raises
    ------
    ValueError
        If inputs are misaligned, nonfinite, or do not contain two classes.
    """
    values = _validate_vector(target_values, "target_values")
    scores = _validate_vector(positive_scores, "positive_scores").astype(float)
    if values.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("positive_scores must be finite and align with target_values.")
    classes = np.unique(values)
    if classes.size != 2 or positive_class not in classes:
        raise ValueError("Categorical metrics require held-out values from both classes.")
    negative_class = classes[classes != positive_class][0]
    predicted = np.where(scores >= 0.5, positive_class, negative_class)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(values, predicted)),
        "auc": float(roc_auc_score(values == positive_class, scores)),
    }


def regression_r2(target_values: np.ndarray, predictions: np.ndarray) -> float | None:
    """Return native-target held-out R2 or None when it is mathematically undefined.

    Parameters
    ----------
    target_values, predictions : np.ndarray
        One-dimensional aligned held-out numerical targets and predictions,
        each shaped ``(trial,)`` in the target's original native units.

    Returns
    -------
    float or None
        Finite R2 in native-target units, including negative values, or None
        for constant/singleton/nonfinite held-out target values.
    """
    try:
        values = _validate_vector(target_values, "target_values").astype(float)
        predicted = _validate_vector(predictions, "predictions").astype(float)
    except ValueError:
        return None
    if (
        values.shape != predicted.shape
        or not np.isfinite(values).all()
        or not np.isfinite(predicted).all()
    ):
        return None
    if values.size < 2 or np.unique(values).size < 2:
        return None
    score = float(r2_score(values, predicted))
    return score if math.isfinite(score) else None


def _invalid_fit_result(reason: str, feature_count: int, *, convergence_status: str) -> FitResult:
    """Build a compact invalid fit result without suppressing unexpected errors.

    Parameters
    ----------
    reason : str
        Declared scientific invalidity reason.
    feature_count : int
        Number of input feature columns represented by the returned vector.
    convergence_status : str
        Recorded estimator convergence state.

    Returns
    -------
    FitResult
        Invalid result with NaN coefficients/intercept and no metrics.
    """
    return FitResult(
        is_valid=False,
        reason=reason,
        coefficients=np.full(feature_count, np.nan, dtype=float),
        intercept=float("nan"),
        metrics={},
        positive_class=None,
        positive_scores=None,
        convergence_status=convergence_status,
    )


def fit_and_score(
    estimator: object,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    held_out_features: np.ndarray,
    held_out_targets: np.ndarray,
    *,
    target_family: str,
) -> FitResult:
    """Fit once, reject warnings/nonfinite artifacts, and score one held-out set.

    Parameters
    ----------
    estimator : object
        Unfitted sklearn-compatible estimator exposing ``fit`` and either
        ``predict_proba`` or ``predict`` plus fitted coefficient attributes.
    train_features, held_out_features : np.ndarray
        Finite matrices shaped ``(trial, feature)`` in standardized-unit or
        PCA-score feature scale.
    train_targets, held_out_targets : np.ndarray
        One-dimensional aligned categorical labels or native numerical target
        values with shapes ``(training_trial,)`` and ``(held_out_trial,)``.
    target_family : {"categorical", "numerical"}
        Chooses held-out prediction and metric behavior.

    Returns
    -------
    FitResult
        Valid finite fit metrics/coefficients or declared scientific invalidity
        for convergence warnings, nonfinite outputs, or undefined test R2.

    Raises
    ------
    Exception
        Unexpected estimator/programming exceptions intentionally propagate.
    """
    train_matrix = _validate_feature_matrix(train_features, "train_features")
    held_out_matrix = _validate_feature_matrix(held_out_features, "held_out_features")
    train_values = _validate_vector(train_targets, "train_targets")
    held_out_values = _validate_vector(held_out_targets, "held_out_targets")
    if (
        train_matrix.shape[0] != train_values.size
        or held_out_matrix.shape[0] != held_out_values.size
    ):
        raise ValueError("Feature rows must match target values for every fit partition.")
    if train_matrix.shape[1] != held_out_matrix.shape[1]:
        raise ValueError("Train and held-out feature matrices must have equal feature width.")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_matrix, train_values)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught_warnings):
        return _invalid_fit_result(
            "convergence_warning",
            train_matrix.shape[1],
            convergence_status="warning",
        )
    coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    intercept_values = np.asarray(estimator.intercept_, dtype=float).reshape(-1)
    if coefficients.size != train_matrix.shape[1] or intercept_values.size != 1:
        return _invalid_fit_result(
            "invalid coefficient or intercept shape",
            train_matrix.shape[1],
            convergence_status="invalid",
        )
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept_values).all():
        return _invalid_fit_result(
            "non-finite coefficient or intercept",
            train_matrix.shape[1],
            convergence_status="invalid",
        )
    intercept = float(intercept_values[0])

    if target_family == "categorical":
        classes = np.asarray(estimator.classes_)
        if classes.size != 2 or 1 not in classes:
            return _invalid_fit_result(
                "categorical estimator must expose binary classes including positive class 1",
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        positive_class = 1
        probability_matrix = np.asarray(estimator.predict_proba(held_out_matrix), dtype=float)
        if probability_matrix.shape != (held_out_matrix.shape[0], classes.size):
            return _invalid_fit_result(
                "invalid probability shape",
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        if not np.isfinite(probability_matrix).all():
            return _invalid_fit_result(
                "non-finite probability or score",
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        positive_column = np.flatnonzero(classes == positive_class)
        if positive_column.size != 1:
            return _invalid_fit_result(
                "positive class is ambiguous",
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        positive_scores = probability_matrix[:, int(positive_column[0])]
        try:
            metrics = classification_metrics(
                held_out_values,
                positive_scores,
                positive_class=positive_class,
            )
        except ValueError as error:
            return _invalid_fit_result(
                str(error),
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        if classes[0] == positive_class:
            coefficients = -coefficients
            intercept = -intercept
        return FitResult(
            is_valid=True,
            reason=None,
            coefficients=coefficients,
            intercept=intercept,
            metrics=metrics,
            positive_class=positive_class,
            positive_scores=positive_scores,
            convergence_status="converged",
        )
    if target_family == "numerical":
        predictions = np.asarray(estimator.predict(held_out_matrix), dtype=float).reshape(-1)
        if predictions.shape != held_out_values.shape or not np.isfinite(predictions).all():
            return _invalid_fit_result(
                "non-finite prediction or score",
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        score = regression_r2(held_out_values, predictions)
        if score is None:
            return _invalid_fit_result(
                "constant, singleton, or non-finite held-out numerical target",
                train_matrix.shape[1],
                convergence_status="invalid",
            )
        return FitResult(
            is_valid=True,
            reason=None,
            coefficients=coefficients,
            intercept=intercept,
            metrics={"r2": score},
            positive_class=None,
            positive_scores=None,
            convergence_status="converged",
        )
    raise ValueError("target_family must be 'categorical' or 'numerical'.")


def _validate_feature_matrix(features: object, name: str) -> np.ndarray:
    """Validate a finite two-dimensional decoder feature matrix.

    Parameters
    ----------
    features : object
        Array-like feature values shaped ``(trial, feature)`` in standardized
        unit or PCA-score scale.
    name : str
        Name used in validation errors.

    Returns
    -------
    np.ndarray
        Finite float64 matrix retaining its trial/feature shape.
    """
    try:
        normalized = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric feature matrix.") from error
    if normalized.ndim != 2 or normalized.shape[0] < 1 or normalized.shape[1] < 1:
        raise ValueError(f"{name} must have shape (trial, feature) with nonzero axes.")
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} must contain finite values.")
    return normalized


def aggregate_complete_folds(
    fold_scores: Sequence[float | None],
    *,
    expected_fold_count: int,
    invalid_reasons: Sequence[str | None] | None = None,
) -> CellSummary:
    """Aggregate an unweighted primary score only when every fold is valid.

    Parameters
    ----------
    fold_scores : sequence[float or None]
        Ordered primary scores, one per requested outer fold.  Negative and
        below-chance finite values remain valid; None/NaN are invalid.
    expected_fold_count : int
        Number of requested grouped outer folds.
    invalid_reasons : sequence[str or None] or None
        Optional ordered scientific invalidity reasons aligned to fold scores.

    Returns
    -------
    CellSummary
        Complete-fold primary mean or unavailable summary preserving all scores
        and valid-count diagnostics.
    """
    normalized_scores: list[float | None] = []
    reasons = tuple(invalid_reasons or (None,) * len(fold_scores))
    if len(reasons) != len(fold_scores):
        raise ValueError("invalid_reasons must align one-to-one with fold_scores.")
    for score in fold_scores:
        if score is None:
            normalized_scores.append(None)
        else:
            normalized_score = float(score)
            normalized_scores.append(normalized_score if math.isfinite(normalized_score) else None)
    valid_scores = [score for score in normalized_scores if score is not None]
    complete = (
        len(normalized_scores) == expected_fold_count
        and len(valid_scores) == expected_fold_count
    )
    if complete:
        return CellSummary(
            is_available=True,
            primary_mean=float(np.mean(valid_scores)),
            valid_fold_count=len(valid_scores),
            expected_fold_count=expected_fold_count,
            fold_scores=tuple(normalized_scores),
            unavailable_reason=None,
        )
    reason = next((item for item in reasons if item), None)
    if reason is None and any(score is None for score in normalized_scores):
        reason = "non-finite or missing fold score"
    if reason is None:
        reason = "incomplete outer-fold coverage"
    return CellSummary(
        is_available=False,
        primary_mean=None,
        valid_fold_count=len(valid_scores),
        expected_fold_count=expected_fold_count,
        fold_scores=tuple(normalized_scores),
        unavailable_reason=reason,
    )


def summarize_direct_unit_coefficients(
    *,
    feature_ids: Sequence[str],
    regions: Sequence[str],
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    feature_statuses: Sequence[str],
) -> CoefficientSummary:
    """Summarize direct-unit standardized coefficients across evaluable folds.

    Parameters
    ----------
    feature_ids : sequence[str]
        Stable direct-unit IDs ordered as coefficient columns.
    regions : sequence[str]
        Region label per feature ID, same shape/order as ``feature_ids``.
    coefficients : np.ndarray
        Fold-by-feature matrix shaped ``(fold, feature)`` on standardized
        direct-unit feature scale. NaN marks unavailable in a fold.
    intercepts : np.ndarray
        One fitted intercept per fold, shape ``(fold,)``. Intercepts are kept
        for provenance but excluded from every nonzero-feature count.
    feature_statuses : sequence[str]
        Inspectable availability status per original feature.

    Returns
    -------
    CoefficientSummary
        Sorted direct-unit coefficient summaries and per-fit nonzero counts
        using the immutable ``1e-8`` tolerance.
    """
    identifiers = tuple(feature_ids)
    region_values = tuple(regions)
    statuses = tuple(feature_statuses)
    matrix = np.asarray(coefficients, dtype=float)
    intercept_values = np.asarray(intercepts, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(identifiers):
        raise ValueError("coefficients must have shape (fold, feature_ids).")
    if len(region_values) != len(identifiers) or len(statuses) != len(identifiers):
        raise ValueError("regions and feature_statuses must align with feature_ids.")
    if intercept_values.shape != (matrix.shape[0],):
        raise ValueError("intercepts must contain one value per coefficient fold.")
    available = np.isfinite(matrix)
    contributing_counts = available.sum(axis=0)
    mean_absolute = np.full(matrix.shape[1], np.nan, dtype=float)
    medians = np.full(matrix.shape[1], np.nan, dtype=float)
    iqrs = np.full(matrix.shape[1], np.nan, dtype=float)
    selection = np.full(matrix.shape[1], np.nan, dtype=float)
    for feature_index in range(matrix.shape[1]):
        values = matrix[available[:, feature_index], feature_index]
        if values.size == 0:
            continue
        mean_absolute[feature_index] = float(np.mean(np.abs(values)))
        medians[feature_index] = float(np.median(values))
        iqrs[feature_index] = float(np.percentile(values, 75) - np.percentile(values, 25))
        selection[feature_index] = float(np.mean(np.abs(values) > COEFFICIENT_TOLERANCE))
    nonzero_mask = available & (np.abs(matrix) > COEFFICIENT_TOLERANCE)
    nonzero_counts = nonzero_mask.sum(axis=1).astype(int)
    available_counts = available.sum(axis=1)
    nonzero_fractions = np.divide(
        nonzero_counts,
        available_counts,
        out=np.zeros(matrix.shape[0], dtype=float),
        where=available_counts > 0,
    )
    sort_key = np.where(np.isfinite(mean_absolute), -mean_absolute, np.inf)
    order = np.argsort(sort_key, kind="stable")
    return CoefficientSummary(
        feature_ids=tuple(identifiers[index] for index in order),
        regions=tuple(region_values[index] for index in order),
        feature_statuses=tuple(statuses[index] for index in order),
        mean_absolute_coefficients=mean_absolute[order],
        signed_median_coefficients=medians[order],
        signed_iqr_coefficients=iqrs[order],
        selection_frequencies=selection[order],
        contributing_fold_counts=contributing_counts[order].astype(int),
        nonzero_counts_per_fit=nonzero_counts,
        nonzero_fractions_per_fit=nonzero_fractions,
        intercepts=intercept_values,
    )


def _class_counts(target_values: np.ndarray, target_family: str) -> dict[int, int] | None:
    """Return binary categorical counts or no count mapping for numerical targets.

    Parameters
    ----------
    target_values : np.ndarray
        One-dimensional train or test target values.
    target_family : {"categorical", "numerical"}
        Target family determining whether class counts apply.

    Returns
    -------
    dict[int, int] or None
        Integer categorical label counts, or None for numerical targets.
    """
    if target_family != "categorical":
        return None
    labels, counts = np.unique(target_values, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def _component_regions(region_configuration: str) -> tuple[str, ...]:
    """Return component regional names required by one decoder configuration.

    Parameters
    ----------
    region_configuration : {"PFC", "HPC", "PFC+HPC"}
        Requested standalone or combined decoder population.

    Returns
    -------
    tuple[str, ...]
        Required component regions in feature concatenation order.
    """
    if region_configuration == "PFC":
        return ("PFC",)
    if region_configuration == "HPC":
        return ("HPC",)
    if region_configuration == "PFC+HPC":
        return ("PFC", "HPC")
    raise ValueError(f"Unknown region configuration {region_configuration!r}.")


def _requested_feature_count(
    region_configuration: str,
    representation: str,
    *,
    pfc_unit_count: int,
    hpc_unit_count: int,
    pfc_requested_pc_count: int,
    hpc_requested_pc_count: int,
) -> int:
    """Return requested direct-unit or PCA feature count before fold-local removal.

    Parameters
    ----------
    region_configuration, representation : str
        Requested cell selectors.
    pfc_unit_count, hpc_unit_count : int
        Original stable unit-axis widths.
    pfc_requested_pc_count, hpc_requested_pc_count : int
        Requested region-specific PCA counts.

    Returns
    -------
    int
        Dimensionless requested feature count for one record.
    """
    use_pca = representation == "pca"
    pfc_count = pfc_requested_pc_count if use_pca else pfc_unit_count
    hpc_count = hpc_requested_pc_count if use_pca else hpc_unit_count
    if region_configuration == "PFC":
        return pfc_count
    if region_configuration == "HPC":
        return hpc_count
    return pfc_count + hpc_count


def _fixed_estimator_parameters(target_family: str) -> dict[str, object]:
    """Copy recorded fixed estimator parameters from the authoritative config mapping.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Chooses the frozen logistic or ElasticNet controls.

    Returns
    -------
    dict[str, object]
        Fresh mutable-safe parameter dictionary suitable for fold provenance.
    """
    name = "LogisticRegression" if target_family == "categorical" else "ElasticNet"
    return dict(FROZEN_ESTIMATOR_CONTROLS[name])


def _recorded_estimator_parameters(
    target_family: str,
    selected_parameters: Mapping[str, float] | None,
) -> dict[str, object]:
    """Return recorded estimator controls including an optional tuned candidate.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Chooses the frozen logistic or ElasticNet baseline controls.
    selected_parameters : mapping[str, float] or None
        Family-whitelisted selected tuning values. None preserves frozen
        scientific controls unchanged.

    Returns
    -------
    dict[str, object]
        Fresh full estimator parameter mapping used for one recorded fit.
    """
    recorded_parameters = _fixed_estimator_parameters(target_family)
    if selected_parameters is not None:
        recorded_parameters.update(selected_parameters)
    return recorded_parameters


def _coefficient_scale_label(target_family: str) -> str:
    """Return the documented direct-feature coefficient-scale description.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Decoder target family.

    Returns
    -------
    str
        Human-readable per-pooled-training-standard-deviation coefficient
        scale label.
    """
    if target_family == "categorical":
        return "log-odds change per pooled training standard deviation"
    return "native target units per pooled training standard deviation"


def _unavailable_record(
    *,
    split: GroupedSplit,
    target_identifier: str,
    inner_fold_count: int,
    time_bin_index: int,
    region_configuration: str,
    representation: str,
    target_family: str,
    reason: str,
    requested_feature_count: int,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    regularization_mode: str = "fixed",
    selected_parameters: dict[str, object] | None = None,
    candidate_inner_scores: tuple[tuple[float | None, ...], ...] = (),
    candidate_inner_statuses: tuple[tuple[str, ...], ...] = (),
    candidate_inner_reasons: tuple[tuple[str | None, ...], ...] = (),
    selected_candidate_index: int | None = None,
) -> FoldRecord:
    """Create one unavailable scientific cell record without fitting an estimator.

    Parameters
    ----------
    split : GroupedSplit
        Current outer train/test partition in matched-tensor rows.
    target_identifier : str
        Original saved target name retained in the unavailable audit record.
    inner_fold_count : int
        Configured three-fold inner-CV count. It is inactive for fixed records
        and identifies the attempted tuned inner plan when relevant.
    time_bin_index, region_configuration, representation : object
        Cell identity values.
    target_family : {"categorical", "numerical"}
        Target family used for labels/count provenance.
    reason : str
        Declared scientific unavailable reason.
    requested_feature_count : int
        Requested pre-removal direct/PCA feature count.
    train_targets, test_targets : np.ndarray
        Partition targets for count provenance.
    regularization_mode : {"fixed", "tuned"}
        Declared fit mode for the unavailable audit record.
    selected_parameters : dict[str, object] or None
        Selected tuning mapping when one exists before a later unavailable fit.
    candidate_inner_scores, candidate_inner_statuses, candidate_inner_reasons : tuple
        Compact tuned candidate-by-inner audit arrays, empty in fixed mode.
    selected_candidate_index : int or None
        Declared-grid selection index, or None when no candidate was selected.

    Returns
    -------
    FoldRecord
        Unavailable record retaining partition/count/identity audit fields.
    """
    inner_selection_indices = (
        np.array(split.train_indices, dtype=int, copy=True)
        if regularization_mode == "tuned"
        else np.empty(0, dtype=int)
    )
    return FoldRecord(
        record_key=(split.fold_id, time_bin_index, region_configuration, representation),
        target_identifier=target_identifier,
        inner_fold_count=inner_fold_count,
        outer_fold_id=split.fold_id,
        outer_test_indices=split.test_indices,
        inner_selection_indices=inner_selection_indices,
        time_bin_index=time_bin_index,
        region_configuration=region_configuration,
        representation=representation,
        regularization_mode=regularization_mode,
        is_valid=False,
        status="unavailable",
        reason=reason,
        train_count=split.train_indices.size,
        test_count=split.test_indices.size,
        train_class_counts=_class_counts(train_targets, target_family),
        test_class_counts=_class_counts(test_targets, target_family),
        requested_feature_count=requested_feature_count,
        effective_feature_count=0,
        estimator_class=None,
        estimator_parameters=_recorded_estimator_parameters(
            target_family,
            selected_parameters,
        ),
        convergence_status="not_run",
        metrics={},
        coefficients=np.empty(0, dtype=float),
        intercept=float("nan"),
        feature_ids=(),
        positive_class=1 if target_family == "categorical" else None,
        coefficient_scale_label=_coefficient_scale_label(target_family),
        selected_parameters=selected_parameters,
        candidate_inner_scores=candidate_inner_scores,
        candidate_inner_statuses=candidate_inner_statuses,
        candidate_inner_reasons=candidate_inner_reasons,
        selected_candidate_index=selected_candidate_index,
    )


def _freeze_candidate_audit(
    scores: Sequence[Sequence[float | None]],
    statuses: Sequence[Sequence[str]],
    reasons: Sequence[Sequence[str | None]],
) -> tuple[
    tuple[tuple[float | None, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str | None, ...], ...],
]:
    """Convert one mutable candidate-by-inner audit into immutable tuples.

    Parameters
    ----------
    scores : sequence[sequence[float or None]]
        Candidate-by-inner primary metric values in balanced-accuracy or R2
        units. None denotes an invalid inner evaluation.
    statuses : sequence[sequence[str]]
        Candidate-by-inner ``"valid"`` or ``"invalid"`` state labels aligned
        exactly to ``scores``.
    reasons : sequence[sequence[str or None]]
        Candidate-by-inner scientific invalidity messages aligned exactly to
        ``scores``.

    Returns
    -------
    tuple
        Immutable ``(scores, statuses, reasons)`` tuples retaining the shape
        ``(candidate, inner_fold)`` for audit persistence.

    Raises
    ------
    ValueError
        If the three candidate-by-inner collections do not share one shape.
    """
    if len(scores) != len(statuses) or len(scores) != len(reasons):
        raise ValueError("Candidate audit collections must have the same candidate count.")
    frozen_scores: list[tuple[float | None, ...]] = []
    frozen_statuses: list[tuple[str, ...]] = []
    frozen_reasons: list[tuple[str | None, ...]] = []
    for score_row, status_row, reason_row in zip(scores, statuses, reasons, strict=True):
        if len(score_row) != len(status_row) or len(score_row) != len(reason_row):
            raise ValueError("Candidate audit rows must have the same inner-fold count.")
        frozen_scores.append(tuple(score_row))
        frozen_statuses.append(tuple(status_row))
        frozen_reasons.append(tuple(reason_row))
    return tuple(frozen_scores), tuple(frozen_statuses), tuple(frozen_reasons)


def _summarize_target_cells(
    records: Sequence[FoldRecord],
    *,
    time_bin_count: int,
    target_family: str,
    expected_fold_count: int,
) -> dict[tuple[int, str, str], CellSummary]:
    """Aggregate complete outer-fold summaries from already recorded cell fits.

    Parameters
    ----------
    records : sequence[FoldRecord]
        Outer-fold audit records for one target. Record metrics use balanced
        accuracy for categorical targets or native-target R2 for numerical ones.
    time_bin_count : int
        Number of common event-relative tensor bins, dimensionless.
    target_family : {"categorical", "numerical"}
        Chooses the requested primary metric.
    expected_fold_count : int
        Configured grouped outer-fold count.

    Returns
    -------
    dict[tuple[int, str, str], CellSummary]
        Complete-fold summaries for every time/region/representation cell.
    """
    summaries: dict[tuple[int, str, str], CellSummary] = {}
    primary_metric = "balanced_accuracy" if target_family == "categorical" else "r2"
    for time_bin_index in range(time_bin_count):
        for region_configuration in _REGION_CONFIGURATIONS:
            for representation in _REPRESENTATIONS:
                cell_records = [
                    record
                    for record in records
                    if record.time_bin_index == time_bin_index
                    and record.region_configuration == region_configuration
                    and record.representation == representation
                ]
                ordered_records = sorted(cell_records, key=lambda record: record.outer_fold_id)
                scores = tuple(
                    record.metrics.get(primary_metric) if record.is_valid else None
                    for record in ordered_records
                )
                reasons = tuple(record.reason for record in ordered_records)
                summaries[(time_bin_index, region_configuration, representation)] = (
                    aggregate_complete_folds(
                        scores,
                        expected_fold_count=expected_fold_count,
                        invalid_reasons=reasons,
                    )
                )
    return summaries


def _decode_tuned_target(
    *,
    target_identifier: str,
    target_family: str,
    target_values: np.ndarray,
    block_ids: np.ndarray,
    pfc_rate_tensor_hz: np.ndarray,
    hpc_rate_tensor_hz: np.ndarray,
    pfc_unit_ids: tuple[str, ...],
    hpc_unit_ids: tuple[str, ...],
    pfc_requested_pc_count: int,
    hpc_requested_pc_count: int,
    outer_split_plan: SplitPlan,
    outer_fold_count: int,
    inner_fold_count: int,
    timing_callback: Callable[[dict[str, object]], None] | None,
) -> TargetDecodingResult:
    """Run leakage-safe nested grouped CV for one already validated target.

    Parameters
    ----------
    target_identifier : str
        Stable target name retained on every result record.
    target_family : {"categorical", "numerical"}
        Chooses grouped split validation, estimator family, and primary metric.
    target_values, block_ids : np.ndarray
        Matched trial-aligned vectors shaped ``(trial,)``. Numerical targets
        retain native units; blocks are dimensionless group identities.
    pfc_rate_tensor_hz, hpc_rate_tensor_hz : np.ndarray
        Finite matched rate tensors shaped ``(trial, time_bin, unit)`` in Hz.
    pfc_unit_ids, hpc_unit_ids : tuple[str, ...]
        Unique stable IDs aligned to the regional unit axes.
    pfc_requested_pc_count, hpc_requested_pc_count : int
        Positive separate regional PCA requests, dimensionless.
    outer_split_plan : SplitPlan
        Available deterministic grouped outer plan expressed in global rows.
    outer_fold_count, inner_fold_count : int
        Configured grouped-CV counts. Inner folds are exactly three.
    timing_callback : callable or None
        Optional consumer of successful split, transform, feature-construction,
        and estimator timing events measured in seconds.

    Returns
    -------
    TargetDecodingResult
        Tuned outer records, complete summaries, global inner plans, and
        candidate-by-inner audit arrays. Inner preprocessing is fit only on
        each inner training subset; outer preprocessing is refit on all outer
        training rows before one held-out evaluation.
    """
    candidates = tuning_candidates(target_family)
    primary_metric = "balanced_accuracy" if target_family == "categorical" else "r2"
    records: list[FoldRecord] = []
    inner_split_plans: dict[int, SplitPlan] = {}
    candidate_scores_by_key: dict[tuple[int, int, str, str], tuple] = {}
    candidate_statuses_by_key: dict[tuple[int, int, str, str], tuple] = {}
    candidate_reasons_by_key: dict[tuple[int, int, str, str], tuple] = {}
    selected_indices_by_key: dict[tuple[int, int, str, str], int | None] = {}
    cell_definitions = tuple(
        (time_bin_index, region_configuration, representation)
        for time_bin_index in range(pfc_rate_tensor_hz.shape[1])
        for region_configuration in _REGION_CONFIGURATIONS
        for representation in _REPRESENTATIONS
    )

    for outer_split in outer_split_plan.splits:
        train_targets = target_values[outer_split.train_indices]
        test_targets = target_values[outer_split.test_indices]
        inner_plan = _timed_call(
            make_inner_splits,
            target_values,
            block_ids,
            outer_split.train_indices,
            target_family=target_family,
            fold_count=inner_fold_count,
            timing_callback=timing_callback,
            timing_operation="split",
            timing_region=None,
            timing_representation=None,
        )
        inner_split_plans[outer_split.fold_id] = inner_plan
        audit_scores = {
            cell: [[None] * len(inner_plan.splits) for _candidate in candidates]
            for cell in cell_definitions
        }
        audit_statuses = {
            cell: [["invalid"] * len(inner_plan.splits) for _candidate in candidates]
            for cell in cell_definitions
        }
        audit_reasons = {
            cell: [[None] * len(inner_plan.splits) for _candidate in candidates]
            for cell in cell_definitions
        }

        if not inner_plan.is_available:
            inner_reason = f"inner split unavailable: {inner_plan.unavailable_reason}"
            for time_bin_index, region_configuration, representation in cell_definitions:
                record_key = (
                    outer_split.fold_id,
                    time_bin_index,
                    region_configuration,
                    representation,
                )
                frozen_audit = _freeze_candidate_audit(
                    audit_scores[(time_bin_index, region_configuration, representation)],
                    audit_statuses[(time_bin_index, region_configuration, representation)],
                    audit_reasons[(time_bin_index, region_configuration, representation)],
                )
                candidate_scores_by_key[record_key] = frozen_audit[0]
                candidate_statuses_by_key[record_key] = frozen_audit[1]
                candidate_reasons_by_key[record_key] = frozen_audit[2]
                selected_indices_by_key[record_key] = None
                requested_count = _requested_feature_count(
                    region_configuration,
                    representation,
                    pfc_unit_count=len(pfc_unit_ids),
                    hpc_unit_count=len(hpc_unit_ids),
                    pfc_requested_pc_count=pfc_requested_pc_count,
                    hpc_requested_pc_count=hpc_requested_pc_count,
                )
                records.append(
                    _unavailable_record(
                        split=outer_split,
                        target_identifier=target_identifier,
                        inner_fold_count=inner_fold_count,
                        time_bin_index=time_bin_index,
                        region_configuration=region_configuration,
                        representation=representation,
                        target_family=target_family,
                        reason=inner_reason,
                        requested_feature_count=requested_count,
                        train_targets=train_targets,
                        test_targets=test_targets,
                        regularization_mode="tuned",
                        candidate_inner_scores=frozen_audit[0],
                        candidate_inner_statuses=frozen_audit[1],
                        candidate_inner_reasons=frozen_audit[2],
                    )
                )
            continue

        for inner_split in inner_plan.splits:
            pfc_inner_transform = _timed_call(
                fit_region_transform,
                pfc_rate_tensor_hz[inner_split.train_indices],
                pfc_requested_pc_count,
                timing_callback=timing_callback,
                timing_operation="regional_transform",
                timing_region="PFC",
                timing_representation=None,
            )
            hpc_inner_transform = _timed_call(
                fit_region_transform,
                hpc_rate_tensor_hz[inner_split.train_indices],
                hpc_requested_pc_count,
                timing_callback=timing_callback,
                timing_operation="regional_transform",
                timing_region="HPC",
                timing_representation=None,
            )
            inner_train_targets = target_values[inner_split.train_indices]
            inner_test_targets = target_values[inner_split.test_indices]
            for time_bin_index, region_configuration, representation in cell_definitions:
                cell = (time_bin_index, region_configuration, representation)
                unavailable_regions = [
                    region
                    for region, transform in (
                        ("PFC", pfc_inner_transform),
                        ("HPC", hpc_inner_transform),
                    )
                    if region in _component_regions(region_configuration)
                    and not transform.is_available
                ]
                if unavailable_regions:
                    invalid_reason = (
                        "unavailable inner regional training features: "
                        f"{', '.join(unavailable_regions)}"
                    )
                    for candidate_index in range(len(candidates)):
                        audit_reasons[cell][candidate_index][inner_split.fold_id] = invalid_reason
                    continue
                inner_train_features = _timed_call(
                    build_region_features,
                    pfc_inner_transform,
                    hpc_inner_transform,
                    pfc_rate_tensor_hz[inner_split.train_indices],
                    hpc_rate_tensor_hz[inner_split.train_indices],
                    time_bin_index=time_bin_index,
                    region_configuration=region_configuration,
                    representation=representation,
                    pfc_unit_ids=pfc_unit_ids,
                    hpc_unit_ids=hpc_unit_ids,
                    timing_callback=timing_callback,
                    timing_operation="feature_construction",
                    timing_region=region_configuration,
                    timing_representation=representation,
                )
                inner_test_features = _timed_call(
                    build_region_features,
                    pfc_inner_transform,
                    hpc_inner_transform,
                    pfc_rate_tensor_hz[inner_split.test_indices],
                    hpc_rate_tensor_hz[inner_split.test_indices],
                    time_bin_index=time_bin_index,
                    region_configuration=region_configuration,
                    representation=representation,
                    pfc_unit_ids=pfc_unit_ids,
                    hpc_unit_ids=hpc_unit_ids,
                    timing_callback=timing_callback,
                    timing_operation="feature_construction",
                    timing_region=region_configuration,
                    timing_representation=representation,
                )
                for candidate_index, candidate in enumerate(candidates):
                    estimator = make_estimator(
                        target_family=target_family,
                        parameters=candidate,
                    )
                    fit_result = _timed_call(
                        fit_and_score,
                        estimator,
                        inner_train_features.values,
                        inner_train_targets,
                        inner_test_features.values,
                        inner_test_targets,
                        target_family=target_family,
                        timing_callback=timing_callback,
                        timing_operation="estimator",
                        timing_region=region_configuration,
                        timing_representation=representation,
                    )
                    score = fit_result.metrics.get(primary_metric)
                    score_is_valid = (
                        fit_result.is_valid
                        and score is not None
                        and math.isfinite(score)
                    )
                    if score_is_valid:
                        audit_scores[cell][candidate_index][inner_split.fold_id] = float(score)
                        audit_statuses[cell][candidate_index][inner_split.fold_id] = "valid"
                    else:
                        audit_reasons[cell][candidate_index][inner_split.fold_id] = (
                            fit_result.reason or "non-finite inner score"
                        )

        pfc_outer_transform = _timed_call(
            fit_region_transform,
            pfc_rate_tensor_hz[outer_split.train_indices],
            pfc_requested_pc_count,
            timing_callback=timing_callback,
            timing_operation="regional_transform",
            timing_region="PFC",
            timing_representation=None,
        )
        hpc_outer_transform = _timed_call(
            fit_region_transform,
            hpc_rate_tensor_hz[outer_split.train_indices],
            hpc_requested_pc_count,
            timing_callback=timing_callback,
            timing_operation="regional_transform",
            timing_region="HPC",
            timing_representation=None,
        )
        for time_bin_index, region_configuration, representation in cell_definitions:
            cell = (time_bin_index, region_configuration, representation)
            record_key = (
                outer_split.fold_id,
                time_bin_index,
                region_configuration,
                representation,
            )
            frozen_audit = _freeze_candidate_audit(
                audit_scores[cell],
                audit_statuses[cell],
                audit_reasons[cell],
            )
            candidate_scores_by_key[record_key] = frozen_audit[0]
            candidate_statuses_by_key[record_key] = frozen_audit[1]
            candidate_reasons_by_key[record_key] = frozen_audit[2]
            candidate_valid = tuple(
                all(status == "valid" for status in status_row)
                for status_row in frozen_audit[1]
            )
            selected_candidate_index = select_tuning_candidate(
                candidates,
                candidate_scores=frozen_audit[0],
                candidate_valid=candidate_valid,
            )
            selected_indices_by_key[record_key] = selected_candidate_index
            requested_count = _requested_feature_count(
                region_configuration,
                representation,
                pfc_unit_count=len(pfc_unit_ids),
                hpc_unit_count=len(hpc_unit_ids),
                pfc_requested_pc_count=pfc_requested_pc_count,
                hpc_requested_pc_count=hpc_requested_pc_count,
            )
            if selected_candidate_index is None:
                invalid_reason = "no valid inner tuning candidate"
                records.append(
                    _unavailable_record(
                        split=outer_split,
                        target_identifier=target_identifier,
                        inner_fold_count=inner_fold_count,
                        time_bin_index=time_bin_index,
                        region_configuration=region_configuration,
                        representation=representation,
                        target_family=target_family,
                        reason=invalid_reason,
                        requested_feature_count=requested_count,
                        train_targets=train_targets,
                        test_targets=test_targets,
                        regularization_mode="tuned",
                        candidate_inner_scores=frozen_audit[0],
                        candidate_inner_statuses=frozen_audit[1],
                        candidate_inner_reasons=frozen_audit[2],
                    )
                )
                continue
            selected_parameters = dict(candidates[selected_candidate_index])
            unavailable_regions = [
                region
                for region, transform in (
                    ("PFC", pfc_outer_transform),
                    ("HPC", hpc_outer_transform),
                )
                if region in _component_regions(region_configuration) and not transform.is_available
            ]
            if unavailable_regions:
                invalid_reason = (
                    "unavailable outer regional training features: "
                    f"{', '.join(unavailable_regions)}"
                )
                records.append(
                    _unavailable_record(
                        split=outer_split,
                        target_identifier=target_identifier,
                        inner_fold_count=inner_fold_count,
                        time_bin_index=time_bin_index,
                        region_configuration=region_configuration,
                        representation=representation,
                        target_family=target_family,
                        reason=invalid_reason,
                        requested_feature_count=requested_count,
                        train_targets=train_targets,
                        test_targets=test_targets,
                        regularization_mode="tuned",
                        selected_parameters=selected_parameters,
                        candidate_inner_scores=frozen_audit[0],
                        candidate_inner_statuses=frozen_audit[1],
                        candidate_inner_reasons=frozen_audit[2],
                        selected_candidate_index=selected_candidate_index,
                    )
                )
                continue
            outer_train_features = _timed_call(
                build_region_features,
                pfc_outer_transform,
                hpc_outer_transform,
                pfc_rate_tensor_hz[outer_split.train_indices],
                hpc_rate_tensor_hz[outer_split.train_indices],
                time_bin_index=time_bin_index,
                region_configuration=region_configuration,
                representation=representation,
                pfc_unit_ids=pfc_unit_ids,
                hpc_unit_ids=hpc_unit_ids,
                timing_callback=timing_callback,
                timing_operation="feature_construction",
                timing_region=region_configuration,
                timing_representation=representation,
            )
            outer_test_features = _timed_call(
                build_region_features,
                pfc_outer_transform,
                hpc_outer_transform,
                pfc_rate_tensor_hz[outer_split.test_indices],
                hpc_rate_tensor_hz[outer_split.test_indices],
                time_bin_index=time_bin_index,
                region_configuration=region_configuration,
                representation=representation,
                pfc_unit_ids=pfc_unit_ids,
                hpc_unit_ids=hpc_unit_ids,
                timing_callback=timing_callback,
                timing_operation="feature_construction",
                timing_region=region_configuration,
                timing_representation=representation,
            )
            estimator = make_estimator(
                target_family=target_family,
                parameters=selected_parameters,
            )
            fit_result = _timed_call(
                fit_and_score,
                estimator,
                outer_train_features.values,
                train_targets,
                outer_test_features.values,
                test_targets,
                target_family=target_family,
                timing_callback=timing_callback,
                timing_operation="estimator",
                timing_region=region_configuration,
                timing_representation=representation,
            )
            records.append(
                FoldRecord(
                    record_key=record_key,
                    target_identifier=target_identifier,
                    inner_fold_count=inner_fold_count,
                    outer_fold_id=outer_split.fold_id,
                    outer_test_indices=outer_split.test_indices,
                    inner_selection_indices=np.array(
                        outer_split.train_indices,
                        dtype=int,
                        copy=True,
                    ),
                    time_bin_index=time_bin_index,
                    region_configuration=region_configuration,
                    representation=representation,
                    regularization_mode="tuned",
                    is_valid=fit_result.is_valid,
                    status="valid" if fit_result.is_valid else "unavailable",
                    reason=fit_result.reason,
                    train_count=outer_split.train_indices.size,
                    test_count=outer_split.test_indices.size,
                    train_class_counts=_class_counts(train_targets, target_family),
                    test_class_counts=_class_counts(test_targets, target_family),
                    requested_feature_count=requested_count,
                    effective_feature_count=len(outer_train_features.feature_ids),
                    estimator_class=type(estimator).__name__,
                    estimator_parameters=_recorded_estimator_parameters(
                        target_family,
                        selected_parameters,
                    ),
                    convergence_status=fit_result.convergence_status,
                    metrics=fit_result.metrics,
                    coefficients=fit_result.coefficients,
                    intercept=fit_result.intercept,
                    feature_ids=outer_train_features.feature_ids,
                    positive_class=fit_result.positive_class,
                    coefficient_scale_label=_coefficient_scale_label(target_family),
                    selected_parameters=selected_parameters,
                    candidate_inner_scores=frozen_audit[0],
                    candidate_inner_statuses=frozen_audit[1],
                    candidate_inner_reasons=frozen_audit[2],
                    selected_candidate_index=selected_candidate_index,
                )
            )

    summaries = _summarize_target_cells(
        records,
        time_bin_count=pfc_rate_tensor_hz.shape[1],
        target_family=target_family,
        expected_fold_count=outer_fold_count,
    )
    return TargetDecodingResult(
        target_identifier=target_identifier,
        inner_fold_count=inner_fold_count,
        is_available=True,
        status="available",
        unavailable_reason=None,
        outer_fold_ids=outer_split_plan.fold_ids,
        inner_split_plans=inner_split_plans,
        fold_records=tuple(records),
        cell_summaries=summaries,
        candidate_inner_scores=candidate_scores_by_key,
        candidate_inner_statuses=candidate_statuses_by_key,
        candidate_inner_reasons=candidate_reasons_by_key,
        selected_candidate_indices=selected_indices_by_key,
    )


def decode_target(
    *,
    target_identifier: str,
    target_family: str,
    target_values: np.ndarray,
    block_ids: np.ndarray,
    pfc_rate_tensor_hz: np.ndarray,
    hpc_rate_tensor_hz: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    pfc_requested_pc_count: int,
    hpc_requested_pc_count: int,
    outer_fold_count: int,
    inner_fold_count: int,
    regularization_mode: str,
    timing_callback: Callable[[dict[str, object]], None] | None = None,
) -> TargetDecodingResult:
    """Decode one matched target across all regions and representations.

    Parameters
    ----------
    target_identifier : str
        Saved target identifier used for higher-level provenance.  It does not
        alter numerical values in this function.
    target_family : {"categorical", "numerical"}
        Chooses grouped splitter, estimator, and held-out metric family.
    target_values : np.ndarray
        One-dimensional matched target vector shaped ``(trial,)``. Numerical
        values remain in native target units.
    block_ids : np.ndarray
        One-dimensional behavioral block IDs shaped ``(trial,)``.
    pfc_rate_tensor_hz, hpc_rate_tensor_hz : np.ndarray
        Finite matched tensors shaped ``(trial, time_bin, unit)`` in Hz.
    pfc_unit_ids, hpc_unit_ids : sequence[str]
        Stable IDs aligned with the corresponding tensor unit axes.
    pfc_requested_pc_count, hpc_requested_pc_count : int
        Positive requested separate regional PCA counts.
    outer_fold_count, inner_fold_count : int
        Requested grouped fold counts. Inner count is recorded but inactive in
        fixed mode and defines leakage-safe nested CV in tuned mode.
    regularization_mode : {"fixed", "tuned"}
        Fixed mode performs one outer fit per cell. Tuned mode fits transforms
        and candidates inside grouped inner folds, then refits the selected
        candidate on the complete outer-training partition.
    timing_callback : callable or None, default=None
        Optional consumer receiving JSON-safe events for successful split,
        transform, feature-construction, and estimator operations. Durations
        are seconds; omitting the callback preserves the prior interface.

    Returns
    -------
    TargetDecodingResult
        Outer assignments, audit fold records, complete-fold cell summaries,
        and, for tuned runs, global inner plans plus candidate-by-inner audit
        arrays. Rate tensors remain in Hz; model features are pooled-training
        standardized direct units or PCA scores.

    Raises
    ------
    ValueError
        If array shape, unit identity, target family, or mode contracts fail.
    """
    if regularization_mode not in {"fixed", "tuned"}:
        raise ValueError("regularization_mode must be 'fixed' or 'tuned'.")
    if target_family not in {"categorical", "numerical"}:
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    approved_inner_fold_count = _validate_fold_count(
        inner_fold_count,
        allowed_counts=(3,),
        name="inner_fold_count",
    )
    approved_pfc_pc_count = _validate_positive_builtin_int(
        pfc_requested_pc_count,
        "pfc_requested_pc_count",
    )
    approved_hpc_pc_count = _validate_positive_builtin_int(
        hpc_requested_pc_count,
        "hpc_requested_pc_count",
    )
    target = _validate_vector(target_values, "target_values")
    blocks = _validate_vector(block_ids, "block_ids")
    pfc_tensor = _validate_rate_tensor(pfc_rate_tensor_hz, "pfc_rate_tensor_hz")
    hpc_tensor = _validate_rate_tensor(hpc_rate_tensor_hz, "hpc_rate_tensor_hz")
    if target.shape != blocks.shape or target.size != pfc_tensor.shape[0]:
        raise ValueError("Targets, blocks, and regional tensors must share the trial axis.")
    if pfc_tensor.shape[:2] != hpc_tensor.shape[:2]:
        raise ValueError("PFC and HPC tensors must share trial and time-bin axes.")
    pfc_ids = _validate_unit_ids(pfc_unit_ids, pfc_tensor.shape[2], "pfc_unit_ids")
    hpc_ids = _validate_unit_ids(hpc_unit_ids, hpc_tensor.shape[2], "hpc_unit_ids")
    split_plan = _timed_call(
        make_outer_splits,
        target,
        blocks,
        target_family=target_family,
        fold_count=outer_fold_count,
        timing_callback=timing_callback,
        timing_operation="split",
        timing_region=None,
        timing_representation=None,
    )
    if not split_plan.is_available:
        return TargetDecodingResult(
            target_identifier=target_identifier,
            inner_fold_count=approved_inner_fold_count,
            is_available=False,
            status="unavailable",
            unavailable_reason=split_plan.unavailable_reason,
            outer_fold_ids=split_plan.fold_ids,
            inner_split_plans={},
            fold_records=(),
            cell_summaries={},
        )
    if regularization_mode == "tuned":
        return _decode_tuned_target(
            target_identifier=target_identifier,
            target_family=target_family,
            target_values=target,
            block_ids=blocks,
            pfc_rate_tensor_hz=pfc_tensor,
            hpc_rate_tensor_hz=hpc_tensor,
            pfc_unit_ids=pfc_ids,
            hpc_unit_ids=hpc_ids,
            pfc_requested_pc_count=approved_pfc_pc_count,
            hpc_requested_pc_count=approved_hpc_pc_count,
            outer_split_plan=split_plan,
            outer_fold_count=outer_fold_count,
            inner_fold_count=approved_inner_fold_count,
            timing_callback=timing_callback,
        )

    records: list[FoldRecord] = []
    for split in split_plan.splits:
        train_targets = target[split.train_indices]
        test_targets = target[split.test_indices]
        pfc_transform = _timed_call(
            fit_region_transform,
            pfc_tensor[split.train_indices],
            approved_pfc_pc_count,
            timing_callback=timing_callback,
            timing_operation="regional_transform",
            timing_region="PFC",
            timing_representation=None,
        )
        hpc_transform = _timed_call(
            fit_region_transform,
            hpc_tensor[split.train_indices],
            approved_hpc_pc_count,
            timing_callback=timing_callback,
            timing_operation="regional_transform",
            timing_region="HPC",
            timing_representation=None,
        )
        for time_bin_index in range(pfc_tensor.shape[1]):
            for region_configuration in _REGION_CONFIGURATIONS:
                required_regions = _component_regions(region_configuration)
                unavailable_regions = [
                    region
                    for region, transform in (("PFC", pfc_transform), ("HPC", hpc_transform))
                    if region in required_regions and not transform.is_available
                ]
                for representation in _REPRESENTATIONS:
                    requested_count = _requested_feature_count(
                        region_configuration,
                        representation,
                        pfc_unit_count=len(pfc_ids),
                        hpc_unit_count=len(hpc_ids),
                        pfc_requested_pc_count=approved_pfc_pc_count,
                        hpc_requested_pc_count=approved_hpc_pc_count,
                    )
                    if unavailable_regions:
                        records.append(
                            _unavailable_record(
                                split=split,
                                target_identifier=target_identifier,
                                inner_fold_count=approved_inner_fold_count,
                                time_bin_index=time_bin_index,
                                region_configuration=region_configuration,
                                representation=representation,
                                target_family=target_family,
                                reason=(
                                    f"unavailable regional training features: "
                                    f"{', '.join(unavailable_regions)}"
                                ),
                                requested_feature_count=requested_count,
                                train_targets=train_targets,
                                test_targets=test_targets,
                            )
                        )
                        continue
                    train_features = _timed_call(
                        build_region_features,
                        pfc_transform,
                        hpc_transform,
                        pfc_tensor[split.train_indices],
                        hpc_tensor[split.train_indices],
                        time_bin_index=time_bin_index,
                        region_configuration=region_configuration,
                        representation=representation,
                        pfc_unit_ids=pfc_ids,
                        hpc_unit_ids=hpc_ids,
                        timing_callback=timing_callback,
                        timing_operation="feature_construction",
                        timing_region=region_configuration,
                        timing_representation=representation,
                    )
                    test_features = _timed_call(
                        build_region_features,
                        pfc_transform,
                        hpc_transform,
                        pfc_tensor[split.test_indices],
                        hpc_tensor[split.test_indices],
                        time_bin_index=time_bin_index,
                        region_configuration=region_configuration,
                        representation=representation,
                        pfc_unit_ids=pfc_ids,
                        hpc_unit_ids=hpc_ids,
                        timing_callback=timing_callback,
                        timing_operation="feature_construction",
                        timing_region=region_configuration,
                        timing_representation=representation,
                    )
                    estimator = make_estimator(target_family=target_family)
                    fit_result = _timed_call(
                        fit_and_score,
                        estimator,
                        train_features.values,
                        train_targets,
                        test_features.values,
                        test_targets,
                        target_family=target_family,
                        timing_callback=timing_callback,
                        timing_operation="estimator",
                        timing_region=region_configuration,
                        timing_representation=representation,
                    )
                    records.append(
                        FoldRecord(
                            record_key=(
                                split.fold_id,
                                time_bin_index,
                                region_configuration,
                                representation,
                            ),
                            target_identifier=target_identifier,
                            inner_fold_count=approved_inner_fold_count,
                            outer_fold_id=split.fold_id,
                            outer_test_indices=split.test_indices,
                            inner_selection_indices=np.empty(0, dtype=int),
                            time_bin_index=time_bin_index,
                            region_configuration=region_configuration,
                            representation=representation,
                            regularization_mode="fixed",
                            is_valid=fit_result.is_valid,
                            status="valid" if fit_result.is_valid else "unavailable",
                            reason=fit_result.reason,
                            train_count=split.train_indices.size,
                            test_count=split.test_indices.size,
                            train_class_counts=_class_counts(train_targets, target_family),
                            test_class_counts=_class_counts(test_targets, target_family),
                            requested_feature_count=requested_count,
                            effective_feature_count=len(train_features.feature_ids),
                            estimator_class=type(estimator).__name__,
                            estimator_parameters=_fixed_estimator_parameters(target_family),
                            convergence_status=fit_result.convergence_status,
                            metrics=fit_result.metrics,
                            coefficients=fit_result.coefficients,
                            intercept=fit_result.intercept,
                            feature_ids=train_features.feature_ids,
                            positive_class=fit_result.positive_class,
                            coefficient_scale_label=_coefficient_scale_label(target_family),
                            selected_parameters=None,
                        )
                    )
    summaries: dict[tuple[int, str, str], CellSummary] = {}
    primary_metric = "balanced_accuracy" if target_family == "categorical" else "r2"
    for time_bin_index in range(pfc_tensor.shape[1]):
        for region_configuration in _REGION_CONFIGURATIONS:
            for representation in _REPRESENTATIONS:
                cell_records = [
                    record
                    for record in records
                    if record.time_bin_index == time_bin_index
                    and record.region_configuration == region_configuration
                    and record.representation == representation
                ]
                ordered_records = sorted(cell_records, key=lambda record: record.outer_fold_id)
                scores = tuple(
                    record.metrics.get(primary_metric) if record.is_valid else None
                    for record in ordered_records
                )
                reasons = tuple(record.reason for record in ordered_records)
                summaries[(time_bin_index, region_configuration, representation)] = (
                    aggregate_complete_folds(
                        scores,
                        expected_fold_count=outer_fold_count,
                        invalid_reasons=reasons,
                    )
                )
    return TargetDecodingResult(
        target_identifier=target_identifier,
        inner_fold_count=approved_inner_fold_count,
        is_available=True,
        status="available",
        unavailable_reason=None,
        outer_fold_ids=split_plan.fold_ids,
        inner_split_plans={},
        fold_records=tuple(records),
        cell_summaries=summaries,
    )
