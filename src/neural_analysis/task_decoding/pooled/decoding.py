"""Eligibility and fold primitives for pooled-window task-variable decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from src.neural_analysis.task_decoding import modeling
from src.neural_analysis.task_decoding.pooled.config import REFERENCE_CONDITION, TrainingWindow


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
METRIC_NAMES = ("balanced_accuracy", "auc", "r2")
REGION_CONFIGURATIONS = ("PFC", "HPC", "PFC+HPC")
REPRESENTATIONS = ("pca", "units")


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


@dataclass(frozen=True)
class PooledTransforms:
    """Two frozen anatomical transforms shared by all pooled model variants.

    Parameters
    ----------
    pfc_transform, hpc_transform : modeling.RegionTransform
        Independently fitted training-window transform state.
    pfc_unit_ids, hpc_unit_ids : tuple[str, ...]
        Stable identifiers aligned to each input regional unit axis.
    pfc_reason, hpc_reason : str or None
        Compact anatomical unavailability code, or None when usable.
    """

    pfc_transform: modeling.RegionTransform
    hpc_transform: modeling.RegionTransform
    pfc_unit_ids: tuple[str, ...]
    hpc_unit_ids: tuple[str, ...]
    pfc_reason: str | None
    hpc_reason: str | None


@dataclass(frozen=True)
class PooledModelRecord:
    """Audit values for one pooled model identity.

    Parameters
    ----------
    target_identifier, target_family : str
        Stable target name and categorical/numerical family.
    region_configuration, representation : str
        Requested regional feature set and direct/PCA representation.
    is_valid : bool
        Whether a reusable estimator was fitted successfully.
    reason : str or None
        Compact pooled reason code when unavailable.
    train_trial_count, stacked_row_count : int
        Independent training trials and trial-time observations.
    requested_feature_count, feature_count : int
        Pre-removal requested width and effective fitted width.
    estimator_class : str or None
        Concrete estimator class when an attempt was reached.
    estimator_parameters : dict[str, object]
        Frozen estimator parameter payload.
    convergence_status : str
        ``"not_run"``, ``"converged"``, ``"warning"``, or ``"invalid"``.
    coefficients : numpy.ndarray
        Oriented coefficients shaped ``(feature_count,)`` on the saved scale.
    intercept : float
        Scalar fitted intercept, or NaN when unavailable.
    feature_ids : tuple[str, ...]
        Stable coefficient-column identities.
    positive_class : int or float or None
        Class one for categorical fits, otherwise None.
    coefficient_scale_label : str
        Explicit feature/target scale description.
    """

    target_identifier: str
    target_family: str
    region_configuration: str
    representation: str
    is_valid: bool
    reason: str | None
    train_trial_count: int
    stacked_row_count: int
    requested_feature_count: int
    feature_count: int
    estimator_class: str | None
    estimator_parameters: dict[str, object]
    convergence_status: str
    coefficients: np.ndarray
    intercept: float
    feature_ids: tuple[str, ...]
    positive_class: int | float | None
    coefficient_scale_label: str


@dataclass(frozen=True)
class FittedPooledModel:
    """One pooled model record plus runtime objects needed for evaluation.

    Parameters
    ----------
    record : PooledModelRecord
        Durable audit fields for this model identity.
    transforms : PooledTransforms or None
        Frozen anatomical scaling/PCA state used for both fit and evaluation,
        or None when a training subset failed before transform fitting.
    fitted_estimator : modeling.FittedEstimator or None
        Runtime estimator state, absent when fitting was not reached.
    """

    record: PooledModelRecord
    transforms: PooledTransforms | None
    fitted_estimator: modeling.FittedEstimator | None


@dataclass(frozen=True)
class PooledEvaluationResult:
    """Dense through-time metrics from one fixed pooled model.

    Parameters
    ----------
    evaluation_time_centers_s : numpy.ndarray
        Choice-relative time centers, shape ``(evaluation_time,)`` in seconds.
    metric_names : tuple[str, ...]
        Fixed metric column order.
    status, reason_codes : numpy.ndarray
        Unicode arrays shaped ``(evaluation_time, metric)``. Status is
        ``valid``, ``unavailable``, or ``not_applicable``; reasons are compact
        codes or empty strings.
    scores : numpy.ndarray
        Float64 metric array shaped ``(evaluation_time, metric)``.
    test_trial_count : int
        Number of independent destination trials.
    test_class_counts : tuple[int, int] or None
        Observed class-zero/class-one counts, or None for numerical targets.
    test_indices : numpy.ndarray
        One-dimensional destination tensor-row positions. It is stored once,
        never repeated over time or metric axes.
    """

    evaluation_time_centers_s: np.ndarray
    metric_names: tuple[str, ...]
    status: np.ndarray
    reason_codes: np.ndarray
    scores: np.ndarray
    test_trial_count: int
    test_class_counts: tuple[int, int] | None
    test_indices: np.ndarray


@dataclass(frozen=True)
class PooledGapResult:
    """Dense native-minus-reference metrics for one destination model cell.

    Parameters
    ----------
    status, reason_codes : numpy.ndarray
        Unicode arrays shaped ``(evaluation_time, metric)``.
    scores : numpy.ndarray
        Float64 native-minus-transfer scores of the same shape.
    """

    status: np.ndarray
    reason_codes: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class PooledWindowTargetResult:
    """Runtime scientific result for one training-window/target cell.

    Parameters
    ----------
    models : mapping
        Keys are ``(condition, fold, region, representation)``.
    evaluations : mapping
        Keys are ``(train_condition, test_condition, fold, region,
        representation)`` and values hold dense time-by-metric arrays.
    gaps : mapping
        Keys are ``(destination, fold, region, representation)``.
    transforms : mapping
        Keys are ``(condition, fold)`` and values are shared anatomical fits.
    unavailable_reason : str or None
        ``target_outer_unavailable`` when orchestration cannot begin.
    target_identifier, target_family : str
        Stable target name and categorical/numerical family.
    outer_fold_ids : numpy.ndarray
        Target-wide fold map shaped ``(tensor_trial,)`` with ``-1`` for
        target-ineligible rows.
    """

    models: Mapping[tuple[str, int, str, str], FittedPooledModel]
    evaluations: Mapping[tuple[str, str, int, str, str], PooledEvaluationResult]
    gaps: Mapping[tuple[str, int, str, str], PooledGapResult]
    transforms: Mapping[tuple[str, int], PooledTransforms]
    unavailable_reason: str | None
    target_identifier: str
    target_family: str
    outer_fold_ids: np.ndarray


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


def stack_training_window(
    feature_tensor: np.ndarray,
    trial_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a pooled window in trial-major/time-major order with trial weights.

    Parameters
    ----------
    feature_tensor : numpy.ndarray
        Finite values shaped ``(training_trial, training_time, feature)`` in
        standardized-unit or PCA-score units.
    trial_targets : numpy.ndarray
        Finite categorical labels or native-unit numerical targets shaped
        ``(training_trial,)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Feature matrix shaped ``(training_trial * training_time, feature)``,
        aligned repeated targets, and float64 row weights. Each trial's row
        weights sum exactly to one.
    """
    try:
        tensor = np.asarray(feature_tensor, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("feature_tensor must be a numeric three-dimensional array.") from error
    targets = np.asarray(trial_targets)
    if tensor.ndim != 3 or any(size < 1 for size in tensor.shape):
        raise ValueError("feature_tensor must have three nonempty axes.")
    if not np.isfinite(tensor).all():
        raise ValueError("feature_tensor must contain finite values.")
    if targets.ndim != 1 or targets.size != tensor.shape[0]:
        raise ValueError("trial_targets must align with the feature-tensor trial axis.")
    try:
        targets_are_finite = bool(np.isfinite(targets).all())
    except TypeError as error:
        raise ValueError("trial_targets must contain finite numeric values.") from error
    if not targets_are_finite:
        raise ValueError("trial_targets must contain finite values.")

    trial_count, time_count, feature_count = tensor.shape
    stacked_features = tensor.reshape(trial_count * time_count, feature_count)
    repeated_targets = np.repeat(targets, time_count)
    sample_weights = np.full(trial_count * time_count, 1.0 / time_count, dtype=float)
    return stacked_features, repeated_targets, sample_weights


def _validate_unit_ids(
    unit_ids: Sequence[str],
    expected_count: int,
    name: str,
) -> tuple[str, ...]:
    """Validate stable identifiers aligned to one regional unit axis.

    Parameters
    ----------
    unit_ids : sequence[str]
        Candidate ordered identifiers.
    expected_count : int
        Required unit-axis length.
    name : str
        Field name used in errors.

    Returns
    -------
    tuple[str, ...]
        Immutable ordered identifiers.
    """
    if isinstance(unit_ids, (str, bytes)):
        raise ValueError(f"{name} must contain one string per unit.")
    identifiers = tuple(unit_ids)
    if (
        len(identifiers) != expected_count
        or any(not isinstance(value, str) or not value for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError(f"{name} must contain unique nonempty strings aligned to units.")
    return identifiers


def _validate_rate_pair(
    pfc_rates_hz: np.ndarray,
    hpc_rates_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite bilateral rate tensors with common trial/time axes.

    Parameters
    ----------
    pfc_rates_hz, hpc_rates_hz : numpy.ndarray
        Regional firing rates shaped ``(trial, evaluation_time, unit)`` in Hz.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Float64 PFC and HPC arrays retaining their original axis order.
    """
    tensors = []
    for value, name in (
        (pfc_rates_hz, "pfc_rates_hz"),
        (hpc_rates_hz, "hpc_rates_hz"),
    ):
        try:
            tensor = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a numeric firing-rate tensor in Hz.") from error
        if tensor.ndim != 3 or any(size < 1 for size in tensor.shape):
            raise ValueError(f"{name} must have three nonempty axes.")
        if not np.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite firing rates in Hz.")
        tensors.append(tensor)
    pfc_tensor, hpc_tensor = tensors
    if pfc_tensor.shape[:2] != hpc_tensor.shape[:2]:
        raise ValueError("PFC and HPC tensors must share trial and evaluation-time axes.")
    return pfc_tensor, hpc_tensor


def _validate_training_bin_indices(
    training_bin_indices: np.ndarray,
    time_count: int,
) -> np.ndarray:
    """Validate an ascending nonempty selection on the evaluation-time axis.

    Parameters
    ----------
    training_bin_indices : numpy.ndarray
        Candidate dimensionless zero-based bin positions.
    time_count : int
        Available evaluation-time axis length.

    Returns
    -------
    numpy.ndarray
        Integer positions shaped ``(training_time,)``.
    """
    raw = np.asarray(training_bin_indices)
    if (
        raw.ndim != 1
        or raw.size == 0
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("training_bin_indices must be a nonempty integer vector.")
    indices = raw.astype(int, copy=False)
    if (
        (indices < 0).any()
        or (indices >= time_count).any()
        or np.unique(indices).size != indices.size
        or not np.all(np.diff(indices) > 0)
    ):
        raise ValueError("training_bin_indices must be distinct, ascending, and in range.")
    return indices


def fit_pooled_transforms(
    pfc_training_rates_hz: np.ndarray,
    hpc_training_rates_hz: np.ndarray,
    training_bin_indices: np.ndarray,
    *,
    pfc_pc_count: int,
    hpc_pc_count: int,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
) -> PooledTransforms:
    """Fit exactly one training-window transform for each anatomical region.

    Parameters
    ----------
    pfc_training_rates_hz, hpc_training_rates_hz : numpy.ndarray
        Finite regional rates shaped ``(training_trial, evaluation_time,
        regional_unit)`` in Hz.
    training_bin_indices : numpy.ndarray
        Distinct ascending selected positions, shape ``(training_time,)``.
    pfc_pc_count, hpc_pc_count : int
        Positive requested regional component counts.
    pfc_unit_ids, hpc_unit_ids : sequence[str]
        Stable identifiers aligned to regional unit axes.

    Returns
    -------
    PooledTransforms
        Frozen regional transformations and compact anatomical availability.
    """
    pfc_tensor, hpc_tensor = _validate_rate_pair(
        pfc_training_rates_hz,
        hpc_training_rates_hz,
    )
    indices = _validate_training_bin_indices(training_bin_indices, pfc_tensor.shape[1])
    pfc_ids = _validate_unit_ids(pfc_unit_ids, pfc_tensor.shape[2], "pfc_unit_ids")
    hpc_ids = _validate_unit_ids(hpc_unit_ids, hpc_tensor.shape[2], "hpc_unit_ids")
    pfc_transform = modeling.fit_region_transform(pfc_tensor[:, indices, :], pfc_pc_count)
    hpc_transform = modeling.fit_region_transform(hpc_tensor[:, indices, :], hpc_pc_count)
    return PooledTransforms(
        pfc_transform=pfc_transform,
        hpc_transform=hpc_transform,
        pfc_unit_ids=pfc_ids,
        hpc_unit_ids=hpc_ids,
        pfc_reason=None if pfc_transform.is_available else "pfc_transform_unavailable",
        hpc_reason=None if hpc_transform.is_available else "hpc_transform_unavailable",
    )


def _pooled_feature_tensor(
    transforms: PooledTransforms,
    pfc_rates_hz: np.ndarray,
    hpc_rates_hz: np.ndarray,
    *,
    region_configuration: str,
    representation: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply frozen transforms and assemble one regional feature tensor.

    Parameters
    ----------
    transforms : PooledTransforms
        Training-only transform state.
    pfc_rates_hz, hpc_rates_hz : numpy.ndarray
        Aligned regional arrays shaped ``(trial, time, unit)`` in Hz.
    region_configuration : {"PFC", "HPC", "PFC+HPC"}
        Requested anatomical features.
    representation : {"pca", "units"}
        PCA scores or standardized direct units.

    Returns
    -------
    tuple[numpy.ndarray, tuple[str, ...]]
        Feature tensor shaped ``(trial, time, feature)`` and stable feature IDs.
    """
    if region_configuration not in REGION_CONFIGURATIONS:
        raise ValueError(f"Unknown region configuration {region_configuration!r}.")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}.")
    pfc_tensor, hpc_tensor = _validate_rate_pair(pfc_rates_hz, hpc_rates_hz)
    _validate_unit_ids(transforms.pfc_unit_ids, pfc_tensor.shape[2], "pfc_unit_ids")
    _validate_unit_ids(transforms.hpc_unit_ids, hpc_tensor.shape[2], "hpc_unit_ids")

    def regional(
        region_name: str,
        transform: modeling.RegionTransform,
        tensor: np.ndarray,
        unit_ids: tuple[str, ...],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Transform one regional tensor and construct its coefficient IDs."""
        if representation == "units":
            values = transform.transform_units(tensor)
            identifiers = tuple(unit_ids[index] for index in transform.usable_unit_indices)
        else:
            values = transform.transform_pca(tensor)
            identifiers = tuple(
                f"{region_name}:PC{index + 1}"
                for index in range(transform.effective_component_count)
            )
        return values, identifiers

    pfc_features: tuple[np.ndarray, tuple[str, ...]] | None = None
    hpc_features: tuple[np.ndarray, tuple[str, ...]] | None = None
    if region_configuration in {"PFC", "PFC+HPC"}:
        pfc_features = regional(
            "PFC",
            transforms.pfc_transform,
            pfc_tensor,
            transforms.pfc_unit_ids,
        )
    if region_configuration in {"HPC", "PFC+HPC"}:
        hpc_features = regional(
            "HPC",
            transforms.hpc_transform,
            hpc_tensor,
            transforms.hpc_unit_ids,
        )
    if region_configuration == "PFC":
        assert pfc_features is not None
        return pfc_features
    if region_configuration == "HPC":
        assert hpc_features is not None
        return hpc_features
    assert pfc_features is not None and hpc_features is not None
    return (
        np.concatenate((pfc_features[0], hpc_features[0]), axis=2),
        (*pfc_features[1], *hpc_features[1]),
    )


def _pooled_reason_for_modeling_failure(
    failure_kind: modeling.FitFailureKind | modeling.EvaluationFailureKind,
) -> str:
    """Map one closed modeling failure to the pooled compact vocabulary.

    Parameters
    ----------
    failure_kind : modeling.FitFailureKind or modeling.EvaluationFailureKind
        Reviewed fit, prediction, or metric failure.

    Returns
    -------
    str
        Declared compact pooled reason code of at most 32 characters.
    """
    reasons = {
        modeling.FitFailureKind.CONVERGENCE_WARNING: "fit_convergence_failure",
        modeling.FitFailureKind.INVALID_COEFFICIENT_OR_INTERCEPT_SHAPE: (
            "outer_estimator_failure"
        ),
        modeling.FitFailureKind.NONFINITE_COEFFICIENT_OR_INTERCEPT: (
            "outer_estimator_failure"
        ),
        modeling.FitFailureKind.INVALID_CLASS_STRUCTURE: "outer_estimator_failure",
        modeling.FitFailureKind.AMBIGUOUS_POSITIVE_CLASS: "outer_estimator_failure",
        modeling.EvaluationFailureKind.INVALID_PROBABILITY_SHAPE: "invalid_probability",
        modeling.EvaluationFailureKind.NONFINITE_PROBABILITY: "invalid_probability",
        modeling.EvaluationFailureKind.POSITIVE_SCORE_ALIGNMENT_INVALID: "invalid_metric",
        modeling.EvaluationFailureKind.CATEGORICAL_TEST_CLASSES_INVALID: "invalid_metric",
        modeling.EvaluationFailureKind.INVALID_NUMERICAL_PREDICTION: "invalid_prediction",
        modeling.EvaluationFailureKind.INVALID_REGRESSION_METRIC: "invalid_metric",
    }
    if not isinstance(
        failure_kind,
        (modeling.FitFailureKind, modeling.EvaluationFailureKind),
    ):
        raise ValueError(f"Unknown modeling failure {failure_kind!r}.")
    try:
        return reasons[failure_kind]
    except KeyError as error:
        raise ValueError(f"Unknown modeling failure {failure_kind!r}.") from error


def _requested_feature_count(
    transforms: PooledTransforms,
    region_configuration: str,
    representation: str,
) -> int:
    """Return pre-removal feature width for one pooled model identity.

    Parameters
    ----------
    transforms : PooledTransforms
        Regional transform provenance.
    region_configuration : {"PFC", "HPC", "PFC+HPC"}
        Requested anatomical sources.
    representation : {"pca", "units"}
        Requested feature representation.

    Returns
    -------
    int
        Requested dimensionless feature-column count.
    """
    if representation == "units":
        counts = {
            "PFC": transforms.pfc_transform.input_unit_count,
            "HPC": transforms.hpc_transform.input_unit_count,
        }
    elif representation == "pca":
        counts = {
            "PFC": transforms.pfc_transform.requested_pc_count,
            "HPC": transforms.hpc_transform.requested_pc_count,
        }
    else:
        raise ValueError(f"Unknown representation {representation!r}.")
    if region_configuration == "PFC":
        return counts["PFC"]
    if region_configuration == "HPC":
        return counts["HPC"]
    if region_configuration == "PFC+HPC":
        return counts["PFC"] + counts["HPC"]
    raise ValueError(f"Unknown region configuration {region_configuration!r}.")


def _transform_reason_for_region(
    transforms: PooledTransforms,
    region_configuration: str,
) -> str | None:
    """Return the exact dependent-model reason for unavailable transforms.

    Parameters
    ----------
    transforms : PooledTransforms
        Regional availability state.
    region_configuration : {"PFC", "HPC", "PFC+HPC"}
        Model region configuration.

    Returns
    -------
    str or None
        Anatomical standalone reason, combined regional reason, or None.
    """
    if region_configuration == "PFC":
        return transforms.pfc_reason
    if region_configuration == "HPC":
        return transforms.hpc_reason
    if region_configuration == "PFC+HPC":
        if transforms.pfc_reason is not None or transforms.hpc_reason is not None:
            return "regional_transform_unavailable"
        return None
    raise ValueError(f"Unknown region configuration {region_configuration!r}.")


def _coefficient_scale_label(target_family: str, representation: str) -> str:
    """Describe coefficient response and feature scales without unit conversion.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Model response family.
    representation : {"pca", "units"}
        Standardized-unit or PCA-score feature scale.

    Returns
    -------
    str
        Human-readable coefficient unit description.
    """
    response = "log-odds" if target_family == "categorical" else "native target units"
    feature_scale = "PCA score" if representation == "pca" else "pooled training standard deviation"
    return f"{response} per {feature_scale}"


def _unavailable_pooled_model(
    transforms: PooledTransforms,
    *,
    target_identifier: str,
    target_family: str,
    region_configuration: str,
    representation: str,
    reason: str,
    train_trial_count: int,
) -> FittedPooledModel:
    """Build one model state when a reviewed upstream stage prevents fitting.

    Parameters
    ----------
    transforms : PooledTransforms
        Regional provenance, including possibly unavailable transforms.
    target_identifier, target_family : str
        Target identity and response family.
    region_configuration, representation : str
        Model feature identity.
    reason : str
        Declared compact pooled reason code.
    train_trial_count : int
        Observed independent training-trial count.

    Returns
    -------
    FittedPooledModel
        Unavailable record with no runtime estimator.
    """
    requested_count = _requested_feature_count(
        transforms,
        region_configuration,
        representation,
    )
    return FittedPooledModel(
        record=PooledModelRecord(
            target_identifier=target_identifier,
            target_family=target_family,
            region_configuration=region_configuration,
            representation=representation,
            is_valid=False,
            reason=reason,
            train_trial_count=train_trial_count,
            stacked_row_count=-1,
            requested_feature_count=requested_count,
            feature_count=-1,
            estimator_class=None,
            estimator_parameters={},
            convergence_status="not_run",
            coefficients=np.empty(0, dtype=float),
            intercept=float("nan"),
            feature_ids=(),
            positive_class=None,
            coefficient_scale_label=_coefficient_scale_label(target_family, representation),
        ),
        transforms=transforms,
        fitted_estimator=None,
    )


def fit_pooled_model(
    transforms: PooledTransforms,
    pfc_training_rates_hz: np.ndarray,
    hpc_training_rates_hz: np.ndarray,
    training_targets: np.ndarray,
    training_bin_indices: np.ndarray,
    *,
    target_identifier: str,
    target_family: str,
    region_configuration: str,
    representation: str,
) -> FittedPooledModel:
    """Fit one weighted pooled estimator using already-fitted transforms.

    Parameters
    ----------
    transforms : PooledTransforms
        Frozen transformations fit on this same outer-training subset/window.
    pfc_training_rates_hz, hpc_training_rates_hz : numpy.ndarray
        Regional rates shaped ``(training_trial, evaluation_time, unit)`` in Hz.
    training_targets : numpy.ndarray
        Trial-level labels or native-unit targets, shape ``(training_trial,)``.
    training_bin_indices : numpy.ndarray
        Selected ascending evaluation-time positions, shape ``(training_time,)``.
    target_identifier : str
        Stable target name.
    target_family : {"categorical", "numerical"}
        Estimator response family.
    region_configuration : {"PFC", "HPC", "PFC+HPC"}
        Requested anatomical source set.
    representation : {"pca", "units"}
        Requested transformed feature scale.

    Returns
    -------
    FittedPooledModel
        Audit record plus reusable runtime estimator and transformations.
    """
    pfc_tensor, hpc_tensor = _validate_rate_pair(
        pfc_training_rates_hz,
        hpc_training_rates_hz,
    )
    indices = _validate_training_bin_indices(training_bin_indices, pfc_tensor.shape[1])
    targets = np.asarray(training_targets)
    if targets.ndim != 1 or targets.size != pfc_tensor.shape[0]:
        raise ValueError("training_targets must align with the training-trial axis.")
    transform_reason = _transform_reason_for_region(transforms, region_configuration)
    if transform_reason is not None:
        return _unavailable_pooled_model(
            transforms,
            target_identifier=target_identifier,
            target_family=target_family,
            region_configuration=region_configuration,
            representation=representation,
            reason=transform_reason,
            train_trial_count=targets.size,
        )

    feature_tensor, feature_ids = _pooled_feature_tensor(
        transforms,
        pfc_tensor,
        hpc_tensor,
        region_configuration=region_configuration,
        representation=representation,
    )
    selected_features = feature_tensor[:, indices, :]
    if selected_features.shape[2] == 0:
        return _unavailable_pooled_model(
            transforms,
            target_identifier=target_identifier,
            target_family=target_family,
            region_configuration=region_configuration,
            representation=representation,
            reason="no_features",
            train_trial_count=targets.size,
        )
    stacked_features, stacked_targets, sample_weights = stack_training_window(
        selected_features,
        targets,
    )
    estimator = modeling.make_estimator(target_family=target_family)
    estimator_parameters = (
        dict(estimator.get_params(deep=False)) if hasattr(estimator, "get_params") else {}
    )
    fitted = modeling.fit_estimator(
        estimator,
        stacked_features,
        stacked_targets,
        target_family=target_family,
        sample_weight=sample_weights,
    )
    reason = (
        None
        if fitted.failure_kind is None
        else _pooled_reason_for_modeling_failure(fitted.failure_kind)
    )
    record = PooledModelRecord(
        target_identifier=target_identifier,
        target_family=target_family,
        region_configuration=region_configuration,
        representation=representation,
        is_valid=fitted.is_valid,
        reason=reason,
        train_trial_count=targets.size,
        stacked_row_count=stacked_features.shape[0],
        requested_feature_count=_requested_feature_count(
            transforms,
            region_configuration,
            representation,
        ),
        feature_count=stacked_features.shape[1],
        estimator_class=estimator.__class__.__name__,
        estimator_parameters=estimator_parameters,
        convergence_status=fitted.convergence_status,
        coefficients=fitted.coefficients,
        intercept=fitted.intercept,
        feature_ids=feature_ids,
        positive_class=fitted.positive_class,
        coefficient_scale_label=_coefficient_scale_label(target_family, representation),
    )
    return FittedPooledModel(
        record=record,
        transforms=transforms,
        fitted_estimator=fitted if fitted.is_valid else None,
    )


def _metric_template(
    time_count: int,
    target_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate dense metric arrays with family-mismatched cells inapplicable.

    Parameters
    ----------
    time_count : int
        Positive evaluation-time axis length.
    target_family : {"categorical", "numerical"}
        Determines applicable metric columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Status ``U16``, reason ``U32``, and float64 score arrays, each shaped
        ``(evaluation_time, 3)`` in :data:`METRIC_NAMES` order.
    """
    if not isinstance(time_count, int) or time_count < 1:
        raise ValueError("time_count must be a positive integer.")
    if target_family == "categorical":
        applicable = (0, 1)
    elif target_family == "numerical":
        applicable = (2,)
    else:
        raise ValueError("target_family must be 'categorical' or 'numerical'.")
    status = np.full((time_count, len(METRIC_NAMES)), "not_applicable", dtype="U16")
    reasons = np.full((time_count, len(METRIC_NAMES)), "", dtype="U32")
    scores = np.full((time_count, len(METRIC_NAMES)), np.nan, dtype=float)
    status[:, applicable] = "unavailable"
    return status, reasons, scores


def _unavailable_evaluation(
    *,
    target_family: str,
    evaluation_time_centers_s: np.ndarray,
    reason: str,
    test_targets: np.ndarray,
    test_indices: np.ndarray,
) -> PooledEvaluationResult:
    """Construct dense unavailable applicable metrics with observed test counts.

    Parameters
    ----------
    target_family : {"categorical", "numerical"}
        Determines applicable metric columns.
    evaluation_time_centers_s : numpy.ndarray
        Finite centers shaped ``(evaluation_time,)`` in seconds.
    reason : str
        Declared compact pooled reason code.
    test_targets : numpy.ndarray
        Destination trial targets shaped ``(test_trial,)``.
    test_indices : numpy.ndarray
        Destination tensor-row positions shaped ``(test_trial,)``.

    Returns
    -------
    PooledEvaluationResult
        Dense unavailable result; family-mismatched metrics remain inapplicable.
    """
    centers = np.asarray(evaluation_time_centers_s, dtype=float)
    targets = np.asarray(test_targets)
    indices = np.asarray(test_indices)
    status, reasons, scores = _metric_template(centers.size, target_family)
    reasons[status == "unavailable"] = reason
    class_counts = None
    if target_family == "categorical":
        class_counts = (int(np.count_nonzero(targets == 0)), int(np.count_nonzero(targets == 1)))
    return PooledEvaluationResult(
        evaluation_time_centers_s=centers,
        metric_names=METRIC_NAMES,
        status=status,
        reason_codes=reasons,
        scores=scores,
        test_trial_count=targets.size,
        test_class_counts=class_counts,
        test_indices=indices,
    )


def evaluate_pooled_model(
    model: FittedPooledModel,
    pfc_test_rates_hz: np.ndarray,
    hpc_test_rates_hz: np.ndarray,
    test_targets: np.ndarray,
    evaluation_time_centers_s: np.ndarray,
) -> PooledEvaluationResult:
    """Apply one pooled fit unchanged at every evaluation time.

    Parameters
    ----------
    model : FittedPooledModel
        Pooled model and training-only transforms. It is never modified or
        refitted by this function.
    pfc_test_rates_hz, hpc_test_rates_hz : numpy.ndarray
        Destination rates shaped ``(test_trial, evaluation_time, unit)`` in Hz.
    test_targets : numpy.ndarray
        Destination labels or native-unit targets shaped ``(test_trial,)``.
    evaluation_time_centers_s : numpy.ndarray
        Finite choice-relative centers shaped ``(evaluation_time,)`` in seconds.

    Returns
    -------
    PooledEvaluationResult
        Dense time-by-metric scores and compact status codes. No fit, scaling,
        PCA, or calibration state is learned from destination data.
    """
    pfc_tensor, hpc_tensor = _validate_rate_pair(pfc_test_rates_hz, hpc_test_rates_hz)
    targets = np.asarray(test_targets)
    centers = np.asarray(evaluation_time_centers_s, dtype=float)
    if targets.ndim != 1 or targets.size != pfc_tensor.shape[0]:
        raise ValueError("test_targets must align with the test-trial axis.")
    if centers.ndim != 1 or centers.size != pfc_tensor.shape[1] or not np.isfinite(centers).all():
        raise ValueError("evaluation_time_centers_s must align with time and be finite.")
    local_indices = np.arange(targets.size, dtype=int)
    if not model.record.is_valid:
        if model.record.reason is None:
            raise ValueError("Unavailable pooled model lacks a compact reason.")
        return _unavailable_evaluation(
            target_family=model.record.target_family,
            evaluation_time_centers_s=centers,
            reason=model.record.reason,
            test_targets=targets,
            test_indices=local_indices,
        )
    if model.transforms is None or model.fitted_estimator is None:
        raise ValueError("Valid pooled model lacks runtime transform or estimator state.")

    feature_tensor, feature_ids = _pooled_feature_tensor(
        model.transforms,
        pfc_tensor,
        hpc_tensor,
        region_configuration=model.record.region_configuration,
        representation=model.record.representation,
    )
    if feature_ids != model.record.feature_ids:
        raise ValueError("Evaluation feature identities do not match the fitted model.")
    status, reasons, scores = _metric_template(centers.size, model.record.target_family)
    applicable_metrics = (
        ("balanced_accuracy", "auc")
        if model.record.target_family == "categorical"
        else ("r2",)
    )
    for time_index in range(centers.size):
        evaluation = modeling.evaluate_fitted_estimator(
            model.fitted_estimator,
            feature_tensor[:, time_index, :],
            targets,
            target_family=model.record.target_family,
        )
        if evaluation.is_valid:
            for metric_name in applicable_metrics:
                metric_index = METRIC_NAMES.index(metric_name)
                status[time_index, metric_index] = "valid"
                scores[time_index, metric_index] = evaluation.metrics[metric_name]
        else:
            if evaluation.failure_kind is None:
                raise ValueError("Unavailable estimator evaluation lacks a failure kind.")
            reason = _pooled_reason_for_modeling_failure(evaluation.failure_kind)
            for metric_name in applicable_metrics:
                metric_index = METRIC_NAMES.index(metric_name)
                reasons[time_index, metric_index] = reason
    class_counts = None
    if model.record.target_family == "categorical":
        class_counts = (int(np.count_nonzero(targets == 0)), int(np.count_nonzero(targets == 1)))
    return PooledEvaluationResult(
        evaluation_time_centers_s=centers,
        metric_names=METRIC_NAMES,
        status=status,
        reason_codes=reasons,
        scores=scores,
        test_trial_count=targets.size,
        test_class_counts=class_counts,
        test_indices=local_indices,
    )


def _native_minus_transfer_gap(
    native: PooledEvaluationResult,
    transfer: PooledEvaluationResult,
) -> PooledGapResult:
    """Subtract aligned native and reference-transfer metric arrays.

    Parameters
    ----------
    native, transfer : PooledEvaluationResult
        Evaluations on the exact same destination tensor rows and time/metric
        axes. Scores are dimensionless decoding metrics.

    Returns
    -------
    PooledGapResult
        Native-minus-transfer scores where both inputs are valid; unavailable
        applicable sources map to ``source_score_unavailable``.
    """
    if native.metric_names != transfer.metric_names:
        raise ValueError("Native and transfer metric identities must match.")
    if native.scores.shape != transfer.scores.shape:
        raise ValueError("Native and transfer score arrays must have equal shape.")
    if native.test_indices is not transfer.test_indices:
        raise ValueError("Native and transfer evaluations must share destination rows.")
    status = np.full(native.status.shape, "not_applicable", dtype="U16")
    reasons = np.full(native.status.shape, "", dtype="U32")
    scores = np.full(native.status.shape, np.nan, dtype=float)
    applicable = (native.status != "not_applicable") | (transfer.status != "not_applicable")
    both_valid = (native.status == "valid") & (transfer.status == "valid")
    status[applicable] = "unavailable"
    reasons[applicable] = "source_score_unavailable"
    status[both_valid] = "valid"
    reasons[both_valid] = ""
    scores[both_valid] = native.scores[both_valid] - transfer.scores[both_valid]
    return PooledGapResult(status=status, reason_codes=reasons, scores=scores)


def _condition_subset_reason(
    targets: np.ndarray,
    *,
    target_family: str,
    partition: str,
) -> str | None:
    """Validate one condition/fold subset with trial-level family rules.

    Parameters
    ----------
    targets : numpy.ndarray
        One-dimensional subset labels or native-unit numerical values.
    target_family : {"categorical", "numerical"}
        Selects exact class or finite-nonconstant requirements.
    partition : {"training", "test"}
        Selects the declared compact reason namespace.

    Returns
    -------
    str or None
        Exact compact subset reason, or None when scientifically usable.
    """
    values = np.asarray(targets)
    if values.ndim != 1:
        raise ValueError("Condition subset targets must be one-dimensional.")
    if partition not in {"training", "test"}:
        raise ValueError("partition must be 'training' or 'test'.")
    if values.size == 0:
        return "empty_training_subset" if partition == "training" else "empty_test_subset"
    if target_family == "categorical":
        try:
            finite = bool(np.isfinite(values).all())
        except TypeError:
            finite = False
        if not finite or set(np.unique(values).tolist()) != {0, 1}:
            return (
                "training_class_unavailable"
                if partition == "training"
                else "test_class_unavailable"
            )
        return None
    if target_family == "numerical":
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            numeric = np.full(values.shape, np.nan)
        if numeric.size < 2 or not np.isfinite(numeric).all() or np.unique(numeric).size < 2:
            return (
                "training_target_unavailable"
                if partition == "training"
                else "test_target_unavailable"
            )
        return None
    raise ValueError("target_family must be 'categorical' or 'numerical'.")


def _unavailable_model_before_transforms(
    *,
    target_identifier: str,
    target_family: str,
    region_configuration: str,
    representation: str,
    reason: str,
    train_trial_count: int,
    pfc_unit_count: int,
    hpc_unit_count: int,
    pfc_pc_count: int,
    hpc_pc_count: int,
) -> FittedPooledModel:
    """Build one unavailable model when fitting stops before transformations.

    Parameters
    ----------
    target_identifier, target_family : str
        Target identity and response family.
    region_configuration, representation : str
        Requested model features.
    reason : str
        Compact upstream unavailability code.
    train_trial_count : int
        Observed independent training rows.
    pfc_unit_count, hpc_unit_count : int
        Requested direct-unit widths.
    pfc_pc_count, hpc_pc_count : int
        Requested PCA widths.

    Returns
    -------
    FittedPooledModel
        Unavailable record with no transformations or estimator.
    """
    if representation == "units":
        regional_counts = {"PFC": pfc_unit_count, "HPC": hpc_unit_count}
    elif representation == "pca":
        regional_counts = {"PFC": pfc_pc_count, "HPC": hpc_pc_count}
    else:
        raise ValueError(f"Unknown representation {representation!r}.")
    if region_configuration == "PFC":
        requested_count = regional_counts["PFC"]
    elif region_configuration == "HPC":
        requested_count = regional_counts["HPC"]
    elif region_configuration == "PFC+HPC":
        requested_count = regional_counts["PFC"] + regional_counts["HPC"]
    else:
        raise ValueError(f"Unknown region configuration {region_configuration!r}.")
    record = PooledModelRecord(
        target_identifier=target_identifier,
        target_family=target_family,
        region_configuration=region_configuration,
        representation=representation,
        is_valid=False,
        reason=reason,
        train_trial_count=train_trial_count,
        stacked_row_count=-1,
        requested_feature_count=requested_count,
        feature_count=-1,
        estimator_class=None,
        estimator_parameters={},
        convergence_status="not_run",
        coefficients=np.empty(0, dtype=float),
        intercept=float("nan"),
        feature_ids=(),
        positive_class=None,
        coefficient_scale_label=_coefficient_scale_label(target_family, representation),
    )
    return FittedPooledModel(record=record, transforms=None, fitted_estimator=None)


def decode_pooled_window_target(
    *,
    training_window: TrainingWindow,
    target_identifier: str,
    target_family: str,
    target_values: np.ndarray,
    master_eligible: np.ndarray,
    block_ids: np.ndarray,
    condition_masks: Mapping[str, np.ndarray],
    outer_split: modeling.SplitPlan,
    pfc_rate_tensor_hz: np.ndarray,
    hpc_rate_tensor_hz: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    evaluation_time_centers_s: np.ndarray,
    pfc_pc_count: int,
    hpc_pc_count: int,
) -> PooledWindowTargetResult:
    """Fit native pooled models and evaluate native/reference generalization.

    Parameters
    ----------
    training_window : TrainingWindow
        Named half-open interval and selected bin positions.
    target_identifier : str
        Stable target name.
    target_family : {"categorical", "numerical"}
        Estimator and metric family.
    target_values, master_eligible, block_ids : numpy.ndarray
        Aligned one-dimensional arrays shaped ``(tensor_trial,)``. Targets are
        categorical labels or native values; eligibility is Boolean; blocks
        contain exact dimensionless behavioral group identities.
    condition_masks : mapping[str, numpy.ndarray]
        Nonempty mapping of aligned Boolean destination/training masks and
        including ``correct_rewarded``.
    outer_split : modeling.SplitPlan
        One target-wide grouped fold map on the same tensor-row axis.
    pfc_rate_tensor_hz, hpc_rate_tensor_hz : numpy.ndarray
        Aligned rates shaped ``(tensor_trial, evaluation_time, unit)`` in Hz.
    pfc_unit_ids, hpc_unit_ids : sequence[str]
        Stable identifiers aligned to regional unit axes.
    evaluation_time_centers_s : numpy.ndarray
        Finite choice-relative centers shaped ``(evaluation_time,)`` in seconds.
    pfc_pc_count, hpc_pc_count : int
        Positive requested regional PCA widths.

    Returns
    -------
    PooledWindowTargetResult
        Model identities, dense evaluation/gap arrays, and shared transforms.
        Each model is fit once and every later evaluation is read-only.
    """
    if not isinstance(training_window, TrainingWindow):
        raise ValueError("training_window must be a resolved TrainingWindow.")
    pfc_tensor, hpc_tensor = _validate_rate_pair(
        pfc_rate_tensor_hz,
        hpc_rate_tensor_hz,
    )
    training_indices = _validate_training_bin_indices(
        training_window.bin_indices,
        pfc_tensor.shape[1],
    )
    pfc_ids = _validate_unit_ids(pfc_unit_ids, pfc_tensor.shape[2], "pfc_unit_ids")
    hpc_ids = _validate_unit_ids(hpc_unit_ids, hpc_tensor.shape[2], "hpc_unit_ids")
    targets = np.asarray(target_values)
    eligible = _boolean_trial_mask(master_eligible, "master_eligible")
    groups = np.asarray(block_ids)
    centers = np.asarray(evaluation_time_centers_s, dtype=float)
    trial_count = pfc_tensor.shape[0]
    if (
        targets.ndim != 1
        or groups.ndim != 1
        or targets.size != trial_count
        or groups.size != trial_count
        or eligible.size != trial_count
    ):
        raise ValueError("Target, eligibility, block, and tensor trial axes must align.")
    if centers.ndim != 1 or centers.size != pfc_tensor.shape[1] or not np.isfinite(centers).all():
        raise ValueError("evaluation_time_centers_s must align with time and be finite.")
    if not isinstance(condition_masks, Mapping) or not condition_masks:
        raise ValueError("condition_masks must be a nonempty mapping.")
    if REFERENCE_CONDITION not in condition_masks:
        raise ValueError("condition_masks must include correct_rewarded.")
    normalized_conditions = {
        name: _boolean_trial_mask(mask, f"condition_masks[{name!r}]")
        for name, mask in condition_masks.items()
    }
    if any(mask.size != trial_count for mask in normalized_conditions.values()):
        raise ValueError("Every condition mask must align with the tensor-trial axis.")
    if not outer_split.is_available:
        return PooledWindowTargetResult(
            models={},
            evaluations={},
            gaps={},
            transforms={},
            unavailable_reason="target_outer_unavailable",
            target_identifier=target_identifier,
            target_family=target_family,
            outer_fold_ids=np.asarray(outer_split.fold_ids, dtype=int).copy(),
        )

    models: dict[tuple[str, int, str, str], FittedPooledModel] = {}
    transforms_by_cell: dict[tuple[str, int], PooledTransforms] = {}
    for condition_name, condition_mask in normalized_conditions.items():
        for split in outer_split.splits:
            train_indices = split.train_indices[
                eligible[split.train_indices] & condition_mask[split.train_indices]
            ]
            training_reason = _condition_subset_reason(
                targets[train_indices],
                target_family=target_family,
                partition="training",
            )
            if training_reason is None:
                transforms = fit_pooled_transforms(
                    pfc_tensor[train_indices],
                    hpc_tensor[train_indices],
                    training_indices,
                    pfc_pc_count=pfc_pc_count,
                    hpc_pc_count=hpc_pc_count,
                    pfc_unit_ids=pfc_ids,
                    hpc_unit_ids=hpc_ids,
                )
                transforms_by_cell[(condition_name, split.fold_id)] = transforms
            else:
                transforms = None
            for region_configuration in REGION_CONFIGURATIONS:
                for representation in REPRESENTATIONS:
                    key = (
                        condition_name,
                        split.fold_id,
                        region_configuration,
                        representation,
                    )
                    if transforms is None:
                        assert training_reason is not None
                        models[key] = _unavailable_model_before_transforms(
                            target_identifier=target_identifier,
                            target_family=target_family,
                            region_configuration=region_configuration,
                            representation=representation,
                            reason=training_reason,
                            train_trial_count=train_indices.size,
                            pfc_unit_count=pfc_tensor.shape[2],
                            hpc_unit_count=hpc_tensor.shape[2],
                            pfc_pc_count=pfc_pc_count,
                            hpc_pc_count=hpc_pc_count,
                        )
                    else:
                        models[key] = fit_pooled_model(
                            transforms,
                            pfc_tensor[train_indices],
                            hpc_tensor[train_indices],
                            targets[train_indices],
                            training_indices,
                            target_identifier=target_identifier,
                            target_family=target_family,
                            region_configuration=region_configuration,
                            representation=representation,
                        )

    evaluation_pairs = [
        (REFERENCE_CONDITION, destination)
        for destination in normalized_conditions
    ]
    evaluation_pairs.extend(
        (condition_name, condition_name)
        for condition_name in normalized_conditions
        if condition_name != REFERENCE_CONDITION
    )
    test_indices_by_destination: dict[tuple[str, int], np.ndarray] = {}
    for destination, condition_mask in normalized_conditions.items():
        for split in outer_split.splits:
            test_indices_by_destination[(destination, split.fold_id)] = split.test_indices[
                eligible[split.test_indices] & condition_mask[split.test_indices]
            ]

    evaluations: dict[tuple[str, str, int, str, str], PooledEvaluationResult] = {}
    for train_condition, test_condition in evaluation_pairs:
        for split in outer_split.splits:
            test_indices = test_indices_by_destination[(test_condition, split.fold_id)]
            test_reason = _condition_subset_reason(
                targets[test_indices],
                target_family=target_family,
                partition="test",
            )
            for region_configuration in REGION_CONFIGURATIONS:
                for representation in REPRESENTATIONS:
                    model = models[
                        (
                            train_condition,
                            split.fold_id,
                            region_configuration,
                            representation,
                        )
                    ]
                    if not model.record.is_valid:
                        assert model.record.reason is not None
                        evaluation = _unavailable_evaluation(
                            target_family=target_family,
                            evaluation_time_centers_s=centers,
                            reason=model.record.reason,
                            test_targets=targets[test_indices],
                            test_indices=test_indices,
                        )
                    elif test_reason is not None:
                        evaluation = _unavailable_evaluation(
                            target_family=target_family,
                            evaluation_time_centers_s=centers,
                            reason=test_reason,
                            test_targets=targets[test_indices],
                            test_indices=test_indices,
                        )
                    else:
                        evaluation = evaluate_pooled_model(
                            model,
                            pfc_tensor[test_indices],
                            hpc_tensor[test_indices],
                            targets[test_indices],
                            centers,
                        )
                        evaluation = replace(evaluation, test_indices=test_indices)
                    evaluations[
                        (
                            train_condition,
                            test_condition,
                            split.fold_id,
                            region_configuration,
                            representation,
                        )
                    ] = evaluation

    gaps: dict[tuple[str, int, str, str], PooledGapResult] = {}
    for destination in normalized_conditions:
        for split in outer_split.splits:
            for region_configuration in REGION_CONFIGURATIONS:
                for representation in REPRESENTATIONS:
                    native = evaluations[
                        (
                            destination,
                            destination,
                            split.fold_id,
                            region_configuration,
                            representation,
                        )
                    ]
                    transfer = evaluations[
                        (
                            REFERENCE_CONDITION,
                            destination,
                            split.fold_id,
                            region_configuration,
                            representation,
                        )
                    ]
                    gaps[
                        (
                            destination,
                            split.fold_id,
                            region_configuration,
                            representation,
                        )
                    ] = _native_minus_transfer_gap(native, transfer)
    return PooledWindowTargetResult(
        models=models,
        evaluations=evaluations,
        gaps=gaps,
        transforms=transforms_by_cell,
        unavailable_reason=None,
        target_identifier=target_identifier,
        target_family=target_family,
        outer_fold_ids=np.asarray(outer_split.fold_ids, dtype=int).copy(),
    )
