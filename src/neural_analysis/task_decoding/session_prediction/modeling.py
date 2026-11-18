"""Weighted fixed-penalty readouts on original session-PC scores."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

from src.neural_analysis.task_decoding.pooled.decoding import stack_training_window

from .config import FOLD_COUNT, LOGISTIC_C, RIDGE_ALPHA


@dataclass(frozen=True)
class RegionPredictions:
    """One region's models and held-out predictions, with explicit PC identities.

    N=shared choices, W=regional windows, B=bins, T=targets, F=5, K=PCs.
    predictions is float64 (W,B,T), in native target units or class-1 probability;
    prediction_reason is str (W,T). coefficients/intercepts are (T,F,K)/(T,F),
    in target-or-logit units per PC score / target-or-logit units respectively.
    pc_ids is (K,) original one-based rank. fit_status/reason and counts are
    (T,F); training_choice_mask is bool (T,F,N). training_pc_sd and contribution
    strength are (T,F,K), in PC-score / target-or-logit units. Means/orders are
    (T,K); contributing_fold_count/color_limit are (T,). NaN parameters and
    nonempty reasons mark unavailable fits. No estimator or source score is saved.
    """

    pc_ids: np.ndarray
    predictions: np.ndarray
    prediction_reason: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    fit_status: np.ndarray
    fit_reason: np.ndarray
    training_choice_mask: np.ndarray
    training_trial_count: np.ndarray
    training_row_count: np.ndarray
    training_pc_sd: np.ndarray
    contribution_strength: np.ndarray
    mean_contribution_strength: np.ndarray
    contribution_pc_order: np.ndarray
    contributing_fold_count: np.ndarray
    contribution_color_limit: np.ndarray


def fit_region_prediction_models(
    scores: np.ndarray,
    choice_positions: np.ndarray,
    target_values: np.ndarray,
    target_valid: np.ndarray,
    baseline_valid: np.ndarray,
    correct_rewarded: np.ndarray,
    fold_ids: np.ndarray,
    training_bin_mask: np.ndarray,
    target_families: tuple[str, ...],
    *,
    training_condition_mask: np.ndarray | None = None,
) -> RegionPredictions:
    """Fit up to T*5 weighted models and predict every assigned regional bin.

    scores is finite float (W,B,K) in saved unwhitened PC units; integer
    choice_positions (W,) indexes N shared choices. target_values/target_valid
    are float/bool (N,T) in native units or 0/1 labels. baseline_valid and
    correct_rewarded are bool (N,); fold_ids is integer (N,), -1 or 0..4.
    training_bin_mask is bool (B,) selecting the supervised interval; families is T categorical/
    numerical strings. Return RegionPredictions; all input arrays stay unchanged.
    Optional bool training_condition_mask (N,) selects mode-specific membership;
    None preserves the existing correct_rewarded training rule. All other
    eligibility intersections and equal total weight per trial are unchanged.
    """
    _validate_inputs(scores, choice_positions, target_values, target_valid,
                     baseline_valid, correct_rewarded, fold_ids, training_bin_mask,
                     target_families)
    membership = correct_rewarded if training_condition_mask is None else training_condition_mask
    if membership.shape != (len(target_values),) or membership.dtype != bool:
        raise ValueError("Training condition mask must be Boolean with shape (N,)")
    values = np.asarray(scores, dtype=np.float64)
    positions = np.asarray(choice_positions)
    targets = np.asarray(target_values, dtype=np.float64)
    target_count = targets.shape[1]
    component_count = values.shape[2]
    model_shape = (target_count, FOLD_COUNT)
    coefficients = np.full((*model_shape, component_count), np.nan)
    intercepts = np.full(model_shape, np.nan)
    statuses = np.full(model_shape, "unavailable", dtype="U11")
    reasons = np.full(model_shape, "empty_training", dtype="U160")
    training_choices = np.zeros((*model_shape, len(targets)), dtype=bool)
    trial_counts = np.zeros(model_shape, dtype=np.int64)
    row_counts = np.zeros(model_shape, dtype=np.int64)
    scales = np.full_like(coefficients, np.nan)
    strengths = np.full_like(coefficients, np.nan)
    regional_coverage = np.zeros(len(targets), dtype=bool)
    regional_coverage[positions] = True
    for target, family in enumerate(target_families):
        eligible = (regional_coverage & baseline_valid & membership
                    & target_valid[:, target] & (fold_ids >= 0))
        for fold in range(FOLD_COUNT):
            mask = eligible & (fold_ids != fold)
            training_choices[target, fold] = mask
            trial_counts[target, fold] = int(mask.sum())
            row_counts[target, fold] = int(mask.sum() * training_bin_mask.sum())
            if not mask.any():
                if family == "numerical" and not target_valid[:, target].any():
                    reasons[target, fold] = "target_unavailable_no_values"
                continue
            tensor = values[mask[positions]][:, training_bin_mask, :]
            matrix, labels, weights = stack_training_window(tensor, targets[mask, target])
            beta, intercept, reason = fit_readout(matrix, labels, weights, family)
            if reason:
                reasons[target, fold] = reason
                continue
            coefficients[target, fold] = beta
            intercepts[target, fold] = intercept
            statuses[target, fold] = "ok"
            reasons[target, fold] = ""
            scales[target, fold] = matrix.std(axis=0, ddof=0)
            strengths[target, fold] = abs(beta) * scales[target, fold]
    predictions, prediction_reasons = predict_region_windows(
        values, positions, fold_ids, coefficients, intercepts, statuses,
        reasons, target_families,
    )
    means = np.full((target_count, component_count), np.nan)
    orders = np.tile(np.arange(1, component_count + 1), (target_count, 1))
    fold_counts = (statuses == "ok").sum(axis=1)
    limits = np.zeros(target_count)
    for target in range(target_count):
        fitted = statuses[target] == "ok"
        if fitted.any():
            means[target] = strengths[target, fitted].mean(axis=0)
            orders[target] = np.lexsort((np.arange(component_count), -means[target])) + 1
        for fold in np.flatnonzero(fitted):
            selected = ((fold_ids[positions] == fold) & (prediction_reasons[:, target] == ""))
            if selected.any():
                # max(|score*weight|) factors by PC, avoiding a contribution tensor.
                maximum = np.abs(values[selected]).max(axis=(0, 1))
                limit = float((maximum * abs(coefficients[target, fold])).max())
                limits[target] = max(limits[target], limit)
    limits[limits == 0] = 1.0
    return RegionPredictions(
        np.arange(1, component_count + 1), predictions, prediction_reasons,
        coefficients, intercepts, statuses, reasons, training_choices,
        trial_counts, row_counts, scales, strengths, means, orders,
        fold_counts, limits,
    )


def predict_region_windows(
    scores: np.ndarray, choice_positions: np.ndarray, fold_ids: np.ndarray,
    coefficients: np.ndarray, intercepts: np.ndarray, fit_status: np.ndarray,
    fit_reason: np.ndarray, target_families: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return float predictions (W,B,T) and reason strings (W,T), without fitting.

    scores is (W,B,K) in saved PC units; choice_positions (W,) maps to shared
    integer fold_ids (N,), -1 or 0..4. Coefficients/intercepts/status/reason have
    (T,5,K)/(T,5) axes; families identifies probability versus native outputs.
    Missing actual targets are irrelevant. Nonfinite outputs invalidate every
    destination window of that model; no clipping or fallback occurs.
    """
    output = np.full((*scores.shape[:2], len(target_families)), np.nan)
    reasons = np.full((len(scores), len(target_families)), "missing_block", dtype="U160")
    window_folds = fold_ids[choice_positions]
    for target, family in enumerate(target_families):
        for fold in range(FOLD_COUNT):
            selected = window_folds == fold
            if fit_status[target, fold] != "ok":
                reasons[selected, target] = fit_reason[target, fold]
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                linear = intercepts[target, fold] + scores[selected] @ coefficients[target, fold]
                values = expit(linear) if family == "categorical" else linear
            if not np.isfinite(linear).all() or not np.isfinite(values).all():
                reasons[selected, target] = "nonfinite_prediction"
                continue
            output[selected, :, target] = values
            reasons[selected, target] = ""
    return output, reasons


def build_pc_contributions(
    scores: np.ndarray, window_fold_ids: np.ndarray, result: RegionPredictions,
    target: int, displayed_pc_ids: tuple[int, ...],
    *, window_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return signed contributions (selected W,B,P), in target or log-odds units.

    scores is original finite (W,B,K), window_fold_ids is (W,), target a saved
    zero-based target index. displayed_pc_ids are P unique original one-based
    PC ranks; optional integer window_indices selects viewport windows first.
    Unavailable predictions stay NaN, including missing folds. No input changes.
    """
    ids = np.asarray(displayed_pc_ids)
    if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or not np.isin(ids, result.pc_ids).all()):
        raise ValueError("Displayed PC IDs must be unique original model PC ranks")
    windows = np.arange(len(scores)) if window_indices is None else np.asarray(window_indices)
    weights_axis = np.array([np.flatnonzero(result.pc_ids == rank)[0] for rank in ids])
    output = np.full((len(windows), scores.shape[1], len(ids)), np.nan)
    folds = window_fold_ids[windows]
    for fold in range(FOLD_COUNT):
        selected = (folds == fold) & (result.prediction_reason[windows, target] == "")
        if result.fit_status[target, fold] == "ok" and selected.any():
            output[selected] = (scores[windows[selected]][:, :, ids - 1]
                                * result.coefficients[target, fold, weights_axis])
    return output


# Shared readout fitting and input validation.
def fit_readout(
    matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray, family: str,
) -> tuple[np.ndarray | None, float, str]:
    """Fit an affine readout with an unpenalized intercept and supplied row weights.

    matrix is finite float (M,K) in original PC-score units. labels is finite
    float (M,): dimensionless 0/1 for family="categorical", with generic class 1
    positive, or native target units for family="numerical". weights is positive
    finite float (M,) in dimensionless units; arbitrary row weights are used
    directly without normalization or additional class weighting.

    Return float coefficients (K,) and a scalar float intercept in log-odds per
    PC-score unit / log-odds for categorical labels, or target units per PC-score
    unit / target units for numerical labels, plus an empty reason string.
    Unavailable fits return (None, NaN, nonempty reason). Inputs are unchanged.
    """
    if family == "categorical" and np.unique(labels).size < 2:
        return None, np.nan, "single_class_training"
    estimator = (
        LogisticRegression(C=LOGISTIC_C, l1_ratio=0.0, solver="lbfgs",
                           fit_intercept=True, class_weight=None, tol=1e-4, max_iter=5000)
        if family == "categorical" else Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(matrix, labels, sample_weight=weights)
    except ConvergenceWarning:
        return None, np.nan, "nonconvergence"
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        return None, np.nan, f"numerical_fit_failure: {error}"
    beta = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
    if not np.isfinite(beta).all() or not np.isfinite(intercept):
        return None, np.nan, "nonfinite_parameters"
    return beta, intercept, ""


def _validate_inputs(
    scores: np.ndarray, positions: np.ndarray, targets: np.ndarray,
    valid: np.ndarray, baseline: np.ndarray, condition: np.ndarray,
    folds: np.ndarray, bins: np.ndarray, families: tuple[str, ...],
) -> None:
    """Validate W,B,K scores, W->N mapping, N,T labels/masks and N/B masks; return None.

    Score/target units are untouched. Raise on structural errors before any fit.
    """
    if (scores.ndim != 3 or not all(scores.shape) or not np.isfinite(scores).all()
            or targets.ndim != 2 or valid.shape != targets.shape
            or targets.shape[1] != len(families)):
        raise ValueError("Expected finite scores(W,B,K) and aligned targets/validity(N,T)")
    n = len(targets)
    if (positions.shape != (len(scores),) or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions < 0) or np.any(positions >= n) or np.any(np.diff(positions) <= 0)):
        raise ValueError("Regional choice positions must be ordered unique indices into N")
    if (folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer)
            or not np.isin(folds, [-1, 0, 1, 2, 3, 4]).all()):
        raise ValueError("Fold IDs must be integer (N,) values -1..4")
    for mask, shape in ((valid, targets.shape), (baseline, (n,)),
                        (condition, (n,)), (bins, (scores.shape[1],))):
        if mask.shape != shape or mask.dtype != bool:
            raise ValueError("Eligibility masks must be Boolean with explicit target/choice/bin axes")
    if not bins.any() or not np.isfinite(targets[valid]).all():
        raise ValueError("Training bins must be nonempty and valid target values finite")
    for target, family in enumerate(families):
        if family not in {"categorical", "numerical"}:
            raise ValueError("Target families must be categorical or numerical")
        if family == "categorical" and not np.isin(targets[valid[:, target], target], [0, 1]).all():
            raise ValueError("Categorical targets must have exact 0/1 labels")
