"""Pure multi-response ridge/OLS on native predictors and unit-Hz or PC responses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


@dataclass(frozen=True)
class LinearFit:
    """Numerical fit with slopes(P,R), intercept(R,), and actual slope penalties(P,).

    Responses retain their supplied physical units; slopes are response units
    per native predictor unit. configured_alpha is positive for ridge, None
    for OLS; OLS penalties are zero. No analysis/representation dispatch occurs.
    """

    slopes: np.ndarray
    intercept: np.ndarray
    estimator: str
    configured_alpha: float | None
    penalties: np.ndarray
    coefficient_convention: str
    limitations: tuple[str, ...]


# Public numerical computations use native arrays and explicit feature penalties.
def fit_ridge_responses(
    predictors: np.ndarray,
    responses: np.ndarray,
    alpha: float = 1.0,
    *,
    penalty_weights: np.ndarray | None = None,
) -> LinearFit:
    """Fit X(N,P), Y(N,R) with summed SSE, an unpenalized intercept and feature penalties.

    Inputs must be finite two-dimensional arrays in native predictor/response
    units. alpha is positive; penalty_weights(P,) is positive dimensionless
    feature weights, default ones. Actual penalties are alpha*weights.
    Reparameterize X by 1/sqrt(weights) for the existing Ridge SVD solver,
    then map slopes back to native units. Return LinearFit with slopes(P,R),
    intercept(R,) and actual penalties(P,). Inputs and their axes are unchanged.
    """
    x, y = _validate_arrays(predictors, responses)
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive number")
    weights = _penalty_weights(penalty_weights, x.shape[1])
    scale = 1 / np.sqrt(weights)
    estimator = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    estimator.fit(x * scale, y)
    slopes, intercept = _coefficient_arrays(
        estimator.coef_, estimator.intercept_, x.shape[1], y.shape[1]
    )
    slopes *= scale[:, None]
    return LinearFit(
        slopes,
        intercept,
        "ridge",
        float(alpha),
        float(alpha) * weights,
        "Feature-weighted L2 slopes; unpenalized intercept",
        _fit_limitations(x, y),
    )


def fit_ols_responses(predictors: np.ndarray, responses: np.ndarray) -> LinearFit:
    """Fit centered dense least squares with minimum-Euclidean-norm slopes.

    Parameters
    ----------
    predictors : np.ndarray
        Finite X (N,P) in native dimensionless predictor units, no ones column.
    responses : np.ndarray
        Finite Y (N,K) in supplied response units.

    Returns
    -------
    LinearFit
        Slopes (P,K), intercept (K,), zero penalties (P,). LinearRegression's dense
        centered solve minimizes slope norm, then recovers the unpenalized
        intercept. Redundant indicators are retained. Inputs are not changed.
    """
    x, y = _validate_arrays(predictors, responses)
    estimator = LinearRegression(fit_intercept=True)
    estimator.fit(x, y)
    slopes, intercept = _coefficient_arrays(
        estimator.coef_, estimator.intercept_, x.shape[1], y.shape[1]
    )
    return LinearFit(
        slopes,
        intercept,
        "ols",
        None,
        np.zeros(x.shape[1]),
        "Centered minimum-Euclidean-norm slopes; unpenalized intercept",
        _fit_limitations(x, y),
    )


def predict_responses(predictors: np.ndarray, fit: LinearFit) -> np.ndarray:
    """Apply a fixed fit without rescaling predictors or responses.

    Parameters
    ----------
    predictors : np.ndarray
        Finite native X (N,P), same column order/coding used to fit.
    fit : LinearFit
        Slopes (P,K) and intercept (K,) in original response units.

    Returns
    -------
    np.ndarray
        Float64 predicted responses (N,K), original response units. Nonfinite
        inputs/results and mismatched predictor axes raise ValueError.
    """
    x = np.asarray(predictors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != fit.slopes.shape[0]:
        raise ValueError("predictor shape must be (rows, fitted predictor columns)")
    if not np.isfinite(x).all():
        raise ValueError("predictors must contain only finite values")
    predicted = x @ fit.slopes + fit.intercept
    if not np.isfinite(predicted).all():
        raise ValueError("predictions must contain only finite values")
    return predicted


def standardize_coefficients(
    fit: LinearFit,
    predictors: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    """Calculate beta*SD(X)/SD(Y) on exactly the supplied reference-fit rows.

    Parameters
    ----------
    fit : LinearFit
        Raw fitted slopes (P,K), never modified by this presentation operation.
    predictors, responses : np.ndarray
        Reference-fit X (N,P) and Y (N,K), preserving native predictor/response units.
        Both arrays must contain exactly the eligible rows used for this fit.

    Returns
    -------
    np.ndarray
        Dimensionless display slopes (P,K). Constant predictors yield valid zero
        values for variable responses. Mathematically constant responses yield
        NaN even when floating-point mean roundoff produces a tiny computed SD.
        This does not refit, whiten or normalize the saved neural activity.
    """
    x, y = _validate_arrays(predictors, responses)
    if fit.slopes.shape != (x.shape[1], y.shape[1]):
        raise ValueError("fit slopes must match supplied predictor/response shape")
    response_sd = y.std(axis=0, ddof=0)
    response_sd[_constant_columns(y)] = np.nan
    predictor_sd = x.std(axis=0, ddof=0)
    predictor_sd[_constant_columns(x)] = 0
    return fit.slopes * predictor_sd[:, None] / response_sd[None, :]


def evaluate_block_cv(
    predictors: np.ndarray,
    responses: np.ndarray,
    fold_ids: np.ndarray,
    *,
    estimator: str = "ridge",
    alpha: float = 1.0,
    penalty_weights: np.ndarray | None = None,
) -> dict:
    """Fit reference and five preassigned training folds, scoring pooled held-out bins.

    Parameters
    ----------
    predictors, responses : np.ndarray
        Finite X (N,P), Y (N,K) in native predictor and supplied response units.
        The caller has already selected the fit window and valid predictors.
    fold_ids : np.ndarray
        Integer (N,) window/block assignments repeated across bins, 0..4.
        -1 marks unavailable assignments: these rows remain in the reference
        fit but enter no training/held-out fold. No reassignment occurs.
    estimator : str
        ridge or ols. OLS has zero feature penalties.
    alpha, penalty_weights : float, np.ndarray or None
        Positive summed-SSE alpha and (P,) feature weights, propagated to every ridge fit.

    Returns
    -------
    dict
        Separate reference LinearFit/predictions (N,K); five fold_fits and
        fold_slopes (5,P,K)/fold_intercepts (5,K); held_out_predictions (N,K)
        in original row order; fold_valid/reasons/train/test counts (5,);
        fold_mse/fold_r2/fold_r2_reason (5,K); pooled mse/r2 (K,) plus
        r2_reason (K,). Metric reasons are separate from fold-fit validity.
        MSE is squared response units; R2 is dimensionless and may be negative.
        cv_available/cv_reason describe complete five-fold scoring, independently
        of a valid reference or completed folds. Missing/failed folds leave NaN
        predictions and unavailable primary scores; completed products survive.
    """
    x, y = _validate_arrays(predictors, responses)
    _penalty_weights(penalty_weights, x.shape[1])
    if estimator not in ("ridge", "ols"):
        raise ValueError("estimator must be ridge or ols")
    folds = np.asarray(fold_ids)
    if (
        folds.shape != (len(x),)
        or not np.issubdtype(folds.dtype, np.integer)
        or not np.isin(folds, [-1, 0, 1, 2, 3, 4]).all()
    ):
        raise ValueError("fold_ids must have shape (rows,) and integer values -1..4")
    reference = _fit_selected(x, y, estimator, alpha, penalty_weights)
    reference_predictions = predict_responses(x, reference)
    if not np.isfinite(reference_predictions).all():
        raise ValueError("reference predictions must be finite")
    response_count = y.shape[1]
    result = {
        "reference_fit": reference,
        "reference_predictions": reference_predictions,
        "reference_scope": "all-eligible-data reference fit",
        "score_scope": "held-out regression on supplied fixed responses",
        "held_out_predictions": np.full_like(y, np.nan),
        "fold_ids": folds.copy(),
        "fold_fits": [None] * 5,
        "fold_slopes": np.full((5, x.shape[1], response_count), np.nan),
        "fold_intercepts": np.full((5, response_count), np.nan),
        "fold_valid": np.zeros(5, dtype=bool),
        "fold_reasons": [""] * 5,
        "fold_train_counts": np.zeros(5, dtype=int),
        "fold_test_counts": np.zeros(5, dtype=int),
        "fold_mse": np.full((5, response_count), np.nan),
        "fold_r2": np.full((5, response_count), np.nan),
        "fold_r2_reason": [
            ["Fold evaluation unavailable"] * response_count for _ in range(5)
        ],
        "mse": np.full(response_count, np.nan),
        "r2": np.full(response_count, np.nan),
        "r2_reason": tuple(
            "Incomplete five-fold evaluation" for _ in range(response_count)
        ),
        "cv_available": False,
        "cv_reason": "",
    }
    for fold in range(5):
        train = (folds >= 0) & (folds != fold)
        test = folds == fold
        result["fold_train_counts"][fold] = train.sum()
        result["fold_test_counts"][fold] = test.sum()
        if not train.any() or not test.any():
            result["fold_reasons"][fold] = (
                "Required fold has empty training or held-out rows"
            )
            continue
        try:
            fitted = _fit_selected(
                x[train], y[train], estimator, alpha, penalty_weights
            )
            result["fold_fits"][fold] = fitted
            result["fold_slopes"][fold] = fitted.slopes
            result["fold_intercepts"][fold] = fitted.intercept
            predicted = predict_responses(x[test], fitted)
            if predicted.shape != y[test].shape or not np.isfinite(predicted).all():
                raise ValueError(
                    "Held-out predictions must have the response shape and be finite"
                )
            result["held_out_predictions"][test] = predicted
            mse, r2, r2_reasons = _prediction_scores(y[test], predicted)
            result["fold_mse"][fold] = mse
            result["fold_r2"][fold] = r2
            result["fold_r2_reason"][fold] = list(r2_reasons)
            result["fold_valid"][fold] = True
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            result["fold_reasons"][fold] = str(error)
    if result["fold_valid"].all() and np.isfinite(result["held_out_predictions"]).all():
        result["mse"], result["r2"], result["r2_reason"] = _prediction_scores(
            y, result["held_out_predictions"]
        )
        result["cv_available"] = True
    else:
        result["cv_reason"] = (
            "Complete finite predictions from all five required folds are unavailable"
        )
    return result


def compute_reward_contrasts(
    fit: LinearFit, predictors: np.ndarray, column_names: tuple[str, ...]
) -> dict:
    """Compute raw reward-minus-omission values(2,R), support and estimability from one native design.

    Input fit has slopes(P,R), X(N,P) is its exact finite fit pool, and names(P,)
    identify native columns. Values retain response units; valid(2,) and
    reasons(2,) describe unpenalized-design estimability, independently of
    regularized numerical predictions. Also return centered design_rank,
    predictor_supported(P,) (nonconstant) and nonzero counts(P,). No inference,
    refitting, standardized-slope subtraction or input mutation occurs.
    """
    x = np.asarray(predictors, dtype=float)
    if (
        x.ndim != 2
        or not len(x)
        or not np.isfinite(x).all()
        or x.shape[1] != len(column_names)
        or fit.slopes.shape[0] != x.shape[1]
    ):
        raise ValueError("Contrast design/name/fit shapes must agree and be finite")
    centered = x - x.mean(0)
    supported = ~_constant_columns(x)
    centered[:, ~supported] = 0
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(float).eps * singular.max(initial=0)
    basis = vectors[singular > tolerance]
    values = np.full((2, fit.slopes.shape[1]), np.nan)
    valid = np.zeros(2, dtype=bool)
    reasons = []
    for index, side in enumerate(("left", "right")):
        reward, omission = f"{side}_reward", f"{side}_omission"
        if reward not in column_names or omission not in column_names:
            reasons.append("Required reward/omission columns are absent")
            continue
        contrast = np.zeros(x.shape[1])
        contrast[column_names.index(reward)] = 1
        contrast[column_names.index(omission)] = -1
        values[index] = contrast @ fit.slopes
        residual = contrast - (contrast @ basis.T) @ basis
        valid[index] = np.linalg.norm(residual) <= 100 * max(x.shape) * np.finfo(
            float
        ).eps * np.linalg.norm(contrast)
        reasons.append(
            ""
            if valid[index]
            else "Reward/omission contrast unsupported or aliased in this fit pool"
        )
    return {
        "values": values,
        "valid": valid,
        "reasons": np.asarray(reasons, dtype=str),
        "names": ("left_reward_minus_omission", "right_reward_minus_omission"),
        "design_rank": len(basis),
        "predictor_supported": supported,
        "predictor_nonzero_counts": np.count_nonzero(x, axis=0),
    }


# Shape, method-selection and scoring helpers.
def _validate_arrays(
    predictors: np.ndarray, responses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Input native X(N,P)/response Y(N,K); return finite float64 arrays unchanged in shape/units."""
    x, y = (
        np.asarray(predictors, dtype=np.float64),
        np.asarray(responses, dtype=np.float64),
    )
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("predictors and responses must be two-dimensional")
    if not x.size or not y.size or len(x) != len(y):
        raise ValueError("predictor and response rows must match and must not be empty")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("predictors and responses must contain only finite values")
    return x, y


def _penalty_weights(weights: np.ndarray | None, columns: int) -> np.ndarray:
    """Input optional dimensionless feature vector/column count; return positive finite (P,) weights."""
    result = np.ones(columns) if weights is None else np.asarray(weights, dtype=float)
    if (
        result.shape != (columns,)
        or not np.isfinite(result).all()
        or np.any(result <= 0)
    ):
        raise ValueError(
            "penalty_weights must be a finite positive feature vector with shape (P,)"
        )
    return result


def _coefficient_arrays(
    coefficients: np.ndarray,
    intercept: np.ndarray,
    predictor_count: int,
    response_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize sklearn (K,P) or single-target (P,) to (P,K), preserving response/native units."""
    raw = np.asarray(coefficients, dtype=np.float64)
    # Ridge squeezes its coefficient response axis when K=1, even for matrix Y.
    if response_count == 1 and raw.shape == (predictor_count,):
        raw = raw[None, :]
    slopes = raw.T.copy()
    constant = np.asarray(intercept, dtype=np.float64).copy()
    if slopes.shape != (predictor_count, response_count) or constant.shape != (
        response_count,
    ):
        raise ValueError("Estimator returned unexpected coefficient shape")
    if not np.isfinite(slopes).all() or not np.isfinite(constant).all():
        raise ValueError("Estimator coefficients and intercept must be finite")
    return slopes, constant


def _constant_columns(values: np.ndarray) -> np.ndarray:
    """Input nonempty finite (N,C); return Boolean (C,) exact-constant columns, preserving units."""
    return np.all(values == values[0], axis=0)


def _fit_limitations(x: np.ndarray, y: np.ndarray) -> tuple[str, ...]:
    """Input finite native X(N,P)/response Y(N,K); return descriptive constant-column reason tuple."""
    reasons = []
    if _constant_columns(x).any():
        reasons.append("Constant predictor columns or one observed context class")
    if _constant_columns(y).any():
        reasons.append(
            "Constant responses: R-squared and standardized coefficient display are unavailable"
        )
    return tuple(reasons)


def _fit_selected(
    x: np.ndarray,
    y: np.ndarray,
    estimator: str,
    alpha: float,
    penalty_weights: np.ndarray | None,
) -> LinearFit:
    """Input native X(N,P)/response Y(N,K)/scalar controls; return selected reference-or-training fit."""
    if estimator == "ols":
        return fit_ols_responses(x, y)
    return fit_ridge_responses(x, y, alpha, penalty_weights=penalty_weights)


def _prediction_scores(
    y: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Input finite response Y/predictions(N,K); return MSE(K),R2(K),reasons(K), squared-response/dimensionless units."""
    squared_errors = np.square(y - predicted)
    mse = squared_errors.mean(axis=0)
    sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    constant = _constant_columns(y)
    defined = ~constant & (sst > 0)
    r2 = np.full(y.shape[1], np.nan)
    r2[defined] = 1 - squared_errors[:, defined].sum(axis=0) / sst[defined]
    reasons = tuple("" if valid else "Zero response SST" for valid in defined)
    return mse, r2, reasons
