"""Explicit unpenalized OLS fitting, held-out scoring, and CV summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .configuration import N_CV_FOLDS
from .records import result_table_from_rows


METRIC_RTOL = 1e-9
METRIC_ATOL = 1e-12
_CV_METRICS = (
    "r2_restricted",
    "r2_full",
    "delta_r2",
    "deviance_explained_restricted",
    "deviance_explained_full",
    "delta_deviance_explained",
    "mse_restricted",
    "mse_full",
)


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
    """Return an owned read-only float64 array."""
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool_array(value: np.ndarray) -> np.ndarray:
    """Return an owned read-only Boolean array."""
    result = np.array(value, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OLSFit:
    """One multi-target ordinary least-squares fit.

    ``coefficients`` has shape ``(feature, target)``. Rank, feature count, and
    residual degrees of freedom are unitless design diagnostics.
    """

    coefficients: np.ndarray
    rank: int
    feature_count: int
    df_resid: int
    constant_targets: np.ndarray


@dataclass(frozen=True)
class OLSHeldOutScores:
    """Per-target held-out OLS metrics.

    Every numeric field has shape ``(target,)``. R-squared fields are NaN when
    held-out target variance is zero; MSE remains defined in squared response
    units.
    """

    r2_restricted: np.ndarray
    r2_full: np.ndarray
    delta_r2: np.ndarray
    mse_restricted: np.ndarray
    mse_full: np.ndarray
    r2_available: np.ndarray


def add_intercept(features: np.ndarray) -> np.ndarray:
    """Prepend exactly one explicit intercept to a feature matrix.

    Parameters
    ----------
    features : numpy.ndarray
        Finite matrix with shape ``(observation, feature)`` in caller-defined
        feature units. It must not already contain an intercept column.

    Returns
    -------
    numpy.ndarray
        Float64 design with shape ``(observation, feature + 1)`` whose first
        column is one.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a finite two-dimensional array.")
    return np.column_stack((np.ones(values.shape[0], dtype=np.float64), values))


def fit_ols_targets(design: np.ndarray, responses: np.ndarray) -> OLSFit:
    """Fit full-rank unpenalized OLS for targets sharing one design.

    Parameters
    ----------
    design : numpy.ndarray
        Finite float-like matrix with shape ``(observation, coefficient)`` and
        one caller-supplied explicit intercept column.
    responses : numpy.ndarray
        Finite matrix with shape ``(observation, target)`` in response units.

    Returns
    -------
    OLSFit
        Coefficients and shared design diagnostics. ``constant_targets`` marks
        zero-variance training responses without invalidating other targets.
    """
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("design and responses must be two-dimensional with matching rows.")
    if x.shape[0] == 0 or x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("design and responses must have nonempty axes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("design and responses must contain only finite values.")
    rank = int(np.linalg.matrix_rank(x))
    feature_count = int(x.shape[1])
    if rank != feature_count:
        raise ValueError(
            f"OLS design is rank deficient: rank {rank}, features {feature_count}."
        )
    df_resid = int(x.shape[0] - feature_count)
    if df_resid <= 0:
        raise ValueError("OLS design must have positive residual degrees of freedom.")
    coefficients, _, solved_rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if int(solved_rank) != rank or not np.all(np.isfinite(coefficients)):
        raise ValueError("OLS solve returned inconsistent rank or nonfinite coefficients.")
    constant_targets = np.var(y, axis=0, ddof=0) == 0.0
    return OLSFit(
        coefficients=_readonly_float_array(coefficients),
        rank=rank,
        feature_count=feature_count,
        df_resid=df_resid,
        constant_targets=_readonly_bool_array(constant_targets),
    )


def predict_ols_targets(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict target responses from one explicit OLS coefficient matrix.

    ``design`` has shape ``(observation, coefficient)`` and ``coefficients``
    has shape ``(coefficient, target)``. The returned float64 matrix has shape
    ``(observation, target)`` in response units.
    """
    x = np.asarray(design, dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    if x.ndim != 2 or beta.ndim != 2 or x.shape[1] != beta.shape[0]:
        raise ValueError("design and coefficients have incompatible two-dimensional shapes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(beta)):
        raise ValueError("design and coefficients must contain only finite values.")
    predictions = x @ beta
    if not np.all(np.isfinite(predictions)):
        raise ValueError("OLS predictions must be finite.")
    return predictions


def score_ols_predictions(
    observed: np.ndarray,
    restricted_predictions: np.ndarray,
    full_predictions: np.ndarray,
) -> OLSHeldOutScores:
    """Calculate per-target held-out absolute and incremental OLS metrics.

    All inputs have shape ``(observation, target)`` and share response units.
    MSE is returned in squared response units. R-squared and its increment are
    dimensionless and remain negative when predictions underperform the mean.
    """
    y = np.asarray(observed, dtype=np.float64)
    restricted = np.asarray(restricted_predictions, dtype=np.float64)
    full = np.asarray(full_predictions, dtype=np.float64)
    if y.ndim != 2 or restricted.shape != y.shape or full.shape != y.shape:
        raise ValueError("Observed and predicted arrays must share (observation, target) shape.")
    if y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("Held-out score arrays must have nonempty axes.")
    if not all(np.all(np.isfinite(value)) for value in (y, restricted, full)):
        raise ValueError("Observed and predicted arrays must be finite.")
    restricted_error = y - restricted
    full_error = y - full
    restricted_sse = np.sum(restricted_error**2, axis=0)
    full_sse = np.sum(full_error**2, axis=0)
    sst = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    r2_available = ~np.isclose(sst, 0.0, rtol=METRIC_RTOL, atol=METRIC_ATOL)
    r2_restricted = np.full(y.shape[1], np.nan, dtype=np.float64)
    r2_full = np.full(y.shape[1], np.nan, dtype=np.float64)
    r2_restricted[r2_available] = 1.0 - restricted_sse[r2_available] / sst[r2_available]
    r2_full[r2_available] = 1.0 - full_sse[r2_available] / sst[r2_available]
    return OLSHeldOutScores(
        r2_restricted=_readonly_float_array(r2_restricted),
        r2_full=_readonly_float_array(r2_full),
        delta_r2=_readonly_float_array(r2_full - r2_restricted),
        mse_restricted=_readonly_float_array(np.mean(restricted_error**2, axis=0)),
        mse_full=_readonly_float_array(np.mean(full_error**2, axis=0)),
        r2_available=_readonly_bool_array(r2_available),
    )


def summarize_complete_cv_targets(
    fold_scores: pd.DataFrame,
    metric_names: Sequence[str] = (
        "r2_restricted",
        "r2_full",
        "delta_r2",
        "mse_restricted",
        "mse_full",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate complete five-fold target means, then population quartiles.

    Parameters
    ----------
    fold_scores : pandas.DataFrame
        One row per target/fold with fold-score identity columns, ``status`` and
        ``reason``, and requested metric columns. Metrics are dimensionless or
        squared response units according to their names.
    metric_names : sequence of str
        OLS or Poisson CV metrics to summarize independently. Raw Poisson
        deviance is intentionally excluded.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Exact typed ``target_summaries`` and ``population_summaries`` tables.
        Population quartiles use complete target means, never pooled folds.
    """
    requested_metrics = tuple(metric_names)
    if not requested_metrics or any(metric not in _CV_METRICS for metric in requested_metrics):
        raise ValueError(f"metric_names must be selected from {_CV_METRICS}.")
    required = {
        "session_id",
        "direction",
        "representation",
        "model_family",
        "condition",
        "window",
        "target_id",
        "fold_id",
        "evaluation_scope",
        "status",
        "reason",
        *requested_metrics,
    }
    missing = required - set(fold_scores.columns)
    if missing:
        raise ValueError(f"fold_scores is missing summary columns: {sorted(missing)}")
    target_group_columns = [
        "session_id",
        "evaluation_scope",
        "direction",
        "representation",
        "model_family",
        "condition",
        "window",
        "target_id",
    ]
    target_rows: list[dict[str, object]] = []
    grouped = fold_scores.groupby(target_group_columns, sort=True, dropna=False)
    for identity, group in grouped:
        target_rank = group["target_rank"].iloc[0] if "target_rank" in group else None
        for metric_name in requested_metrics:
            values = pd.to_numeric(group[metric_name], errors="coerce")
            if metric_name.endswith("_restricted") and "restricted_status" in group:
                fit_valid = group["restricted_status"].eq("ok")
            elif metric_name.endswith("_full") and "full_status" in group:
                fit_valid = group["full_status"].eq("ok")
            elif metric_name in {"delta_r2", "delta_deviance_explained"} and {
                "restricted_status",
                "full_status",
            } <= set(group.columns):
                fit_valid = group["restricted_status"].eq("ok") & group[
                    "full_status"
                ].eq("ok")
            else:
                fit_valid = group["status"].eq("ok")
            valid = fit_valid & np.isfinite(values)
            complete = (
                len(group) == N_CV_FOLDS
                and set(group["fold_id"].astype(int)) == set(range(N_CV_FOLDS))
                and int(valid.sum()) == N_CV_FOLDS
            )
            target_rows.append(
                {
                    **dict(zip(target_group_columns, identity, strict=True)),
                    "metric_name": metric_name,
                    "target_rank": target_rank,
                    "status": "ok" if complete else "incomplete_folds",
                    "reason": "" if complete else "incomplete_requested_folds",
                    "requested_folds": N_CV_FOLDS,
                    "valid_folds": int(valid.sum()),
                    "mean_value": float(values.mean()) if complete else None,
                }
            )
    target_summaries = result_table_from_rows("target_summaries", target_rows)

    population_group_columns = [
        "session_id",
        "evaluation_scope",
        "direction",
        "representation",
        "model_family",
        "condition",
        "window",
        "metric_name",
    ]
    population_rows: list[dict[str, object]] = []
    population_groups = target_summaries.groupby(
        population_group_columns, sort=True, dropna=False
    )
    for identity, group in population_groups:
        complete_values = group.loc[group["status"].eq("ok"), "mean_value"].to_numpy(
            dtype=float
        )
        if complete_values.size:
            quartiles = np.quantile(
                complete_values, [0.25, 0.5, 0.75], method="linear"
            )
            status, reason = "ok", ""
            q25, median, q75 = (float(value) for value in quartiles)
        else:
            status, reason = "not_applicable", "no_complete_targets"
            q25 = median = q75 = None
        population_rows.append(
            {
                **dict(zip(population_group_columns, identity, strict=True)),
                "status": status,
                "reason": reason,
                "n_targets": int(complete_values.size),
                "q25": q25,
                "median": median,
                "q75": q75,
            }
        )
    population_summaries = result_table_from_rows(
        "population_summaries", population_rows
    )
    return target_summaries, population_summaries
