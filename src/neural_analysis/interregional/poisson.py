"""Target-wise unpenalized Poisson fitting, scoring, and MSE comparison."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from .configuration import N_CV_FOLDS
from .linear import METRIC_ATOL, METRIC_RTOL


_KNOWN_FIT_ERRORS = (
    PerfectSeparationError,
    FloatingPointError,
    np.linalg.LinAlgError,
    ValueError,
)
_PAIR_KEYS = (
    "session_id",
    "direction",
    "condition",
    "window",
    "target_id",
    "fold_id",
    "evaluation_scope",
)


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
    """Return an owned read-only float64 array with its input shape unchanged."""
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_count_vector(counts: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and return one float64 count vector with shape ``(observation,)``."""
    values = np.asarray(counts, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.all(values == np.floor(values))
    ):
        raise ValueError(f"{name} must be finite nonnegative integer counts.")
    return values


def _validate_expected_counts(
    expected_counts: np.ndarray, *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    """Validate positive finite expected counts with one required observation shape."""
    values = np.asarray(expected_counts, dtype=np.float64)
    if values.shape != shape or values.ndim != 1:
        raise ValueError(f"{name} must have the same one-dimensional shape as observed.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise PoissonFitUnavailable(
            "nonpositive_poisson_mean",
            f"{name} must contain only positive finite expected counts.",
        )
    return values


@dataclass(frozen=True)
class PoissonFit:
    """One target-wise unpenalized Poisson GLM fit.

    ``parameters`` has shape ``(coefficient,)`` on the log-count scale. Rank,
    feature count, residual degrees of freedom, convergence, and IRLS iteration
    count are unitless fit diagnostics.
    """

    parameters: np.ndarray
    rank: int
    feature_count: int
    df_resid: int
    converged: bool
    iterations: int


@dataclass(frozen=True)
class PoissonHeldOutScores:
    """Held-out metrics for one count target and one restricted/full pair.

    Deviances and deviance-explained values are dimensionless. MSE values are
    in squared counts per bin. Normalized values are NaN when the held-out null
    deviance is zero.
    """

    deviance_restricted: float
    deviance_full: float
    null_deviance: float
    deviance_explained_restricted: float
    deviance_explained_full: float
    delta_deviance_explained: float
    mse_restricted: float
    mse_full: float
    normalized_available: bool


class PoissonFitUnavailable(RuntimeError):
    """A recognized target-local Poisson failure with a stable reason code."""

    def __init__(self, reason: str, detail: str) -> None:
        """Store a machine-readable reason and human-readable failure detail."""
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def fit_poisson_target(design: np.ndarray, count_response: np.ndarray) -> PoissonFit:
    """Fit one strict unpenalized Poisson GLM with a log link.

    Parameters
    ----------
    design : numpy.ndarray
        Finite matrix with shape ``(observation, coefficient)``. The first
        column must be an explicit intercept of ones; other columns retain
        caller-defined predictor units.
    count_response : numpy.ndarray
        Vector with shape ``(observation,)`` containing nonnegative integer
        spike counts per bin. Counts are neither scaled nor converted to rates.

    Returns
    -------
    PoissonFit
        Log-count coefficients with shape ``(coefficient,)`` and minimal
        unpenalized-fit diagnostics.

    Raises
    ------
    PoissonFitUnavailable
        If this otherwise valid target is constant, fails in a recognized
        numerical way, warns about convergence/separation, or does not converge.
    """
    x = np.asarray(design, dtype=np.float64)
    y = _validate_count_vector(count_response, name="count_response")
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] == 0:
        raise ValueError("design must be a nonempty matrix with one row per count.")
    if not np.all(np.isfinite(x)):
        raise ValueError("design must contain only finite values.")
    if not np.array_equal(x[:, 0], np.ones(x.shape[0], dtype=np.float64)):
        raise ValueError("design must contain an explicit intercept in its first column.")

    feature_count = int(x.shape[1])
    rank = int(np.linalg.matrix_rank(x))
    if rank != feature_count:
        raise ValueError(
            f"Poisson design is rank deficient: rank {rank}, features {feature_count}."
        )
    df_resid = int(x.shape[0] - feature_count)
    if df_resid <= 0:
        raise ValueError("Poisson design must have positive residual degrees of freedom.")
    if np.var(y, ddof=0) == 0.0:
        raise PoissonFitUnavailable(
            "constant_training_target",
            "Poisson count_response must vary in the training rows.",
        )

    model = sm.GLM(
        y,
        x,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        missing="raise",
    )
    try:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", PerfectSeparationWarning)
            result = model.fit(
                method="IRLS",
                maxiter=100,
                tol=1e-8,
                scale=None,
                cov_type="nonrobust",
                full_output=True,
                disp=False,
            )
    except _KNOWN_FIT_ERRORS as error:
        raise PoissonFitUnavailable(
            "poisson_fit_error",
            f"Poisson IRLS failed with {type(error).__name__}: {error}",
        ) from error

    warning_types = {type(item.message) for item in fit_warnings}
    if any(issubclass(kind, PerfectSeparationWarning) for kind in warning_types):
        raise PoissonFitUnavailable(
            "poisson_fit_error", "Poisson IRLS emitted a perfect-separation warning."
        )
    if any(issubclass(kind, ConvergenceWarning) for kind in warning_types):
        raise PoissonFitUnavailable(
            "poisson_nonconverged", "Poisson IRLS emitted a convergence warning."
        )

    converged_value = result.converged
    if not isinstance(converged_value, (bool, np.bool_)):
        raise RuntimeError("statsmodels returned a non-Boolean convergence diagnostic.")
    if not bool(converged_value):
        raise PoissonFitUnavailable(
            "poisson_nonconverged", "Poisson IRLS did not converge."
        )

    iteration_value = result.fit_history["iteration"]
    if not isinstance(iteration_value, (int, np.integer)) or int(iteration_value) < 0:
        raise RuntimeError("statsmodels returned an invalid IRLS iteration count.")
    parameters = np.asarray(result.params, dtype=np.float64)
    if parameters.shape != (feature_count,) or not np.all(np.isfinite(parameters)):
        raise PoissonFitUnavailable(
            "poisson_fit_error", "Poisson IRLS returned invalid or nonfinite parameters."
        )
    return PoissonFit(
        parameters=_readonly_float_array(parameters),
        rank=rank,
        feature_count=feature_count,
        df_resid=df_resid,
        converged=True,
        iterations=int(iteration_value),
    )


def predict_poisson_mean(design: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Predict positive expected counts for one fitted Poisson target.

    ``design`` has shape ``(observation, coefficient)`` and ``parameters`` has
    shape ``(coefficient,)`` on the log-count scale. The returned float64 vector
    has shape ``(observation,)`` in expected counts per bin.
    """
    x = np.asarray(design, dtype=np.float64)
    beta = np.asarray(parameters, dtype=np.float64)
    if x.ndim != 2 or beta.ndim != 1 or x.shape[1] != beta.size:
        raise ValueError("design and parameters have incompatible shapes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(beta)):
        raise ValueError("design and parameters must contain only finite values.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected_counts = np.exp(x @ beta)
    if not np.all(np.isfinite(expected_counts)) or np.any(expected_counts <= 0.0):
        raise PoissonFitUnavailable(
            "nonpositive_poisson_mean",
            "Poisson prediction returned a nonpositive or nonfinite expected count.",
        )
    return _readonly_float_array(expected_counts)


def poisson_deviance(observed_counts: np.ndarray, expected_counts: np.ndarray) -> float:
    """Calculate conventional Poisson deviance for one held-out target.

    Both inputs have shape ``(observation,)`` and units of counts per bin.
    ``observed_counts`` must contain finite nonnegative integers and
    ``expected_counts`` must be positive and finite. The returned deviance is a
    dimensionless scalar; zero-count logarithmic terms use their zero limit.
    """
    observed = _validate_count_vector(observed_counts, name="observed_counts")
    expected = _validate_expected_counts(
        expected_counts, shape=observed.shape, name="expected_counts"
    )
    positive = observed > 0.0
    terms = np.array(expected, copy=True)
    terms[positive] = (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - (observed[positive] - expected[positive])
    )
    deviance = float(2.0 * np.sum(terms))
    if not np.isfinite(deviance):
        raise ValueError("Poisson deviance must be finite.")
    return deviance


def score_poisson_predictions(
    observed_counts: np.ndarray,
    restricted_expected_counts: np.ndarray,
    full_expected_counts: np.ndarray,
) -> PoissonHeldOutScores:
    """Score one restricted/full Poisson pair on identical held-out rows.

    All inputs have shape ``(observation,)`` in counts per bin. Restricted and
    full raw deviance share the null prediction formed from the held-out count
    mean. Returned MSE values have squared-count units; other metrics are
    dimensionless.
    """
    observed = _validate_count_vector(observed_counts, name="observed_counts")
    restricted = _validate_expected_counts(
        restricted_expected_counts,
        shape=observed.shape,
        name="restricted_expected_counts",
    )
    full = _validate_expected_counts(
        full_expected_counts, shape=observed.shape, name="full_expected_counts"
    )
    restricted_deviance = poisson_deviance(observed, restricted)
    full_deviance = poisson_deviance(observed, full)
    null_mean = float(np.mean(observed))
    if np.isclose(null_mean, 0.0, rtol=METRIC_RTOL, atol=METRIC_ATOL):
        null_deviance = 0.0
    else:
        null_deviance = poisson_deviance(
            observed, np.full(observed.shape, null_mean, dtype=np.float64)
        )
    normalized_available = bool(
        np.isfinite(null_deviance)
        and not np.isclose(
            null_deviance, 0.0, rtol=METRIC_RTOL, atol=METRIC_ATOL
        )
    )
    if normalized_available:
        restricted_explained = 1.0 - restricted_deviance / null_deviance
        full_explained = 1.0 - full_deviance / null_deviance
        incremental_explained = full_explained - restricted_explained
    else:
        restricted_explained = np.nan
        full_explained = np.nan
        incremental_explained = np.nan
    return PoissonHeldOutScores(
        deviance_restricted=restricted_deviance,
        deviance_full=full_deviance,
        null_deviance=null_deviance,
        deviance_explained_restricted=float(restricted_explained),
        deviance_explained_full=float(full_explained),
        delta_deviance_explained=float(incremental_explained),
        mse_restricted=float(np.mean((observed - restricted) ** 2)),
        mse_full=float(np.mean((observed - full) ** 2)),
        normalized_available=normalized_available,
    )


def derive_mse_comparison(
    fold_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive exact paired OLS/Poisson count-MSE views from saved fold rows.

    Parameters
    ----------
    fold_scores : pandas.DataFrame
        Fold rows containing OLS and Poisson direct-unit fits. Identity columns
        use scalar strings and integer fold IDs; MSE columns are squared counts
        per bin; row-set columns are canonical SHA-256 strings.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The first table contains defined fold pairs for both ``restricted`` and
        ``full`` comparisons. The second contains one five-fold target summary
        per comparison. Positive advantages mean lower Poisson MSE. These are
        derived display views and are not persistence-schema tables.
    """
    required = {
        *_PAIR_KEYS,
        "representation",
        "model_family",
        "restricted_status",
        "full_status",
        "train_row_set_sha256",
        "test_row_set_sha256",
        "mse_restricted",
        "mse_full",
    }
    missing = required - set(fold_scores.columns)
    if missing:
        raise ValueError(f"fold_scores is missing MSE comparison columns: {sorted(missing)}")

    poisson_rows = fold_scores.loc[fold_scores["model_family"].eq("poisson")]
    if not poisson_rows["representation"].eq("units").all():
        raise ValueError("Poisson MSE comparison accepts unit targets only.")
    ols_rows = fold_scores.loc[
        fold_scores["model_family"].eq("ols")
        & fold_scores["representation"].eq("units")
    ]
    if ols_rows.duplicated(list(_PAIR_KEYS)).any() or poisson_rows.duplicated(
        list(_PAIR_KEYS)
    ).any():
        raise ValueError("Each model family must have at most one row per comparison key.")

    paired = ols_rows.merge(
        poisson_rows,
        on=list(_PAIR_KEYS),
        how="inner",
        suffixes=("_ols", "_poisson"),
        validate="one_to_one",
    )
    for fingerprint in ("train_row_set_sha256", "test_row_set_sha256"):
        if not paired[f"{fingerprint}_ols"].eq(
            paired[f"{fingerprint}_poisson"]
        ).all():
            raise ValueError(
                "OLS and Poisson row-set fingerprints must match for every paired fold."
            )

    fold_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    target_keys = [key for key in _PAIR_KEYS if key != "fold_id"]
    for comparison in ("restricted", "full"):
        ols_mse = pd.to_numeric(paired[f"mse_{comparison}_ols"], errors="coerce")
        poisson_mse = pd.to_numeric(
            paired[f"mse_{comparison}_poisson"], errors="coerce"
        )
        valid = (
            paired[f"{comparison}_status_ols"].eq("ok")
            & paired[f"{comparison}_status_poisson"].eq("ok")
            & np.isfinite(ols_mse)
            & np.isfinite(poisson_mse)
        )
        candidate = paired.assign(
            _valid=valid,
            _mse_ols=ols_mse,
            _mse_poisson=poisson_mse,
        )
        for _, row in candidate.loc[candidate["_valid"]].iterrows():
            fold_rows.append(
                {
                    **{key: row[key] for key in _PAIR_KEYS},
                    "comparison": comparison,
                    "train_row_set_sha256": row["train_row_set_sha256_ols"],
                    "test_row_set_sha256": row["test_row_set_sha256_ols"],
                    "mse_ols": float(row["_mse_ols"]),
                    "mse_poisson": float(row["_mse_poisson"]),
                    "mse_advantage_poisson": float(
                        row["_mse_ols"] - row["_mse_poisson"]
                    ),
                }
            )
        grouped = candidate.groupby(target_keys, sort=True, dropna=False)
        for identity, group in grouped:
            valid_group = group.loc[group["_valid"]]
            complete = (
                len(group) == N_CV_FOLDS
                and set(group["fold_id"].astype(int)) == set(range(N_CV_FOLDS))
                and len(valid_group) == N_CV_FOLDS
            )
            advantages = valid_group["_mse_ols"] - valid_group["_mse_poisson"]
            target_rows.append(
                {
                    **dict(zip(target_keys, identity, strict=True)),
                    "comparison": comparison,
                    "status": "ok" if complete else "incomplete_folds",
                    "reason": "" if complete else "incomplete_requested_folds",
                    "requested_folds": N_CV_FOLDS,
                    "valid_folds": int(len(valid_group)),
                    "mean_mse_advantage_poisson": (
                        float(advantages.mean()) if complete else None
                    ),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(target_rows)
