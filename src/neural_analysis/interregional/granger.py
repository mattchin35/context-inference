"""Descriptive in-sample linear and Poisson Granger-style scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from .linear import METRIC_ATOL, METRIC_RTOL
from .poisson import poisson_deviance


@dataclass(frozen=True)
class LinearGrangerScores:
    """Linear nested-fit diagnostics for one target on identical rows.

    Residual sums of squares retain squared response units. ``linear_granger``
    is the dimensionless natural logarithm of their restricted/full ratio.
    """

    sse_restricted: float
    sse_full: float
    linear_granger: float
    diagnostic: str


@dataclass(frozen=True)
class PoissonGrangerScores:
    """Poisson nested-fit diagnostics for one count target on identical rows.

    Log likelihoods, deviances, and likelihood ratio are dimensionless.
    ``mean_deviance_improvement`` is the likelihood ratio per fitted row.
    """

    llf_restricted: float
    llf_full: float
    deviance_restricted: float
    deviance_full: float
    likelihood_ratio: float
    mean_deviance_improvement: float
    diagnostic: str


class GrangerUnavailable(RuntimeError):
    """A recognized target-local Granger failure with a frozen reason code."""

    def __init__(self, reason: str, detail: str) -> None:
        """Store one machine-readable reason and explanatory detail."""
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def validate_nested_fit_improvement(
    restricted_value: float,
    full_value: float,
    *,
    larger_is_better: bool,
) -> tuple[float, str]:
    """Validate one theoretically nonnegative nested-model improvement.

    Parameters
    ----------
    restricted_value, full_value : float
        Finite scalar fit criteria on identical observation rows and in the
        same units. For sums/deviances, lower is better; for log likelihood,
        higher is better.
    larger_is_better : bool
        ``True`` for log likelihood and ``False`` for sums/deviances.

    Returns
    -------
    tuple[float, str]
        Nonnegative improvement in the criterion's units and either an empty
        diagnostic or ``"nested_roundoff"`` when a tolerated negative value
        was clamped to zero.

    Raises
    ------
    GrangerUnavailable
        If either criterion is nonfinite or the full model is substantively
        worse than its nested restricted model.
    """
    restricted = float(restricted_value)
    full = float(full_value)
    if not np.isfinite(restricted) or not np.isfinite(full):
        raise GrangerUnavailable(
            "nested_fit_inconsistency", "Nested-fit criteria must be finite."
        )
    improvement = full - restricted if larger_is_better else restricted - full
    tolerance = METRIC_ATOL + METRIC_RTOL * max(
        abs(restricted), abs(full), 1.0
    )
    if improvement < -tolerance:
        raise GrangerUnavailable(
            "nested_fit_inconsistency",
            "The full in-sample model is substantively worse than the restricted model.",
        )
    if improvement < 0.0:
        return 0.0, "nested_roundoff"
    return float(improvement), ""


def _matching_finite_vectors(
    observed: np.ndarray,
    restricted: np.ndarray,
    full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three finite float64 vectors sharing one nonempty row axis."""
    values = tuple(
        np.asarray(array, dtype=np.float64)
        for array in (observed, restricted, full)
    )
    if (
        any(array.ndim != 1 for array in values)
        or values[0].size == 0
        or values[1].shape != values[0].shape
        or values[2].shape != values[0].shape
    ):
        raise ValueError("Granger inputs must be nonempty vectors with identical shapes.")
    if not all(np.all(np.isfinite(array)) for array in values):
        raise ValueError("Granger inputs must contain only finite values.")
    return values


def compute_linear_granger(
    observed: np.ndarray,
    restricted_predictions: np.ndarray,
    full_predictions: np.ndarray,
) -> LinearGrangerScores:
    """Calculate one in-sample linear Granger magnitude on matched rows.

    All inputs have shape ``(observation,)`` in the same caller-defined
    response units. The returned SSE values have squared response units and
    the log ratio is dimensionless.
    """
    response, restricted, full = _matching_finite_vectors(
        observed, restricted_predictions, full_predictions
    )
    sse_restricted = float(np.sum((response - restricted) ** 2))
    sse_full = float(np.sum((response - full) ** 2))
    if np.isclose(sse_restricted, 0.0, rtol=METRIC_RTOL, atol=METRIC_ATOL) or np.isclose(
        sse_full, 0.0, rtol=METRIC_RTOL, atol=METRIC_ATOL
    ):
        raise GrangerUnavailable(
            "zero_granger_residual",
            "Linear Granger magnitude requires positive restricted and full residual sums.",
        )
    _, diagnostic = validate_nested_fit_improvement(
        sse_restricted, sse_full, larger_is_better=False
    )
    if diagnostic:
        magnitude = 0.0
    else:
        magnitude = float(np.log(sse_restricted / sse_full))
    return LinearGrangerScores(
        sse_restricted=sse_restricted,
        sse_full=sse_full,
        linear_granger=magnitude,
        diagnostic=diagnostic,
    )


def _poisson_log_likelihood(observed: np.ndarray, expected: np.ndarray) -> float:
    """Return summed Poisson log likelihood for count/mean vectors."""
    positive = observed > 0.0
    terms = -expected - gammaln(observed + 1.0)
    terms[positive] += observed[positive] * np.log(expected[positive])
    value = float(np.sum(terms))
    if not np.isfinite(value):
        raise ValueError("Poisson log likelihood must be finite.")
    return value


def compute_poisson_granger(
    observed_counts: np.ndarray,
    restricted_expected_counts: np.ndarray,
    full_expected_counts: np.ndarray,
) -> PoissonGrangerScores:
    """Calculate one in-sample Poisson Granger-style magnitude.

    Inputs have shape ``(observation,)`` in counts per bin. Observations must
    be nonnegative integers and expected counts must be positive. Returned log
    likelihoods, deviances, LR, and mean deviance improvement are dimensionless.
    """
    observed, restricted, full = _matching_finite_vectors(
        observed_counts, restricted_expected_counts, full_expected_counts
    )
    if np.any(observed < 0.0) or not np.all(observed == np.floor(observed)):
        raise ValueError("observed_counts must contain nonnegative integer counts.")
    if np.any(restricted <= 0.0) or np.any(full <= 0.0):
        raise ValueError("Poisson expected counts must be positive.")
    llf_restricted = _poisson_log_likelihood(observed, restricted)
    llf_full = _poisson_log_likelihood(observed, full)
    deviance_restricted = poisson_deviance(observed, restricted)
    deviance_full = poisson_deviance(observed, full)
    llf_gain, llf_diagnostic = validate_nested_fit_improvement(
        llf_restricted, llf_full, larger_is_better=True
    )
    deviance_gain, deviance_diagnostic = validate_nested_fit_improvement(
        deviance_restricted, deviance_full, larger_is_better=False
    )
    likelihood_ratio = 2.0 * llf_gain
    tolerance = METRIC_ATOL + METRIC_RTOL * max(
        abs(likelihood_ratio), abs(deviance_gain), 1.0
    )
    if not np.isclose(
        likelihood_ratio, deviance_gain, rtol=METRIC_RTOL, atol=tolerance
    ):
        raise GrangerUnavailable(
            "nested_fit_inconsistency",
            "Poisson likelihood-ratio and deviance improvements disagree.",
        )
    diagnostic = llf_diagnostic or deviance_diagnostic
    if diagnostic:
        likelihood_ratio = 0.0
    return PoissonGrangerScores(
        llf_restricted=llf_restricted,
        llf_full=llf_full,
        deviance_restricted=deviance_restricted,
        deviance_full=deviance_full,
        likelihood_ratio=float(likelihood_ratio),
        mean_deviance_improvement=float(likelihood_ratio / observed.size),
        diagnostic=diagnostic,
    )
