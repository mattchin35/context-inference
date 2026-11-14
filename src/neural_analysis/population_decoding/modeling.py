"""Weighted fixed-L2 classifiers with portable numeric prediction parameters."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ClassifierFit:
    """Affine logit parameters: coefficients(L,F), intercepts(L,), classes(C,).

    L=1 for binary class-1 log odds, L=C=4 for multinomial logits. Parameters
    map standardized-unit/PC-score units to dimensionless logits. Unavailable
    fits carry empty arrays plus an explicit reason, never invented classes.
    """

    coefficients: np.ndarray
    intercepts: np.ndarray
    classes: tuple[int, ...]
    available: bool
    reason: str
    iterations: int = 0


def fit_classifier(matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                   classes: tuple[int, ...]) -> ClassifierFit:
    """Fit finite X(M,F), integer y(M,), positive weights(M,) with fixed L2 C=1.

    classes is (0,1) or (0,1,2,3) in probability order. An unpenalized intercept,
    lbfgs solver, and supplied row weights are used without normalization/class
    weighting. Missing classes or numeric/convergence failures return unavailable
    ClassifierFit. Structural misuse raises ValueError; inputs are unchanged.
    """
    x, y, weight = np.asarray(matrix, float), np.asarray(labels), np.asarray(weights, float)
    if (classes not in ((0, 1), (0, 1, 2, 3)) or x.ndim != 2 or not all(x.shape)
            or y.shape != (len(x),) or weight.shape != y.shape or not np.isfinite(x).all()
            or not np.isin(y, classes).all() or not np.isfinite(weight).all() or (weight <= 0).any()):
        raise ValueError("Expected finite X(M,F), class labels(M,), positive weights(M,)")
    empty = ClassifierFit(np.empty((0, x.shape[1])), np.empty(0), classes, False, "training_class_missing")
    if set(np.unique(y)) != set(classes):
        return empty
    estimator = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
                                   class_weight=None, tol=1e-4, max_iter=5000)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x, y, sample_weight=weight)
    except (ConvergenceWarning, np.linalg.LinAlgError, FloatingPointError) as error:
        return ClassifierFit(empty.coefficients, empty.intercepts, classes, False,
                             f"numeric_or_convergence_failure: {error}")
    beta, intercept = np.asarray(estimator.coef_), np.asarray(estimator.intercept_)
    if not np.isfinite(beta).all() or not np.isfinite(intercept).all():
        return ClassifierFit(empty.coefficients, empty.intercepts, classes, False, "nonfinite_parameters")
    return ClassifierFit(beta, intercept, classes, True, "", int(estimator.n_iter_.max()))


def predict_probabilities(matrix: np.ndarray, fitted: ClassifierFit) -> tuple[np.ndarray, np.ndarray]:
    """Predict from finite X(M,F); return ordered probability/log-probability (M,C).

    Use stable binary log-sigmoids or joint multinomial log-softmax. No model
    fitting or normalization across observations occurs. Unavailable fits raise
    ValueError; input features and stored parameters are unchanged.
    """
    x = np.asarray(matrix, float)
    if not fitted.available or x.ndim != 2 or x.shape[1] != fitted.coefficients.shape[1] or not np.isfinite(x).all():
        raise ValueError("Prediction requires available fit and matching finite X(M,F)")
    logits = x @ fitted.coefficients.T + fitted.intercepts
    if len(fitted.classes) == 2:
        z = logits[:, 0]
        logp = np.column_stack([-np.logaddexp(0, z), -np.logaddexp(0, -z)])
    else:
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logp), logp
