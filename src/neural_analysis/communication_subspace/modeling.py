"""Training transforms and ridge-regularized reduced-rank regression.

Numerical inputs have observation rows and neuron columns. Fitting uses only the
supplied training rows; callers own the choice-level train/test split.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrainingTransform:
    """Training means/scales in counts/bin and variable mask, each shape (neuron,)."""

    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    variable_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class RidgeDecomposition:
    """Dimensionless (source, target) ridge slope and ordered target basis.

    right_vectors has shape (basis, target) with descending singular-value
    order. max_rank is the supported coefficient rank based on the training
    source design and number of variable targets, even if X.T @ Y is zero.
    """

    ridge_coefficients: NDArray[np.float64]
    right_vectors: NDArray[np.float64]
    max_rank: int


def fit_training_transform(training_counts: NDArray[np.float64]) -> TrainingTransform:
    """Fit per-neuron means/SDs on (observation, neuron) training counts/bin.

    Return means, population SDs (ddof=0), and an exactly-nonzero-variance mask,
    each length neuron. Zero-variance columns remain represented in the record.
    """
    counts = _require_finite_matrix(training_counts, "training_counts")
    means = counts.mean(axis=0)
    scales = counts.std(axis=0, ddof=0)
    return TrainingTransform(means=means, scales=scales, variable_mask=scales != 0)


def transform_counts(
    counts: NDArray[np.float64], transform: TrainingTransform
) -> NDArray[np.float64]:
    """Apply training z-scores to (observation, neuron) counts/bin.

    Return dimensionless float64 (observation, variable neuron) columns in
    their original order. Constant training columns are omitted from the solve.
    """
    matrix = _require_finite_matrix(counts, "counts")
    if matrix.shape[1] != transform.means.size:
        raise ValueError("counts neuron axis does not match the training transform")
    variable = transform.variable_mask
    return (matrix[:, variable] - transform.means[variable]) / transform.scales[variable]


def inverse_transform_predictions(
    standardized_predictions: NDArray[np.float64], transform: TrainingTransform
) -> NDArray[np.float64]:
    """Restore (observation, variable target) z predictions to all counts/bin.

    Return float64 (observation, original target neuron) predictions. Each
    constant training target is predicted by its training mean.
    """
    predictions = np.asarray(standardized_predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != np.count_nonzero(transform.variable_mask):
        raise ValueError("prediction target axis does not match the training transform")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must contain only finite values")
    restored = np.broadcast_to(transform.means, (predictions.shape[0], transform.means.size)).copy()
    variable = transform.variable_mask
    restored[:, variable] = predictions * transform.scales[variable] + transform.means[variable]
    if not np.isfinite(restored).all():
        raise ValueError("restored predictions must contain only finite values")
    return restored


def fit_ridge_decomposition(
    source: NDArray[np.float64], target: NDArray[np.float64], alpha: float
) -> RidgeDecomposition:
    """Fit ridge and an ordered RRR basis from standardized training matrices.

    source is float64 (N, p), target is float64 (N, q), both dimensionless and
    training-centered; alpha is positive and multiplies ||B||_F^2 in the
    objective ||Y-XB||_F^2/N + alpha||B||_F^2. Return a dimensionless
    coefficient matrix (p, q), right singular vectors (min(p,q), q), and the
    supported rank min(numerical_rank(X), q). The intercept was removed by the
    training transform and is never penalized.
    """
    x_matrix = _require_finite_matrix(source, "source")
    y_matrix = _require_finite_matrix(target, "target")
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError("source and target must have the same observation count")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")

    sample_count, source_count = x_matrix.shape
    cross_covariance = x_matrix.T @ y_matrix
    regularized_gram = x_matrix.T @ x_matrix + sample_count * alpha * np.eye(source_count)
    lower_factor = np.linalg.cholesky(regularized_gram)
    whitened_cross_covariance = np.linalg.solve(lower_factor, cross_covariance)
    ridge_coefficients = np.linalg.solve(lower_factor.T, whitened_cross_covariance)
    _, _, right_vectors = np.linalg.svd(whitened_cross_covariance, full_matrices=False)

    singular_values = np.linalg.svd(x_matrix, compute_uv=False)
    tolerance = max(x_matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    source_rank = int(np.count_nonzero(singular_values > tolerance))
    max_rank = min(source_rank, y_matrix.shape[1])
    return RidgeDecomposition(ridge_coefficients, right_vectors, max_rank)


def predict_rank(
    source: NDArray[np.float64], decomposition: RidgeDecomposition, rank: int | None
) -> NDArray[np.float64]:
    """Predict dimensionless (observation, target) responses from source z-scores.

    source is float64 (observation, source neuron). rank=None uses unrestricted
    ridge; integer rank in 1..max_rank uses the penalized RRR projection and
    bounds the slope rank. Return float64 (observation, target neuron).
    """
    x_matrix = _require_finite_matrix(source, "source")
    coefficients = decomposition.ridge_coefficients
    if x_matrix.shape[1] != coefficients.shape[0]:
        raise ValueError("source neuron axis does not match fitted coefficients")
    if rank is None:
        predictions = x_matrix @ coefficients
    else:
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
            raise ValueError("rank must be an integer")
        if rank < 1 or rank > decomposition.max_rank:
            raise ValueError("rank is outside the supported range")
        target_basis = decomposition.right_vectors[:rank, :]
        predictions = (x_matrix @ coefficients @ target_basis.T) @ target_basis
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must contain only finite values")
    return predictions


def select_alpha(
    alphas: NDArray[np.float64], scores: NDArray[np.float64], tolerance: float = 1e-12
) -> float | None:
    """Choose a penalty from 1D alpha and inner-validation mean-R2 arrays.

    alphas are positive dimensionless penalties; scores are dimensionless R2,
    with nonfinite entries marking failed/undefined candidates. Return the
    largest alpha within absolute tolerance of the best finite score, or None
    when no candidate has a finite score. Neither input is modified.
    """
    penalties = np.asarray(alphas, dtype=np.float64)
    validation_scores = np.asarray(scores, dtype=np.float64)
    if penalties.ndim != 1 or validation_scores.shape != penalties.shape or not penalties.size:
        raise ValueError("alphas and scores must be nonempty matching 1D arrays")
    if not np.isfinite(penalties).all() or np.any(penalties <= 0):
        raise ValueError("alphas must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    finite = np.isfinite(validation_scores)
    if not finite.any():
        return None
    best_score = np.max(validation_scores[finite])
    tied = finite & (validation_scores >= best_score - tolerance)
    return float(np.max(penalties[tied]))


# Helper methods
def _require_finite_matrix(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return a float64 (observation, neuron) matrix with finite entries; raise otherwise."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix
