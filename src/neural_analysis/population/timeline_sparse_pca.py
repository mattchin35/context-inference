"""Up-to-30-component SparsePCA for timeline counts; no source loading or plotting."""

from numbers import Real
from time import perf_counter
import warnings

import numpy as np
from sklearn.decomposition import SparsePCA

from src.neural_analysis.interregional.pca import standardize_unit_activity
from src.neural_analysis.population.session_timeline import (
    RegionalTimelinePCA,
    SparseTimelineTransform,
    choice_bin_edges,
)

COMPONENT_LIMIT = 30


def sparse_pca_parameters(alpha: float = 1.0) -> dict:
    """Return explicit sklearn settings except the data-dependent component count.

    alpha is a finite nonnegative real sparsity penalty (dimensionless after
    count standardization). Return a new JSON-compatible dictionary; reject
    flags/text rather than silently coercing scientific settings.
    """
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError("SparsePCA alpha must be a finite nonnegative number.")
    return dict(
        alpha=float(alpha),
        ridge_alpha=0.01,
        method="lars",
        max_iter=1000,
        tol=1e-8,
        random_state=0,
        n_jobs=1,
        U_init=None,
        V_init=None,
        verbose=False,
    )


def fit_sparse_timeline_activity(
    counts: np.ndarray,
    unit_ids: tuple[str, ...],
    choice_positions: np.ndarray,
    region: str,
    probe_id: str,
    bin_width_s: float,
    *,
    alpha: float = 1.0,
) -> RegionalTimelinePCA:
    """Fit up to 30 sparse atoms and ridge scores to float64 counts (W,B,U).

    unit_ids uniquely label the unit axis; choice_positions (W,) indexes the
    common choice table. region/probe_id identify the population. bin_width_s
    is a supported width in seconds for [-2,+2) s bins; alpha is the sparsity
    penalty. Return the existing regional envelope with float64 (W,B,K) ridge
    coefficients, K=min(30,W*B,variable units), native count means/SDs, sparse
    parameters/diagnostics, and direct standardized-unit variance totals.
    No observation sampling, sorting of components, whitening or mutation.
    """
    parameters = sparse_pca_parameters(alpha)
    values = np.asarray(counts, dtype=np.float64)
    ids = tuple(str(unit) for unit in unit_ids)
    positions = np.asarray(choice_positions)
    edges = choice_bin_edges(bin_width_s)
    if (
        values.ndim != 3
        or not all(values.shape)
        or values.shape[:2] != (positions.size, edges.size - 1)
        or positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or values.shape[2] != len(ids)
        or len(set(ids)) != len(ids)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Sparse activity needs finite (windows,bins,units) counts and aligned unique IDs."
        )
    started = perf_counter()
    standardized, mask, means, scales = standardize_unit_activity(
        values.reshape(-1, len(ids))
    )
    standardization_s = perf_counter() - started
    parameters["n_components"] = min(COMPONENT_LIMIT, *standardized.shape)
    estimator = SparsePCA(**parameters)
    # Keep sklearn's exact transform, which uses a ridge solve, not a rotation.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = perf_counter()
        estimator.fit(standardized)
        fit_s = perf_counter() - started
        started = perf_counter()
        coefficients = estimator.transform(standardized)
        transform_s = perf_counter() - started
    if not np.isfinite(coefficients).all():
        raise ValueError("SparsePCA returned nonfinite coefficients.")
    fitted = SparseTimelineTransform(
        input_unit_ids=ids,
        retained_unit_ids=tuple(unit for unit, keep in zip(ids, mask) if keep),
        omitted_unit_ids=tuple(unit for unit, keep in zip(ids, mask) if not keep),
        training_mean=means,
        training_scale=scales,
        center=estimator.mean_,
        components=estimator.components_,
        n_training_observations=len(standardized),
        parameters=parameters,
        n_iter=int(estimator.n_iter_),
        objective_history=np.asarray(estimator.error_),
        warnings=tuple(str(item.message) for item in caught),
    )
    scores = np.asarray(coefficients, dtype=np.float64).reshape(*values.shape[:2], -1)
    standardized_windows = standardized.reshape(*values.shape[:2], -1)
    return RegionalTimelinePCA(
        region,
        probe_id,
        positions.copy(),
        edges,
        scores,
        fitted,
        float(standardized.var(axis=0, ddof=1).sum()),
        standardized_windows.var(axis=1, ddof=1).sum(axis=1),
        max(float(np.abs(scores).max()), 1e-12),
        {
            "standardization": standardization_s,
            "sparse_fit": fit_s,
            "sparse_transform": transform_s,
        },
    )
