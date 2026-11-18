"""Fixed session-wide CCA of paired unit counts; no loading, plotting, or file I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from statsmodels.multivariate.cancorr import CanCorr


NUMERICAL_VERSION = "cca-unit-rank-adapter-v1"
PARAMETERS = {
    "unit_scale_ddof": 0,
    "score_scale_ddof": 1,
    "rank_rule": "eps_float64 * max(shape) * largest_singular_value",
    "package_tolerance": 1e-8,
    "covariance_atol": 1e-8,
    "projection_rtol": 1e-8,
    "projection_atol": 1e-10,
    "correlation_bound_atol": 1e-10,
}


@dataclass(frozen=True)
class RegionalCCATransform:
    """One region's fixed affine projection from count activity to canonical scores.

    IDs label the original U input columns; retained IDs exclude exact constants.
    count_mean/count_scale are float64 (U,) in counts per bin. coefficients is
    (P,K), with P nonconstant units, in canonical SD per standardized unit.
    score_offset is (K,) in canonical SD. singular_values spans the regional
    standardized observation matrix; rank/cutoff record numerical null removal.
    Projection preserves leading (window,bin) axes and returns (window,bin,K).
    """

    input_unit_ids: tuple[str, ...]
    retained_unit_ids: tuple[str, ...]
    omitted_unit_ids: tuple[str, ...]
    count_mean: np.ndarray
    count_scale: np.ndarray
    coefficients: np.ndarray
    score_offset: np.ndarray
    singular_values: np.ndarray
    rank: int
    rank_cutoff: float


@dataclass(frozen=True)
class PairedCCAFit:
    """One fixed paired CCA fit, with PFC/HPC transforms and canonical-score arrays.

    scores maps each region to float64 (W,B,K), in dimensionless sample-SD-one
    units over all N=W*B fitting observations. correlations is descending (K,)
    nonnegative Pearson r; joint_signs is (K,) +/-1 applied to both regions.
    color_limit is the full-score maximum absolute value in canonical SD.
    diagnostics stores dimensionless validation/rank quantities and measured
    stage times in seconds. This record contains no raw counts or UI state.
    """

    transforms: dict[str, RegionalCCATransform]
    scores: dict[str, np.ndarray]
    correlations: np.ndarray
    n_observations: int
    color_limit: float
    diagnostics: dict[str, object]
    joint_signs: np.ndarray


def fit_paired_cca(
    pfc_counts: np.ndarray,
    hpc_counts: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
) -> PairedCCAFit:
    """Fit direct-unit CCA to paired, unsmoothed regional count tensors.

    Parameters are finite nonnegative arrays (W,B,U_PFC)/(W,B,U_HPC), in counts
    per bin, and unique ordered string IDs for their unit axes. Both regions
    must share W/B axes; the caller establishes their actual trial/bin pairing.
    Return all K=min(regional numerical ranks) pairs as PairedCCAFit, preserving
    W/B order. Pooled unit scaling uses population SD; scores use sample SD.
    Inputs are unchanged. Invalid, constant, or numerically inconsistent data
    raise ValueError; no scientific variance cutoff or regularization is used.
    """
    pfc, pfc_ids = _validated_counts(pfc_counts, pfc_unit_ids)
    hpc, hpc_ids = _validated_counts(hpc_counts, hpc_unit_ids)
    if pfc.shape[:2] != hpc.shape[:2]:
        raise ValueError("PFC/HPC must share paired window and bin axes.")
    observations = pfc.shape[0] * pfc.shape[1]
    if observations < 2:
        raise ValueError("CCA needs at least two paired bin observations.")

    started = perf_counter()
    bases = {"PFC": _prepare_region(pfc, pfc_ids), "HPC": _prepare_region(hpc, hpc_ids)}
    rank_elapsed = perf_counter() - started
    started = perf_counter()
    correlations, coordinate_coefficients = _fit_coordinate_cca(
        bases["PFC"].coordinates, bases["HPC"].coordinates
    )
    package_elapsed = perf_counter() - started
    order = np.argsort(-correlations, kind="stable")
    correlations = correlations[order]

    started = perf_counter()
    transforms = {}
    scores = {}
    for region, values in (("PFC", pfc), ("HPC", hpc)):
        transform, regional_scores = _finish_region(
            values, bases[region], coordinate_coefficients[region][:, order]
        )
        transforms[region] = transform
        scores[region] = regional_scores

    # CCA signs are arbitrary; anchor each paired direction to the PFC loadings.
    coefficients = transforms["PFC"].coefficients
    dominant = np.argmax(np.abs(coefficients), axis=0)
    signs = np.where(coefficients[dominant, np.arange(len(order))] < 0, -1, 1).astype(np.int8)
    for region in ("PFC", "HPC"):
        transforms[region] = replace(
            transforms[region],
            coefficients=transforms[region].coefficients * signs,
            score_offset=transforms[region].score_offset * signs,
        )
        scores[region] = scores[region] * signs

    diagnostics = _validate_scores(scores, correlations)
    for region, values in (("PFC", pfc), ("HPC", hpc)):
        replayed = transform_cca_activity(values, bases[region].unit_ids, transforms[region])
        if not np.allclose(replayed, scores[region], rtol=PARAMETERS["projection_rtol"], atol=PARAMETERS["projection_atol"]):
            raise ValueError(f"{region} CCA coefficients do not reconstruct the saved scores.")
    ranks = {region: basis.coordinates.shape[1] for region, basis in bases.items()}
    diagnostics["forced_unit_correlations"] = max(0, sum(ranks.values()) - (observations - 1))
    diagnostics["retained_conditioning"] = {
        region: float(basis.singular_values[0] / basis.singular_values[ranks[region] - 1])
        for region, basis in bases.items()
    }
    diagnostics["timings_s"] = {
        "rank_adapter": rank_elapsed,
        "package_cca": package_elapsed,
        "projection_and_validation": perf_counter() - started,
    }
    return PairedCCAFit(
        transforms, scores, correlations, observations,
        max(float(np.abs(values).max()) for values in scores.values()),
        diagnostics, signs,
    )


def transform_cca_activity(
    counts: np.ndarray,
    unit_ids: Sequence[str],
    fitted_transform: RegionalCCATransform,
) -> np.ndarray:
    """Project counts with the saved unit order, scaling, coefficients and offset.

    counts is finite nonnegative (W,B,U) activity in counts per bin; unit_ids
    must exactly equal the transform's ordered input IDs. Return float64
    (W,B,K) in canonical session-SD units, with no local fitting or scaling.
    Counts and fitted arrays are unchanged. Invalid inputs raise ValueError.
    """
    values, ids = _validated_counts(counts, unit_ids)
    if ids != fitted_transform.input_unit_ids:
        raise ValueError("unit_ids must match the fitted CCA input order.")
    retained = [ids.index(unit_id) for unit_id in fitted_transform.retained_unit_ids]
    standardized = (
        values[..., retained] - fitted_transform.count_mean[retained]
    ) / fitted_transform.count_scale[retained]
    return np.asarray(
        standardized @ fitted_transform.coefficients - fitted_transform.score_offset,
        dtype=np.float64,
    )


# Internal preparation, library adaptation, and numerical validation helpers.
@dataclass(frozen=True)
class _RegionalBasis:
    """Temporary numerical inputs; all arrays use float64 except unit positions.

    count means/SDs are (U,) in counts; positions labels P retained units.
    coordinates is (N,rank), coefficient_map (P,rank), standardized (N,P),
    singular_values (min(N,P),). The cutoff is dimensionless in standardized
    activity. This helper record is discarded after fitting.
    """

    unit_ids: tuple[str, ...]
    positions: np.ndarray
    count_mean: np.ndarray
    count_scale: np.ndarray
    coordinates: np.ndarray
    coefficient_map: np.ndarray
    standardized: np.ndarray
    singular_values: np.ndarray
    rank_cutoff: float


def _validated_counts(
    counts: np.ndarray, unit_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Validate count tensor/identity axes; return float64 (W,B,U) and string IDs.

    Input counts are real finite nonnegative counts per bin with nonempty axes;
    unit_ids uniquely labels U. Return arrays without modifying caller data.
    """
    if np.iscomplexobj(counts):
        raise ValueError("CCA counts must be real nonnegative activity.")
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError("CCA counts need nonempty (window, bin, unit) axes.")
    if isinstance(unit_ids, (str, bytes)):
        raise ValueError("unit_ids must be an ordered sequence of unit labels.")
    ids = tuple(unit_ids)
    if (len(ids) != values.shape[-1] or any(not isinstance(unit, str) or not unit for unit in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("unit_ids must uniquely label every count-axis unit.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("CCA counts must be finite and nonnegative.")
    return values, ids


def _prepare_region(values: np.ndarray, unit_ids: tuple[str, ...]) -> _RegionalBasis:
    """Standardize (W,B,U) count values and retain all numerical non-null directions.

    Return temporary pooled (N,P) activity and its thin-SVD observation basis;
    unit means/SDs retain all U input columns, in counts per bin. Constant units
    are identified by exact min/max equality, independent of mean roundoff.
    """
    flat = values.reshape(-1, values.shape[-1])
    variable = flat.max(axis=0) > flat.min(axis=0)
    positions = np.flatnonzero(variable)
    if not positions.size:
        raise ValueError("CCA region has no variable units.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        means = flat.mean(axis=0)
        scales = flat.std(axis=0, ddof=PARAMETERS["unit_scale_ddof"])
        scales[~variable] = 0
        standardized = (flat[:, positions] - means[positions]) / scales[positions]
    if not np.isfinite(standardized).all():
        raise ValueError("CCA unit standardization is numerically undefined.")
    try:
        left, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("CCA regional singular-value decomposition failed.") from error
    cutoff = np.finfo(np.float64).eps * max(standardized.shape) * singular_values[0]
    keep = singular_values > cutoff
    if not np.any(keep):
        raise ValueError("CCA region has zero numerical rank.")
    return _RegionalBasis(
        unit_ids, positions, means, scales, left[:, keep],
        right[keep].T / singular_values[keep], standardized,
        singular_values, float(cutoff),
    )


def _fit_coordinate_cca(
    pfc: np.ndarray, hpc: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit full CCA to dimensionless (N,rank) regional observation bases.

    Return (K,) correlations and regional (rank,K) coordinate coefficients.
    Statsmodels squeezes one-column endog, so swap sides if only HPC has rank
    one and use the scalar correlation definition if both ranks are one.
    """
    if pfc.shape[1] == hpc.shape[1] == 1:
        x = pfc[:, 0] - pfc[:, 0].mean()
        y = hpc[:, 0] - hpc[:, 0].mean()
        correlation = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        return np.asarray([abs(correlation)]), {
            "PFC": np.ones((1, 1)),
            "HPC": np.asarray([[-1.0 if correlation < 0 else 1.0]]),
        }
    swapped = hpc.shape[1] == 1
    try:
        fitted = CanCorr(
            endog=pfc if swapped else hpc,
            exog=hpc if swapped else pfc,
            missing="raise", hasconst=False,
            tolerance=PARAMETERS["package_tolerance"],
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"CCA coordinate fitting failed: {error}") from error
    return np.asarray(fitted.cancorr, dtype=np.float64), {
        "PFC": fitted.y_cancoef if swapped else fitted.x_cancoef,
        "HPC": fitted.x_cancoef if swapped else fitted.y_cancoef,
    }


def _finish_region(
    values: np.ndarray, basis: _RegionalBasis, coordinate_coefficients: np.ndarray,
) -> tuple[RegionalCCATransform, np.ndarray]:
    """Map (rank,K) coefficients back to units; return transform and (W,B,K) scores.

    values is the original regional count tensor. Normalize pooled scores to
    sample SD one, incorporating normalization into coefficients/offsets rather
    than applying a window-specific adjustment. Output scores are dimensionless.
    """
    observations = basis.standardized.shape[0]
    coefficients = basis.coefficient_map @ coordinate_coefficients * np.sqrt(observations - 1)
    raw_scores = basis.standardized @ coefficients
    means = raw_scores.mean(axis=0)
    scales = raw_scores.std(axis=0, ddof=PARAMETERS["score_scale_ddof"])
    if not np.isfinite(raw_scores).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("CCA canonical score normalization is undefined.")
    coefficients = coefficients / scales
    offset = means / scales
    retained = tuple(basis.unit_ids[index] for index in basis.positions)
    omitted = tuple(unit for index, unit in enumerate(basis.unit_ids) if index not in basis.positions)
    transform = RegionalCCATransform(
        basis.unit_ids, retained, omitted, basis.count_mean, basis.count_scale,
        coefficients, offset, basis.singular_values, basis.coordinates.shape[1],
        basis.rank_cutoff,
    )
    scores = ((raw_scores - means) / scales).reshape(*values.shape[:2], -1)
    return transform, scores


def _validate_scores(
    scores: dict[str, np.ndarray], correlations: np.ndarray,
) -> dict[str, object]:
    """Validate canonical identities; return dimensionless maximum-error diagnostics.

    scores holds regional (W,B,K) canonical-SD arrays; correlations is (K,)
    Pearson r. Material violations raise ValueError, never repaired by clipping.
    """
    pair_count = len(correlations)
    x = scores["PFC"].reshape(-1, pair_count)
    y = scores["HPC"].reshape(-1, pair_count)
    denominator = len(x) - 1
    covariances = {"PFC": x.T @ x / denominator, "HPC": y.T @ y / denominator}
    cross_covariance = x.T @ y / denominator
    paired = np.diag(cross_covariance) / np.sqrt(
        np.diag(covariances["PFC"]) * np.diag(covariances["HPC"])
    )
    errors = {
        "mean_max_abs": max(float(np.abs(values.mean(axis=0)).max()) for values in (x, y)),
        "within_covariance_max_error": max(float(np.abs(value - np.eye(pair_count)).max()) for value in covariances.values()),
        "cross_covariance_max_error": float(np.abs(cross_covariance - np.diag(correlations)).max()),
        "correlation_max_error": float(np.abs(paired - correlations).max()),
    }
    if not np.isfinite(correlations).all() or not all(np.isfinite(value) for value in errors.values()):
        raise ValueError("CCA produced nonfinite canonical validation quantities.")
    bound = PARAMETERS["correlation_bound_atol"]
    if (np.any(correlations < -bound) or np.any(correlations > 1 + bound)
            or np.any(paired < -bound) or np.any(paired > 1 + bound)):
        raise ValueError("CCA canonical correlations violate their numerical bounds.")
    if any(value > PARAMETERS["covariance_atol"] for value in errors.values()):
        raise ValueError(f"CCA canonical covariance validation failed: {errors}")
    return errors
