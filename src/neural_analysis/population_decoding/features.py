"""One training-only Hz scaler followed by PCA of standardized unit vectors."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class UnitScaling:
    """Training Hz means/SDs (U,), Boolean retained mask (U,), ordered unit IDs.

    Arrays retain original input-unit axes, including omitted constants. Applying
    this record produces only P variable-unit columns in dimensionless SD units.
    """

    unit_ids: tuple[str, ...]
    mean_hz: np.ndarray
    scale_hz: np.ndarray
    retained_mask: np.ndarray


@dataclass(frozen=True)
class FoldPCA:
    """Array-only unwhitened PCA fitted to standardized training vectors (M,P).

    components(K,P), center(P,), singular_values(K,), explained_variance(K,) and
    explained_variance_ratio(K,) are dimensionless. rank_tolerance is the saved
    numerical singular-value cutoff; training_observations is a count.
    """

    components: np.ndarray
    center: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    rank_tolerance: float
    training_observations: int


def fit_unit_scaling(training_rates_hz: np.ndarray, unit_ids: tuple[str, ...]) -> UnitScaling:
    """Fit one scaler from finite nonnegative rates (...,U) in Hz; return UnitScaling.

    Leading axes are pooled into observations. Means and population SDs (ddof=0)
    use training observations only. Constant units are omitted; no variable
    units raises ValueError. Input arrays and unit order remain unchanged.
    """
    values = np.asarray(training_rates_hz, dtype=float)
    ids = tuple(unit_ids)
    if (values.ndim < 2 or not all(values.shape) or values.shape[-1] != len(ids)
            or len(set(ids)) != len(ids) or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("Expected finite nonnegative Hz (...,U) and unique ordered unit IDs")
    flat = values.reshape(-1, len(ids))
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0, ddof=0)
    # Exact constants must remain excluded even if summation roundoff gives a tiny SD.
    retained = (np.ptp(flat, axis=0) > 0) & (scale > 0)
    if not retained.any():
        raise ValueError("No variable training units")
    return UnitScaling(ids, mean, scale, retained)


def transform_unit_activity(rates_hz: np.ndarray, unit_ids: tuple[str, ...],
                            fitted: UnitScaling) -> np.ndarray:
    """Apply saved Hz scaling to (...,U); return standardized (...,P), no refit.

    Ordered IDs must match training. Nonfinite/negative rates fail rather than
    changing row eligibility or feature selection. Input data are unchanged.
    """
    values = np.asarray(rates_hz, dtype=float)
    if tuple(unit_ids) != fitted.unit_ids:
        raise ValueError("Unit order must match the fitted scaler")
    if values.ndim < 2 or values.shape[-1] != len(unit_ids) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative rates (...,U) in Hz")
    keep = fitted.retained_mask
    return (values[..., keep] - fitted.mean_hz[keep]) / fitted.scale_hz[keep]


def fit_fold_pca(training_standardized: np.ndarray) -> FoldPCA:
    """Fit full-SVD PCA to finite standardized (M,P); return all nonzero PC axes.

    Inputs are dimensionless unit SDs. PCA centers but never rescales/reselects
    units. No whitening or randomness is used. Numerical rank uses
    max(M,P)*eps(float64)*largest singular value. Empty rank raises ValueError.
    """
    values = np.asarray(training_standardized, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("PCA requires finite standardized (M,P), M>=2")
    estimator = PCA(svd_solver="full", whiten=False).fit(values)
    tolerance = max(values.shape) * np.finfo(float).eps * estimator.singular_values_.max(initial=0)
    keep = estimator.singular_values_ > tolerance
    if not keep.any():
        raise ValueError("No numerically nonzero training PCs")
    return FoldPCA(estimator.components_[keep], estimator.mean_, estimator.singular_values_[keep],
                   estimator.explained_variance_[keep], estimator.explained_variance_ratio_[keep],
                   float(tolerance), len(values))


def transform_pc_activity(standardized: np.ndarray, fitted: FoldPCA) -> np.ndarray:
    """Transform standardized (...,P) to dimensionless scores (...,K), no whitening.

    Use only the saved PCA center/components; inputs are not modified. Raise
    ValueError on incompatible feature axes or nonfinite observations.
    """
    values = np.asarray(standardized, dtype=float)
    if values.ndim < 2 or values.shape[-1] != len(fitted.center) or not np.isfinite(values).all():
        raise ValueError("Standardized feature axis must match fitted PCA")
    return (values - fitted.center) @ fitted.components.T
