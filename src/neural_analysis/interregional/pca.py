"""Training-only regional PCA transforms for inter-regional regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
import warnings

import numpy as np
from sklearn.decomposition import PCA

from .records import RegionalPCATransform


@dataclass(frozen=True)
class RegionalPCATransforms:
    """Paired PFC/HPC transforms fitted for one declared analysis scope.

    ``scope`` is ``"fold"`` for training-only CV transforms or
    ``"descriptive"`` for an all-eligible-data transform. ``fold_id`` is a
    zero-based fold identifier only for fold scope. ``n_training_trials`` is a
    dimensionless count of unique trials in the fitting pool.
    """

    scope: str
    fold_id: int | None
    pfc: RegionalPCATransform
    hpc: RegionalPCATransform
    n_training_trials: int

    def __post_init__(self) -> None:
        """Validate that scope, fold identity, and trial count agree."""
        if self.scope not in {"fold", "descriptive"}:
            raise ValueError("scope must be 'fold' or 'descriptive'.")
        if self.scope == "fold":
            if (
                isinstance(self.fold_id, bool)
                or not isinstance(self.fold_id, Integral)
                or int(self.fold_id) < 0
            ):
                raise ValueError("Fold-scoped PCA requires a nonnegative fold_id.")
            object.__setattr__(self, "fold_id", int(self.fold_id))
        elif self.fold_id is not None:
            raise ValueError("Descriptive PCA must not carry a fold_id.")
        if (
            isinstance(self.n_training_trials, bool)
            or not isinstance(self.n_training_trials, Integral)
            or int(self.n_training_trials) <= 0
        ):
            raise ValueError("n_training_trials must be a positive integer.")
        object.__setattr__(self, "n_training_trials", int(self.n_training_trials))


def _validated_activity(
    activity: np.ndarray, unit_ids: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return float64 activity and unique IDs for a ``(..., unit)`` array."""
    values = np.asarray(activity, dtype=np.float64)
    ids = tuple(str(value) for value in unit_ids)
    if values.ndim < 2:
        raise ValueError("activity must have at least observation and unit axes.")
    if values.shape[-1] != len(ids) or not ids or len(set(ids)) != len(ids):
        raise ValueError("unit_ids must uniquely label the final activity axis.")
    if values.size == 0 or int(np.prod(values.shape[:-1])) == 0:
        raise ValueError("activity must contain at least one observation.")
    return values, ids


def fit_regional_pca(
    training_activity: np.ndarray,
    unit_ids: Sequence[str],
    requested_components: int,
) -> RegionalPCATransform:
    """Fit standardized, unwhitened PCA to one regional training pool.

    Parameters
    ----------
    training_activity : numpy.ndarray
        Activity with shape ``(..., unit)``. Leading axes are pooled into the
        observation axis. Values are spike counts per configured time bin.
    unit_ids : sequence[str]
        Stable labels for the final unit axis.
    requested_components : int
        Positive dimensionless upper bound on retained components.

    Returns
    -------
    RegionalPCATransform
        Immutable float64 standardization and PCA parameters. Units with zero
        or nonfinite training population standard deviation are omitted.
    """
    values, ids = _validated_activity(training_activity, unit_ids)
    if (
        isinstance(requested_components, bool)
        or not isinstance(requested_components, Integral)
        or int(requested_components) <= 0
    ):
        raise ValueError("requested_components must be a positive integer.")
    observations = values.reshape(-1, values.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        all_means = observations.mean(axis=0, dtype=np.float64)
        all_scales = observations.std(axis=0, ddof=0, dtype=np.float64)
    retained_mask = (
        np.isfinite(all_means) & np.isfinite(all_scales) & (all_scales > 0.0)
    )
    if not np.any(retained_mask):
        raise ValueError("Regional PCA has no variable finite training units.")
    retained_ids = tuple(unit_id for unit_id, keep in zip(ids, retained_mask) if keep)
    omitted_ids = tuple(unit_id for unit_id, keep in zip(ids, retained_mask) if not keep)
    means = all_means[retained_mask]
    scales = all_scales[retained_mask]
    standardized = (observations[:, retained_mask] - means) / scales
    component_count = min(
        int(requested_components), standardized.shape[0], standardized.shape[1]
    )
    estimator = PCA(
        n_components=component_count,
        svd_solver="full",
        whiten=False,
        random_state=None,
    )
    estimator.fit(standardized)
    return RegionalPCATransform(
        retained_unit_ids=retained_ids,
        omitted_unit_ids=omitted_ids,
        training_mean=means,
        training_scale=scales,
        components=estimator.components_,
        explained_variance=estimator.explained_variance_,
        explained_variance_ratio=estimator.explained_variance_ratio_,
        n_training_observations=standardized.shape[0],
        input_unit_ids=ids,
        svd_solver="full",
        whiten=False,
        randomness_used=False,
        random_seed=None,
    )


def transform_regional_activity(
    activity: np.ndarray,
    unit_ids: Sequence[str],
    fitted_transform: RegionalPCATransform,
) -> np.ndarray:
    """Apply one training-fitted PCA transform without refitting or rescaling.

    Parameters
    ----------
    activity : numpy.ndarray
        Activity with shape ``(..., unit)`` in spike counts per configured bin.
    unit_ids : sequence[str]
        Ordered labels for the final axis; they must exactly match the fit input.
    fitted_transform : RegionalPCATransform
        Training-derived standardization and unwhitened PCA rotation.

    Returns
    -------
    numpy.ndarray
        Float64 scores with shape ``(..., component)`` and arbitrary PCA-score
        units; no post-PCA standardization is applied.
    """
    values, ids = _validated_activity(activity, unit_ids)
    if ids != fitted_transform.input_unit_ids:
        raise ValueError("unit_ids must match the fitted transform input order.")
    retained_positions = np.asarray(
        [ids.index(unit_id) for unit_id in fitted_transform.retained_unit_ids],
        dtype=np.int64,
    )
    retained = values[..., retained_positions]
    standardized = (
        retained - fitted_transform.training_mean
    ) / fitted_transform.training_scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError("activity is nonfinite after training standardization.")
    return np.asarray(
        standardized @ fitted_transform.components.T, dtype=np.float64
    )


def _condition_union(
    n_trials: int,
    condition_masks: Mapping[str, np.ndarray],
    requested_conditions: Sequence[str],
) -> np.ndarray:
    """Return the unique-trial union of requested boolean condition masks."""
    requested = tuple(str(value) for value in requested_conditions)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested_conditions must be nonempty and unique.")
    combined = np.zeros(n_trials, dtype=bool)
    for condition in requested:
        if condition not in condition_masks:
            raise ValueError(f"Missing condition mask {condition!r}.")
        mask = np.asarray(condition_masks[condition], dtype=bool)
        if mask.shape != (n_trials,):
            raise ValueError("Each condition mask must have shape (trial,).")
        combined |= mask
    return combined


def _fit_paired_regional_pcas(
    *,
    pfc_activity: np.ndarray,
    hpc_activity: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    fitting_trials: np.ndarray,
    condition_masks: Mapping[str, np.ndarray],
    requested_conditions: Sequence[str],
    requested_components: int,
    scope: str,
    fold_id: int | None,
) -> RegionalPCATransforms:
    """Fit both regions on one unique-trial condition-union pool."""
    pfc_values, _ = _validated_activity(pfc_activity, pfc_unit_ids)
    hpc_values, _ = _validated_activity(hpc_activity, hpc_unit_ids)
    if pfc_values.shape[:-1] != hpc_values.shape[:-1] or pfc_values.ndim != 3:
        raise ValueError("Regional activity must share (trial, whole_window_bin) axes.")
    trial_selector = np.asarray(fitting_trials, dtype=bool)
    if trial_selector.shape != (pfc_values.shape[0],):
        raise ValueError("fitting trial mask must have shape (trial,).")
    pooled_trials = trial_selector & _condition_union(
        pfc_values.shape[0], condition_masks, requested_conditions
    )
    if not np.any(pooled_trials):
        raise ValueError("Regional PCA fitting pool contains no trials.")
    return RegionalPCATransforms(
        scope=scope,
        fold_id=fold_id,
        pfc=fit_regional_pca(
            pfc_values[pooled_trials], pfc_unit_ids, requested_components
        ),
        hpc=fit_regional_pca(
            hpc_values[pooled_trials], hpc_unit_ids, requested_components
        ),
        n_training_trials=int(np.count_nonzero(pooled_trials)),
    )


def fit_fold_regional_pcas(
    *,
    pfc_activity: np.ndarray,
    hpc_activity: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    training_trials: np.ndarray,
    condition_masks: Mapping[str, np.ndarray],
    requested_conditions: Sequence[str],
    requested_components: int,
    fold_id: int,
) -> RegionalPCATransforms:
    """Fit paired PCA transforms from one fold's training trials.

    Regional arrays have shape ``(trial, whole_window_bin, unit)`` in spike
    counts per bin. Trial and condition masks have shape ``(trial,)``. The
    returned transforms use only the unique-trial union selected by both masks.
    """
    return _fit_paired_regional_pcas(
        pfc_activity=pfc_activity,
        hpc_activity=hpc_activity,
        pfc_unit_ids=pfc_unit_ids,
        hpc_unit_ids=hpc_unit_ids,
        fitting_trials=training_trials,
        condition_masks=condition_masks,
        requested_conditions=requested_conditions,
        requested_components=requested_components,
        scope="fold",
        fold_id=fold_id,
    )


def fit_descriptive_regional_pcas(
    *,
    pfc_activity: np.ndarray,
    hpc_activity: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    scientific_trials: np.ndarray,
    condition_masks: Mapping[str, np.ndarray],
    requested_conditions: Sequence[str],
    requested_components: int,
) -> RegionalPCATransforms:
    """Fit paired PCA transforms from all scientifically eligible trials.

    Regional arrays have shape ``(trial, whole_window_bin, unit)`` in spike
    counts per bin. This distinct descriptive scope is not valid for held-out
    cross-validation.
    """
    return _fit_paired_regional_pcas(
        pfc_activity=pfc_activity,
        hpc_activity=hpc_activity,
        pfc_unit_ids=pfc_unit_ids,
        hpc_unit_ids=hpc_unit_ids,
        fitting_trials=scientific_trials,
        condition_masks=condition_masks,
        requested_conditions=requested_conditions,
        requested_components=requested_components,
        scope="descriptive",
        fold_id=None,
    )
