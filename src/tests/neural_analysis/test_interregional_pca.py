"""Tests for training-only regional PCA used by inter-regional regression."""

from __future__ import annotations

import numpy as np
import pytest

from src.neural_analysis.interregional.pca import (
    fit_descriptive_regional_pcas,
    fit_fold_regional_pcas,
    fit_regional_pca,
    transform_regional_activity,
)


def test_fit_uses_population_scaling_full_svd_and_omits_constant_units() -> None:
    """The saved transform exactly describes standardized, unwhitened PCA."""
    activity = np.array(
        [
            [[1.0, 10.0, 5.0], [2.0, 12.0, 5.0]],
            [[3.0, 14.0, 5.0], [4.0, 18.0, 5.0]],
        ],
        dtype=np.float32,
    )

    fitted = fit_regional_pca(
        activity, ("probe:1", "probe:2", "probe:3"), requested_components=3
    )

    flattened = activity.astype(np.float64).reshape(-1, 3)
    np.testing.assert_allclose(fitted.training_mean, flattened[:, :2].mean(axis=0))
    np.testing.assert_allclose(
        fitted.training_scale, flattened[:, :2].std(axis=0, ddof=0)
    )
    assert fitted.retained_unit_ids == ("probe:1", "probe:2")
    assert fitted.omitted_unit_ids == ("probe:3",)
    assert fitted.components.shape == (2, 2)
    assert fitted.components.dtype == np.float64
    assert fitted.n_training_observations == 4
    assert fitted.svd_solver == "full"
    assert fitted.whiten is False
    assert fitted.randomness_used is False
    assert fitted.random_seed is None


def test_transform_applies_training_parameters_without_post_pca_scaling() -> None:
    """Held-out scores equal the recorded affine transform and PCA rotation."""
    training = np.array(
        [
            [[0.0, 10.0, 2.0], [1.0, 13.0, 2.0]],
            [[3.0, 11.0, 2.0], [4.0, 18.0, 2.0]],
        ]
    )
    held_out = np.array([[[8.0, 6.0, 200.0], [2.0, 20.0, -50.0]]])
    unit_ids = ("a", "b", "constant")
    fitted = fit_regional_pca(training, unit_ids, requested_components=2)

    scores = transform_regional_activity(held_out, unit_ids, fitted)

    retained = held_out[..., :2]
    expected = ((retained - fitted.training_mean) / fitted.training_scale) @ (
        fitted.components.T
    )
    np.testing.assert_allclose(scores, expected)
    assert scores.shape == (1, 2, 2)
    assert not np.allclose(scores.std(axis=(0, 1), ddof=0), np.ones(2))


def test_component_count_is_limited_by_observations_units_and_request() -> None:
    """Actual dimensionality is the documented minimum, never the request alone."""
    two_observations = np.array([[[1.0, 2.0, 4.0], [2.0, 5.0, 9.0]]])

    fitted = fit_regional_pca(
        two_observations, ("a", "b", "c"), requested_components=10
    )

    assert fitted.actual_components == 2
    assert fitted.components.shape == (2, 3)


def test_fold_fit_pools_unique_training_trials_from_requested_condition_union() -> None:
    """Overlapping conditions do not duplicate a trial in either regional fit."""
    base = np.arange(6 * 2 * 2, dtype=float).reshape(6, 2, 2)
    condition_masks = {
        "rewarded": np.array([True, True, False, False, False, False]),
        "right": np.array([False, True, True, False, False, False]),
        "unused": np.array([False, False, False, True, False, False]),
    }
    training_trials = np.array([True, True, True, True, False, False])

    fitted = fit_fold_regional_pcas(
        pfc_activity=base,
        hpc_activity=base + 100.0,
        pfc_unit_ids=("p1", "p2"),
        hpc_unit_ids=("h1", "h2"),
        training_trials=training_trials,
        condition_masks=condition_masks,
        requested_conditions=("rewarded", "right"),
        requested_components=2,
        fold_id=3,
    )

    expected_trials = np.array([True, True, True, False, False, False])
    expected_pfc = base[expected_trials].reshape(-1, 2)
    assert fitted.scope == "fold"
    assert fitted.fold_id == 3
    assert fitted.n_training_trials == 3
    assert fitted.pfc.n_training_observations == 6
    np.testing.assert_allclose(fitted.pfc.training_mean, expected_pfc.mean(axis=0))
    np.testing.assert_allclose(
        fitted.hpc.training_mean, (expected_pfc + 100.0).mean(axis=0)
    )


def test_changing_held_out_activity_cannot_change_fold_transform() -> None:
    """Fold fitting reads only the declared training-trial subset."""
    rng = np.random.default_rng(51)
    pfc = rng.normal(size=(8, 3, 3))
    hpc = rng.normal(size=(8, 3, 2))
    training = np.array([True, True, True, True, False, False, False, False])
    conditions = {"all": np.ones(8, dtype=bool)}
    common = dict(
        pfc_unit_ids=("p1", "p2", "p3"),
        hpc_unit_ids=("h1", "h2"),
        training_trials=training,
        condition_masks=conditions,
        requested_conditions=("all",),
        requested_components=2,
        fold_id=0,
    )
    original = fit_fold_regional_pcas(
        pfc_activity=pfc, hpc_activity=hpc, **common
    )
    changed_pfc = pfc.copy()
    changed_hpc = hpc.copy()
    changed_pfc[~training] = 1e12
    changed_hpc[~training] = -1e12

    changed = fit_fold_regional_pcas(
        pfc_activity=changed_pfc, hpc_activity=changed_hpc, **common
    )

    np.testing.assert_array_equal(original.pfc.components, changed.pfc.components)
    np.testing.assert_array_equal(original.hpc.components, changed.hpc.components)
    np.testing.assert_array_equal(original.pfc.training_mean, changed.pfc.training_mean)
    np.testing.assert_array_equal(original.hpc.training_scale, changed.hpc.training_scale)


def test_descriptive_scope_is_distinct_and_transform_rejects_wrong_unit_order() -> None:
    """All-data fits are labeled separately and unit-axis mismatches fail loudly."""
    activity = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
    conditions = {"all": np.ones(5, dtype=bool)}

    descriptive = fit_descriptive_regional_pcas(
        pfc_activity=activity,
        hpc_activity=activity + 1.0,
        pfc_unit_ids=("p1", "p2"),
        hpc_unit_ids=("h1", "h2"),
        scientific_trials=np.ones(5, dtype=bool),
        condition_masks=conditions,
        requested_conditions=("all",),
        requested_components=1,
    )

    assert descriptive.scope == "descriptive"
    assert descriptive.fold_id is None
    with pytest.raises(ValueError, match="unit_ids"):
        transform_regional_activity(activity, ("p2", "p1"), descriptive.pfc)
