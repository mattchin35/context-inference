"""Tests for explicit unpenalized inter-regional OLS and CV scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.interregional import linear


def test_add_intercept_and_fit_multiple_targets_exactly() -> None:
    """One explicit intercept supports one multi-right-hand-side OLS solve."""
    feature = np.arange(6, dtype=float)[:, None]
    design = linear.add_intercept(feature)
    responses = np.column_stack((2.0 + 3.0 * feature[:, 0], -1.0 + 0.5 * feature[:, 0]))

    fit = linear.fit_ols_targets(design, responses)

    np.testing.assert_array_equal(design[:, 0], np.ones(6))
    np.testing.assert_allclose(fit.coefficients, np.array([[2.0, -1.0], [3.0, 0.5]]))
    np.testing.assert_allclose(linear.predict_ols_targets(design, fit.coefficients), responses)
    assert fit.rank == 2
    assert fit.feature_count == 2
    assert fit.df_resid == 4
    np.testing.assert_array_equal(fit.constant_targets, np.array([False, False]))


def test_ols_rejects_rank_deficiency_and_nonpositive_residual_df() -> None:
    """The strict unpenalized model has no rank-recovery fallback."""
    with pytest.raises(ValueError, match="rank deficient"):
        linear.fit_ols_targets(
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            np.arange(3, dtype=float)[:, None],
        )
    with pytest.raises(ValueError, match="degrees of freedom"):
        linear.fit_ols_targets(
            np.array([[1.0, 0.0], [1.0, 1.0]]),
            np.arange(2, dtype=float)[:, None],
        )


def test_constant_training_targets_are_reported_per_target() -> None:
    """A constant target does not invalidate other targets sharing the design."""
    x = np.arange(5, dtype=float)
    fit = linear.fit_ols_targets(
        linear.add_intercept(x[:, None]),
        np.column_stack((np.ones(5), x)),
    )
    np.testing.assert_array_equal(fit.constant_targets, np.array([True, False]))


def test_held_out_scores_match_hand_calculation_and_preserve_negative_values() -> None:
    """R-squared, MSE, and incremental R-squared use direct held-out formulas."""
    observed = np.array([[0.0], [1.0], [2.0]])
    restricted = np.array([[1.0], [1.0], [1.0]])
    full = np.array([[3.0], [3.0], [3.0]])

    scores = linear.score_ols_predictions(observed, restricted, full)

    assert scores.r2_restricted[0] == 0.0
    assert scores.r2_full[0] == -6.0
    assert scores.delta_r2[0] == -6.0
    assert scores.mse_restricted[0] == pytest.approx(2.0 / 3.0)
    assert scores.mse_full[0] == pytest.approx(14.0 / 3.0)
    np.testing.assert_array_equal(scores.r2_available, np.array([True]))


def test_constant_test_target_keeps_mse_and_marks_r2_unavailable() -> None:
    """Undefined held-out R-squared never erases a defined prediction error."""
    observed = np.ones((4, 1))
    restricted = np.zeros((4, 1))
    full = np.full((4, 1), 2.0)

    scores = linear.score_ols_predictions(observed, restricted, full)

    assert np.isnan(scores.r2_restricted[0])
    assert np.isnan(scores.r2_full[0])
    assert np.isnan(scores.delta_r2[0])
    assert scores.mse_restricted[0] == 1.0
    assert scores.mse_full[0] == 1.0
    np.testing.assert_array_equal(scores.r2_available, np.array([False]))


def test_complete_fold_summaries_use_target_means_then_population_quantiles() -> None:
    """Incomplete targets are retained but excluded from population quartiles."""
    rows: list[dict[str, object]] = []
    for target_id, values in (("pfc:1", [1, 2, 3, 4, 5]), ("pfc:2", [2, 4, 6, 8, 10])):
        for fold_id, value in enumerate(values):
            rows.append(
                {
                    "session_id": "s",
                    "direction": "HPC_to_PFC",
                    "representation": "units",
                    "model_family": "ols",
                    "condition": "all",
                    "window": "before",
                    "target_id": target_id,
                    "fold_id": fold_id,
                    "evaluation_scope": "held_out_cv",
                    "status": "ok",
                    "reason": "",
                    "delta_r2": float(value),
                }
            )
    rows[-1]["delta_r2"] = np.nan
    fold_scores = pd.DataFrame(rows)

    targets, population = linear.summarize_complete_cv_targets(
        fold_scores, metric_names=("delta_r2",)
    )

    complete = targets.loc[targets["target_id"] == "pfc:1"].iloc[0]
    incomplete = targets.loc[targets["target_id"] == "pfc:2"].iloc[0]
    assert complete["status"] == "ok"
    assert complete["mean_value"] == 3.0
    assert incomplete["status"] == "incomplete_folds"
    assert pd.isna(incomplete["mean_value"])
    assert population.iloc[0]["n_targets"] == 1
    assert population.iloc[0]["median"] == 3.0
