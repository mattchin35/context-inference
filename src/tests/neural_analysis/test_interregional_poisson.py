"""Tests for inter-regional Poisson CV fitting, scoring, and MSE pairing."""

from __future__ import annotations

import gc
import warnings
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from src.neural_analysis.interregional import linear, poisson


def _design(feature: np.ndarray) -> np.ndarray:
    """Return a finite design with observation rows and an explicit intercept."""
    values = np.asarray(feature, dtype=np.float64)
    return np.column_stack((np.ones(values.size), values))


def _fold_rows(*, missing_poisson_fold: int | None = None) -> pd.DataFrame:
    """Build matched OLS/Poisson fold rows with count-squared MSE units."""
    rows: list[dict[str, object]] = []
    for model_family in ("ols", "poisson"):
        for fold_id in range(5):
            poisson_missing = (
                model_family == "poisson" and fold_id == missing_poisson_fold
            )
            rows.append(
                {
                    "session_id": "session",
                    "direction": "HPC_to_PFC",
                    "representation": "units",
                    "model_family": model_family,
                    "condition": "all",
                    "window": "before",
                    "target_id": "pfc:1",
                    "fold_id": fold_id,
                    "evaluation_scope": "held_out_cv",
                    "restricted_status": "ok",
                    "full_status": "fit_unavailable" if poisson_missing else "ok",
                    "train_row_set_sha256": "a" * 64,
                    "test_row_set_sha256": "b" * 64,
                    "mse_restricted": float(4 + fold_id)
                    if model_family == "ols"
                    else float(2 + fold_id),
                    "mse_full": (
                        float(3 + fold_id)
                        if model_family == "ols"
                        else None if poisson_missing else float(1 + fold_id)
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "counts",
    (
        np.array([0.0, -1.0, 2.0]),
        np.array([0.0, 1.5, 2.0]),
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, np.inf, 2.0]),
    ),
)
def test_poisson_fit_rejects_invalid_count_responses(counts: np.ndarray) -> None:
    """Poisson responses must be finite nonnegative integer counts per bin."""
    with pytest.raises(ValueError, match="nonnegative integer counts"):
        poisson.fit_poisson_target(_design(np.arange(3)), counts)


def test_poisson_fit_uses_explicit_unpenalized_statsmodels_path(monkeypatch) -> None:
    """The fit fixes the family, log link, intercept, and IRLS arguments."""
    captured: dict[str, object] = {}

    class FakeModel:
        """Record the ordinary fit call without exposing a regularized method."""

        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return SimpleNamespace(
                params=np.array([0.2, -0.1]),
                converged=True,
                fit_history={"iteration": 3},
                llf=-4.0,
            )

    def fake_glm(endog, exog, *, family, missing):
        captured.update(endog=endog, exog=exog, family=family, missing=missing)
        return FakeModel()

    monkeypatch.setattr(poisson.sm, "GLM", fake_glm)
    fit = poisson.fit_poisson_target(_design(np.arange(5)), np.array([1, 2, 1, 3, 2]))

    np.testing.assert_array_equal(np.asarray(captured["exog"])[:, 0], np.ones(5))
    assert isinstance(captured["family"], poisson.sm.families.Poisson)
    assert isinstance(captured["family"].link, poisson.sm.families.links.Log)
    assert captured["missing"] == "raise"
    assert captured["fit_kwargs"] == {
        "method": "IRLS",
        "maxiter": 100,
        "tol": 1e-8,
        "scale": None,
        "cov_type": "nonrobust",
        "full_output": True,
        "disp": False,
    }
    assert fit.converged is True
    assert fit.iterations == 3


def test_poisson_fit_matches_default_irls_reference() -> None:
    """The public fit preserves default IRLS parameters, predictions, and iterations."""
    rng = np.random.default_rng(20261008)
    features = rng.normal(size=(800, 4))
    design = np.column_stack((np.ones(features.shape[0]), features))
    true_parameters = np.array([-0.2, 0.15, -0.1, 0.08, 0.04])
    counts = rng.poisson(np.exp(design @ true_parameters))

    reference = poisson.sm.GLM(
        counts,
        design,
        family=poisson.sm.families.Poisson(link=poisson.sm.families.links.Log()),
        missing="raise",
    ).fit(
        method="IRLS",
        maxiter=100,
        tol=1e-8,
        scale=None,
        cov_type="nonrobust",
        full_output=True,
        disp=False,
    )
    fitted = poisson.fit_poisson_target(design, counts)

    np.testing.assert_allclose(
        fitted.parameters, reference.params, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        poisson.predict_poisson_mean(design, fitted.parameters),
        np.exp(design @ reference.params),
        rtol=1e-12,
        atol=1e-12,
    )
    assert fitted.converged is True
    assert fitted.iterations == reference.fit_history["iteration"]


def test_poisson_fit_collects_cyclic_statsmodels_result(monkeypatch) -> None:
    """The public boundary releases a self-cyclic external result after success."""
    result_reference: weakref.ReferenceType[object] | None = None

    class CyclicResult:
        """Mimic a Statsmodels result cycle retaining numerical workspace."""

        def __init__(self) -> None:
            self.params = np.array([0.2, -0.1])
            self.converged = True
            self.fit_history = {"iteration": 3}
            self.llf = -4.0
            self.retained_workspace = np.ones((32, 32))
            self.self_reference = self

    class FakeModel:
        """Return one cyclic result and retain only a weak test reference."""

        def fit(self, **kwargs):
            del kwargs
            nonlocal result_reference
            result = CyclicResult()
            result_reference = weakref.ref(result)
            return result

    monkeypatch.setattr(poisson.sm, "GLM", lambda *args, **kwargs: FakeModel())
    gc.collect()
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        poisson.fit_poisson_target(
            _design(np.arange(5)), np.array([1, 2, 1, 3, 2])
        )
        assert result_reference is not None
        assert result_reference() is None
    finally:
        if collection_was_enabled:
            gc.enable()
        gc.collect()


def test_poisson_fit_runs_cleanup_after_known_fit_error(monkeypatch) -> None:
    """A recognized estimator exception still crosses the cleanup boundary."""
    cleanup_calls: list[str] = []

    class ErrorModel:
        """Raise a recognized error from the external estimator."""

        def fit(self, **kwargs):
            del kwargs
            raise ValueError("bad data")

    monkeypatch.setattr(poisson.sm, "GLM", lambda *args, **kwargs: ErrorModel())
    monkeypatch.setattr(gc, "collect", lambda: cleanup_calls.append("collect"))

    with pytest.raises(poisson.PoissonFitUnavailable):
        poisson.fit_poisson_target(
            _design(np.arange(5)), np.array([0, 1, 2, 1, 3])
        )
    assert cleanup_calls == ["collect"]


def test_seeded_poisson_fit_returns_finite_positive_means() -> None:
    """A seeded identifiable count model yields finite target-wise diagnostics."""
    rng = np.random.default_rng(91)
    feature = rng.normal(size=400)
    design = _design(feature)
    counts = rng.poisson(np.exp(0.4 + 0.3 * feature))

    fit = poisson.fit_poisson_target(design, counts)
    expected_counts = poisson.predict_poisson_mean(design, fit.parameters)

    assert fit.converged is True
    assert isinstance(fit.iterations, int)
    assert fit.iterations > 0
    assert fit.parameters.shape == (2,)
    assert np.all(np.isfinite(expected_counts))
    assert np.all(expected_counts > 0.0)


def test_poisson_fit_rejects_missing_intercept_and_constant_response() -> None:
    """Strict fits require one explicit intercept and a varying training target."""
    with pytest.raises(ValueError, match="explicit intercept"):
        poisson.fit_poisson_target(
            np.arange(10, dtype=float).reshape(5, 2), np.arange(5)
        )
    with pytest.raises(poisson.PoissonFitUnavailable) as caught:
        poisson.fit_poisson_target(_design(np.arange(5)), np.ones(5, dtype=int))
    assert caught.value.reason == "constant_training_target"


@pytest.mark.parametrize("warning_type", (ConvergenceWarning, PerfectSeparationWarning))
def test_statsmodels_fit_warnings_are_local_unavailability(monkeypatch, warning_type) -> None:
    """Known convergence warnings become stable target-local failure reasons."""

    class WarningModel:
        """Emit one known statsmodels warning during ordinary fitting."""

        def fit(self, **kwargs):
            warnings.warn("known fit warning", warning_type)
            return SimpleNamespace(
                params=np.array([0.0, 0.0]),
                converged=True,
                fit_history={"iteration": 1},
                llf=-1.0,
            )

    monkeypatch.setattr(poisson.sm, "GLM", lambda *args, **kwargs: WarningModel())
    with pytest.raises(poisson.PoissonFitUnavailable) as caught:
        poisson.fit_poisson_target(_design(np.arange(5)), np.array([0, 1, 2, 1, 3]))
    expected = (
        "poisson_nonconverged"
        if warning_type is ConvergenceWarning
        else "poisson_fit_error"
    )
    assert caught.value.reason == expected


@pytest.mark.parametrize(
    "error",
    (
        PerfectSeparationError("separation"),
        FloatingPointError("overflow"),
        np.linalg.LinAlgError("singular"),
        ValueError("bad data"),
    ),
)
def test_known_statsmodels_fit_errors_are_local(monkeypatch, error: Exception) -> None:
    """Known numerical/data exceptions do not masquerade as programming failures."""

    class ErrorModel:
        """Raise a recognized exception only from the statsmodels fit call."""

        def fit(self, **kwargs):
            raise error

    monkeypatch.setattr(poisson.sm, "GLM", lambda *args, **kwargs: ErrorModel())
    with pytest.raises(poisson.PoissonFitUnavailable) as caught:
        poisson.fit_poisson_target(_design(np.arange(5)), np.array([0, 1, 2, 1, 3]))
    assert caught.value.reason == "poisson_fit_error"


@pytest.mark.parametrize(
    ("params", "converged", "reason"),
    (
        (np.array([0.0, 0.0]), False, "poisson_nonconverged"),
        (np.array([0.0, np.nan]), True, "poisson_fit_error"),
    ),
)
def test_returned_fit_diagnostics_are_validated(
    monkeypatch, params: np.ndarray, converged: bool, reason: str
) -> None:
    """Returned nonconvergence and nonfinite parameters have stable reasons."""
    result = SimpleNamespace(
        params=params,
        converged=converged,
        fit_history={"iteration": 7},
        llf=-1.0,
    )
    model = SimpleNamespace(fit=lambda **kwargs: result)
    monkeypatch.setattr(poisson.sm, "GLM", lambda *args, **kwargs: model)

    with pytest.raises(poisson.PoissonFitUnavailable) as caught:
        poisson.fit_poisson_target(_design(np.arange(5)), np.array([0, 1, 2, 1, 3]))
    assert caught.value.reason == reason


def test_nonfinite_or_nonpositive_predicted_means_are_local() -> None:
    """The log-link prediction boundary rejects unusable expected counts."""
    with pytest.raises(poisson.PoissonFitUnavailable) as caught:
        poisson.predict_poisson_mean(np.array([[1.0, 1e308]]), np.array([0.0, 1.0]))
    assert caught.value.reason == "nonpositive_poisson_mean"
    with pytest.raises(poisson.PoissonFitUnavailable) as caught:
        poisson.predict_poisson_mean(np.array([[1.0, 1e308]]), np.array([0.0, -1.0]))
    assert caught.value.reason == "nonpositive_poisson_mean"


def test_poisson_deviance_matches_zero_count_hand_calculation() -> None:
    """Zero observations contribute twice their expected count."""
    observed = np.array([0.0, 2.0, 4.0])
    expected = np.array([0.5, 1.5, 5.0])
    hand = 2.0 * (
        0.5
        + (2.0 * np.log(2.0 / 1.5) - (2.0 - 1.5))
        + (4.0 * np.log(4.0 / 5.0) - (4.0 - 5.0))
    )

    assert poisson.poisson_deviance(observed, expected) == pytest.approx(hand)


def test_held_out_deviance_scores_match_one_shared_null() -> None:
    """Restricted and full scores share the held-out count-mean denominator."""
    observed = np.array([0.0, 1.0, 4.0])
    restricted = np.array([1.0, 1.0, 2.0])
    full = np.array([0.5, 1.5, 3.0])

    scores = poisson.score_poisson_predictions(observed, restricted, full)
    null_expected = np.full(observed.shape, observed.mean())
    null_deviance = poisson.poisson_deviance(observed, null_expected)

    assert scores.null_deviance == pytest.approx(null_deviance)
    assert scores.deviance_restricted == pytest.approx(
        poisson.poisson_deviance(observed, restricted)
    )
    assert scores.deviance_full == pytest.approx(poisson.poisson_deviance(observed, full))
    assert scores.deviance_explained_restricted == pytest.approx(
        1.0 - scores.deviance_restricted / null_deviance
    )
    assert scores.deviance_explained_full == pytest.approx(
        1.0 - scores.deviance_full / null_deviance
    )
    assert scores.delta_deviance_explained == pytest.approx(
        scores.deviance_explained_full - scores.deviance_explained_restricted
    )


def test_deviance_scores_preserve_negative_values_and_increments() -> None:
    """Bad predictions remain negative scientific scores rather than being clipped."""
    scores = poisson.score_poisson_predictions(
        np.array([0.0, 1.0, 4.0]),
        np.array([0.5, 1.0, 4.0]),
        np.array([8.0, 8.0, 8.0]),
    )

    assert scores.deviance_explained_full < 0.0
    assert scores.delta_deviance_explained < 0.0


def test_zero_null_deviance_only_disables_normalized_metrics() -> None:
    """Constant held-out counts retain deviance and MSE diagnostics."""
    scores = poisson.score_poisson_predictions(
        np.ones(4), np.full(4, 0.5), np.full(4, 1.5)
    )

    assert scores.normalized_available is False
    assert np.isnan(scores.deviance_explained_restricted)
    assert np.isnan(scores.deviance_explained_full)
    assert np.isnan(scores.delta_deviance_explained)
    assert np.isfinite(scores.deviance_restricted)
    assert scores.mse_restricted == 0.25
    assert scores.mse_full == 0.25


def test_deviance_pair_rejects_different_evaluation_rows() -> None:
    """Restricted/full models cannot use different held-out rows or null denominators."""
    with pytest.raises(ValueError, match="same one-dimensional shape"):
        poisson.score_poisson_predictions(
            np.array([0, 1, 2]), np.ones(3), np.ones(2)
        )


def test_mse_comparison_is_exact_paired_join_for_both_models() -> None:
    """Derived fold and target comparisons preserve restricted/full pairing."""
    fold_comparison, target_comparison = poisson.derive_mse_comparison(_fold_rows())

    assert len(fold_comparison) == 10
    assert set(fold_comparison["comparison"]) == {"restricted", "full"}
    assert set(fold_comparison["mse_advantage_poisson"]) == {2.0}
    assert len(target_comparison) == 2
    assert set(target_comparison["status"]) == {"ok"}
    assert set(target_comparison["valid_folds"]) == {5}
    assert set(target_comparison["mean_mse_advantage_poisson"]) == {2.0}


def test_count_mse_uses_raw_ols_and_poisson_predictions_without_adjustment() -> None:
    """Count-MSE comparison neither clips OLS values nor rounds Poisson means."""
    observed = np.array([[0.0], [2.0]])
    raw_ols = np.array([[-1.25], [2.75]])
    raw_poisson = np.array([0.6, 1.4])

    ols_scores = linear.score_ols_predictions(observed, raw_ols, raw_ols)
    poisson_scores = poisson.score_poisson_predictions(
        observed[:, 0], raw_poisson, raw_poisson
    )

    assert ols_scores.mse_full[0] == pytest.approx((1.25**2 + 0.75**2) / 2.0)
    assert poisson_scores.mse_full == pytest.approx((0.6**2 + 0.6**2) / 2.0)
    assert ols_scores.mse_full[0] - poisson_scores.mse_full == pytest.approx(
        0.7025
    )


def test_mse_comparison_requires_units_and_identical_row_hashes() -> None:
    """PC targets and nonidentical OLS/Poisson row sets are not comparable."""
    pc_rows = _fold_rows()
    pc_rows["representation"] = "pcs"
    with pytest.raises(ValueError, match="unit"):
        poisson.derive_mse_comparison(pc_rows)

    mismatched_rows = _fold_rows()
    mismatch = (
        (mismatched_rows["model_family"] == "poisson")
        & (mismatched_rows["fold_id"] == 0)
    )
    mismatched_rows.loc[mismatch, "test_row_set_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="row-set fingerprints"):
        poisson.derive_mse_comparison(mismatched_rows)


def test_mse_target_comparison_requires_all_five_defined_pairs() -> None:
    """One missing model/fold metric makes only that model pairing incomplete."""
    fold_comparison, target_comparison = poisson.derive_mse_comparison(
        _fold_rows(missing_poisson_fold=4)
    )

    restricted = target_comparison.loc[
        target_comparison["comparison"] == "restricted"
    ].iloc[0]
    full = target_comparison.loc[target_comparison["comparison"] == "full"].iloc[0]
    assert restricted["status"] == "ok"
    assert restricted["valid_folds"] == 5
    assert full["status"] == "incomplete_folds"
    assert full["reason"] == "incomplete_requested_folds"
    assert full["valid_folds"] == 4
    assert pd.isna(full["mean_mse_advantage_poisson"])
    assert len(fold_comparison.loc[fold_comparison["comparison"] == "full"]) == 4
