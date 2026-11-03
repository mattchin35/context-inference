"""Numeric ridge/OLS and grouped pooled-prediction contracts on fixed responses."""

import numpy as np
import pytest

from src.neural_analysis.encoding import modeling
from src.neural_analysis.encoding.modeling import fit_ridge_responses as original_ridge_fit
from src.neural_analysis.encoding.modeling import predict_responses as original_predict
from src.neural_analysis.encoding.predictors import encode_context


@pytest.fixture
def model_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native X(15,1), Y(15,2), folds(15,) with unequal grouped sizes.

    No inputs. X is dimensionless signed context. Y has arbitrary fixed PC-score
    units; folds 0..4 are dimensionless block assignments of sizes 1,2,3,4,5.
    Values are deterministic, with no random draws or source loading.
    """
    signed = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1], dtype=float)
    residual = np.linspace(-1.0, 1.0, len(signed))
    responses = np.column_stack((3 + 2 * signed + residual, -4 - 0.7 * signed + residual**2))
    return signed[:, None], responses, np.repeat(np.arange(5), [1, 2, 3, 4, 5])


def fail_smallest_training_pool(
    predictors: np.ndarray, responses: np.ndarray, alpha: float = 1.0, **settings: object,
) -> object:
    """Input X(N,P)/Y(N,K)/penalty/settings; fail N=10, otherwise return real LinearFit.

    X is native dimensionless context and Y retains PC-score units. This mocks
    one fold failure while allowing the all-data reference and four folds.
    """
    if len(predictors) == 10:
        raise ValueError("synthetic fold fitting failure")
    return original_ridge_fit(predictors, responses, alpha, **settings)


def nonfinite_last_fold_prediction(predictors: np.ndarray, fit: object) -> np.ndarray:
    """Input X(N,P)/LinearFit; return PC predictions(N,K), NaN only for N=5 held-out fold."""
    values = original_predict(predictors, fit)
    return np.full_like(values, np.nan) if len(predictors) == 5 else values


def test_ridge_matches_analytic_summed_sse_with_unpenalized_intercept() -> None:
    """No inputs; native X(5,1)/PC Y(5,1) analytic fit uses summed SSE, return None."""
    predictors = np.array([[-1.0], [-1.0], [-1.0], [1.0], [1.0]])
    responses = np.array([[1.0], [3.0], [4.0], [9.0], [10.0]])
    alpha = 2.3
    fit = modeling.fit_ridge_responses(predictors, responses, alpha)
    centered_x = predictors - predictors.mean(axis=0)
    centered_y = responses - responses.mean(axis=0)
    expected = centered_x.T @ centered_y / (np.square(centered_x).sum() + alpha)
    intercept = responses.mean(axis=0) - predictors.mean(axis=0) @ expected

    np.testing.assert_allclose(fit.slopes, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.intercept, intercept)
    assert fit.slopes.shape == (1, 1) and fit.intercept.shape == (1,)
    assert fit.estimator == "ridge"
    assert fit.configured_alpha == fit.effective_alpha == alpha
    assert fit.penalty_mode == "same_numeric"
    predictions = modeling.predict_responses(predictors, fit)
    residual = responses - predictions
    np.testing.assert_allclose(centered_x.T @ residual, alpha * fit.slopes, atol=1e-12)
    np.testing.assert_allclose(residual.sum(axis=0), 0, atol=1e-12)


def test_intercept_offset_does_not_change_ridge_slopes(model_data: tuple) -> None:
    """Input X(15,1)/Y(15,2)/folds tuple; PC offsets affect intercept only, return None."""
    predictors, responses, _ = model_data
    offset = np.array([1000.0, -50.0])
    first = modeling.fit_ridge_responses(predictors, responses, 4.0)
    shifted = modeling.fit_ridge_responses(predictors, responses + offset, 4.0)
    np.testing.assert_allclose(first.slopes, shifted.slopes, atol=1e-12)
    np.testing.assert_allclose(shifted.intercept, first.intercept + offset)


@pytest.mark.parametrize("estimator", ["ridge", "ols"])
def test_matrix_response_fitting_matches_independent_columns(model_data: tuple, estimator: str) -> None:
    """Input numeric fixture/estimator string; matrix slopes(P,K) match K scalar fits, return None."""
    predictors, responses, _ = model_data
    fit_function = modeling.fit_ridge_responses if estimator == "ridge" else modeling.fit_ols_responses
    combined = fit_function(predictors, responses)
    for column in range(responses.shape[1]):
        separate = fit_function(predictors, responses[:, column:column + 1])
        np.testing.assert_allclose(combined.slopes[:, column], separate.slopes[:, 0])
        np.testing.assert_allclose(combined.intercept[column], separate.intercept[0])
    assert modeling.predict_responses(predictors, combined).shape == responses.shape


def test_ols_centered_minimum_norm_indicator_convention_and_equivalent_predictions(model_data: tuple) -> None:
    """Input native signed X/PC Y; redundant (N,2) OLS centers slopes, return None."""
    signed, responses, _ = model_data
    indicators, _, _ = encode_context((signed[:, 0] + 1) / 2, "separate_context")
    signed_fit = modeling.fit_ols_responses(signed, responses)
    separate_fit = modeling.fit_ols_responses(indicators, responses)
    centered_x = indicators - indicators.mean(axis=0)
    centered_y = responses - responses.mean(axis=0)
    expected, _, _, _ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
    np.testing.assert_allclose(separate_fit.slopes, expected, atol=1e-12)
    np.testing.assert_allclose(separate_fit.slopes[0], -separate_fit.slopes[1], atol=1e-12)
    np.testing.assert_allclose(separate_fit.intercept,
                               responses.mean(axis=0) - indicators.mean(axis=0) @ expected)
    np.testing.assert_allclose(modeling.predict_responses(signed, signed_fit),
                               modeling.predict_responses(indicators, separate_fit), atol=1e-12)
    assert separate_fit.estimator == "ols"
    assert separate_fit.effective_alpha is None and separate_fit.configured_alpha is None
    assert separate_fit.penalty_mode is None
    assert "minimum" in separate_fit.coefficient_convention.lower()


def test_matched_context_ridge_maps_indicator_alpha_to_half(model_data: tuple) -> None:
    """Input native X/PC Y; alpha and alpha/2 yield equal predictions, return None."""
    signed, responses, _ = model_data
    indicators, _, _ = encode_context((signed[:, 0] + 1) / 2, "separate_context")
    signed_fit = modeling.fit_ridge_responses(signed, responses, 7.0,
        encoding="signed_context", penalty_mode="matched_context")
    separate_fit = modeling.fit_ridge_responses(indicators, responses, 7.0,
        encoding="separate_context", penalty_mode="matched_context")
    assert signed_fit.configured_alpha == separate_fit.configured_alpha == 7.0
    assert signed_fit.effective_alpha == 7.0 and separate_fit.effective_alpha == 3.5
    assert separate_fit.penalty_mode == "matched_context"
    np.testing.assert_allclose(modeling.predict_responses(signed, signed_fit),
                               modeling.predict_responses(indicators, separate_fit), atol=1e-12)
    np.testing.assert_allclose(separate_fit.slopes[0], signed_fit.slopes[0], atol=1e-12)
    np.testing.assert_allclose(separate_fit.slopes[1], -signed_fit.slopes[0], atol=1e-12)


def test_same_numeric_ridge_preserves_encoding_dependent_shrinkage(model_data: tuple) -> None:
    """Input native X/PC Y; same summed-SSE alpha intentionally differs by encoding, return None."""
    signed, responses, _ = model_data
    indicators, _, _ = encode_context((signed[:, 0] + 1) / 2, "separate_context")
    signed_fit = modeling.fit_ridge_responses(signed, responses, 7.0)
    separate_fit = modeling.fit_ridge_responses(indicators, responses, 7.0, encoding="separate_context")
    assert signed_fit.effective_alpha == separate_fit.effective_alpha == 7.0
    assert np.all(np.abs(separate_fit.slopes[0]) < np.abs(signed_fit.slopes[0]))
    assert not np.allclose(modeling.predict_responses(signed, signed_fit),
                           modeling.predict_responses(indicators, separate_fit))


def test_standardized_display_uses_only_supplied_reference_rows(model_data: tuple) -> None:
    """Input X/Y/folds; raw slopes remain unchanged while display uses eligible-row SD, return None."""
    predictors, responses, _ = model_data
    predictors, responses = predictors[:11], responses[:11]
    fit = modeling.fit_ridge_responses(predictors, responses)
    raw = fit.slopes.copy()
    expected = raw * predictors.std(axis=0)[:, None] / responses.std(axis=0)[None, :]
    displayed = modeling.standardize_coefficients(fit, predictors, responses)
    np.testing.assert_allclose(displayed, expected)
    np.testing.assert_array_equal(fit.slopes, raw)
    assert displayed.shape == fit.slopes.shape


@pytest.mark.parametrize("estimator", ["ridge", "ols"])
def test_one_context_class_and_constant_predictors_are_valid(estimator: str) -> None:
    """Input estimator; (8,2) one-class indicators retain intercept and zero slopes, return None."""
    predictors = np.tile([1.0, 0.0], (8, 1))
    responses = np.arange(16, dtype=float).reshape(8, 2)
    fit = (modeling.fit_ridge_responses(predictors, responses, encoding="separate_context")
           if estimator == "ridge" else modeling.fit_ols_responses(predictors, responses))
    np.testing.assert_allclose(fit.slopes, 0, atol=1e-12)
    np.testing.assert_allclose(fit.intercept, responses.mean(axis=0))
    np.testing.assert_allclose(modeling.standardize_coefficients(fit, predictors, responses), 0)
    assert fit.limitations
    assert any("constant" in reason.lower() or "class" in reason.lower() for reason in fit.limitations)


@pytest.mark.parametrize("constant", [0.0, 0.1])
def test_constant_response_has_finite_fit_mse_and_unavailable_display_r2(
    model_data: tuple, constant: float,
) -> None:
    """Input numeric fixture/PC constant; per-response validity survives mean roundoff, return None."""
    predictors, responses, folds = model_data
    responses = np.column_stack((responses[:, 0], np.full(len(responses), constant)))
    result = modeling.evaluate_block_cv(predictors, responses, folds)
    display = modeling.standardize_coefficients(result["reference_fit"], predictors, responses)
    assert result["cv_available"]
    assert np.isfinite(result["mse"]).all()
    assert np.isfinite(result["r2"][0]) and np.isnan(result["r2"][1])
    assert result["r2_reason"][1]
    assert np.isfinite(display[:, 0]).all() and np.isnan(display[:, 1]).all()
    np.testing.assert_allclose(result["reference_fit"].slopes[:, 1], 0, atol=1e-12)
    np.testing.assert_allclose(result["reference_fit"].intercept[1], constant, atol=1e-12)


@pytest.mark.parametrize("estimator", ["ridge", "ols"])
def test_cv_uses_supplied_block_map_and_saves_one_oof_prediction_per_bin(
    model_data: tuple, estimator: str,
) -> None:
    """Input X(15,1)/Y(15,2)/folds(15,) and estimator; preserve row order, return None."""
    predictors, responses, folds = model_data
    original = tuple(array.copy() for array in model_data)
    result = modeling.evaluate_block_cv(predictors, responses, folds, estimator=estimator)
    assert result["cv_available"] and result["cv_reason"] == ""
    assert result["held_out_predictions"].shape == responses.shape
    assert result["reference_predictions"].shape == responses.shape
    assert result["fold_slopes"].shape == (5, 1, 2)
    assert result["fold_intercepts"].shape == (5, 2)
    np.testing.assert_array_equal(result["fold_valid"], np.ones(5, dtype=bool))
    np.testing.assert_array_equal(result["fold_test_counts"], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result["fold_train_counts"], [14, 13, 12, 11, 10])
    assert np.isnan(result["fold_r2"][0]).all()  # One held-out row has zero SST.
    for fold in range(5):
        fit = (modeling.fit_ridge_responses(predictors[folds != fold], responses[folds != fold])
               if estimator == "ridge" else modeling.fit_ols_responses(predictors[folds != fold], responses[folds != fold]))
        expected = modeling.predict_responses(predictors[folds == fold], fit)
        np.testing.assert_allclose(result["held_out_predictions"][folds == fold], expected)
        np.testing.assert_allclose(result["fold_slopes"][fold], fit.slopes)
        np.testing.assert_allclose(result["fold_intercepts"][fold], fit.intercept)
        np.testing.assert_allclose(result["fold_fits"][fold].slopes, fit.slopes)
    np.testing.assert_allclose(result["reference_predictions"],
                               modeling.predict_responses(predictors, result["reference_fit"]))
    assert not np.allclose(result["held_out_predictions"], result["reference_predictions"])
    assert "reference" in result["reference_scope"]
    assert "held" in result["score_scope"] and "PCA" in result["score_scope"]
    for actual, before in zip(model_data, original):
        np.testing.assert_array_equal(actual, before)


def test_cv_pools_unequal_folds_and_keeps_negative_r_squared() -> None:
    """No inputs; unequal groups/PC Y(15,1) require pooled SSE/SST, return None."""
    predictors = np.ones((15, 1))
    responses = np.arange(15, dtype=float)[:, None] ** 2
    folds = np.repeat(np.arange(5), [1, 2, 3, 4, 5])
    result = modeling.evaluate_block_cv(predictors, responses, folds)
    expected_prediction = np.empty_like(responses)
    for fold in range(5):
        expected_prediction[folds == fold] = responses[folds != fold].mean(axis=0)
    squared_error = (responses - expected_prediction) ** 2
    mse = squared_error.mean(axis=0)
    r2 = 1 - squared_error.sum(axis=0) / np.square(responses - responses.mean(axis=0)).sum(axis=0)
    np.testing.assert_allclose(result["held_out_predictions"], expected_prediction)
    np.testing.assert_allclose(result["mse"], mse)
    np.testing.assert_allclose(result["r2"], r2)
    assert result["r2"][0] < 0
    assert not np.allclose(result["mse"], result["fold_mse"].mean(axis=0))


@pytest.mark.parametrize("estimator,penalty_mode", [("ols", "same_numeric"), ("ridge", "matched_context")])
def test_cv_propagates_encoding_and_penalty_settings_to_every_fit(
    model_data: tuple, estimator: str, penalty_mode: str,
) -> None:
    """Input native X/PC Y/folds plus settings; equivalent codes match all predictions, return None."""
    signed, responses, folds = model_data
    indicators, _, _ = encode_context((signed[:, 0] + 1) / 2, "separate_context")
    common = {"estimator": estimator, "penalty_mode": penalty_mode, "alpha": 7.0}
    signed_result = modeling.evaluate_block_cv(signed, responses, folds,
                                              encoding="signed_context", **common)
    separate_result = modeling.evaluate_block_cv(indicators, responses, folds,
                                                encoding="separate_context", **common)
    assert signed_result["cv_available"] and separate_result["cv_available"]
    for key in ("reference_predictions", "held_out_predictions", "mse", "r2"):
        np.testing.assert_allclose(signed_result[key], separate_result[key], atol=1e-12)
    signed_fits = (signed_result["reference_fit"], *signed_result["fold_fits"])
    separate_fits = (separate_result["reference_fit"], *separate_result["fold_fits"])
    for signed_fit, separate_fit in zip(signed_fits, separate_fits):
        assert signed_fit.estimator == separate_fit.estimator == estimator
        np.testing.assert_allclose(separate_fit.slopes[0], signed_fit.slopes[0], atol=1e-12)
        np.testing.assert_allclose(separate_fit.slopes[1], -signed_fit.slopes[0], atol=1e-12)
        if estimator == "ridge":
            assert signed_fit.configured_alpha == separate_fit.configured_alpha == 7.0
            assert signed_fit.effective_alpha == 7.0 and separate_fit.effective_alpha == 3.5
            assert signed_fit.penalty_mode == separate_fit.penalty_mode == "matched_context"
        else:
            assert signed_fit.effective_alpha is separate_fit.effective_alpha is None
            assert signed_fit.penalty_mode is separate_fit.penalty_mode is None
    assert separate_result["fold_slopes"].shape == (5, 2, 2)
    for fold, fit in enumerate(separate_result["fold_fits"]):
        np.testing.assert_allclose(separate_result["fold_slopes"][fold], fit.slopes)


@pytest.mark.parametrize("missing", ["one_fold", "all_folds"])
def test_unusable_fold_map_preserves_reference_without_primary_partial_scores(
    model_data: tuple, missing: str,
) -> None:
    """Input numeric fixture/missing name; incomplete five-fold predictions unavailable, return None."""
    predictors, responses, folds = model_data
    folds = np.where(folds == 4, 3, folds) if missing == "one_fold" else np.full(len(folds), -1)
    result = modeling.evaluate_block_cv(predictors, responses, folds)
    assert not result["cv_available"] and result["cv_reason"]
    assert np.isfinite(result["reference_predictions"]).all()
    assert np.isfinite(result["reference_fit"].slopes).all()
    assert np.isnan(result["mse"]).all() and np.isnan(result["r2"]).all()
    assert not result["fold_valid"][4]
    assert result["fold_reasons"][4]


def test_failed_fold_preserves_reference_and_completed_fold_results(
    model_data: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input numeric fixture/patch; fold failure keeps completed predictions/coefficients, return None."""
    predictors, responses, folds = model_data
    monkeypatch.setattr(modeling, "fit_ridge_responses", fail_smallest_training_pool)
    result = modeling.evaluate_block_cv(predictors, responses, folds)
    assert not result["cv_available"]
    assert "synthetic" in result["fold_reasons"][4]
    assert np.isfinite(result["reference_predictions"]).all()
    assert np.isfinite(result["held_out_predictions"][folds != 4]).all()
    assert np.isnan(result["held_out_predictions"][folds == 4]).all()
    assert np.isfinite(result["fold_slopes"][:4]).all()
    assert np.isnan(result["fold_slopes"][4]).all()
    assert result["fold_fits"][4] is None
    assert np.isnan(result["mse"]).all()


def test_nonfinite_fold_prediction_cannot_produce_primary_score(
    model_data: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input numeric fixture/patch; nonfinite held-out PC output invalidates fold, return None."""
    predictors, responses, folds = model_data
    monkeypatch.setattr(modeling, "predict_responses", nonfinite_last_fold_prediction)
    result = modeling.evaluate_block_cv(predictors, responses, folds)
    assert not result["cv_available"] and not result["fold_valid"][4]
    assert result["fold_reasons"][4]
    assert np.isfinite(result["reference_predictions"]).all()
    assert np.isfinite(result["held_out_predictions"][folds != 4]).all()
    assert np.isnan(result["mse"]).all() and np.isnan(result["r2"]).all()


@pytest.mark.parametrize("field", ["predictors", "responses"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_invalid_before_fit_or_cv(model_data: tuple, field: str, bad: float) -> None:
    """Input X/Y fixture, field name, nonfinite scalar; native finite contract fails, return None."""
    predictors, responses, folds = (array.copy() for array in model_data)
    (predictors if field == "predictors" else responses)[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        modeling.fit_ridge_responses(predictors, responses)
    with pytest.raises(ValueError, match="finite"):
        modeling.fit_ols_responses(predictors, responses)
    with pytest.raises(ValueError, match="finite"):
        modeling.evaluate_block_cv(predictors, responses, folds)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, True])
def test_ridge_rejects_invalid_penalty(model_data: tuple, alpha: float) -> None:
    """Input native fixture/unitless penalty; invalid fixed alpha raises, return None."""
    predictors, responses, _ = model_data
    with pytest.raises(ValueError, match="alpha"):
        modeling.fit_ridge_responses(predictors, responses, alpha)


def test_predictor_column_count_must_match_explicit_context_encoding(model_data: tuple) -> None:
    """Input native fixture; two columns cannot silently claim signed encoding, return None."""
    predictors, responses, _ = model_data
    indicators, _, _ = encode_context((predictors[:, 0] + 1) / 2, "separate_context")
    with pytest.raises(ValueError, match="encoding|column|shape"):
        modeling.fit_ridge_responses(indicators, responses)


@pytest.mark.parametrize("settings", [
    {"encoding": "current_action"}, {"penalty_mode": "automatic"}, {"estimator": "poisson"},
])
def test_unsupported_model_settings_are_not_silently_substituted(model_data: tuple, settings: dict) -> None:
    """Input numeric fixture/unitless model setting mapping; unknown controls fail, return None."""
    predictors, responses, folds = model_data
    with pytest.raises(ValueError, match="encoding|penalty|estimator"):
        modeling.evaluate_block_cv(predictors, responses, folds, **settings)


@pytest.mark.parametrize("failure", ["vector_y", "different_rows", "empty", "bad_fold_shape", "fractional_fold", "unknown_fold"])
def test_numeric_array_contracts_reject_implicit_axis_changes(model_data: tuple, failure: str) -> None:
    """Input numeric fixture/failure; require X(N,P),Y(N,K),folds(N,), return None."""
    predictors, responses, folds = model_data
    if failure == "vector_y":
        responses = responses[:, 0]
    elif failure == "different_rows":
        responses = responses[:-1]
    elif failure == "empty":
        predictors, responses, folds = predictors[:0], responses[:0], folds[:0]
    elif failure == "bad_fold_shape":
        folds = folds[:, None]
    elif failure == "fractional_fold":
        folds = folds.astype(float)
        folds[0] = 0.5
    else:
        folds = folds.copy()
        folds[0] = 5
    with pytest.raises(ValueError, match="shape|rows|empty|fold|two-dimensional"):
        modeling.evaluate_block_cv(predictors, responses, folds)
