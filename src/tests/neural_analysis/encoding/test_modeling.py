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
    assert fit.configured_alpha == alpha
    np.testing.assert_array_equal(fit.penalties, [alpha])
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
    assert separate_fit.configured_alpha is None
    assert not separate_fit.penalties.any()
    assert "minimum" in separate_fit.coefficient_convention.lower()






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
    fit = (modeling.fit_ridge_responses(predictors, responses)
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
    assert np.asarray(result["fold_r2_reason"]).shape == (5, 2)
    assert all(row[1] for row in result["fold_r2_reason"])
    assert result["fold_valid"].all()  # Undefined R2 does not invalidate finite fold fits/MSE.
    assert np.isfinite(result["fold_mse"]).all()
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
    assert np.asarray(result["fold_r2_reason"]).shape == (5, 2)
    assert all(result["fold_r2_reason"][0])
    assert result["fold_valid"][0]  # Finite one-row predictions still define MSE.
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
    assert "held" in result["score_scope"] and "supplied" in result["score_scope"]
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


def test_unassigned_rows_are_excluded_from_training_but_retained_in_reference(model_data: tuple) -> None:
    """Input X(15,1)/PC Y(15,2)/folds; mixed -1 rows train no fold, return None."""
    predictors, responses, original_folds = model_data
    folds = original_folds.copy()
    folds[-1] = -1
    result = modeling.evaluate_block_cv(predictors, responses, folds)
    reference = modeling.fit_ridge_responses(predictors, responses)
    np.testing.assert_allclose(result["reference_fit"].slopes, reference.slopes)
    np.testing.assert_allclose(result["reference_predictions"], modeling.predict_responses(predictors, reference))
    assert not result["cv_available"] and result["cv_reason"]
    assert np.isnan(result["held_out_predictions"][-1]).all()
    assert np.isnan(result["mse"]).all() and np.isnan(result["r2"]).all()
    np.testing.assert_array_equal(result["fold_test_counts"], [1, 2, 3, 4, 4])
    np.testing.assert_array_equal(result["fold_train_counts"], [13, 12, 11, 10, 10])
    for fold in range(5):
        train = (folds >= 0) & (folds != fold)
        fitted = modeling.fit_ridge_responses(predictors[train], responses[train])
        np.testing.assert_allclose(result["fold_fits"][fold].slopes, fitted.slopes)
        np.testing.assert_allclose(result["fold_fits"][fold].intercept, fitted.intercept)
        np.testing.assert_allclose(result["held_out_predictions"][folds == fold],
                                   modeling.predict_responses(predictors[folds == fold], fitted))


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


def joint_design(counts=(10,10,10,10,10,10), indicators=False):
    """Return independently constructed X(N,P), column names, and trial-fold IDs; binary/signed columns are dimensionless."""
    categories=("left_reward","left_omission","left_incorrect","right_reward","right_omission","right_incorrect")
    kind=np.repeat(np.arange(6),counts)
    context=np.repeat(np.array([1,1,-1,-1,-1,1])[kind],4)
    choice=np.repeat(np.array([1,1,1,-1,-1,-1])[kind],4)
    post=np.tile([0.,0.,1.,1.],len(kind))
    outcome_names=("left_reward","right_reward","left_omission","right_omission","left_incorrect","right_incorrect")
    outcomes=np.column_stack([np.repeat(np.asarray(categories)[kind]==name,4)*post for name in outcome_names])
    c=np.column_stack(((context+1)/2,(1-context)/2)) if indicators else context[:,None]
    names=(("left_context","right_context") if indicators else ("signed_context",))+("signed_choice",)+outcome_names
    return np.column_stack((c,choice,outcomes)),names,np.repeat(np.arange(len(kind))%5,4)


def penalized_oracle(x,y,penalties):
    """Input native X(N,P), Y(N,R), penalties(P,); return independent centered augmented-system slopes/intercept in Y units."""
    xc,yc=x-x.mean(0),y-y.mean(0)
    augmented=np.vstack((xc,np.diag(np.sqrt(penalties))))
    targets=np.vstack((yc,np.zeros((x.shape[1],y.shape[1]))))
    slopes=np.linalg.lstsq(augmented,targets,rcond=None)[0]
    return slopes,y.mean(0)-x.mean(0)@slopes


@pytest.mark.parametrize("indicators",[False,True])
def test_joint_feature_penalties_match_augmented_system_and_all_cv_fits(indicators):
    """Feature-weighted ridge must leave choice/outcome penalties unchanged in reference and every grouped fold."""
    x,names,folds=joint_design()
    y=np.column_stack((np.arange(len(x))%7,4+3*x[:,-6]))
    weights=np.ones(x.shape[1]); weights[:2]=.5 if indicators else 1
    result=modeling.evaluate_block_cv(x,y,folds,alpha=2.3,penalty_weights=weights)
    for fold,fit in [(-1,result["reference_fit"])]+list(enumerate(result["fold_fits"])):
        train=np.ones(len(x),dtype=bool) if fold==-1 else folds!=fold
        b,a=penalized_oracle(x[train],y[train],2.3*weights)
        np.testing.assert_allclose(fit.slopes,b,atol=1e-11)
        np.testing.assert_allclose(fit.intercept,a,atol=1e-11)
        np.testing.assert_allclose(fit.penalties,2.3*weights)
    assert result["cv_available"]


def test_joint_matched_context_changes_only_context_penalties_and_preserves_predictions():
    """Signed and redundant indicator context parameterizations agree with half context penalties, all others alpha."""
    signed,names,folds=joint_design((15,4,8,11,5,9))
    indicators,_,_=joint_design((15,4,8,11,5,9),True)
    y=np.column_stack((np.sin(np.arange(len(signed))),2+signed[:,0]+4*signed[:,2]))
    weights=np.r_[.5,.5,np.ones(7)]
    a=modeling.evaluate_block_cv(signed,y,folds,alpha=3)
    b=modeling.evaluate_block_cv(indicators,y,folds,alpha=3,penalty_weights=weights)
    for key in ("reference_predictions","held_out_predictions","mse","r2"):
        np.testing.assert_allclose(a[key],b[key],atol=1e-11)
    np.testing.assert_array_equal(b["reference_fit"].penalties[2:],np.full(7,3.))


def test_imbalanced_generic_post_response_has_ols_zero_but_expected_nonzero_ridge_contrasts():
    """Unequal native outcome counts cause ridge contrast bias; compare analytic shrinkage instead of forcing zero."""
    x,names,_=joint_design((60,6,12,40,4,8))
    post=x[:,-6:].sum(1)
    y=(10+3*post)[:,None]
    ols=modeling.fit_ols_responses(x,y)
    ridge=modeling.fit_ridge_responses(x,y)
    oc=modeling.compute_reward_contrasts(ols,x,names)
    rc=modeling.compute_reward_contrasts(ridge,x,names)
    assert oc["valid"].all() and rc["valid"].all()
    np.testing.assert_allclose(oc["values"],0,atol=1e-11)
    expected,_=penalized_oracle(x,y,np.ones(x.shape[1]))
    expected_values=np.vstack((expected[names.index("left_reward")]-expected[names.index("left_omission")],expected[names.index("right_reward")]-expected[names.index("right_omission")]))
    np.testing.assert_allclose(rc["values"],expected_values,atol=1e-11)
    assert (rc["values"]>.1).all()


def test_contrast_support_is_independent_of_finite_regularized_predictions():
    """Absent left omission permits prediction but not a supported left reward-minus-omission contrast."""
    x,names,_=joint_design((10,0,10,10,10,10))
    y=np.arange(len(x))[:,None].astype(float)
    fit=modeling.fit_ridge_responses(x,y)
    contrast=modeling.compute_reward_contrasts(fit,x,names)
    assert contrast["valid"].tolist()==[False,True]
    assert contrast["reasons"][0] and not contrast["reasons"][1]
    assert np.isfinite(contrast["values"]).all() and np.isfinite(modeling.predict_responses(x,fit)).all()
    assert not contrast["predictor_supported"][names.index("left_omission")]


@pytest.mark.parametrize("weights",[np.ones(7),np.array([0.]*8),np.full(8,np.nan),np.ones((8,1))])
def test_feature_penalty_contract_rejects_invalid_vectors(weights):
    """Weights are a finite positive (P,) feature vector, never a target alpha vector or implicit reshape."""
    x,_,_=joint_design()
    with pytest.raises(ValueError,match="penalty|weight"):
        modeling.fit_ridge_responses(x,np.ones((len(x),2)),penalty_weights=weights)


def test_unknown_estimator_is_rejected(model_data):
    """Numeric model selection is limited to ridge/OLS, without silent substitution."""
    x,y,folds=model_data
    with pytest.raises(ValueError,match="estimator"):
        modeling.evaluate_block_cv(x,y,folds,estimator="poisson")
