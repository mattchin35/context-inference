"""Weighted model oracles and held-out availability on independent regions."""

from dataclasses import replace
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.neural_analysis.task_decoding.session_prediction import modeling
from src.neural_analysis.task_decoding.session_prediction.preparation import prepare_prediction_inputs


def fit_region(prepared, role="PFC"):
    """Fit prepared N-choice targets to regional (W,B,K) original scores; return records."""
    region = prepared.timeline.regions[role]
    return modeling.fit_region_prediction_models(
        region.scores, region.choice_positions, prepared.target_values,
        prepared.target_valid, prepared.baseline_valid, prepared.correct_rewarded,
        prepared.fold_ids, prepared.training_bin_mask, prepared.target_families,
    )


def test_weighted_ridge_matches_independent_normal_equation(prediction_inputs):
    """Compare native-unit affine ridge to a weighted, intercept-exempt matrix oracle."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0])
    region = prepared.timeline.regions["PFC"]
    result = fit_region(prepared)
    for fold in range(5):
        target = 2
        mask = result.training_choice_mask[target, fold]
        assert not np.any(mask & (prepared.fold_ids == fold))
        assert np.all(prepared.correct_rewarded[mask])
        local = mask[region.choice_positions]
        matrix = region.scores[local][:, prepared.training_bin_mask, :].reshape(-1, 3)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        labels = np.repeat(prepared.target_values[mask, target], 4)
        row_weights = np.full(len(matrix), 1 / 4)
        penalty = np.diag([0, 1, 1, 1])
        oracle = np.linalg.solve(design.T @ (row_weights[:, None] * design) + penalty,
                                 design.T @ (row_weights * labels))
        np.testing.assert_allclose(result.intercepts[target, fold], oracle[0], atol=1e-12)
        np.testing.assert_allclose(result.coefficients[target, fold], oracle[1:], atol=1e-12)
        assert result.training_row_count[target, fold] == 4 * mask.sum()
        held = prepared.fold_ids[region.choice_positions] == fold
        np.testing.assert_allclose(result.predictions[held, :, target],
                                   oracle[0] + region.scores[held] @ oracle[1:])


def test_binary_orientation_matches_objective_gradient_and_identical_labels(prediction_inputs):
    """Class 1 means left; fitted logistic weights satisfy the selected weighted loss."""
    base, trials, timeline = prediction_inputs
    prepared = prepare_prediction_inputs(trials, timeline, base)
    result = fit_region(prepared)
    np.testing.assert_allclose(result.predictions[:, :, 0], result.predictions[:, :, 1])
    region = timeline.regions["PFC"]
    for fold in range(5):
        mask = result.training_choice_mask[1, fold]
        x = region.scores[mask[region.choice_positions]][:, prepared.training_bin_mask].reshape(-1, 3)
        y = np.repeat(prepared.target_values[mask, 1], 4)
        beta, intercept = result.coefficients[1, fold], result.intercepts[1, fold]
        probability = 1 / (1 + np.exp(-(x @ beta + intercept)))
        np.testing.assert_allclose(x.T @ ((probability - y) / 4) + beta, 0, atol=0.003)
        assert abs(np.sum((probability - y) / 4)) < 0.003
    assert np.all((result.predictions[:, :, :2] >= 0) & (result.predictions[:, :, :2] <= 1))


def test_destination_label_missingness_does_not_suppress_predictions(prediction_inputs):
    """Missing actuals and missing blocks are distinct; no -1 fold indexes the last fit."""
    base, trials, timeline = prediction_inputs
    trials.loc[100, "Qlearning_rel_value"] = np.nan
    trials.loc[103, "cur_block"] = np.nan
    prepared = prepare_prediction_inputs(trials, timeline, base)
    result = fit_region(prepared)
    assert np.isfinite(result.predictions[0, :, 2]).all()
    missing_window = np.flatnonzero(timeline.choices.row_position.to_numpy() == 3)[0]
    assert np.isnan(result.predictions[missing_window]).all()
    assert np.all(result.prediction_reason[missing_window] == "missing_block")
    assert not result.training_choice_mask[:, :, prepared.fold_ids < 0].any()


def test_empty_target_and_single_class_are_local_failures_but_constants_fit(prediction_inputs):
    """An unfit target leaves other targets/regions usable; constant ridge stays valid."""
    base, trials, timeline = prediction_inputs
    trials["HMM_rel_value_logodds"] = np.nan
    trials["Qlearning_rel_value"] = 5.0
    trials["state_int"] = 1
    result = fit_region(prepare_prediction_inputs(trials, timeline, base))
    assert np.all(result.fit_status[0] == "unavailable")
    assert np.all(result.fit_reason[0] == "single_class_training")
    assert np.all(result.fit_reason[4] == "target_unavailable_no_values")
    np.testing.assert_allclose(result.predictions[:, :, 2], 5)
    np.testing.assert_allclose(result.coefficients[2], 0, atol=1e-12)
    assert np.all(result.fit_status[1] == "ok")
    prepared = prepare_prediction_inputs(trials, timeline, base)
    empty = replace(prepared, correct_rewarded=np.zeros_like(prepared.correct_rewarded))
    assert np.all(fit_region(empty).fit_status == "unavailable")


def test_nonconvergence_invalidates_model_without_fallback(prediction_inputs, monkeypatch):
    """A convergence warning makes binary models unavailable while ridge fits continue."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0])

    def fail_to_converge(self, *args, **kwargs):
        """Estimator test seam: emit the real sklearn warning; return the estimator."""
        warnings.warn("iteration limit", ConvergenceWarning)
        return self

    monkeypatch.setattr(modeling.LogisticRegression, "fit", fail_to_converge)
    result = fit_region(prepared)
    assert np.all(result.fit_reason[:2] == "nonconvergence")
    assert np.all(result.fit_status[2:] == "ok")


def test_regional_coverage_and_source_arrays_remain_independent(prediction_inputs):
    """Different W counts remain independent; fitting does not mutate saved PCA data."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0])
    original = {role: region.scores.copy() for role, region in prepared.timeline.regions.items()}
    for role, scores in original.items():
        result = fit_region(prepared, role)
        assert result.predictions.shape == (*scores.shape[:2], 7)
        np.testing.assert_array_equal(prepared.timeline.regions[role].scores, scores)


def test_invalid_structural_inputs_raise_before_model_fitting(prediction_inputs):
    """Mismatched target axes are input errors, not seventy unavailable model records."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0])
    with pytest.raises(ValueError):
        fit_region(replace(prepared, target_values=prepared.target_values[:-1]))
