"""Weighted model oracles and held-out availability on independent regions."""

from dataclasses import replace
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.neural_analysis.task_decoding.session_prediction import modeling
from src.neural_analysis.task_decoding.session_prediction.preparation import prepare_prediction_inputs


def fit_region(prepared, role="PFC", *, use_training_membership=False):
    """Fit prepared N-choice targets to regional (W,B,K) original scores; return records."""
    region = prepared.timeline.regions[role]
    return modeling.fit_region_prediction_models(
        region.scores, region.choice_positions, prepared.target_values,
        prepared.target_valid, prepared.baseline_valid, prepared.correct_rewarded,
        prepared.fold_ids, prepared.training_bin_mask, prepared.target_families,
        **({"training_condition_mask": prepared.training_condition_mask}
           if use_training_membership else {}),
    )


def test_public_categorical_readout_matches_nonuniform_weight_sklearn_oracle():
    """Fit float X(36,3), binary y(36,) and positive dimensionless row weights.

    Return contract is beta(3,) in log-odds/PC units, a log-odds scalar
    intercept and an empty reason; original PC scales and inputs stay unchanged.
    """
    generator = np.random.default_rng(460)
    matrix = generator.normal(size=(36, 3)) * [0.2, 1.0, 5.0] + [1.5, -0.4, 3.0]
    labels = np.tile([0.0, 0.0, 1.0], 12)
    weights = np.tile([0.2, 0.7, 3.0], 12)
    originals = [array.copy() for array in (matrix, labels, weights)]
    oracle = LogisticRegression(
        C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
        class_weight=None, tol=1e-4, max_iter=5000,
    ).fit(matrix, labels, sample_weight=weights)

    beta, intercept, reason = modeling.fit_readout(matrix, labels, weights, "categorical")

    assert reason == ""
    assert beta.shape == (3,)
    assert isinstance(intercept, float)
    np.testing.assert_allclose(beta, oracle.coef_[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(intercept, oracle.intercept_[0], rtol=1e-12, atol=1e-12)
    for actual, original in zip((matrix, labels, weights), originals, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_public_numerical_readout_matches_weighted_normal_equation():
    """Fit float X(20,3), native-unit y(20,) and positive row weights.

    Check beta(3,) in target/PC units and scalar target-unit intercept against
    an independent alpha=1 ridge oracle whose intercept is exempt from penalty.
    """
    generator = np.random.default_rng(191)
    matrix = generator.normal(size=(20, 3)) * [0.3, 2.0, 4.0] + [2.0, -1.0, 3.0]
    labels = 4.5 + matrix @ np.array([0.8, -0.2, 1.1]) + generator.normal(0, 0.2, 20)
    weights = np.linspace(0.1, 2.0, len(matrix))
    design = np.column_stack([np.ones(len(matrix)), matrix])
    penalty = np.diag([0.0, 1.0, 1.0, 1.0])
    oracle = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights * labels),
    )

    beta, intercept, reason = modeling.fit_readout(matrix, labels, weights, "numerical")

    assert reason == ""
    assert beta.shape == (3,)
    assert isinstance(intercept, float)
    np.testing.assert_allclose(beta, oracle[1:], atol=1e-12)
    np.testing.assert_allclose(intercept, oracle[0], atol=1e-12)


@pytest.mark.parametrize("window, bin_count", [("full", 4), ("before", 2), ("after", 2)])
def test_regional_categorical_predictions_match_independent_sklearn_fit(prediction_inputs, window, bin_count):
    """Preserve regional (W,B,2) class-1 probabilities and (2,5,K) affine fits.

    Independently stack selected training bins with total weight one per trial;
    compare predictions on every held-out bin in original regional PC units.
    """
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0], training_window=window)
    for role, region in prepared.timeline.regions.items():
        result = fit_region(prepared, role)
        coverage = np.zeros(len(prepared.fold_ids), dtype=bool)
        coverage[region.choice_positions] = True
        for target in range(2):
            for fold in range(5):
                training = (coverage & prepared.baseline_valid & prepared.correct_rewarded
                            & prepared.target_valid[:, target] & (prepared.fold_ids >= 0)
                            & (prepared.fold_ids != fold))
                matrix = region.scores[training[region.choice_positions]][:, prepared.training_bin_mask]
                matrix = matrix.reshape(-1, region.scores.shape[2])
                labels = np.repeat(prepared.target_values[training, target], bin_count)
                oracle = LogisticRegression(
                    C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
                    class_weight=None, tol=1e-4, max_iter=5000,
                ).fit(matrix, labels, sample_weight=np.full(len(matrix), 1 / bin_count))
                np.testing.assert_allclose(result.coefficients[target, fold], oracle.coef_[0], atol=1e-12)
                np.testing.assert_allclose(result.intercepts[target, fold], oracle.intercept_[0], atol=1e-12)
                held = prepared.fold_ids[region.choice_positions] == fold
                held_matrix = region.scores[held].reshape(-1, region.scores.shape[2])
                expected = oracle.predict_proba(held_matrix)[:, 1].reshape(held.sum(), region.scores.shape[1])
                np.testing.assert_allclose(result.predictions[held, :, target], expected, atol=1e-12)


@pytest.mark.parametrize("window, bin_count", [("full", 4), ("before", 2), ("after", 2)])
def test_weighted_ridge_matches_independent_normal_equation(prediction_inputs, window, bin_count):
    """Compare native-unit affine ridge to a weighted, intercept-exempt matrix oracle."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0], training_window=window)
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
        labels = np.repeat(prepared.target_values[mask, target], bin_count)
        row_weights = np.full(len(matrix), 1 / bin_count)
        assert row_weights.sum() == mask.sum()
        penalty = np.diag([0, 1, 1, 1])
        oracle = np.linalg.solve(design.T @ (row_weights[:, None] * design) + penalty,
                                 design.T @ (row_weights * labels))
        np.testing.assert_allclose(result.intercepts[target, fold], oracle[0], atol=1e-12)
        np.testing.assert_allclose(result.coefficients[target, fold], oracle[1:], atol=1e-12)
        assert result.training_row_count[target, fold] == bin_count * mask.sum()
        held = prepared.fold_ids[region.choice_positions] == fold
        np.testing.assert_allclose(result.predictions[held, :, target],
                                   oracle[0] + region.scores[held] @ oracle[1:])


@pytest.mark.parametrize("window, bin_count", [("full", 4), ("before", 2), ("after", 2)])
def test_binary_orientation_matches_objective_gradient_and_identical_labels(prediction_inputs, window, bin_count):
    """Class 1 means left; fitted logistic weights satisfy the selected weighted loss."""
    base, trials, timeline = prediction_inputs
    prepared = prepare_prediction_inputs(trials, timeline, base, training_window=window)
    result = fit_region(prepared)
    np.testing.assert_allclose(result.predictions[:, :, 0], result.predictions[:, :, 1])
    region = timeline.regions["PFC"]
    for fold in range(5):
        mask = result.training_choice_mask[1, fold]
        x = region.scores[mask[region.choice_positions]][:, prepared.training_bin_mask].reshape(-1, 3)
        y = np.repeat(prepared.target_values[mask, 1], bin_count)
        beta, intercept = result.coefficients[1, fold], result.intercepts[1, fold]
        probability = 1 / (1 + np.exp(-(x @ beta + intercept)))
        np.testing.assert_allclose(x.T @ ((probability - y) / bin_count) + beta, 0, atol=0.003)
        assert abs(np.sum((probability - y) / bin_count)) < 0.003
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


@pytest.fixture
def mixed_inputs(prediction_inputs):
    """Return N-choice mixed inputs with seeded independent state/action/reward PC signals."""
    base, trials, timeline = prediction_inputs
    trials["state_int"] = np.arange(len(trials)) % 2
    trials["correct"] = (trials.state_int == trials.action).astype(int)
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    generator = np.random.default_rng(318)
    regions = {}
    for role, source in timeline.regions.items():
        labels = prepared.target_values[source.choice_positions]
        scores = generator.normal(0, 0.1, source.scores.shape)
        scores += 3 * (2 * labels[:, None, :] - 1)
        regions[role] = replace(source, scores=scores)
    return replace(prepared, timeline=replace(timeline, regions=regions))


def test_mixed_training_masks_signal_recovery_and_logistic_objective(mixed_inputs):
    """Three classifiers recover separate labels using every eligible trial once."""
    for role, source in mixed_inputs.timeline.regions.items():
        model = fit_region(mixed_inputs, role, use_training_membership=True)
        assert model.predictions.shape == (*source.scores.shape[:2], 3)
        assert np.all(model.fit_status == "ok")
        coverage = np.isin(np.arange(len(mixed_inputs.fold_ids)), source.choice_positions)
        for target in range(3):
            actual = mixed_inputs.target_values[source.choice_positions, target]
            assert np.mean((model.predictions[:, :, target] >= 0.5) == actual[:, None]) > 0.95
            for fold in range(5):
                expected = (coverage & mixed_inputs.baseline_valid & mixed_inputs.target_valid[:, target]
                            & mixed_inputs.training_condition_mask & (mixed_inputs.fold_ids >= 0)
                            & (mixed_inputs.fold_ids != fold))
                np.testing.assert_array_equal(model.training_choice_mask[target, fold], expected)
                assert np.any(expected & ~mixed_inputs.correct_rewarded)
                matrix = source.scores[expected[source.choice_positions]][:, mixed_inputs.training_bin_mask]
                matrix = matrix.reshape(-1, source.scores.shape[2])
                labels = np.repeat(mixed_inputs.target_values[expected, target], 4)
                beta = model.coefficients[target, fold]
                probability = 1 / (1 + np.exp(-(matrix @ beta + model.intercepts[target, fold])))
                residual = (probability - labels) / 4
                np.testing.assert_allclose(matrix.T @ residual + beta, 0, atol=0.003)
                assert abs(residual.sum()) < 0.003
                assert model.training_row_count[target, fold] == 4 * expected.sum()


def test_mixed_mask_exclusions_and_local_failures(mixed_inputs):
    """Membership never overrides coverage/baseline/label/block masks; failures stay local."""
    baseline = mixed_inputs.baseline_valid.copy()
    baseline[0] = False
    valid = mixed_inputs.target_valid.copy()
    valid[1, 0] = False
    values = mixed_inputs.target_values.copy()
    values[1, 0] = np.nan
    folds = mixed_inputs.fold_ids.copy()
    folds[2] = -1
    prepared = replace(mixed_inputs, baseline_valid=baseline, target_valid=valid,
                       target_values=values, fold_ids=folds)
    model = fit_region(prepared, use_training_membership=True)
    assert not model.training_choice_mask[:, :, [0, 2]].any()
    assert not model.training_choice_mask[0, :, 1].any()
    source = prepared.timeline.regions["PFC"]
    missing = source.choice_positions == 2
    assert np.isnan(model.predictions[missing]).all()
    values[:, 0] = 1
    single = fit_region(replace(prepared, target_values=values), use_training_membership=True)
    assert np.all(single.fit_reason[0] == "single_class_training")
    assert np.all(single.fit_status[1] == "ok")
    empty = fit_region(replace(prepared, training_condition_mask=np.zeros(len(values), dtype=bool)),
                       use_training_membership=True)
    assert np.all(empty.fit_reason == "empty_training")


@pytest.mark.parametrize("defect", ["shape", "dtype"])
def test_explicit_membership_is_validated(mixed_inputs, defect):
    """An explicit membership mask must have exactly N Boolean entries."""
    mask = mixed_inputs.training_condition_mask
    mask = mask[:-1] if defect == "shape" else mask.astype(float)
    with pytest.raises(ValueError, match="mask"):
        fit_region(replace(mixed_inputs, training_condition_mask=mask), use_training_membership=True)


@pytest.mark.parametrize("reward", [0., np.nan])
def test_unfit_reward_does_not_disable_state_or_action(prediction_inputs, reward):
    """Single-class or missing reward is local; incorrect membership remains independent."""
    base, trials, timeline = prediction_inputs
    trials["correct"] = 0
    trials["reward"] = reward
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    model = fit_region(prepared, use_training_membership=True)
    assert (model.fit_status[:2] == "ok").all()
    assert (model.fit_status[2] == "unavailable").all()
    assert np.isnan(model.predictions[:, :, 2]).all()
    assert np.isfinite(model.predictions[:, :, :2]).all()
