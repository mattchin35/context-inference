"""Trial-level performance oracles independent of viewport, fitting and raw inputs."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit

from src.neural_analysis.task_decoding.session_prediction import scoring
from src.neural_analysis.task_decoding.session_prediction.preparation import prepare_prediction_inputs
from .test_modeling import fit_region


def test_known_binary_accuracy_and_balanced_accuracy():
    """An imbalanced four-trial example distinguishes ordinary and balanced accuracy."""
    actual = np.array([0., 0., 0., 1.])
    scores = np.array([0.9, 0.1, 0.1, 0.9])
    metrics = scoring.trial_metrics(actual, scores, "categorical")
    assert metrics["n_trials"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == pytest.approx(5 / 6)
    always_left = scoring.trial_metrics(actual, np.ones(4), "categorical")
    assert always_left["accuracy"] == 0.25
    assert always_left["balanced_accuracy"] == 0.5


def test_known_regression_scores_keep_negative_r2_and_native_rmse():
    """Three native-unit targets give R2=-1 and RMSE=sqrt(4/3), without clipping."""
    metrics = scoring.trial_metrics(np.array([1., 2., 3.]), np.array([1., 2., 5.]), "numerical")
    assert metrics["n_trials"] == 3
    assert metrics["r2"] == -1
    assert metrics["rmse"] == pytest.approx(np.sqrt(4 / 3))


@pytest.mark.parametrize("actual, predicted, family, expected", [
    ([1, 1], [0.5, 0.1], "categorical", {"accuracy": 0.5, "balanced_accuracy": None}),
    ([], [], "categorical", {"accuracy": None, "balanced_accuracy": None}),
    ([4, 4], [3, 5], "numerical", {"r2": None, "rmse": 1}),
    ([4], [6], "numerical", {"r2": None, "rmse": 2}),
    ([], [], "numerical", {"r2": None, "rmse": None}),
])
def test_undefined_metrics_remain_unavailable(actual, predicted, family, expected):
    """Empty/single-class/constant groups keep counts and every still-defined metric."""
    metrics = scoring.trial_metrics(np.array(actual, dtype=float), np.array(predicted, dtype=float), family)
    assert metrics == {"n_trials": len(actual), **expected}


@pytest.mark.parametrize("actual, predicted, family", [
    ([0, 1], [0.2], "categorical"), ([0, 1], [0.2, np.nan], "categorical"),
    ([0, 1], [-0.1, 1.1], "categorical"), ([0, 2], [0.2, 0.8], "categorical"),
    ([0, 1], [0.2, 0.8], "unknown"),
])
def test_malformed_metric_vectors_raise(actual, predicted, family):
    """Public scoring requires aligned finite vectors and canonical families/labels."""
    with pytest.raises(ValueError):
        scoring.trial_metrics(np.array(actual, dtype=float), np.array(predicted), family)


def find_row(rows, **selectors):
    """Return the unique small scoring record matching scalar field values."""
    matches = [row for row in rows if all(row[key] == value for key, value in selectors.items())]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("mode", ["correct_rewarded_generalization", "mixed_outcome_decoding"])
def test_training_reconstruction_and_pooled_held_out_trial_counts(prediction_inputs, mode):
    """Exact saved masks/coefficients supply per-fold training, with unique held-out trials."""
    base, trials, timeline = prediction_inputs
    trials.loc[105, "correct"] = 0
    trials.loc[105, "state_int"] = 1 - trials.loc[105, "action"]
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode=mode)
    models = {role: fit_region(prepared, role, use_training_membership=True) for role in timeline.regions}
    originals = {role: region.scores.copy() for role, region in timeline.regions.items()}
    report = scoring.compute_prediction_performance(prepared, models, "current_action")
    for role, source in timeline.regions.items():
        model = models[role]
        actual = prepared.target_values[source.choice_positions, 1]
        test_predictions = model.predictions[:, prepared.training_bin_mask, 1].mean(axis=1)
        metrics = scoring.trial_metrics(actual, test_predictions, "categorical")
        row = find_row(report["held_out"], region=role, population="all")
        assert all(row[key] == value for key, value in metrics.items())
        assert row["n_trials"] == len(source.choice_positions)
        for fold in range(5):
            mask = model.training_choice_mask[1, fold, source.choice_positions]
            matrix = source.scores[mask][:, prepared.training_bin_mask]
            predictions = expit(matrix @ model.coefficients[1, fold] + model.intercepts[1, fold]).mean(axis=1)
            expected = scoring.trial_metrics(actual[mask], predictions, "categorical")
            train = find_row(report["folds"], region=role, fold=fold, split="training")
            assert all(train[key] == value for key, value in expected.items())
            held = ((prepared.fold_ids[source.choice_positions] == fold)
                    & prepared.training_condition_mask[source.choice_positions])
            expected = scoring.trial_metrics(actual[held], test_predictions[held], "categorical")
            test = find_row(report["folds"], region=role, fold=fold, split="held_out")
            assert all(test[key] == value for key, value in expected.items())
        for name in ("correct_rewarded", "omission", "incorrect", "training_outcomes"):
            mask = (prepared.training_condition_mask if name == "training_outcomes" else
                    prepared.condition_masks[:, prepared.condition_names.index(name)])
            assert find_row(report["held_out"], region=role, population=name)["n_trials"] == mask[source.choice_positions].sum()
        np.testing.assert_array_equal(source.scores, originals[role])


def test_probability_mean_precedes_threshold_and_ignores_outer_bins(prediction_inputs):
    """Mean sigmoid([-10,2,2,2]) is left, although sigmoid(mean logits) is right."""
    base, trials, timeline = prediction_inputs
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    source = timeline.regions["PFC"]
    scores = np.full_like(source.scores, -100.)
    scores[:, prepared.training_bin_mask, 0] = [-10., 2., 2., 2.]
    source = replace(source, scores=scores)
    prepared = replace(prepared, timeline=replace(timeline, regions={"PFC": source}))
    model = fit_region(prepared, use_training_membership=True)
    coefficients = np.zeros_like(model.coefficients)
    coefficients[:, :, 0] = 1
    predictions = np.full_like(model.predictions, 0.01)
    predictions[:, prepared.training_bin_mask] = expit(np.array([-10., 2., 2., 2.]))[None, :, None]
    model = replace(model, coefficients=coefficients, intercepts=np.zeros_like(model.intercepts),
                    predictions=predictions)
    report = scoring.compute_prediction_performance(prepared, {"PFC": model}, "current_action")
    for row in report["folds"]:
        mask = (model.training_choice_mask[1, row["fold"], source.choice_positions] if row["split"] == "training"
                else ((prepared.fold_ids[source.choice_positions] == row["fold"])
                      & prepared.training_condition_mask[source.choice_positions]))
        assert row["accuracy"] == pytest.approx(prepared.target_values[source.choice_positions[mask], 1].mean())
    all_test = find_row(report["held_out"], region="PFC", population="all")
    assert all_test["accuracy"] == pytest.approx(prepared.target_values[source.choice_positions, 1].mean())


def test_choice_and_context_laterality_are_independent_on_incorrect_trials(prediction_inputs):
    """Always-left predictions show different choice/context accuracies with factual counts."""
    base, trials, timeline = prediction_inputs
    trials["state_int"] = np.arange(len(trials)) % 2
    trials["correct"] = (trials.state_int == trials.action).astype(int)
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    models = {role: fit_region(prepared, role, use_training_membership=True) for role in timeline.regions}
    models = {role: replace(model, predictions=np.full_like(model.predictions, 0.9)) for role, model in models.items()}
    report = scoring.compute_prediction_performance(prepared, models, "current_state")
    for role, source in timeline.regions.items():
        target = prepared.target_values[source.choice_positions, 0]
        for grouping, index in (("choice", 1), ("state", 0)):
            sides = prepared.target_values[source.choice_positions, index]
            for side, value in (("left", 1), ("right", 0)):
                row = find_row(report["laterality"], region=role, population="all", grouping=grouping, side=side)
                assert row["n_trials"] == (sides == value).sum()
                assert row["accuracy"] == pytest.approx(target[sides == value].mean())
            gap = find_row(report["gaps"], region=role, population="all", grouping=grouping)
            assert gap["accuracy"] == pytest.approx(target[sides == 1].mean() - target[sides == 0].mean())
        assert find_row(report["gaps"], region=role, population="all", grouping="state")["accuracy"] == 1


def test_scoring_exclusions_and_missing_side_labels_do_not_suppress_other_splits(prediction_inputs):
    """Baseline/actual/fold/fit eligibility is separate from choice/context grouping."""
    base, trials, timeline = prediction_inputs
    trials.loc[103, "cur_block"] = np.nan
    trials.loc[104, "state_int"] = np.nan
    prepared = prepare_prediction_inputs(trials, timeline, base)
    baseline = prepared.baseline_valid.copy()
    baseline[0] = False
    prepared = replace(prepared, baseline_valid=baseline)
    models = {role: fit_region(prepared, role, use_training_membership=True) for role in timeline.regions}
    report = scoring.compute_prediction_performance(prepared, models, "current_action")
    for role, source in timeline.regions.items():
        positions = source.choice_positions
        eligible = prepared.baseline_valid[positions] & (prepared.fold_ids[positions] >= 0)
        overall = find_row(report["held_out"], region=role, population="all")
        assert overall["n_trials"] == eligible.sum()
        for grouping, index in (("choice", 1), ("state", 0)):
            rows = [row for row in report["laterality"] if row["region"] == role
                    and row["population"] == "all" and row["grouping"] == grouping]
            assert sum(row["n_trials"] for row in rows) == (eligible & prepared.target_valid[positions, index]).sum()
    target_report = scoring.compute_prediction_performance(prepared, models, "current_state")
    assert find_row(target_report["held_out"], region="PFC", population="all")["n_trials"] < find_row(report["held_out"], region="PFC", population="all")["n_trials"]


def test_continuous_performance_and_unavailable_fits(prediction_inputs):
    """Continuous means use target units; failed folds/empty targets never become zero scores."""
    base, trials, timeline = prediction_inputs
    trials["HMM_rel_value_logodds"] = np.nan
    prepared = prepare_prediction_inputs(trials, timeline, base)
    models = {role: fit_region(prepared, role, use_training_membership=True) for role in timeline.regions}
    report = scoring.compute_prediction_performance(prepared, models, "qlearning_relative_value")
    for role, source in timeline.regions.items():
        actual = prepared.target_values[source.choice_positions, 2]
        predictions = models[role].predictions[:, prepared.training_bin_mask, 2].mean(axis=1)
        expected = scoring.trial_metrics(actual, predictions, "numerical")
        row = find_row(report["held_out"], region=role, population="all")
        assert all(row[key] == value for key, value in expected.items())
    empty = scoring.compute_prediction_performance(prepared, models, "hmm_signed_belief")
    assert all(row["n_trials"] == 0 and row["r2"] is None and row["rmse"] is None for row in empty["held_out"])
    models = {role: replace(model, fit_status=np.full_like(model.fit_status, "unavailable"))
              for role, model in models.items()}
    unavailable = scoring.compute_prediction_performance(prepared, models, "current_action")
    assert all(row["n_trials"] == 0 and row["accuracy"] is None for row in unavailable["held_out"])
