"""Signed additive contributions retain model/PC identity through selection."""

import numpy as np
import pytest

from src.neural_analysis.task_decoding.session_prediction import modeling
from src.neural_analysis.task_decoding.session_prediction.preparation import prepare_prediction_inputs
from .test_modeling import fit_region
from .test_modeling import mixed_inputs as mixed_inputs_fixture

mixed_inputs = mixed_inputs_fixture


def test_contributions_reconstruct_predictions_and_use_original_pc_ids(prediction_inputs):
    """Selected/reordered scores use their own coefficients, with no source mutation."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0])
    region = prepared.timeline.regions["PFC"]
    result = fit_region(prepared)
    folds = prepared.fold_ids[region.choice_positions]
    for target in range(7):
        contributions = modeling.build_pc_contributions(region.scores, folds, result, target, (3, 1, 2))
        linear = contributions.sum(axis=2) + result.intercepts[target, folds, None]
        expected = 1 / (1 + np.exp(-linear)) if target < 2 else linear
        np.testing.assert_allclose(expected, result.predictions[:, :, target])
        np.testing.assert_allclose(contributions[:, :, 0],
                                   region.scores[:, :, 2] * result.coefficients[target, folds, 2, None])


@pytest.mark.parametrize("window", ["full", "before", "after"])
def test_strength_uses_training_sd_and_order_is_fixed_with_rank_ties(prediction_inputs, window):
    """Saved rankings use pooled training rows; tied strengths prefer original PC rank."""
    prepared = prepare_prediction_inputs(*prediction_inputs[1:], prediction_inputs[0], training_window=window)
    result = fit_region(prepared)
    region = prepared.timeline.regions["PFC"]
    for target in range(7):
        for fold in range(5):
            local = result.training_choice_mask[target, fold, region.choice_positions]
            sd = region.scores[local][:, prepared.training_bin_mask].reshape(-1, 3).std(axis=0)
            np.testing.assert_allclose(result.training_pc_sd[target, fold], sd)
            np.testing.assert_allclose(result.contribution_strength[target, fold],
                                       abs(result.coefficients[target, fold]) * sd)
        mean = result.contribution_strength[target].mean(axis=0)
        np.testing.assert_allclose(result.mean_contribution_strength[target], mean)
        np.testing.assert_array_equal(result.contribution_pc_order[target],
                                      np.lexsort((np.arange(3), -mean)) + 1)
        assert result.contributing_fold_count[target] == 5


def test_color_limit_is_global_and_missing_fold_never_borrows_weights(prediction_inputs):
    """Limit spans all destination contributions; missing folds yield only NaN."""
    base, trials, timeline = prediction_inputs
    trials.loc[103, "cur_block"] = np.nan
    prepared = prepare_prediction_inputs(trials, timeline, base)
    region = timeline.regions["PFC"]
    result = fit_region(prepared)
    folds = prepared.fold_ids[region.choice_positions]
    contributions = modeling.build_pc_contributions(region.scores, folds, result, 2, (1, 2, 3))
    assert np.isnan(contributions[folds == -1]).all()
    assert result.contribution_color_limit[2] == np.nanmax(abs(contributions))


@pytest.mark.parametrize("role", ["PFC", "HPC"])
def test_mixed_contributions_reconstruct_own_classifiers(mixed_inputs, role):
    """All mixed target axes reconstruct logits and retain their own global PC ranks."""
    source = mixed_inputs.timeline.regions[role]
    model = fit_region(mixed_inputs, role, use_training_membership=True)
    folds = mixed_inputs.fold_ids[source.choice_positions]
    for target in range(3):
        contributions = modeling.build_pc_contributions(source.scores, folds, model, target, (3, 1, 2))
        linear = contributions.sum(axis=2) + model.intercepts[target, folds, None]
        np.testing.assert_allclose(1 / (1 + np.exp(-linear)), model.predictions[:, :, target])
        assert model.contribution_color_limit[target] == np.max(abs(contributions))
        mean = model.contribution_strength[target].mean(axis=0)
        np.testing.assert_array_equal(model.contribution_pc_order[target],
                                      np.lexsort((np.arange(3), -mean)) + 1)
