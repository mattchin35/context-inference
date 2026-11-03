"""Target alignment precedes every condition, regional and model mask."""

from dataclasses import replace

import numpy as np
import pytest

from src.neural_analysis.interregional.preparation import build_block_fold_assignment
from src.neural_analysis.population.session_timeline import choice_bin_edges
from src.neural_analysis.task_decoding import targets
from src.neural_analysis.task_decoding.config import CONDITION_IDENTIFIERS
from src.neural_analysis.task_decoding.session_prediction.config import TARGET_NAMES
from src.neural_analysis.task_decoding.session_prediction.preparation import (
    prepare_prediction_inputs, training_bin_mask,
)


def test_alignment_targets_conditions_and_full_table_folds(prediction_inputs):
    """Return N-choice arrays aligned by source position, not index or regional row."""
    base, trials, timeline = prediction_inputs
    original = trials.copy(deep=True)
    prepared = prepare_prediction_inputs(trials, timeline, base)
    positions = timeline.choices.row_position.to_numpy()
    assert prepared.target_names == TARGET_NAMES
    assert prepared.target_values.shape == (len(positions), 7)
    np.testing.assert_array_equal(prepared.target_values[:, 1], trials.action.iloc[positions])
    np.testing.assert_array_equal(prepared.target_values[:, 2], trials.Qlearning_rel_value.iloc[positions])
    expected = build_block_fold_assignment(trials)
    np.testing.assert_array_equal(prepared.fold_ids, np.asarray(expected.fold_ids)[positions])
    assert prepared.condition_names == CONDITION_IDENTIFIERS
    np.testing.assert_array_equal(prepared.correct_rewarded, prepared.condition_masks[:, 1])
    assert prepared.condition_masks[:, 0].all()
    assert prepared.condition_masks.sum(axis=1).max() >= 2
    assert not np.array_equal(timeline.regions["PFC"].choice_positions,
                              timeline.regions["HPC"].choice_positions)
    assert trials.equals(original)


@pytest.mark.parametrize("width, count", [(500, 4), (100, 20), (50, 40), (20, 100)])
def test_training_bins_are_fully_inside_exact_interval(width, count):
    """Relative-second edges yield exactly [-1,+1) without shifting or averaging."""
    edges = choice_bin_edges(width / 1000)
    selected = training_bin_mask(edges)
    assert selected.sum() == count
    assert edges[:-1][selected][0] == -1
    assert edges[1:][selected][-1] == 1


def test_missing_blocks_keep_choices_and_all_missing_target_keeps_coordinate(prediction_inputs):
    """Missing blocks have -1 folds; an empty target does not suppress other values."""
    base, trials, timeline = prediction_inputs
    trials.loc[103, "cur_block"] = np.nan
    trials["cur_block"] = trials.cur_block.astype(object)
    trials.loc[104, "cur_block"] = "None"
    trials["HMM_rel_value_logodds"] = np.nan
    prepared = prepare_prediction_inputs(trials, timeline, base)
    source_positions = timeline.choices.row_position.to_numpy()
    np.testing.assert_array_equal(prepared.fold_ids[np.isin(source_positions, [3, 4])], [-1, -1])
    assert np.isnan(prepared.target_values[:, 4]).all()
    assert not prepared.target_valid[:, 4].any()
    assert prepared.target_availability["hmm_signed_belief"] == "target_unavailable_no_values"
    assert prepared.target_valid[:, 2].all()


def test_fewer_than_five_blocks_fails(prediction_inputs):
    """Five folds are mandatory rather than silently reduced for a small session."""
    base, trials, timeline = prediction_inputs
    trials["cur_block"] = np.arange(len(trials)) // 5
    with pytest.raises(ValueError, match="five"):
        prepare_prediction_inputs(trials, timeline, base)


@pytest.mark.parametrize("column", ["row_position", "trial_id", "action", "choice_time"])
def test_mismatched_choice_identity_fails(prediction_inputs, column):
    """A mismatched saved choice is rejected, including subsecond errors at epoch scale."""
    base, trials, timeline = prediction_inputs
    choices = timeline.choices.copy()
    choices.loc[0, column] += 0.001 if column == "choice_time" else 1
    with pytest.raises(ValueError, match="align|match|identity"):
        prepare_prediction_inputs(trials, replace(timeline, choices=choices), base)


def test_csv_timestamp_roundtrip_tolerance_and_reordered_rows(prediction_inputs):
    """Submicrosecond UTC roundoff is accepted; reordered chronological rows are not."""
    base, trials, timeline = prediction_inputs
    trials["choice_time"] += 0.0000005
    prepare_prediction_inputs(trials, timeline, base)
    with pytest.raises(ValueError):
        prepare_prediction_inputs(trials.iloc[::-1], timeline, base)


def test_opt_in_target_validation_preserves_strict_defaults(prediction_inputs):
    """Both public loaders accept opt-ins but continue rejecting malformed present data."""
    base, trials, _ = prediction_inputs
    config = replace(base, target_names=TARGET_NAMES)
    trials.loc[100, "cur_block"] = np.nan
    trials["HMM_rel_value_logodds"] = np.nan
    for function in (targets.validate_augmented_trials, targets.build_target_table):
        with pytest.raises(ValueError):
            function(trials, config)
        function(trials, config, allow_missing_blocks=True, allow_empty_numerical_targets=True)
        bad = trials.copy()
        bad["HMM_rel_value_logodds"] = "malformed"
        with pytest.raises(ValueError, match="malformed"):
            function(bad, config, allow_missing_blocks=True, allow_empty_numerical_targets=True)
    bad = trials.copy()
    bad.loc[117, "cur_block"] = 0
    with pytest.raises(ValueError, match="contiguous"):
        targets.build_target_table(bad, config, allow_missing_blocks=True, allow_empty_numerical_targets=True)
