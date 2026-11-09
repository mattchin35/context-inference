"""Target alignment precedes every condition, regional and model mask."""

from dataclasses import replace

import numpy as np
import pytest

from src.neural_analysis.interregional.preparation import build_block_fold_assignment
from src.neural_analysis.population.session_timeline import choice_bin_edges
from src.neural_analysis.task_decoding import targets
from src.neural_analysis.task_decoding.config import CONDITION_IDENTIFIERS
from src.neural_analysis.task_decoding.conditions import build_condition_masks
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
    assert prepared.analysis_mode == "correct_rewarded_generalization"
    np.testing.assert_array_equal(prepared.training_condition_mask, prepared.correct_rewarded)
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


def test_mixed_membership_targets_and_shared_folds(prediction_inputs):
    """Mixed N-by-3 targets retain canonical outcome unions and full-table folds."""
    base, trials, timeline = prediction_inputs
    trials.loc[105, "correct"] = 0
    trials.loc[105, "state_int"] = 1 - trials.loc[105, "action"]
    trials.loc[106, "correct"] = np.nan
    trials.loc[103, "cur_block"] = np.nan
    original = trials.copy(deep=True)
    score_copies = {role: region.scores.copy() for role, region in timeline.regions.items()}
    reference = prepare_prediction_inputs(trials, timeline, base)
    mixed = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    positions = timeline.choices.row_position.to_numpy()
    masks = build_condition_masks(trials, CONDITION_IDENTIFIERS)
    expected = (masks["correct_rewarded"] | masks["omission"] | masks["incorrect"])[positions]
    assert mixed.analysis_mode == "mixed_outcome_decoding"
    assert mixed.target_names == ("current_state", "current_action", "current_reward")
    assert mixed.target_families == ("categorical", "categorical", "categorical")
    np.testing.assert_array_equal(mixed.target_values[:, :2], reference.target_values[:, :2])
    np.testing.assert_array_equal(mixed.target_valid[:, :2], reference.target_valid[:, :2])
    np.testing.assert_array_equal(mixed.training_condition_mask, expected)
    np.testing.assert_array_equal(mixed.correct_rewarded, reference.correct_rewarded)
    np.testing.assert_array_equal(mixed.fold_ids, reference.fold_ids)
    np.testing.assert_array_equal(mixed.bin_edges_s, reference.bin_edges_s)
    np.testing.assert_array_equal(mixed.baseline_valid, reference.baseline_valid)
    assert expected[positions == 5].all()  # Incorrect choice is included once.
    assert expected[positions == 4].all()  # Correct omission is included.
    assert not expected[positions == 6].any()  # Unknown outcome is not canonical all.
    assert trials.equals(original)
    for role, region in timeline.regions.items():
        np.testing.assert_array_equal(region.scores, score_copies[role])
        np.testing.assert_array_equal(mixed.timeline.regions[role].choice_positions,
                                      reference.timeline.regions[role].choice_positions)


def test_mixed_does_not_require_continuous_source_columns(prediction_inputs):
    """Select three targets before shared validation; strict generalization stays intact."""
    base, trials, timeline = prediction_inputs
    trials = trials.drop(columns=[
        "Qlearning_rel_value", "FQlearning_rel_value", "HMM_rel_value_logodds",
        "HMM_rel_value_logodds_decay", "relative_doubt_index",
    ])
    mixed = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    assert mixed.target_values.shape == (len(timeline.choices), 3)
    with pytest.raises(ValueError, match="missing"):
        prepare_prediction_inputs(trials, timeline, base)


@pytest.mark.parametrize("width", [500, 100, 50, 20])
def test_three_supervised_masks_partition_full_at_zero(width):
    """Half-open masks partition Full exactly, retaining the complete prediction grid."""
    edges = choice_bin_edges(width / 1000)
    masks = {window: training_bin_mask(edges, training_window=window)
             for window in ("full", "before", "after")}
    np.testing.assert_array_equal(masks["before"] | masks["after"], masks["full"])
    assert not (masks["before"] & masks["after"]).any()
    assert masks["full"].sum() == 2000 // width
    assert masks["before"].sum() == masks["after"].sum() == 1000 // width
    assert edges[1:][masks["before"]][-1] == 0
    assert edges[:-1][masks["after"]][0] == 0
    with pytest.raises(ValueError):
        training_bin_mask(np.array([-2., -1., 0.1, 1., 2.]), training_window="before")


def test_window_variants_share_eligibility_folds_and_source_scores(prediction_inputs):
    """Windows restrict bins only; a missing reward affects labels and canonical membership independently."""
    base, trials, timeline = prediction_inputs
    trials.loc[105, "correct"] = 0
    trials.loc[105, "reward"] = np.nan  # Incorrect membership does not require reward.
    trials.loc[106, "reward"] = np.nan  # Correct but unknown outcome leaves the training union.
    reference = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding")
    position = np.flatnonzero(timeline.choices.row_position.to_numpy() == 5)[0]
    assert reference.training_condition_mask[position]
    assert reference.target_valid[position, :2].all()
    assert not reference.target_valid[position, 2]
    assert not reference.training_condition_mask[timeline.choices.row_position.to_numpy() == 6].any()
    for window in ("full", "before", "after"):
        result = prepare_prediction_inputs(trials, timeline, base, analysis_mode="mixed_outcome_decoding", training_window=window)
        assert result.training_window == window
        for field in ("target_values", "target_valid", "baseline_valid", "fold_ids", "training_condition_mask"):
            np.testing.assert_array_equal(getattr(result, field), getattr(reference, field))
        for role, source in timeline.regions.items():
            assert result.timeline.regions[role] is source
