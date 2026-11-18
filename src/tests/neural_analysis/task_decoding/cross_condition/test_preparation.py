"""Hand-enumerated trial identities, cell labels, exclusions and alignment."""

from dataclasses import replace
import inspect

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding import targets
from src.neural_analysis.task_decoding.cross_condition.config import TARGET_NAMES, TRANSFERS
from src.neural_analysis.task_decoding.cross_condition.preparation import prepare_transfer_inputs


# Each tuple explicitly lists source class 0/1, then test class 0/1 original IDs.
# Expectations are independent of registry expressions and preparation masks.
EXPECTED_CELLS = {
    "context_correct_to_incorrect": ((2, 3), (0, 1, 19), (4, 6), (5, 7, 9)),
    "context_left_choice_to_right_choice": ((4, 6), (0, 1, 19), (2, 3), (5, 7, 9)),
    "context_right_choice_to_left_choice": ((2, 3), (5, 7, 9), (4, 6), (0, 1, 19)),
    "context_incorrect_to_correct": ((4, 6), (5, 7, 9), (2, 3), (0, 1, 19)),
    "choice_correct_to_incorrect": ((2, 3), (0, 1, 19), (5, 7, 9), (4, 6)),
    "choice_left_context_to_right_context": ((5, 7, 9), (0, 1, 19), (2, 3), (4, 6)),
    "choice_right_context_to_left_context": ((2, 3), (4, 6), (5, 7, 9), (0, 1, 19)),
    "choice_incorrect_to_correct": ((5, 7, 9), (4, 6), (2, 3), (0, 1, 19)),
    "reward_left_choice_to_right_choice": ((1, 4), (0, 6, 10, 19), (3, 5), (2, 7)),
    "reward_left_rewarded_right_unrewarded": ((3, 5), (0, 6, 10, 19), (1, 4), (2, 7)),
    "reward_right_rewarded_left_unrewarded": ((1, 4), (2, 7), (3, 5), (0, 6, 10, 19)),
    "reward_right_choice_to_left_choice": ((3, 5), (2, 7), (1, 4), (0, 6, 10, 19)),
    "value_left_choice_to_right_choice": ((4, 6), (0, 1, 19), (5, 7, 9), (2, 3)),
    "value_left_correct_right_incorrect": ((5, 7, 9), (0, 1, 19), (4, 6), (2, 3)),
    "value_right_correct_left_incorrect": ((4, 6), (2, 3), (5, 7, 9), (0, 1, 19)),
    "value_right_choice_to_left_choice": ((5, 7, 9), (2, 3), (4, 6), (0, 1, 19)),
    "context_correct_rewarded_to_other_outcomes": ((2,), (0, 19), (3, 4, 6), (1, 5, 7, 9)),
    "context_omission_to_other_outcomes": ((3,), (1,), (2, 4, 6), (0, 5, 7, 9, 19)),
}


def test_all_eighteen_transfers_match_original_trial_ids_and_actual_class_axis(transfer_data):
    """Preserve the original sixteen literal memberships and check appended cells(N=16,S=18,2)."""
    inputs = prepare_transfer_inputs(*transfer_data)
    assert tuple(item.split_id for item in inputs.transfers) == tuple(EXPECTED_CELLS)
    assert inputs.transfers == TRANSFERS
    assert inputs.source_cells.shape == inputs.test_cells.shape == (16, 18, 2)
    assert inputs.source_cells.dtype == inputs.test_cells.dtype == bool
    ids = inputs.choices.trial_id.to_numpy()
    for split, definition in enumerate(inputs.transfers):
        expected = EXPECTED_CELLS[definition.split_id]
        target = inputs.target_names.index(definition.target_name)
        for role, cells in enumerate((inputs.source_cells, inputs.test_cells)):
            for label in (0, 1):
                selected = cells[:, split, label]
                assert tuple(ids[selected]) == expected[2 * role + label]
                np.testing.assert_array_equal(inputs.target_values[selected, target], label)
                assert inputs.target_valid[selected, target].all()
        assert not np.any(inputs.source_cells[:, split].any(axis=1) & inputs.test_cells[:, split].any(axis=1))


@pytest.mark.parametrize("split_id,training_population,source_ids,test_outcomes", [
    ("context_correct_rewarded_to_other_outcomes", "correct_rewarded", (0, 2, 19),
     (("omission", (1, 3)), ("incorrect", (4, 5, 6, 7, 9)))),
    ("context_omission_to_other_outcomes", "omission", (1, 3),
     (("correct_rewarded", (0, 2, 19)), ("incorrect", (4, 5, 6, 7, 9)))),
    ("context_incorrect_to_correct", "incorrect", (4, 5, 6, 7, 9),
     (("correct_rewarded", (0, 2, 19)), ("omission", (1, 3)))),
])
def test_context_outcome_sources_and_two_test_outcomes_have_independent_literal_memberships(
    transfer_data, split_id, training_population, source_ids, test_outcomes,
):
    """Each Context source uses both recorded contexts; its complementary union retains two separate outcomes.

    Original IDs6/7 are rewarded incorrect, ID9 is incorrect with unknown reward,
    and ID8 is correct with unknown reward. The former stay incorrect; the latter
    enters no Context outcome pool. All IDs are dimensionless original trial IDs.
    """
    inputs = prepare_transfer_inputs(*transfer_data)
    split_ids = tuple(item.split_id for item in inputs.transfers)
    assert split_id in split_ids
    split = split_ids.index(split_id)
    assert inputs.transfers[split].target_name == "current_state"
    ids = inputs.choices.trial_id.to_numpy()
    source = inputs.source_cells[:, split].any(axis=1)
    complementary = inputs.test_cells[:, split].any(axis=1)
    assert tuple(ids[source]) == source_ids
    training_condition = inputs.condition_masks[:, inputs.condition_names.index(training_population)]
    np.testing.assert_array_equal(source, training_condition)
    expected_union = np.zeros(len(ids), dtype=bool)
    for population, expected_ids in test_outcomes:
        outcome = inputs.condition_masks[:, inputs.condition_names.index(population)]
        selected = complementary & outcome
        assert tuple(ids[selected]) == expected_ids
        assert not np.any(source & selected)
        assert not np.any(expected_union & selected)
        expected_union |= selected
    np.testing.assert_array_equal(complementary, expected_union)
    assert not source[ids == 8].any() and not complementary[ids == 8].any()
    for cells in (inputs.source_cells, inputs.test_cells):
        for label in (0, 1):
            members = cells[:, split, label]
            np.testing.assert_array_equal(inputs.target_values[members, 0], label)


def test_prepared_identity_dtypes_labels_and_input_immutability(transfer_data):
    """J=20 rows map to N=16 original choices; labels/weights are dimensionless."""
    trials, choices, edges, base = transfer_data
    original_trials, original_choices, original_edges = trials.copy(deep=True), choices.copy(deep=True), edges.copy()
    inputs = prepare_transfer_inputs(*transfer_data)
    assert inputs.target_names == TARGET_NAMES
    assert inputs.target_values.shape == inputs.target_valid.shape == (16, 4)
    assert inputs.target_values.dtype == np.float64
    assert inputs.target_valid.dtype == inputs.baseline_valid.dtype == inputs.conflict_mask.dtype == bool
    assert inputs.baseline_valid.shape == inputs.conflict_mask.shape == (16,)
    assert inputs.choices.row_position.tolist() == list(range(15)) + [19]
    assert inputs.choices.trial_id.tolist() == list(range(15)) + [19]
    for name in TARGET_NAMES:
        assert inputs.target_metadata[name]["positive_class"] == 1
    assert inputs.target_metadata["current_reward"]["positive_label"] == "rewarded"
    assert inputs.target_metadata["current_action_is_correct"]["positive_label"] == "correct"
    assert base.target_names == ("previous_action",) and base.alignment == "start_time"
    pd.testing.assert_frame_equal(trials, original_trials)
    pd.testing.assert_frame_equal(choices, original_choices)
    np.testing.assert_array_equal(edges, original_edges)


@pytest.mark.parametrize("sentinel", [None, "None", "", "nan", "NaN"])
def test_numeric_strings_flags_and_missing_sentinels_preserve_membership(transfer_data, sentinel):
    """String-coded numeric labels/flags yield identical arrays without mutating J-row input."""
    trials, choices, edges, base = transfer_data
    expected = prepare_transfer_inputs(*transfer_data)
    strings = trials.copy(deep=True)
    for column in ("state_int", "action", "correct", "reward", "experimenter_reward_given"):
        strings[column] = strings[column].map(str).where(strings[column].notna(), sentinel)
    original = strings.copy(deep=True)
    actual = prepare_transfer_inputs(strings, choices, edges, base)
    for name in ("target_values", "target_valid", "baseline_valid", "conflict_mask", "condition_masks", "source_cells", "test_cells"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    pd.testing.assert_frame_equal(strings, original)


def test_dark_and_conflicting_trials_are_excluded_with_original_value_audit(transfer_data):
    """Exclude dark ID12 through common eligibility; retain loader labels and raw conflict audit."""
    trials, choices, edges, base = transfer_data
    inputs = prepare_transfer_inputs(*transfer_data)
    assert inputs.dark_state_count == 1
    assert not inputs.baseline_valid[12]
    np.testing.assert_array_equal(inputs.target_valid[12], [False, True, True, True])
    np.testing.assert_array_equal(inputs.target_values[12], [np.nan, 1, 0, 0])
    assert inputs.choices.trial_id[inputs.conflict_mask].tolist() == [13, 14]
    assert not inputs.source_cells[[12, 13, 14]].any()
    assert not inputs.test_cells[[12, 13, 14]].any()
    audit = inputs.conflicts
    assert audit.row_position.tolist() == [13, 14]
    assert audit.trial_id.tolist() == [13, 14]
    for column in ("state_int", "action", "correct"):
        assert audit[column].tolist() == trials.iloc[[13, 14]][column].tolist()
    assert audit.reason.str.len().gt(0).all()
    strings = trials.copy(deep=True)
    for column in ("state_int", "action", "correct"):
        strings[column] = strings[column].map(str)
    string_audit = prepare_transfer_inputs(strings, choices, edges, base).conflicts
    for column in ("state_int", "action", "correct"):
        assert string_audit[column].tolist() == strings.iloc[[13, 14]][column].tolist()


def test_actual_reward_correctness_and_omission_missingness_remain_distinct(transfer_data):
    """Rewarded incorrect IDs6/7 and unknown correctness ID10 retain actual-reward labels."""
    inputs = prepare_transfer_inputs(*transfer_data)
    reward, value = 2, 3
    np.testing.assert_array_equal(inputs.target_values[[1, 3], reward], [0, 0])
    np.testing.assert_array_equal(inputs.target_values[[1, 3], value], [1, 1])
    np.testing.assert_array_equal(inputs.target_valid[[8, 9], reward], [False, False])
    np.testing.assert_array_equal(inputs.target_valid[[10, 11], value], [False, False])
    np.testing.assert_array_equal(inputs.target_values[[6, 7], [reward, reward]], [1, 1])
    canonical = dict(zip(inputs.condition_names, inputs.condition_masks.T, strict=True))
    assert canonical["incorrect"][[6, 7, 9]].all()
    assert not canonical["correct_rewarded"][[6, 7, 10]].any()
    assert canonical["omission"][[1, 3]].all()
    # Correct with unknown reward does not enter the canonical correct union;
    # incorrect with unknown reward still belongs to the canonical incorrect cell.
    assert not inputs.source_cells[8].any() and not inputs.test_cells[8].any()
    assert not inputs.source_cells[11].any() and not inputs.test_cells[11].any()


@pytest.mark.parametrize("excluded_row", [15, 16, 17, 18])
@pytest.mark.parametrize("context", [np.nan, "None", "invalid", 99])
def test_baseline_exclusions_precede_context_validation(transfer_data, excluded_row, context):
    """Manual/no-choice/missing-alignment/missing-flag J rows never require valid context."""
    trials, choices, edges, base = transfer_data
    altered = trials.copy(deep=True)
    altered["state_int"] = altered.state_int.astype(object)
    altered.loc[altered.index[excluded_row], "state_int"] = context
    actual = prepare_transfer_inputs(altered, choices, edges, base)
    assert actual.choices.row_position.tolist() == list(range(15)) + [19]
    assert actual.dark_state_count == 1


@pytest.mark.parametrize("column,value", [
    ("state_int", np.nan), ("state_int", "None"), ("state_int", "invalid"), ("state_int", 3),
    ("action", "invalid"), ("action", 0.5), ("correct", "invalid"), ("correct", 2),
    ("reward", "invalid"), ("reward", -1), ("experimenter_reward_given", "invalid"),
    ("experimenter_reward_given", np.inf),
])
def test_malformed_present_labels_and_missing_ordinary_context_raise(transfer_data, column, value):
    """Reject malformed source values; ordinary binary-context choices cannot lack context."""
    trials, choices, edges, base = transfer_data
    altered = trials.copy(deep=True)
    altered[column] = altered[column].astype(object)
    altered.loc[altered.index[0], column] = value
    with pytest.raises(ValueError, match=column):
        prepare_transfer_inputs(altered, choices, edges, base)


@pytest.mark.parametrize("column", ["row_position", "trial_id", "action", "choice_time"])
def test_saved_choice_identity_or_alignment_mismatch_raises(transfer_data, column):
    """Choice(N,C) must match original rows, IDs/actions and absolute UTC time with rtol=0."""
    trials, choices, edges, base = transfer_data
    altered = choices.copy(deep=True)
    altered.loc[0, column] += 1e-3 if column == "choice_time" else 1
    with pytest.raises(ValueError, match=column):
        prepare_transfer_inputs(trials, altered, edges, base)


def test_one_microsecond_absolute_time_tolerance_and_choice_count(transfer_data):
    """Sub-us UTC offsets pass; missing/reordered original choice coordinates fail."""
    trials, choices, edges, base = transfer_data
    shifted = choices.copy(deep=True)
    shifted["choice_time"] += 0.9e-6
    prepare_transfer_inputs(trials, shifted, edges, base)
    shifted["choice_time"] += 2e-6
    with pytest.raises(ValueError, match="choice_time"):
        prepare_transfer_inputs(trials, shifted, edges, base)
    for altered in (choices.iloc[:-1], choices.iloc[::-1]):
        with pytest.raises(ValueError):
            prepare_transfer_inputs(trials, altered, edges, base)


@pytest.mark.parametrize("sentinel", ["no_choice", "None", "", None])
def test_explicit_no_choice_action_sentinels_remain_baseline_excluded(transfer_data, sentinel):
    """Recognized missing/no-choice action tokens retain original chronology, with no context check."""
    trials, choices, edges, base = transfer_data
    altered = trials.copy(deep=True)
    altered["action"] = altered.action.astype(object)
    altered.loc[altered.index[16], "action"] = sentinel
    inputs = prepare_transfer_inputs(altered, choices, edges, base)
    assert inputs.choices.row_position.tolist() == list(range(15)) + [19]


@pytest.mark.parametrize("blocks", ["absent", "missing", "noncontiguous", "changed"])
def test_blocks_have_no_effect_on_any_prepared_array(transfer_data, blocks):
    """Unused block contents must never affect labels, memberships, exclusions or bin masks."""
    trials, choices, edges, base = transfer_data
    expected = prepare_transfer_inputs(*transfer_data)
    altered = trials.copy(deep=True)
    if blocks == "absent":
        altered = altered.drop(columns="cur_block")
    elif blocks == "missing":
        altered["cur_block"] = "None"
    elif blocks == "noncontiguous":
        altered["cur_block"] = np.arange(len(altered)) % 2
    else:
        altered["cur_block"] = [f"arbitrary-{i}" for i in range(len(altered))]
    actual = prepare_transfer_inputs(altered, choices, edges, base)
    for name in ("target_values", "target_valid", "baseline_valid", "conflict_mask", "condition_masks", "source_cells", "test_cells", "window_bin_masks"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    pd.testing.assert_frame_equal(actual.conflicts, expected.conflicts)
    assert not hasattr(actual, "fold_ids")


@pytest.mark.parametrize("width_ms", [500, 100, 50, 20])
def test_exact_supervised_window_bins_and_zero_belongs_to_after(transfer_data, width_ms):
    """Three Boolean (H=3,B) windows use exact half-open seconds without resampling."""
    trials, choices, _, base = transfer_data
    edges = np.linspace(-2, 2, 4000 // width_ms + 1)
    inputs = prepare_transfer_inputs(trials, choices, edges, base)
    assert inputs.window_names == ("full", "before", "after")
    np.testing.assert_array_equal(inputs.bin_edges_s, edges)
    expected = np.zeros((3, len(edges) - 1), dtype=bool)
    bins_per_second = 1000 // width_ms
    expected[0, bins_per_second:3 * bins_per_second] = True
    expected[1, bins_per_second:2 * bins_per_second] = True
    expected[2, 2 * bins_per_second:3 * bins_per_second] = True
    np.testing.assert_array_equal(inputs.window_bin_masks, expected)
    assert not inputs.window_bin_masks[1, 2 * bins_per_second]
    assert inputs.window_bin_masks[2, 2 * bins_per_second]


@pytest.mark.parametrize("edges", [np.array([-2., -.5, .5, 2.]), np.array([-2., -1., 0., 0., 1., 2.]), np.array([-2., -1., 0., np.nan, 2.])])
def test_invalid_or_boundary_crossing_bin_grids_raise(transfer_data, edges):
    """Finite increasing (B+1,) edges must include exact Full/Before/After boundaries."""
    trials, choices, _, base = transfer_data
    with pytest.raises(ValueError):
        prepare_transfer_inputs(trials, choices, edges, base)


@pytest.mark.parametrize("blocks", ["absent", "missing", "noncontiguous"])
def test_ignore_blocks_is_explicit_opt_in_and_legacy_validation_remains(transfer_data, blocks):
    """Shared target APIs default to strict blocks; keyword-only opt-in ignores source contents."""
    trials, _, _, base = transfer_data
    base = replace(base, target_names=TARGET_NAMES, alignment="choice_time")
    altered = trials.copy(deep=True)
    if blocks == "absent":
        altered = altered.drop(columns="cur_block")
    elif blocks == "missing":
        altered["cur_block"] = np.nan
    else:
        altered["cur_block"] = np.arange(len(altered)) % 2
    for function in (targets.validate_augmented_trials, targets.build_target_table):
        with pytest.raises(ValueError, match="cur_block|block"):
            function(altered, base)
        parameter = inspect.signature(function).parameters["ignore_blocks"]
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.default is False
    targets.validate_augmented_trials(altered, base, ignore_blocks=True)
    output = targets.build_target_table(altered, base, ignore_blocks=True)
    assert output.block_id.isna().all()
    assert output.row_position.tolist() == list(range(20))
    assert output.trial_id.tolist() == list(range(20))
    assert output.baseline_valid.tolist() == [True] * 15 + [False] * 4 + [True]


def test_ignore_blocks_does_not_disable_other_validation(transfer_data):
    """Block opt-in never accepts malformed actions or mutates the caller's table."""
    trials, _, _, base = transfer_data
    base = replace(base, target_names=TARGET_NAMES, alignment="choice_time")
    altered = trials.drop(columns="cur_block")
    altered["action"] = altered.action.astype(object)
    altered.loc[altered.index[0], "action"] = "invalid"
    original = altered.copy(deep=True)
    with pytest.raises(ValueError, match="action"):
        targets.build_target_table(altered, base, ignore_blocks=True)
    pd.testing.assert_frame_equal(altered, original)
