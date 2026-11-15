"""Handcrafted signed coefficients establish the grouping contract independently of fits."""

from copy import deepcopy

import numpy as np
import pytest

from src.neural_analysis.decoder_unit_activity.selection import (
    GROUP_IDS, build_group_weights, summarize_group_weights, assign_unit_groups,
)


@pytest.fixture
def coefficient_cells():
    """Return five saved-cell-shaped mappings for three units and five available folds."""
    cells = {}
    for name in ("context", "choice", "outcome", "left_switch_vs_stay", "right_switch_vs_stay"):
        classes = ["right", "left"]
        side = None
        coefficients = np.array([[3., -2., .5]])
        if name == "outcome":
            classes = ["left_reward", "right_reward", "left_no_reward", "right_no_reward"]
            coefficients = np.array([[3., 1., 0.], [1., 3., 0.], [-1., -1., 0.], [-3., -3., 0.]])
        elif name.endswith("switch_vs_stay"):
            classes = ["stay", "switch"]
            side = name.split("_")[0]
        cell = {"meta": {"target": name, "unit_ids": ["A:0", "A:1", "A:2"],
                "class_names": classes, "conditioning_side": side,
                "representations": {"units": {"folds": []}}},
                "class_codes": np.arange(len(classes))}
        for fold in range(5):
            cell["meta"]["representations"]["units"]["folds"].append(
                {"fold": fold, "available": True, "reason": ""})
            cell[f"fold_{fold}_retained_mask"] = np.ones(3, bool)
            cell[f"fold_{fold}_mean_hz"] = np.zeros(3)
            cell[f"fold_{fold}_scale_hz"] = np.ones(3)
            cell[f"fold_{fold}_units_coefficients"] = coefficients.copy()
        cells[name] = cell
    return cells


def test_exact_group_orientation_and_no_mutation(coefficient_cells):
    """Input saved coefficients; output (5,10,3) preserves signed classes and source arrays."""
    original = deepcopy(coefficient_cells)
    weights = build_group_weights(coefficient_cells, ("A:0", "A:1", "A:2"))
    assert weights.shape == (5, 10, 3)
    np.testing.assert_allclose(weights[:, 0], np.tile([3, -2, .5], (5, 1)))
    np.testing.assert_allclose(weights[:, 1], -weights[:, 0])
    np.testing.assert_allclose(weights[:, 3], -weights[:, 2])
    beta = original["outcome"]["fold_0_units_coefficients"]
    for index in range(4):
        np.testing.assert_allclose(weights[0, 4+index], beta[index] - np.delete(beta, index, axis=0).mean(axis=0))
    np.testing.assert_allclose(weights[:, 8], weights[:, 9])
    np.testing.assert_array_equal(coefficient_cells["outcome"]["fold_0_units_coefficients"], beta)


def test_retained_masks_restore_units_and_unavailable_fold(coefficient_cells):
    """Input omitted/failed fold; missing weights are NaN, with original unit axis intact."""
    cell = coefficient_cells["context"]
    cell["fold_0_retained_mask"] = np.array([True, False, True])
    cell["fold_0_scale_hz"][1] = 0
    cell["fold_0_units_coefficients"] = np.array([[3., .5]])
    cell["meta"]["representations"]["units"]["folds"][1].update(available=False, reason="missing_class")
    del cell["fold_1_units_coefficients"]
    weights = build_group_weights({"context": cell}, ("A:0", "A:1", "A:2"))
    assert np.isnan(weights[0, 0, 1]) and weights[0, 0, 2] == .5
    assert np.isnan(weights[1, :2]).all()
    assert np.isnan(weights[:, 2:]).all()


@pytest.mark.parametrize("problem", ["class_order", "side", "unit_order", "duplicate_ids", "duplicate_fold",
    "missing_fold", "unknown_fold", "mask_shape", "mask_dtype", "coefficient_shape", "infinite", "zero_scale"])
def test_malformed_saved_weights_fail(coefficient_cells, problem):
    """Input malformed class/fold/unit/feature metadata; reject without reinterpreting axes."""
    cell = coefficient_cells["context"]
    ids = ("A:0", "A:1", "A:2")
    if problem == "class_order":
        cell["meta"]["class_names"].reverse()
    elif problem == "side":
        coefficient_cells["left_switch_vs_stay"]["meta"]["conditioning_side"] = "right"
    elif problem == "unit_order":
        cell["meta"]["unit_ids"].reverse()
    elif problem == "duplicate_ids":
        ids = ("A:0", "A:0", "A:2")
    elif problem == "duplicate_fold":
        cell["meta"]["representations"]["units"]["folds"][1]["fold"] = 0
    elif problem == "missing_fold":
        cell["meta"]["representations"]["units"]["folds"].pop()
    elif problem == "unknown_fold":
        cell["meta"]["representations"]["units"]["folds"][0]["fold"] = 5
    elif problem == "mask_shape":
        cell["fold_0_retained_mask"] = np.ones(2, bool)
    elif problem == "mask_dtype":
        cell["fold_0_retained_mask"] = np.ones(3, int)
    elif problem == "coefficient_shape":
        cell["fold_0_units_coefficients"] = np.zeros((2, 3))
    elif problem == "infinite":
        cell["fold_0_units_coefficients"][0, 0] = np.inf
    elif problem == "zero_scale":
        cell["fold_0_scale_hz"][0] = 0
    with pytest.raises(ValueError):
        build_group_weights(coefficient_cells, ids)


def test_fold_ids_are_explicit_not_list_positions(coefficient_cells):
    """Input reordered fold records; fixed fold 0..4 output is unchanged."""
    before = build_group_weights(coefficient_cells, ("A:0", "A:1", "A:2"))
    coefficient_cells["context"]["meta"]["representations"]["units"]["folds"].reverse()
    np.testing.assert_equal(build_group_weights(coefficient_cells, ("A:0", "A:1", "A:2")), before)


def test_signed_means_average_ranks_and_consistency():
    """No inputs; exact expected means/ranks establish five-fold and four-positive eligibility."""
    weights = np.zeros((5, 10, 5))
    weights[:, 0] = [8, 4, 4, -2, 1]
    weights[:, 2, 0] = [1, 1, 1, 1, -1]
    weights[:, 2, 1] = [1, 1, 1, -1, -1]
    weights[0, 2, 2] = np.nan
    summary = summarize_group_weights(weights, GROUP_IDS, tuple(f"u{i}" for i in range(5)))
    group = summary.loc[summary.group_index == 0].sort_values("unit_index")
    np.testing.assert_allclose(group.percentile, [.9, .6, .6, .1, .3])
    np.testing.assert_array_equal(group.candidate, [True, True, True, False, True])
    group = summary.loc[summary.group_index == 2].set_index("unit_index")
    assert group.loc[0, "candidate"] and group.loc[0, "positive_folds"] == 4
    assert not group.loc[1, "candidate"] and group.loc[1, "positive_folds"] == 3
    assert np.isnan(group.loc[2, "mean_weight"]) and group.loc[2, "available_folds"] == 4


def test_unique_assignment_precedes_cap_and_never_backfills():
    """No inputs; overlapping top ranks stay assigned to context even if its cap hides units."""
    weights = np.zeros((5, 10, 3))
    weights[:, 0] = [5, 4, 3]
    weights[:, 2] = [5, 4, 0]
    summary = summarize_group_weights(weights, GROUP_IDS, ("u0", "u1", "u2"))
    small = assign_unit_groups(summary, 1)
    large = assign_unit_groups(summary, 3)
    np.testing.assert_array_equal(small.group_index, [0, 0, 0])
    np.testing.assert_array_equal(small.group_index, large.group_index)
    assert small.displayed.sum() == 1 and large.displayed.sum() == 3
    assert small.loc[small.displayed, "unit_id"].tolist() == ["u0"]
    assert small.unit_id.is_unique and small.loc[0, "tied"]


def test_ties_prefer_positive_fold_count_then_group_order():
    """No inputs; percentile ties use consistency before canonical order, not raw magnitude."""
    weights = np.zeros((5, 10, 1))
    weights[:, 0, 0] = [2, 2, 2, 2, -.1]
    weights[:, 2, 0] = 1
    selected = assign_unit_groups(summarize_group_weights(weights, GROUP_IDS, ("u",)), 10)
    assert selected.loc[0, "group_index"] == 2
    assert selected.loc[0, "runner_up_group"] == GROUP_IDS[0]
    assert selected.loc[0, "percentile"] == .5


@pytest.mark.parametrize("units", [0, 1, 4])
def test_empty_and_unavailable_groups_are_not_filled(units):
    """Input population size; missing weights produce no invented candidates or displayed rows."""
    summary = summarize_group_weights(np.full((5, 10, units), np.nan), GROUP_IDS, tuple(f"u{i}" for i in range(units)))
    assigned = assign_unit_groups(summary, 10)
    assert len(assigned) == units and not assigned.displayed.any()
    assert (assigned.group_index == -1).all()


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_caps_are_rejected(cap):
    """Input invalid dimensionless cap; no implicit truncation or Boolean integer acceptance."""
    summary = summarize_group_weights(np.ones((5, 10, 1)), GROUP_IDS, ("u",))
    with pytest.raises(ValueError):
        assign_unit_groups(summary, cap)
