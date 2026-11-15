"""Current-side eligibility, fold-local fits, and exact matched OOF comparisons."""

from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from src.neural_analysis.population_decoding.targets import (
    TARGET_NAMES, MODEL_TARGET_NAMES, build_population_targets, build_model_target,
)
from src.neural_analysis.population_decoding.pipeline import fit_target, build_switch_comparison
from src.neural_analysis.population_decoding.scoring import score_predictions


def test_seven_models_reuse_five_unchanged_source_columns():
    """No inputs; return no data, verify class semantics without adding source axes."""
    source = build_population_targets([1]*5, [1, 1, 0, 0, 1], [np.nan]*5, [0]*5)
    before = source.values.copy()
    assert len(TARGET_NAMES) == 5 and len(MODEL_TARGET_NAMES) == 7
    left = build_model_target("left_switch_vs_stay", source)
    right = build_model_target("right_switch_vs_stay", source)
    np.testing.assert_array_equal(left.labels, [0, 1, -1, -1, -1])
    np.testing.assert_array_equal(right.labels, [-1, -1, 0, 1, -1])
    assert left.class_names == right.class_names == ("stay", "switch")
    assert left.conditioning_side == "left" and right.conditioning_side == "right"
    assert left.reasons[2] == "opposite_current_side"
    assert left.reasons[-1] == "terminal_trial"
    for i, name in enumerate(TARGET_NAMES):
        model = build_model_target(name, source)
        np.testing.assert_array_equal(model.labels, source.values[:, i])
        np.testing.assert_array_equal(model.valid, source.valid[:, i])
    np.testing.assert_array_equal(source.values, before)


@pytest.mark.parametrize("next_action,manual", [("no_choice", 0), (0, 1), (0, np.nan)])
def test_invalid_immediate_successor_never_becomes_stay(next_action, manual):
    """Input invalid next action/manual flag; new target stays unknown without skipping rows."""
    source = build_population_targets([1, 0, 0], [1, next_action, 0], [0]*3, [0, manual, 0])
    model = build_model_target("left_switch_vs_stay", source)
    assert not model.valid[0] and model.labels[0] == -1


@pytest.mark.parametrize("target,side", [("left_switch_vs_stay", 1), ("right_switch_vs_stay", 0)])
def test_new_fit_scope_signal_and_fold_assignment(synthetic_population, target, side):
    """Input seeded population/model/side; predictions occupy only same-side whole windows."""
    cell = fit_target(synthetic_population, "PFC", target)
    rows = cell["row_positions"]
    assert (synthetic_population.targets.values[rows, 1] == side).all()
    assert cell["units_probabilities"].shape == (len(rows), 4, 2)
    np.testing.assert_array_equal(cell["fold_ids"], synthetic_population.fold_ids[rows])
    assert cell["meta"]["class_names"] == ["stay", "switch"]
    for rep in ("units", "pcs"):
        assert cell["meta"]["representations"][rep]["pooled"]["balanced_accuracy"] > .95


def test_opposite_side_and_heldout_activity_cannot_change_training(synthetic_population):
    """Input Hz population; perturb excluded or fold-zero test rates, leave fitted fold zero unchanged."""
    original = fit_target(synthetic_population, "PFC", "left_switch_vs_stay")
    region = synthetic_population.regions["PFC"]
    rates = region.rates_hz.copy()
    excluded = (synthetic_population.targets.values[:, 1] == 0) | (synthetic_population.fold_ids == 0)
    rates[excluded] += 1000
    changed = replace(synthetic_population, regions={"PFC": replace(region, rates_hz=rates)})
    result = fit_target(changed, "PFC", "left_switch_vs_stay")
    for key in ("fold_0_mean_hz", "fold_0_scale_hz", "fold_0_retained_mask",
                "fold_0_pca_components", "fold_0_units_coefficients", "fold_0_pcs_coefficients"):
        np.testing.assert_array_equal(original[key], result[key])


@pytest.fixture
def switch_cells(synthetic_population):
    """Return pure same-run all-trial/within-left cell dictionaries; probabilities are W,B,2."""
    return (fit_target(synthetic_population, "PFC", "left_choice_switch"),
            fit_target(synthetic_population, "PFC", "left_switch_vs_stay"))


def test_comparison_matches_direct_rescore_without_changing_reference(synthetic_population, switch_cells):
    """Input saved-style cells; compare all-trial/ref-matched/new metrics on exact shared rows."""
    reference, new = switch_cells
    before = deepcopy(reference)
    comparison = build_switch_comparison(synthetic_population.targets.values[:, 1], reference, new)
    indices = comparison["reference_window_indices"]
    np.testing.assert_array_equal(reference["row_positions"][indices], new["row_positions"])
    for rep in ("units", "pcs"):
        expected = score_predictions(np.repeat(new["labels"], 4),
            reference[f"{rep}_probabilities"][indices].reshape(-1, 2), (0, 1),
            reference[f"{rep}_log_probabilities"][indices].reshape(-1, 2))
        record = comparison["meta"]["representations"][rep]
        assert record["reference_matched_side"]["scores"] == expected
        assert record["reference_all_trials"]["scores"] == reference["meta"]["representations"][rep]["pooled"]
        assert record["within_side"]["trial_count"] == len(new["labels"])
        assert len(record["folds"]) == 5
    np.testing.assert_array_equal(reference["units_probabilities"], before["units_probabilities"])
    assert reference["meta"] == before["meta"]


@pytest.mark.parametrize("field", ["row_positions", "bin_indices", "fold_ids", "labels"])
def test_misaligned_comparison_is_rejected(synthetic_population, switch_cells, field):
    """Input named mapping field; wrong rows/bins/folds/labels cannot be silently intersected."""
    reference, new = switch_cells
    new = deepcopy(new)
    new[field] = new[field].copy()
    new[field][0] += 1
    with pytest.raises(ValueError):
        build_switch_comparison(synthetic_population.targets.values[:, 1], reference, new)


def test_incomplete_new_predictions_do_not_shrink_matched_scope(synthetic_population, switch_cells):
    """Input cells; missing one held-out window disables new primary score but preserves reference."""
    reference, new = switch_cells
    for rep in ("units", "pcs"):
        new[f"{rep}_probabilities"][0] = np.nan
        new[f"{rep}_log_probabilities"][0] = np.nan
    comparison = build_switch_comparison(synthetic_population.targets.values[:, 1], reference, new)
    for rep, record in comparison["meta"]["representations"].items():
        assert record["within_side"]["scores"]["required_rows"] == len(new["labels"])*4
        assert record["within_side"]["scores"]["balanced_accuracy"] is None
        assert record["reference_matched_side"]["scores"]["balanced_accuracy"] is not None
        assert record["differences"]["balanced_accuracy"] is None
