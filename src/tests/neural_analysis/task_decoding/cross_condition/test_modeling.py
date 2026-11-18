"""Independent weighted logistic oracles, regional coverage and local failures."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.neural_analysis.task_decoding.cross_condition import modeling


@pytest.mark.parametrize("weighting,expected", [
    ("equal_trial", [1., 1., 1., 1., 1., 0.]),
    ("equal_cell", [5 / 4, 5 / 4, 5 / 6, 5 / 6, 5 / 6, 0.]),
])
def test_exact_trial_weight_normalization_and_off_source_zeros(weighting, expected):
    """Bool source(W=6,2) yields float64 total trial weights(W,), dimensionless sum N=5."""
    cells = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 0]], dtype=bool)
    original = cells.copy()
    weights = modeling.training_trial_weights(cells, weighting)
    assert weights.dtype == np.float64
    np.testing.assert_allclose(weights, expected, atol=1e-15)
    assert weights.sum() == pytest.approx(5)
    np.testing.assert_array_equal(cells, original)


def test_missing_source_class_retains_equal_trial_but_zeroes_equal_cell_weights():
    """A single present class is unavailable for fitting; describe its actual source trials."""
    cells = np.array([[1, 0], [0, 0], [1, 0]], dtype=bool)
    np.testing.assert_array_equal(modeling.training_trial_weights(cells, "equal_trial"), [1, 0, 1])
    np.testing.assert_array_equal(modeling.training_trial_weights(cells, "equal_cell"), [0, 0, 0])
    empty = np.zeros((3, 2), dtype=bool)
    np.testing.assert_array_equal(modeling.training_trial_weights(empty, "equal_trial"), 0)
    np.testing.assert_array_equal(modeling.training_trial_weights(empty, "equal_cell"), 0)


@pytest.mark.parametrize("cells,weighting", [
    (np.zeros(3, dtype=bool), "equal_trial"), (np.zeros((3, 3), dtype=bool), "equal_trial"),
    (np.zeros((3, 2)), "equal_trial"), (np.ones((3, 2), dtype=bool), "equal_cell"),
    (np.zeros((3, 2), dtype=bool), "unknown"),
])
def test_invalid_cell_shapes_overlap_or_scheme_are_input_errors(cells, weighting):
    """Only disjoint bool(W,2) actual-class masks and the two canonical schemes are valid."""
    with pytest.raises(ValueError):
        modeling.training_trial_weights(cells, weighting)


def test_every_regional_window_weight_and_split_matches_independent_sklearn_oracle(regional_transfer_data):
    """Verify all H/G/S affine fits, float64 trial means, weights and exact role counts.

    Scores(W,B,K) keep original PC scales; oracles construct trial-major/bin-major
    rows independently and normalize weights after each region's coverage mask.
    Coefficients/intercepts use log-odds per PC-score unit / log-odds.
    """
    inputs, regions = regional_transfer_data
    expected_bins = ((2, 3, 4, 5), (2, 3), (4, 5))
    for scores, positions in regions.values():
        original_scores = scores.copy()
        model = modeling.fit_region_transfers(scores, positions, inputs)
        component_count = scores.shape[2]
        assert model.coefficients.shape == (3, 2, 18, component_count)
        assert model.intercepts.shape == model.fit_status.shape == model.fit_reason.shape == (3, 2, 18)
        assert model.trial_probabilities.shape == (3, 2, 18, len(positions))
        assert model.source_mask.shape == model.test_mask.shape == (18, len(positions))
        assert model.source_trial_weights.shape == (2, 18, len(positions))
        assert model.eligible_cell_counts.shape == (18, 2, 2)
        assert model.coefficients.dtype == model.intercepts.dtype == model.trial_probabilities.dtype == np.float64
        assert model.source_mask.dtype == model.test_mask.dtype == bool
        assert np.issubdtype(model.eligible_cell_counts.dtype, np.integer)
        np.testing.assert_array_equal(model.pc_ids, np.arange(1, component_count + 1))
        np.testing.assert_array_equal(model.choice_positions, positions)
        assert not hasattr(model, "scores") and not hasattr(model, "fold_ids")
        for split, definition in enumerate(inputs.transfers):
            target = inputs.target_names.index(definition.target_name)
            source_cells = inputs.source_cells[positions, split]
            test_cells = inputs.test_cells[positions, split]
            ids = inputs.choices.trial_id.to_numpy()[positions]
            if definition.split_id == "context_correct_rewarded_to_other_outcomes":
                np.testing.assert_array_equal(source_cells, np.column_stack([ids == 2, np.isin(ids, [0, 19])]))
            elif definition.split_id == "context_omission_to_other_outcomes":
                np.testing.assert_array_equal(source_cells, np.column_stack([ids == 3, ids == 1]))
            source = source_cells.any(axis=1)
            test = test_cells.any(axis=1)
            np.testing.assert_array_equal(model.source_mask[split], source)
            np.testing.assert_array_equal(model.test_mask[split], test)
            np.testing.assert_array_equal(model.eligible_cell_counts[split], [source_cells.sum(axis=0), test_cells.sum(axis=0)])
            actual = inputs.target_values[positions, target]
            both_classes = source_cells.any(axis=0).all()
            for weighting in range(2):
                trial_weights = np.zeros(len(positions))
                for label in (0, 1):
                    selected = source_cells[:, label]
                    if weighting == 0:
                        trial_weights[selected] = 1
                    elif both_classes:
                        trial_weights[selected] = source.sum() / (2 * selected.sum())
                np.testing.assert_allclose(model.source_trial_weights[weighting, split], trial_weights)
                assert trial_weights.sum() == pytest.approx(source.sum() if weighting == 0 or both_classes else 0)
                if not both_classes:
                    assert definition.split_id == "context_correct_rewarded_to_other_outcomes"
                    assert np.all(model.fit_status[:, weighting, split] == "unavailable")
                    assert np.all(model.fit_reason[:, weighting, split] == "single_class_training")
                    assert np.isnan(model.coefficients[:, weighting, split]).all()
                    assert np.isnan(model.intercepts[:, weighting, split]).all()
                    assert np.isnan(model.trial_probabilities[:, weighting, split]).all()
                    continue
                assert np.all(model.fit_status[:, weighting, split] == "ok")
                assert np.all(model.fit_reason[:, weighting, split] == "")
                for window, bins in enumerate(expected_bins):
                    matrix = np.array([scores[trial, time] for trial in np.flatnonzero(source) for time in bins])
                    labels = np.array([actual[trial] for trial in np.flatnonzero(source) for _ in bins])
                    row_weights = np.array([trial_weights[trial] / len(bins) for trial in np.flatnonzero(source) for _ in bins])
                    oracle = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
                                                class_weight=None, tol=1e-4, max_iter=5000)
                    oracle.fit(matrix, labels, sample_weight=row_weights)
                    np.testing.assert_allclose(model.coefficients[window, weighting, split], oracle.coef_[0], atol=1e-11)
                    np.testing.assert_allclose(model.intercepts[window, weighting, split], oracle.intercept_[0], atol=1e-11)
                    expected = expit(scores[source | test][:, bins] @ oracle.coef_[0] + oracle.intercept_[0]).mean(axis=1)
                    probabilities = model.trial_probabilities[window, weighting, split]
                    np.testing.assert_allclose(probabilities[source | test], expected, atol=1e-12)
                    assert np.isnan(probabilities[~(source | test)]).all()
        np.testing.assert_array_equal(scores, original_scores)


def test_regional_coverage_changes_counts_before_weight_normalization(regional_transfer_data):
    """HPC drops original IDs0/6/19: context-correct source counts become (2,1), N=3."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["HPC"]
    model = modeling.fit_region_transfers(scores, positions, inputs)
    np.testing.assert_array_equal(model.eligible_cell_counts[0, 0], [2, 1])
    ids = inputs.choices.trial_id.to_numpy()[positions]
    weights = model.source_trial_weights[1, 0]
    np.testing.assert_array_equal(weights[np.isin(ids, [2, 3])], [.75, .75])
    np.testing.assert_array_equal(weights[ids == 1], [1.5])
    assert weights.sum() == 3


def test_common_masks_and_target_validity_are_intersected_before_weights(regional_transfer_data):
    """Exclude three otherwise source-cell members for baseline, conflict and target missingness."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    baseline, conflict, valid = inputs.baseline_valid.copy(), inputs.conflict_mask.copy(), inputs.target_valid.copy()
    baseline[0], conflict[1], valid[2, 0] = False, True, False
    altered = replace(inputs, baseline_valid=baseline, conflict_mask=conflict, target_valid=valid)
    model = modeling.fit_region_transfers(scores, positions, altered)
    np.testing.assert_array_equal(model.eligible_cell_counts[0], [[1, 1], [2, 3]])
    assert not model.source_mask[0, :3].any() and not model.test_mask[0, :3].any()
    np.testing.assert_array_equal(model.source_trial_weights[:, 0, :3], 0)
    np.testing.assert_array_equal(model.source_trial_weights[:, 0].sum(axis=1), [2, 2])
    assert np.isnan(model.trial_probabilities[:, :, 0, :3]).all()


def test_balanced_cells_give_identical_fits_under_both_weightings(regional_transfer_data):
    """Original IDs0..7 balance every source cell without rescaling original PC scores."""
    inputs, regions = regional_transfer_data
    scores, _ = regions["PFC"]
    positions = np.arange(8)
    model = modeling.fit_region_transfers(scores[positions], positions, inputs)
    np.testing.assert_array_equal(model.source_trial_weights[0], model.source_trial_weights[1])
    np.testing.assert_allclose(model.coefficients[:, 0], model.coefficients[:, 1], atol=1e-12)
    np.testing.assert_allclose(model.intercepts[:, 0], model.intercepts[:, 1], atol=1e-12)
    np.testing.assert_allclose(model.trial_probabilities[:, 0], model.trial_probabilities[:, 1], atol=1e-12)


def test_selected_rows_reach_readout_in_trial_major_bin_major_order(regional_transfer_data, monkeypatch):
    """Observe float X(M,K), binary y(M,) and row weights summing to each trial total."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    calls = []

    def record_readout(matrix, labels, weights, family):
        """Copy finite M-row readout inputs; return zero K-vector/logit and no failure."""
        assert family == "categorical"
        calls.append((matrix.copy(), labels.copy(), weights.copy()))
        return np.zeros(matrix.shape[1]), 0., ""

    monkeypatch.setattr(modeling, "fit_readout", record_readout)
    modeling.fit_region_transfers(scores, positions, inputs)
    assert len(calls) == 108
    assert sum(item.split_id == "context_incorrect_to_correct" for item in inputs.transfers) == 1
    for bins in ((2, 3, 4, 5), (2, 3), (4, 5)):
        expected_matrix = np.array([scores[trial, time] for trial in (0, 1, 2, 3, 15) for time in bins])
        expected_labels = np.repeat([1, 1, 0, 0, 1], len(bins))
        for totals in ([1., 1., 1., 1., 1.], [5 / 6, 5 / 6, 5 / 4, 5 / 4, 5 / 6]):
            expected_weights = np.repeat(np.array(totals) / len(bins), len(bins))
            assert any(np.array_equal(matrix, expected_matrix) and np.array_equal(labels, expected_labels)
                       and np.allclose(weights, expected_weights) for matrix, labels, weights in calls)


def test_new_rewarded_context_model_retains_known_hpc_missing_source_class(regional_transfer_data):
    """HPC excludes both Left rewarded IDs0/19: source counts(1,0), no borrowing from test."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["HPC"]
    model = modeling.fit_region_transfers(scores, positions, inputs)
    split = next(i for i, item in enumerate(inputs.transfers)
                 if item.split_id == "context_correct_rewarded_to_other_outcomes")
    np.testing.assert_array_equal(model.eligible_cell_counts[split], [[1, 0], [2, 4]])
    assert model.source_trial_weights[0, split].sum() == 1
    assert not model.source_trial_weights[1, split].any()
    assert np.all(model.fit_status[:, :, split] == "unavailable")
    assert np.all(model.fit_reason[:, :, split] == "single_class_training")
    assert np.isnan(model.coefficients[:, :, split]).all()
    assert np.isnan(model.trial_probabilities[:, :, split]).all()
    assert np.count_nonzero(model.fit_status == "ok") == 102


@pytest.mark.parametrize("split_id", [
    "context_correct_rewarded_to_other_outcomes", "context_omission_to_other_outcomes",
    "context_incorrect_to_correct",
])
def test_complementary_outcome_scores_cannot_change_source_coefficients(regional_transfer_data, split_id):
    """Perturb only this model's test trials(W,B,K); all H/G source fits stay identical."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    split = next(i for i, item in enumerate(inputs.transfers) if item.split_id == split_id)
    baseline = modeling.fit_region_transfers(scores, positions, inputs)
    changed = scores.copy()
    changed[baseline.test_mask[split]] += 13
    actual = modeling.fit_region_transfers(changed, positions, inputs)
    np.testing.assert_array_equal(actual.coefficients[:, :, split], baseline.coefficients[:, :, split])
    np.testing.assert_array_equal(actual.intercepts[:, :, split], baseline.intercepts[:, :, split])
    np.testing.assert_array_equal(actual.eligible_cell_counts[split], baseline.eligible_cell_counts[split])
    source = baseline.source_mask[split]
    np.testing.assert_array_equal(actual.trial_probabilities[:, :, split, source],
                                  baseline.trial_probabilities[:, :, split, source])


@pytest.mark.parametrize("pattern", [[-10., 2., 2., 2.], [-.1, -.1, -.1, 10.]])
def test_saved_trial_means_apply_expit_before_averaging_and_thresholding(regional_transfer_data, monkeypatch, pattern):
    """Handcrafted bin logits distinguish mean probability from sigmoid(mean)/majority vote."""
    inputs, _ = regional_transfer_data
    scores = np.full((16, 8, 1), -100.)
    scores[:, 2:6, 0] = pattern

    def known_readout(matrix, labels, weights, family):
        """For M-row inputs return beta(1,)=1 and zero intercept in logit units."""
        return np.ones(1), 0., ""

    monkeypatch.setattr(modeling, "fit_readout", known_readout)
    model = modeling.fit_region_transfers(scores, np.arange(16), inputs)
    selected = model.source_mask[0] | model.test_mask[0]
    expected = expit(pattern).mean()
    assert expected > .5
    assert not np.isclose(expected, expit(np.mean(pattern)))
    assert not np.isclose(expected, np.mean(np.array(pattern) >= 0))
    np.testing.assert_allclose(model.trial_probabilities[0, :, 0, selected], expected)
    assert np.isnan(model.trial_probabilities[0, :, 0, ~selected]).all()


@pytest.mark.parametrize("window", [0, 1, 2])
def test_bins_outside_selected_window_cannot_affect_its_fit_or_outputs(regional_transfer_data, window):
    """Mutate only unselected bins in finite W,B,K scores; matched-window outputs stay fixed."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    expected = modeling.fit_region_transfers(scores, positions, inputs)
    changed = scores.copy()
    changed[:, ~inputs.window_bin_masks[window]] = 500.
    actual = modeling.fit_region_transfers(changed, positions, inputs)
    for field in ("coefficients", "intercepts", "trial_probabilities"):
        np.testing.assert_allclose(getattr(actual, field)[window], getattr(expected, field)[window], atol=1e-12)


def test_missing_source_class_is_local_and_retains_eligibility_counts(regional_transfer_data):
    """Removing original IDs2/3 leaves three class-1 source trials; complementary test is untouched."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    retained = ~inputs.choices.trial_id.isin([2, 3]).to_numpy()
    model = modeling.fit_region_transfers(scores[retained], positions[retained], inputs)
    np.testing.assert_array_equal(model.eligible_cell_counts[0], [[0, 3], [2, 3]])
    assert np.all(model.fit_reason[:, :, 0] == "single_class_training")
    assert np.all(model.fit_status[:, :, 0] == "unavailable")
    assert np.isnan(model.coefficients[:, :, 0]).all() and np.isnan(model.intercepts[:, :, 0]).all()
    assert np.isnan(model.trial_probabilities[:, :, 0]).all()
    assert model.source_trial_weights[0, 0].sum() == 3
    assert not model.source_trial_weights[1, 0].any()
    assert np.all(model.fit_status[:, :, 1] == "ok")


@pytest.mark.parametrize("reason", ["nonconvergence", "numerical_fit_failure: synthetic", "nonfinite_parameters"])
def test_one_readout_failure_does_not_erase_counts_weights_or_other_models(regional_transfer_data, monkeypatch, reason):
    """A failed H/G/S fit has NaN parameters/probabilities and retained eligibility, without fallback."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    baseline = modeling.fit_region_transfers(scores, positions, inputs)
    original_fit = modeling.fit_readout
    calls = []

    def fail_once(matrix, labels, weights, family):
        """Return one unavailable readout tuple, then delegate all other M-row fits unchanged."""
        calls.append(1)
        return (None, np.nan, reason) if len(calls) == 1 else original_fit(matrix, labels, weights, family)

    monkeypatch.setattr(modeling, "fit_readout", fail_once)
    model = modeling.fit_region_transfers(scores, positions, inputs)
    failed = model.fit_status != "ok"
    assert failed.sum() == 1
    assert model.fit_reason[failed].item() == reason
    assert np.isnan(model.coefficients[failed]).all() and np.isnan(model.intercepts[failed]).all()
    assert np.isnan(model.trial_probabilities[failed]).all()
    np.testing.assert_array_equal(model.eligible_cell_counts, baseline.eligible_cell_counts)
    np.testing.assert_array_equal(model.source_trial_weights, baseline.source_trial_weights)
    np.testing.assert_allclose(model.trial_probabilities[~failed], baseline.trial_probabilities[~failed])


def test_nonfinite_affine_output_invalidates_only_its_model_without_clipping(regional_transfer_data, monkeypatch):
    """Finite extreme beta causes overflow before expit; invalidate parameters and every selected output."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    calls = []

    def overflowing_once(matrix, labels, weights, family):
        """Return finite beta(K,) extreme once, otherwise zero logits; no estimator randomness."""
        calls.append(1)
        beta = np.full(matrix.shape[1], np.finfo(float).max) if len(calls) == 1 else np.zeros(matrix.shape[1])
        return beta, 0., ""

    monkeypatch.setattr(modeling, "fit_readout", overflowing_once)
    model = modeling.fit_region_transfers(scores, positions, inputs)
    failed = model.fit_status != "ok"
    assert failed.sum() == 1 and model.fit_reason[failed].item() == "nonfinite_prediction"
    assert np.isnan(model.coefficients[failed]).all() and np.isnan(model.intercepts[failed]).all()
    assert np.isnan(model.trial_probabilities[failed]).all()
    assert model.eligible_cell_counts.sum() > 0 and model.source_trial_weights.sum() > 0


def test_choice_only_signal_keeps_controlled_transfer_reversals_below_chance(regional_transfer_data):
    """Context/Reward/Value pairing reversals fail while direct choice transfer succeeds."""
    inputs, _ = regional_transfer_data
    action = inputs.target_values[:, 1]
    scores = np.broadcast_to(3 * (2 * action - 1)[:, None, None], (16, 8, 1)).copy()
    model = modeling.fit_region_transfers(scores, np.arange(16), inputs)
    for split, expected_test_accuracy in ((0, 0.), (4, 1.), (9, 0.), (13, 0.)):
        target = inputs.target_names.index(inputs.transfers[split].target_name)
        actual = inputs.target_values[:, target]
        predicted = model.trial_probabilities[0, 0, split] >= .5
        assert np.mean(predicted[model.source_mask[split]] == actual[model.source_mask[split]]) == 1
        assert np.mean(predicted[model.test_mask[split]] == actual[model.test_mask[split]]) == expected_test_accuracy


@pytest.mark.parametrize("defect", ["duplicate", "reversed", "outside", "nonfinite", "bins"])
def test_structural_region_errors_raise_before_fitting(regional_transfer_data, defect):
    """Require finite scores(W,B,K), ordered unique choice positions(W,) and matching B bins."""
    inputs, regions = regional_transfer_data
    scores, positions = (array.copy() for array in regions["PFC"])
    if defect == "duplicate":
        positions[1] = positions[0]
    elif defect == "reversed":
        positions = positions[::-1]
    elif defect == "outside":
        positions[-1] = len(inputs.choices)
    elif defect == "nonfinite":
        scores[0, 0, 0] = np.nan
    else:
        inputs = replace(inputs, window_bin_masks=inputs.window_bin_masks[:, :-1])
    with pytest.raises(ValueError):
        modeling.fit_region_transfers(scores, positions, inputs)
