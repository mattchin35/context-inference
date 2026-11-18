"""Master-fold and pooled scientific-core tests."""

import numpy as np
import pytest

from src.neural_analysis.task_decoding import modeling
from src.neural_analysis.task_decoding.pooled.config import TrainingWindow
from src.neural_analysis.task_decoding.pooled import decoding


def make_pooled_fixture() -> dict[str, object]:
    """Return a small deterministic binary pooled-decoding fixture.

    Returns
    -------
    dict[str, object]
        Aligned target, condition, split, and Hz tensors. Trial axes have
        length 12, time axes length three, and each regional unit axis length
        two.
    """
    target_values = np.tile(np.array([0, 1], dtype=int), 6)
    trial_signal = target_values[:, None, None].astype(float)
    time_offsets = np.array([0.0, 0.3, -0.2])[None, :, None]
    pfc_rates_hz = 2.0 + trial_signal * np.array([[[2.0, -0.5]]]) + time_offsets
    hpc_rates_hz = 3.0 + trial_signal * np.array([[[1.5, 0.4]]]) - time_offsets
    first_half = np.arange(6, dtype=int)
    second_half = np.arange(6, 12, dtype=int)
    outer_split = modeling.SplitPlan(
        fold_ids=np.repeat(np.array([1, 0], dtype=int), 6),
        splits=(
            modeling.GroupedSplit(0, first_half, second_half),
            modeling.GroupedSplit(1, second_half, first_half),
        ),
        is_available=True,
        unavailable_reason=None,
    )
    return {
        "training_window": TrainingWindow(
            name="pre_choice",
            start_s=-0.2,
            stop_s=0.0,
            bin_indices=np.array([0, 1], dtype=int),
        ),
        "target_identifier": "synthetic_action",
        "target_family": "categorical",
        "target_values": target_values,
        "master_eligible": np.ones(12, dtype=bool),
        "block_ids": np.repeat(np.arange(6), 2),
        "condition_masks": {
            "correct_rewarded": np.ones(12, dtype=bool),
            "incorrect_unrewarded": np.ones(12, dtype=bool),
        },
        "outer_split": outer_split,
        "pfc_rate_tensor_hz": pfc_rates_hz,
        "hpc_rate_tensor_hz": hpc_rates_hz,
        "pfc_unit_ids": ("pfc:1", "pfc:2"),
        "hpc_unit_ids": ("hpc:1", "hpc:2"),
        "evaluation_time_centers_s": np.array([-0.15, -0.05, 0.05]),
        "pfc_pc_count": 2,
        "hpc_pc_count": 2,
    }


def test_master_eligibility_preserves_target_mask_and_reason_precedence():
    """Independent masks and the first failing full-table reason should be stable."""
    baseline = np.array([False, True, True, True, False, True], dtype=bool)
    target = np.array([False, False, True, True, True, True], dtype=bool)
    bilateral = np.array([False, False, False, True, True, True], dtype=bool)

    result = decoding.build_master_eligibility(baseline, target, bilateral)

    np.testing.assert_array_equal(result.target_eligible, target)
    np.testing.assert_array_equal(
        result.master_eligible,
        np.array([False, False, False, True, False, True]),
    )
    reason_names = np.full(result.reason_index.shape, "", dtype="U32")
    unavailable = result.reason_index >= 0
    reason_names[unavailable] = np.asarray(decoding.POOLED_REASON_CODES)[
        result.reason_index[unavailable]
    ]
    assert reason_names.tolist() == [
        "baseline_ineligible",
        "target_ineligible",
        "neural_window_unavailable",
        "",
        "baseline_ineligible",
        "",
    ]


@pytest.mark.parametrize(
    ("baseline", "target", "bilateral"),
    [
        (np.ones((2, 1), dtype=bool), np.ones(2, dtype=bool), np.ones(2, dtype=bool)),
        (np.ones(2, dtype=int), np.ones(2, dtype=bool), np.ones(2, dtype=bool)),
        (np.ones(2, dtype=bool), np.ones(3, dtype=bool), np.ones(2, dtype=bool)),
    ],
)
def test_master_eligibility_rejects_malformed_axes(baseline, target, bilateral):
    """Eligibility inputs must be aligned one-dimensional Boolean trial masks."""
    with pytest.raises(ValueError, match="Boolean|one-dimensional|shape"):
        decoding.build_master_eligibility(baseline, target, bilateral)


def test_master_outer_split_projects_local_folds_back_to_tensor_rows():
    """Excluded tensor rows should retain -1 while eligible blocks remain intact."""
    target_values = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan])
    block_ids = np.array([0, 0, "excluded", 1, 1, 2, 2, 3, 3, 4, 4, "excluded"], dtype=object)
    master_eligible = np.array(
        [True, True, False, True, True, True, True, True, True, True, True, False],
        dtype=bool,
    )

    plan = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="numerical",
        fold_count=5,
    )

    assert plan.is_available
    np.testing.assert_array_equal(plan.fold_ids[~master_eligible], np.array([-1, -1]))
    assert np.all(plan.fold_ids[master_eligible] >= 0)
    for split in plan.splits:
        assert np.all(master_eligible[split.train_indices])
        assert np.all(master_eligible[split.test_indices])
        train_blocks = set(block_ids[split.train_indices])
        test_blocks = set(block_ids[split.test_indices])
        assert train_blocks.isdisjoint(test_blocks)


def test_master_outer_split_is_deterministic_and_condition_independent():
    """One target fold map should be reused even when a condition lacks classes."""
    block_ids = np.repeat(np.arange(6), 2)
    target_values = np.tile(np.array([0, 1], dtype=int), 6)
    master_eligible = np.ones(target_values.size, dtype=bool)

    first = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="categorical",
        fold_count=3,
    )
    repeated = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="categorical",
        fold_count=3,
    )

    assert first.is_available
    np.testing.assert_array_equal(first.fold_ids, repeated.fold_ids)
    condition_mask = target_values == 0
    assert any(
        np.unique(target_values[split.test_indices[condition_mask[split.test_indices]]]).size < 2
        for split in first.splits
    )
    np.testing.assert_array_equal(first.fold_ids, repeated.fold_ids)


def test_master_outer_split_returns_full_length_unavailable_plan():
    """Insufficient eligible groups should not shrink the tensor-row fold axis."""
    target_values = np.array([0, 1, 0, 1, 0, 1])
    block_ids = np.array([0, 0, 1, 1, 2, 2])
    master_eligible = np.array([True, True, False, False, False, False])

    plan = decoding.make_master_outer_split(
        target_values,
        block_ids,
        master_eligible,
        target_family="categorical",
        fold_count=3,
    )

    assert not plan.is_available
    assert plan.fold_ids.shape == target_values.shape
    assert np.all(plan.fold_ids == -1)
    assert plan.splits == ()
    assert plan.unavailable_reason


def test_stack_training_window_is_trial_major_and_gives_each_trial_unit_weight():
    """Stacked rows should align to repeated targets and equal trial influence."""
    feature_tensor = np.arange(24.0).reshape(3, 4, 2)
    targets = np.array([10.0, 20.0, 30.0])

    features, repeated_targets, sample_weights = decoding.stack_training_window(
        feature_tensor,
        targets,
    )

    np.testing.assert_array_equal(features, feature_tensor.reshape(12, 2))
    np.testing.assert_array_equal(repeated_targets, np.repeat(targets, 4))
    np.testing.assert_array_equal(sample_weights, np.full(12, 0.25))
    np.testing.assert_allclose(sample_weights.reshape(3, 4).sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("feature_tensor", "targets"),
    [
        (np.empty((0, 2, 1)), np.empty(0)),
        (np.empty((2, 0, 1)), np.arange(2.0)),
        (np.empty((2, 2, 0)), np.arange(2.0)),
        (np.ones((2, 2, 1)), np.arange(3.0)),
        (np.array([[[np.nan]]]), np.array([1.0])),
    ],
)
def test_stack_training_window_rejects_empty_nonfinite_or_misaligned_inputs(
    feature_tensor,
    targets,
):
    """Stacking rejects malformed tensors before an estimator can run."""
    with pytest.raises(ValueError):
        decoding.stack_training_window(feature_tensor, targets)


def test_fit_pooled_transforms_fits_exactly_two_window_restricted_regions(monkeypatch):
    """A pooled cell should fit one frozen transform per anatomical region."""
    fixture = make_pooled_fixture()
    original_fit = modeling.fit_region_transform
    observed_tensors = []

    def record_fit(training_rate_tensor_hz, requested_pc_count):
        """Record each regional fit tensor before calling production code."""
        observed_tensors.append(np.asarray(training_rate_tensor_hz).copy())
        return original_fit(training_rate_tensor_hz, requested_pc_count)

    monkeypatch.setattr(modeling, "fit_region_transform", record_fit)
    transforms = decoding.fit_pooled_transforms(
        fixture["pfc_rate_tensor_hz"][:6],
        fixture["hpc_rate_tensor_hz"][:6],
        fixture["training_window"].bin_indices,
        pfc_pc_count=2,
        hpc_pc_count=2,
        pfc_unit_ids=fixture["pfc_unit_ids"],
        hpc_unit_ids=fixture["hpc_unit_ids"],
    )

    assert len(observed_tensors) == 2
    assert all(tensor.shape == (6, 2, 2) for tensor in observed_tensors)
    assert transforms.pfc_reason is None
    assert transforms.hpc_reason is None
    np.testing.assert_array_equal(
        observed_tensors[0],
        fixture["pfc_rate_tensor_hz"][:6, :2, :],
    )
    np.testing.assert_array_equal(
        observed_tensors[1],
        fixture["hpc_rate_tensor_hz"][:6, :2, :],
    )


@pytest.mark.parametrize(
    ("region", "representation", "expected_ids"),
    [
        ("PFC", "units", ("pfc:1", "pfc:2")),
        ("HPC", "units", ("hpc:1", "hpc:2")),
        ("PFC+HPC", "units", ("pfc:1", "pfc:2", "hpc:1", "hpc:2")),
        ("PFC", "pca", ("PFC:PC1", "PFC:PC2")),
        ("HPC", "pca", ("HPC:PC1", "HPC:PC2")),
        ("PFC+HPC", "pca", ("PFC:PC1", "PFC:PC2", "HPC:PC1", "HPC:PC2")),
    ],
)
def test_pooled_model_feature_identity_and_coefficient_positions_are_stable(
    region,
    representation,
    expected_ids,
):
    """Regional feature order should be identical in records and coefficients."""
    fixture = make_pooled_fixture()
    training_rows = np.arange(6)
    transforms = decoding.fit_pooled_transforms(
        fixture["pfc_rate_tensor_hz"][training_rows],
        fixture["hpc_rate_tensor_hz"][training_rows],
        fixture["training_window"].bin_indices,
        pfc_pc_count=2,
        hpc_pc_count=2,
        pfc_unit_ids=fixture["pfc_unit_ids"],
        hpc_unit_ids=fixture["hpc_unit_ids"],
    )

    model = decoding.fit_pooled_model(
        transforms,
        fixture["pfc_rate_tensor_hz"][training_rows],
        fixture["hpc_rate_tensor_hz"][training_rows],
        fixture["target_values"][training_rows],
        fixture["training_window"].bin_indices,
        target_identifier=fixture["target_identifier"],
        target_family="categorical",
        region_configuration=region,
        representation=representation,
    )

    assert model.record.is_valid
    assert model.record.feature_ids == expected_ids
    assert model.record.coefficients.shape == (len(expected_ids),)
    assert model.record.train_trial_count == 6
    assert model.record.stacked_row_count == 12


def test_decode_reuses_one_model_across_time_and_reference_destinations(monkeypatch):
    """Fit counts depend on model identities; prediction counts include evaluations."""
    fixture = make_pooled_fixture()
    estimators = []

    class CountingClassifier:
        """Count fit and prediction calls while returning valid model artifacts."""

        def __init__(self):
            """Initialize call counters."""
            self.fit_calls = 0
            self.predict_calls = 0

        def fit(self, features, targets, *, sample_weight):
            """Expose a finite binary linear model and count the fit."""
            self.fit_calls += 1
            self.coef_ = np.arange(1, features.shape[1] + 1, dtype=float)[None, :]
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return deterministic finite probabilities and count evaluation."""
            self.predict_calls += 1
            positive = 1.0 / (1.0 + np.exp(-features[:, 0]))
            return np.column_stack((1.0 - positive, positive))

        def get_params(self, deep=False):
            """Return an empty JSON-compatible synthetic parameter mapping."""
            return {}

    def make_counting_estimator(*, target_family, parameters=None):
        """Return one new counting estimator for each model identity."""
        assert target_family == "categorical"
        assert parameters is None
        estimator = CountingClassifier()
        estimators.append(estimator)
        return estimator

    transform_calls = []
    original_transform_fit = modeling.fit_region_transform

    def count_transform_fit(tensor, requested_pc_count):
        """Count transform fits before delegating to the real implementation."""
        transform_calls.append((tensor.shape, requested_pc_count))
        return original_transform_fit(tensor, requested_pc_count)

    monkeypatch.setattr(modeling, "make_estimator", make_counting_estimator)
    monkeypatch.setattr(modeling, "fit_region_transform", count_transform_fit)
    result = decoding.decode_pooled_window_target(**fixture)

    condition_count = 2
    fold_count = 2
    model_kinds = 3 * 2
    assert len(transform_calls) == condition_count * fold_count * 2
    assert len(estimators) == condition_count * fold_count * model_kinds
    assert all(estimator.fit_calls == 1 for estimator in estimators)
    assert len(result.models) == condition_count * fold_count * model_kinds
    assert len(result.evaluations) == (2 * condition_count - 1) * fold_count * model_kinds
    assert sum(estimator.predict_calls for estimator in estimators) == (
        (2 * condition_count - 1) * fold_count * model_kinds * 3
    )


def test_native_transfer_rows_and_gaps_are_exact_including_reference_self_zero(monkeypatch):
    """Native and transfer use one destination subset and gaps subtract exact scores."""
    fixture = make_pooled_fixture()

    class FiniteClassifier:
        """Return condition-dependent but deterministic valid fitted models."""

        fit_counter = 0

        def fit(self, features, targets, *, sample_weight):
            """Assign a deterministic coefficient based on global fit order."""
            type(self).fit_counter += 1
            self.coef_ = np.full((1, features.shape[1]), float(type(self).fit_counter))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Convert the first feature and fitted coefficient to probabilities."""
            positive = 1.0 / (1.0 + np.exp(-self.coef_[0, 0] * features[:, 0]))
            return np.column_stack((1.0 - positive, positive))

        def get_params(self, deep=False):
            """Return no synthetic hyperparameters."""
            return {}

    monkeypatch.setattr(
        modeling,
        "make_estimator",
        lambda *, target_family, parameters=None: FiniteClassifier(),
    )
    result = decoding.decode_pooled_window_target(**fixture)

    for destination in fixture["condition_masks"]:
        for fold_id in (0, 1):
            for region in ("PFC", "HPC", "PFC+HPC"):
                for representation in ("pca", "units"):
                    native = result.evaluations[
                        (destination, destination, fold_id, region, representation)
                    ]
                    transfer = result.evaluations[
                        ("correct_rewarded", destination, fold_id, region, representation)
                    ]
                    gap = result.gaps[(destination, fold_id, region, representation)]
                    assert native.test_indices is transfer.test_indices
                    valid = (native.status == "valid") & (transfer.status == "valid")
                    np.testing.assert_allclose(
                        gap.scores[valid],
                        native.scores[valid] - transfer.scores[valid],
                    )
                    if destination == "correct_rewarded":
                        np.testing.assert_array_equal(gap.scores[valid], 0.0)


def test_modeling_failure_mapping_is_exhaustive_compact_and_rejects_unknown_states():
    """Every closed seam failure maps directly to one declared compact code."""
    expected = {
        modeling.FitFailureKind.CONVERGENCE_WARNING: "fit_convergence_failure",
        modeling.FitFailureKind.INVALID_COEFFICIENT_OR_INTERCEPT_SHAPE: "outer_estimator_failure",
        modeling.FitFailureKind.NONFINITE_COEFFICIENT_OR_INTERCEPT: "outer_estimator_failure",
        modeling.FitFailureKind.INVALID_CLASS_STRUCTURE: "outer_estimator_failure",
        modeling.FitFailureKind.AMBIGUOUS_POSITIVE_CLASS: "outer_estimator_failure",
        modeling.EvaluationFailureKind.INVALID_PROBABILITY_SHAPE: "invalid_probability",
        modeling.EvaluationFailureKind.NONFINITE_PROBABILITY: "invalid_probability",
        modeling.EvaluationFailureKind.POSITIVE_SCORE_ALIGNMENT_INVALID: "invalid_metric",
        modeling.EvaluationFailureKind.CATEGORICAL_TEST_CLASSES_INVALID: "invalid_metric",
        modeling.EvaluationFailureKind.INVALID_NUMERICAL_PREDICTION: "invalid_prediction",
        modeling.EvaluationFailureKind.INVALID_REGRESSION_METRIC: "invalid_metric",
    }
    assert all(len(code) <= 32 for code in decoding.POOLED_REASON_CODES)
    assert {
        failure: decoding._pooled_reason_for_modeling_failure(failure)
        for failure in (*modeling.FitFailureKind, *modeling.EvaluationFailureKind)
    } == expected
    with pytest.raises(ValueError, match="Unknown modeling failure"):
        decoding._pooled_reason_for_modeling_failure("invalid_metric")
