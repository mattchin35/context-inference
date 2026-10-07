"""Synthetic grouped-decoding contracts for task-variable modeling.

This tests-only module freezes the task-decoding modeling interface:

* ``make_outer_splits`` and ``make_inner_splits`` return validated grouped
  split plans expressed in original matched-tensor row positions.
* ``fit_region_transform`` owns pooled training scaling and separate regional
  PCA; ``build_region_features`` chooses direct units or PCA scores.
* Small public helpers construct estimators, score fits, select tuning
  candidates, aggregate complete folds, and summarize unit coefficients.
* ``decode_target`` accepts matched PFC/HPC tensors shaped
  ``(trial, time_bin, unit)`` in Hz and returns audit-ready split/fold/cell
  records. It always computes both representations and three regions.

No test reads experimental files. Every input is deterministic synthetic data
with common trial rows across the two regional tensors.
"""

from __future__ import annotations

from collections import Counter
import inspect
from pathlib import Path
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.neural_analysis.task_decoding import modeling
from src.neural_analysis.task_decoding.config import (
    RegionConfig,
    TaskDecodingConfig,
    scientific_config_payload,
)


def make_scientific_config() -> TaskDecodingConfig:
    """Build in-memory configuration values for frozen-control comparisons.

    Returns
    -------
    TaskDecodingConfig
        Minimal immutable settings. This helper only passes the object to
        ``scientific_config_payload``; it never accesses synthetic paths.
    """
    pfc_region = RegionConfig(
        region="PFC",
        probe_id="probe-pfc",
        channel_labels=("good",),
        require_inside_brain=True,
        cluster_groups=("good",),
    )
    hpc_region = RegionConfig(
        region="HPC",
        probe_id="probe-hpc",
        channel_labels=("good",),
        require_inside_brain=True,
        cluster_groups=("good",),
    )
    return TaskDecodingConfig(
        session_metadata_path=Path("/synthetic/neural_session.json"),
        augmented_trial_path=Path("/synthetic/augmented.csv"),
        trial_feature_parameter_path=Path("/synthetic/params.json"),
        pfc_region=pfc_region,
        hpc_region=hpc_region,
        alignment="choice_time",
        bin_width_ms=100,
        pfc_pc_count=2,
        hpc_pc_count=2,
        target_names=("current_action",),
        regularization_mode="fixed",
        outer_fold_count=5,
        inner_fold_count=3,
        trusted_utc_bounds={},
        output_root=Path("/synthetic/analysis_runs"),
        session_root=Path("/synthetic"),
    )


def frozen_controls() -> dict[str, object]:
    """Return code-owned scientific controls used by modeling tests.

    Returns
    -------
    dict[str, object]
        The ``frozen_controls`` payload from the existing config module.
    """
    return scientific_config_payload(make_scientific_config())["frozen_controls"]


def make_classification_inputs(
    *,
    n_time_bins: int = 2,
    n_blocks: int = 10,
    trials_per_block: int = 4,
) -> dict[str, object]:
    """Build deterministic matched binary-classification rate tensors.

    Parameters
    ----------
    n_time_bins : int, default=2
        Common event-relative time-bin count.
    n_blocks : int, default=10
        Number of indivisible behavioral groups.
    trials_per_block : int, default=4
        Rows per group; each group has two examples of each class.

    Returns
    -------
    dict[str, object]
        Keyword arguments for ``decode_target``. Both rate arrays are float64
        Hz arrays shaped ``(trial, time_bin, unit)``.
    """
    if trials_per_block != 4:
        raise ValueError("The categorical synthetic fixture requires four rows per block.")
    n_trials = n_blocks * trials_per_block
    block_ids = np.repeat(np.arange(n_blocks), trials_per_block)
    target_values = np.tile(np.array([0, 1, 0, 1], dtype=int), n_blocks)
    rng = np.random.default_rng(23)
    pfc_rates_hz = rng.normal(2.0, 0.03, size=(n_trials, n_time_bins, 2))
    hpc_rates_hz = rng.normal(3.0, 0.03, size=(n_trials, n_time_bins, 2))
    class_signal = target_values[:, None]
    pfc_rates_hz[:, :, 0] += 3.0 * class_signal
    pfc_rates_hz[:, :, 1] += 0.25 * class_signal
    hpc_rates_hz[:, :, 0] += 2.0 * class_signal
    hpc_rates_hz[:, :, 1] -= 0.15 * class_signal
    return {
        "target_identifier": "synthetic_action",
        "target_family": "categorical",
        "target_values": target_values,
        "block_ids": block_ids,
        "pfc_rate_tensor_hz": pfc_rates_hz,
        "hpc_rate_tensor_hz": hpc_rates_hz,
        "pfc_unit_ids": ("probe-pfc:11", "probe-pfc:19"),
        "hpc_unit_ids": ("probe-hpc:5", "probe-hpc:17"),
        "pfc_requested_pc_count": 2,
        "hpc_requested_pc_count": 2,
        "outer_fold_count": 5,
        "inner_fold_count": 3,
        "regularization_mode": "fixed",
    }


def make_numerical_inputs(
    *,
    n_time_bins: int = 1,
    n_blocks: int = 10,
) -> dict[str, object]:
    """Build matched continuous-target tensors with unique native values.

    Parameters
    ----------
    n_time_bins : int, default=1
        Common event-relative time-bin count.
    n_blocks : int, default=10
        Number of behavioral groups, each containing four nonconstant targets.

    Returns
    -------
    dict[str, object]
        Numerical ``decode_target`` inputs. The target remains in native units
        and is unique for every synthetic matched row.
    """
    inputs = make_classification_inputs(n_time_bins=n_time_bins, n_blocks=n_blocks)
    n_trials = len(inputs["target_values"])
    target_values = 10.0 + np.arange(n_trials, dtype=float) / 10.0
    pfc_rates_hz = np.asarray(inputs["pfc_rate_tensor_hz"], dtype=float).copy()
    hpc_rates_hz = np.asarray(inputs["hpc_rate_tensor_hz"], dtype=float).copy()
    pfc_rates_hz[:, :, 0] = target_values[:, None]
    pfc_rates_hz[:, :, 1] = 0.5 * target_values[:, None] + 1.0
    hpc_rates_hz[:, :, 0] = target_values[:, None]
    hpc_rates_hz[:, :, 1] = -0.25 * target_values[:, None] + 3.0
    inputs.update(
        {
            "target_identifier": "synthetic_native_value",
            "target_family": "numerical",
            "target_values": target_values,
            "pfc_rate_tensor_hz": pfc_rates_hz,
            "hpc_rate_tensor_hz": hpc_rates_hz,
        }
    )
    return inputs


def decode_target(**overrides: object):
    """Run ``decode_target`` on ordinary categorical synthetic inputs.

    Parameters
    ----------
    **overrides : object
        Keyword values replacing the fixture defaults.

    Returns
    -------
    modeling.TargetDecodingResult
        Result retaining outer assignments, inner plans, fold records, and
        complete-fold cell summaries.
    """
    inputs = make_classification_inputs()
    inputs.update(overrides)
    return modeling.decode_target(**inputs)


def reference_fold_ids(
    target_values: np.ndarray,
    block_ids: np.ndarray,
    *,
    target_family: str,
    fold_count: int,
) -> np.ndarray:
    """Compute exact installed-scikit-learn deterministic grouped assignments.

    Parameters
    ----------
    target_values : np.ndarray
        One-dimensional categorical labels or continuous native targets.
    block_ids : np.ndarray
        One-dimensional behavioral group identifiers aligned to target rows.
    target_family : {"categorical", "numerical"}
        Determines the documented scikit-learn splitter.
    fold_count : int
        Requested number of deterministic grouped folds.

    Returns
    -------
    np.ndarray
        Integer fold IDs in original matched-tensor row order.
    """
    values = np.asarray(target_values)
    groups = np.asarray(block_ids)
    if target_family == "categorical":
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=False)
        split_arguments = values
    elif target_family == "numerical":
        splitter = GroupKFold(n_splits=fold_count, shuffle=False)
        split_arguments = None
    else:
        raise ValueError(f"Unsupported synthetic family {target_family!r}.")
    fold_ids = np.full(values.shape, -1, dtype=int)
    for fold_id, (_train, test) in enumerate(
        splitter.split(np.zeros((values.size, 1)), split_arguments, groups)
    ):
        fold_ids[test] = fold_id
    return fold_ids


def selected_summary(result, time_bin: int, region: str, representation: str):
    """Return a complete-fold cell summary by its deterministic tuple key.

    Parameters
    ----------
    result : modeling.TargetDecodingResult
        Decoding result returned by the public principal API.
    time_bin : int
        Zero-based event-relative time-bin index.
    region : {"PFC", "HPC", "PFC+HPC"}
        Requested regional feature configuration.
    representation : {"pca", "units"}
        Requested PCA-score or direct-unit feature representation.

    Returns
    -------
    modeling.CellSummary
        Aggregated result for the addressed modeling cell.
    """
    return result.cell_summaries[(time_bin, region, representation)]


def test_outer_splits_exactly_match_installed_stratified_group_kfold_assignments():
    """Categorical outer assignments must equal deterministic StratifiedGroupKFold."""
    inputs = make_classification_inputs()
    expected_fold_ids = reference_fold_ids(
        inputs["target_values"],
        inputs["block_ids"],
        target_family="categorical",
        fold_count=5,
    )

    split_plan = modeling.make_outer_splits(
        inputs["target_values"],
        inputs["block_ids"],
        target_family="categorical",
        fold_count=5,
    )

    assert split_plan.is_available
    np.testing.assert_array_equal(split_plan.fold_ids, expected_fold_ids)
    for split in split_plan.splits:
        train_groups = set(np.asarray(inputs["block_ids"])[split.train_indices])
        test_groups = set(np.asarray(inputs["block_ids"])[split.test_indices])
        assert train_groups.isdisjoint(test_groups)
        assert set(np.asarray(inputs["target_values"])[split.train_indices]) == {0, 1}
        assert set(np.asarray(inputs["target_values"])[split.test_indices]) == {0, 1}


def test_invalid_categorical_or_constant_targets_return_explicit_unavailable_reasons():
    """Invalid class coverage and constant targets are scientific unavailable states."""
    class_invalid = modeling.make_outer_splits(
        np.array([0, 0, 1, 1, 0, 0]),
        np.array([0, 0, 1, 1, 2, 2]),
        target_family="categorical",
        fold_count=3,
    )
    constant = modeling.make_outer_splits(
        np.ones(12),
        np.repeat(np.arange(3), 4),
        target_family="numerical",
        fold_count=3,
    )

    assert not class_invalid.is_available
    assert "class" in class_invalid.unavailable_reason.lower()
    assert not constant.is_available
    assert "constant" in constant.unavailable_reason.lower()


def test_categorical_inner_split_with_pure_group_validation_is_explicitly_unavailable():
    """Three groups can still fail inner CV when a held-out group lacks one class."""
    target_values = np.array([0, 0, 0, 0, 1, 1])
    block_ids = np.array([0, 0, 1, 1, 2, 2])
    outer_train_indices = np.arange(target_values.size)

    inner_plan = modeling.make_inner_splits(
        target_values,
        block_ids,
        outer_train_indices,
        target_family="categorical",
        fold_count=3,
    )

    assert not inner_plan.is_available
    assert "class" in inner_plan.unavailable_reason.lower()


def test_numerical_outer_splits_exactly_match_group_kfold_without_discretization():
    """Continuous targets must use GroupKFold's deterministic assignment unbinned."""
    inputs = make_numerical_inputs()
    values = np.asarray(inputs["target_values"])
    expected_fold_ids = reference_fold_ids(
        values,
        inputs["block_ids"],
        target_family="numerical",
        fold_count=5,
    )

    split_plan = modeling.make_outer_splits(
        values,
        inputs["block_ids"],
        target_family="numerical",
        fold_count=5,
    )

    assert split_plan.is_available
    assert np.unique(values).size == values.size
    np.testing.assert_array_equal(split_plan.fold_ids, expected_fold_ids)
    for split in split_plan.splits:
        assert np.unique(values[split.test_indices]).size > 1


def test_outer_assignments_are_reused_across_every_cell_and_regularization_mode(monkeypatch):
    """All cells and fixed/tuned runs must retain the same outer fold vector."""
    calls = []

    class FastClassifier:
        """Deterministic warning-free classifier used only for orchestration tests."""

        def __init__(self, _parameters):
            self._parameters = _parameters

        def fit(self, features, targets):
            """Store valid binary estimator attributes without iterative fitting."""
            self.coef_ = np.ones((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return finite binary positive-class scores for every test row."""
            return np.tile(np.array([[0.4, 0.6]]), (len(features), 1))

    def fast_factory(*, target_family, parameters=None):
        """Record calls while constructing a fast categorical fake estimator."""
        calls.append((target_family, parameters))
        return FastClassifier(parameters)

    monkeypatch.setattr(modeling, "make_estimator", fast_factory)
    fixed = decode_target()
    tuned = decode_target(regularization_mode="tuned")

    np.testing.assert_array_equal(fixed.outer_fold_ids, tuned.outer_fold_ids)
    assert set(fixed.outer_fold_ids) == set(range(5))
    for result in (fixed, tuned):
        for record in result.fold_records:
            np.testing.assert_array_equal(
                record.outer_test_indices,
                np.flatnonzero(result.outer_fold_ids == record.outer_fold_id),
            )
    assert calls


def test_region_transform_uses_only_training_pool_and_matches_manual_scaling_pca():
    """Regional transforms must match independently fit train-only pooled PCA."""
    training_hz = np.array(
        [
            [[0.0, 2.0], [2.0, 1.0]],
            [[4.0, 5.0], [6.0, 3.0]],
        ]
    )
    held_out_hz = np.array([[[1000.0, -500.0], [900.0, -450.0]]])
    pooled = training_hz.reshape(-1, 2)
    expected_mean = pooled.mean(axis=0)
    expected_scale = pooled.std(axis=0)
    expected_standardized = (pooled - expected_mean) / expected_scale
    pca_controls = frozen_controls()["estimators"]["PCA"]
    expected_pca = PCA(n_components=2, **pca_controls).fit(expected_standardized)

    transform = modeling.fit_region_transform(training_hz, requested_pc_count=9)
    transformed_units = transform.transform_units(held_out_hz)
    transformed_pca = transform.transform_pca(held_out_hz)
    manual_held_out = (held_out_hz - expected_mean) / expected_scale
    expected_pca_scores = expected_pca.transform(manual_held_out.reshape(-1, 2)).reshape(1, 2, 2)

    assert transform.is_available
    np.testing.assert_allclose(transform.unit_means_hz, expected_mean)
    np.testing.assert_allclose(transform.unit_scales_hz, expected_scale)
    np.testing.assert_allclose(transform.pca_model.components_, expected_pca.components_)
    np.testing.assert_allclose(transformed_units, manual_held_out)
    np.testing.assert_allclose(transformed_pca, expected_pca_scores)
    assert transform.pca_model.whiten is False
    assert transform.pca_model.svd_solver == "auto"
    assert transform.pca_model.random_state == 0


def test_pca_component_limit_is_observation_limited_without_rank_check(monkeypatch):
    """PCA keeps its exact dimensional cap and no numerical-rank policy."""
    training_hz = np.array([[[0.0, 0.0, 4.0, 5.0], [1.0, 2.0, 8.0, 9.0]]])

    def fail_if_called(*_args, **_kwargs):
        """Fail if an unapproved matrix-rank decision enters the implementation."""
        raise AssertionError("PCA component selection must not call matrix_rank.")

    monkeypatch.setattr(modeling.np.linalg, "matrix_rank", fail_if_called)
    transform = modeling.fit_region_transform(training_hz, requested_pc_count=99)

    assert transform.is_available
    assert transform.usable_unit_indices.tolist() == [0, 1, 2, 3]
    assert transform.component_limit == min(4, 1 * 2)
    assert transform.effective_component_count == 2
    assert transform.cap_warning is not None


def test_one_usable_unit_is_valid_and_zero_usable_units_are_explicitly_unavailable(monkeypatch):
    """One retained unit must not route through the legacy two-unit PCA helper."""
    one_usable_hz = np.array(
        [[[1.0, 7.0], [2.0, 7.0]], [[3.0, 7.0], [4.0, 7.0]]]
    )

    def fail_legacy_helper(*_args, **_kwargs):
        """Fail if task decoding calls the older two-unit-only PCA utility."""
        raise AssertionError("Task decoding must not use fit_population_pca.")

    monkeypatch.setattr(
        "src.neural_analysis.population.pca.fit_population_pca",
        fail_legacy_helper,
    )
    one_unit = modeling.fit_region_transform(one_usable_hz, requested_pc_count=10)
    zero_units = modeling.fit_region_transform(
        np.full((2, 2, 2), 7.0),
        requested_pc_count=10,
    )

    assert one_unit.is_available
    assert one_unit.component_limit == 1
    assert one_unit.effective_component_count == 1
    assert not zero_units.is_available
    assert "usable" in zero_units.unavailable_reason.lower()


def test_combined_features_concatenate_separate_region_projections_and_stable_units():
    """Combined PCA/direct features must concatenate PFC then HPC, never joint PCA."""
    inputs = make_classification_inputs(n_time_bins=2)
    pfc_hz = np.asarray(inputs["pfc_rate_tensor_hz"])
    hpc_hz = np.asarray(inputs["hpc_rate_tensor_hz"])
    pfc_transform = modeling.fit_region_transform(pfc_hz[:20], requested_pc_count=2)
    hpc_transform = modeling.fit_region_transform(hpc_hz[:20], requested_pc_count=2)

    pfc_pca = modeling.build_region_features(
        pfc_transform,
        hpc_transform,
        pfc_hz[20:],
        hpc_hz[20:],
        time_bin_index=1,
        region_configuration="PFC",
        representation="pca",
        pfc_unit_ids=inputs["pfc_unit_ids"],
        hpc_unit_ids=inputs["hpc_unit_ids"],
    )
    hpc_pca = modeling.build_region_features(
        pfc_transform,
        hpc_transform,
        pfc_hz[20:],
        hpc_hz[20:],
        time_bin_index=1,
        region_configuration="HPC",
        representation="pca",
        pfc_unit_ids=inputs["pfc_unit_ids"],
        hpc_unit_ids=inputs["hpc_unit_ids"],
    )
    combined_pca = modeling.build_region_features(
        pfc_transform,
        hpc_transform,
        pfc_hz[20:],
        hpc_hz[20:],
        time_bin_index=1,
        region_configuration="PFC+HPC",
        representation="pca",
        pfc_unit_ids=inputs["pfc_unit_ids"],
        hpc_unit_ids=inputs["hpc_unit_ids"],
    )
    combined_units = modeling.build_region_features(
        pfc_transform,
        hpc_transform,
        pfc_hz[20:],
        hpc_hz[20:],
        time_bin_index=1,
        region_configuration="PFC+HPC",
        representation="units",
        pfc_unit_ids=inputs["pfc_unit_ids"],
        hpc_unit_ids=inputs["hpc_unit_ids"],
    )

    np.testing.assert_allclose(
        combined_pca.values,
        np.column_stack((pfc_pca.values, hpc_pca.values)),
    )
    assert combined_pca.feature_ids == (*pfc_pca.feature_ids, *hpc_pca.feature_ids)
    assert combined_units.feature_ids == (
        "probe-pfc:11",
        "probe-pfc:19",
        "probe-hpc:5",
        "probe-hpc:17",
    )
    expected_direct_values = np.column_stack(
        (
            pfc_transform.transform_units(pfc_hz[20:])[:, 1, :],
            hpc_transform.transform_units(hpc_hz[20:])[:, 1, :],
        )
    )
    np.testing.assert_allclose(combined_units.values, expected_direct_values)


def test_fixed_mode_builds_no_inner_splits_and_emits_full_cartesian_finite_records(monkeypatch):
    """Fixed mode must cover all cells without inner work or vacuous results."""

    fit_count = []

    class FastClassifier:
        """Small deterministic classifier that avoids iterative solver timing."""

        def fit(self, features, targets):
            """Record valid model attributes for subsequent scoring."""
            fit_count.append((tuple(np.asarray(features).shape), tuple(np.asarray(targets))))
            self.coef_ = np.ones((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return finite scores for one held-out prediction pass."""
            return np.tile(np.array([[0.4, 0.6]]), (len(features), 1))

    def fail_inner_split(*_args, **_kwargs):
        """Fail if fixed mode tries to construct inner grouped folds."""
        raise AssertionError("Fixed mode must not construct inner splits.")

    monkeypatch.setattr(modeling, "make_inner_splits", fail_inner_split)
    monkeypatch.setattr(
        modeling,
        "make_estimator",
        lambda *, target_family, parameters=None: FastClassifier(),
    )
    result = decode_target()
    expected_keys = {
        (fold_id, time_bin, region, representation)
        for fold_id in range(5)
        for time_bin in range(2)
        for region in ("PFC", "HPC", "PFC+HPC")
        for representation in ("pca", "units")
    }

    assert result.inner_split_plans == {}
    assert {record.record_key for record in result.fold_records} == expected_keys
    assert len(result.fold_records) == 5 * 2 * 3 * 2
    assert len(fit_count) == 5 * 2 * 3 * 2
    for record in result.fold_records:
        assert record.is_valid
        assert record.status == "valid"
        assert record.reason is None
        assert record.train_count == 32
        assert record.test_count == 8
        assert record.train_class_counts == {0: 16, 1: 16}
        assert record.test_class_counts == {0: 4, 1: 4}
        assert record.requested_feature_count >= record.effective_feature_count >= 1
        assert len(record.feature_ids) == record.effective_feature_count
        assert record.estimator_parameters["C"] == 1.0
        assert record.estimator_parameters["l1_ratio"] == 0.5
        assert record.convergence_status == "converged"
        assert np.isfinite(record.metrics["balanced_accuracy"])
        assert np.isfinite(record.metrics["auc"])
    assert all(summary.is_available for summary in result.cell_summaries.values())


class RecordingEstimator:
    """Fast fake estimator recording exact target rows used by model fitting.

    Parameters
    ----------
    target_family : str
        ``"categorical"`` or ``"numerical"`` selected decoder family.
    parameters : dict[str, float] or None
        Candidate settings passed by the modeler, retained for assertions.
    fit_log : list[dict[str, object]]
        Shared mutable event list recording training target multisets.
    invalid_parameters : tuple[tuple[tuple[str, float], ...], ...]
        Parameter tuples that force non-finite held-out predictions.
    score_log : list[int] or None
        Optional shared list receiving one held-out prediction call per fit.
    """

    def __init__(
        self,
        target_family,
        parameters,
        fit_log,
        invalid_parameters=(),
        score_log=None,
    ):
        self.target_family = target_family
        self.parameters = parameters
        self.fit_log = fit_log
        self.invalid_parameters = invalid_parameters
        self.score_log = score_log

    def fit(self, features, targets):
        """Record exact training target values and expose fitted attributes."""
        self.fit_log.append(
            {
                "parameters": self.parameters,
                "targets": tuple(sorted(float(value) for value in targets)),
                "feature_shape": tuple(np.asarray(features).shape),
            }
        )
        self.coef_ = np.ones((1, np.asarray(features).shape[1]))
        self.intercept_ = np.array([0.0])
        self.classes_ = np.array([0, 1])
        return self

    def _is_invalid_candidate(self) -> bool:
        """Return whether this configured candidate should yield invalid scores."""
        if self.parameters is None:
            return False
        parameter_tuple = tuple(sorted(self.parameters.items()))
        return parameter_tuple in self.invalid_parameters

    def predict_proba(self, features):
        """Return one finite or deliberately non-finite categorical score matrix."""
        if self.score_log is not None:
            self.score_log.append(len(features))
        if self._is_invalid_candidate():
            return np.full((len(features), 2), np.nan)
        return np.tile(np.array([[0.4, 0.6]]), (len(features), 1))

    def predict(self, features):
        """Return finite or deliberately non-finite native-scale predictions."""
        if self.score_log is not None:
            self.score_log.append(len(features))
        if self._is_invalid_candidate():
            return np.full(len(features), np.nan)
        return np.zeros(len(features), dtype=float)


@pytest.mark.parametrize(
    (
        "regularization_mode",
        "n_time_bins",
        "expected_transform_fits",
        "expected_estimator_fits",
        "expected_inner_split_calls",
    ),
    [
        ("fixed", 1, 3 * 2, 3 * 1 * 3 * 2, 0),
        ("fixed", 2, 3 * 2, 3 * 2 * 3 * 2, 0),
        ("tuned", 1, 3 * (1 + 3) * 2, 828, 3),
        ("tuned", 2, 3 * (1 + 3) * 2, 1656, 3),
    ],
)
def test_transform_fit_count_is_invariant_to_time_bins_and_tuning_candidates(
    monkeypatch,
    regularization_mode,
    n_time_bins,
    expected_transform_fits,
    expected_estimator_fits,
    expected_inner_split_calls,
):
    """Regional transforms are fit once per outer/inner split, never per cell."""
    transform_shapes = []
    fit_log = []
    inner_split_calls = []
    original_transform = modeling.fit_region_transform
    original_inner_splits = modeling.make_inner_splits

    def record_transform(training_hz, requested_pc_count):
        """Record raw fit tensor shapes while delegating to the real transform."""
        transform_shapes.append(tuple(np.asarray(training_hz).shape))
        return original_transform(training_hz, requested_pc_count)

    def factory(*, target_family, parameters=None):
        """Construct a finite fake to keep repeated tuned cases bounded."""
        return RecordingEstimator(target_family, parameters, fit_log)

    def record_inner_splits(*args, **kwargs):
        """Record each grouped inner assignment requested by a tuned outer fold."""
        inner_split_calls.append((args, kwargs))
        return original_inner_splits(*args, **kwargs)

    monkeypatch.setattr(modeling, "fit_region_transform", record_transform)
    monkeypatch.setattr(modeling, "make_estimator", factory)
    monkeypatch.setattr(modeling, "make_inner_splits", record_inner_splits)
    inputs = make_classification_inputs(n_time_bins=n_time_bins, n_blocks=9)
    inputs.update(
        {
            "outer_fold_count": 3,
            "inner_fold_count": 3,
            "regularization_mode": regularization_mode,
        }
    )

    result = modeling.decode_target(**inputs)

    assert len(transform_shapes) == expected_transform_fits
    assert all(shape[1] == n_time_bins for shape in transform_shapes)
    assert len(fit_log) == expected_estimator_fits
    assert len(inner_split_calls) == expected_inner_split_calls
    if regularization_mode == "tuned":
        expected_keys = {
            (fold_id, time_bin, region, representation)
            for fold_id in range(3)
            for time_bin in range(n_time_bins)
            for region in ("PFC", "HPC", "PFC+HPC")
            for representation in ("pca", "units")
        }
        assert {record.record_key for record in result.fold_records} == expected_keys


def test_tuned_mode_uses_exact_grid_order_candidate_counts_and_tie_order(monkeypatch):
    """Tuning must evaluate all 15 ordered candidates with no redundant fit for AUC."""
    expected_candidates = tuple(
        {"C": c_value, "l1_ratio": ratio}
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0)
        for ratio in (0.1, 0.5, 0.9)
    )
    fit_log = []
    score_log = []
    inner_split_calls = []
    original_inner_splits = modeling.make_inner_splits

    def factory(*, target_family, parameters=None):
        """Build an instrumented fast classifier for every fit request."""
        return RecordingEstimator(target_family, parameters, fit_log, score_log=score_log)

    def record_inner_splits(*args, **kwargs):
        """Count one retained grouped inner assignment per outer fold."""
        inner_split_calls.append((args, kwargs))
        return original_inner_splits(*args, **kwargs)

    monkeypatch.setattr(modeling, "make_estimator", factory)
    monkeypatch.setattr(modeling, "make_inner_splits", record_inner_splits)
    inputs = make_classification_inputs(n_time_bins=1, n_blocks=9)
    inputs.update(
        {"outer_fold_count": 3, "inner_fold_count": 3, "regularization_mode": "tuned"}
    )
    result = modeling.decode_target(**inputs)
    inner_fits = [entry for entry in fit_log if len(entry["targets"]) < 24]
    observed_candidates = [entry["parameters"] for entry in inner_fits]

    assert modeling.tuning_candidates("categorical") == expected_candidates
    assert len(fit_log) == 3 * 3 * 3 * 2 * 15 + 3 * 3 * 2
    assert len(inner_fits) == 3 * 3 * 3 * 2 * 15
    assert len(score_log) == len(fit_log)
    assert len(inner_split_calls) == 3
    observed_candidate_counts = Counter(
        tuple(sorted(candidate.items())) for candidate in observed_candidates
    )
    assert observed_candidate_counts == Counter(
        {
            tuple(sorted(candidate.items())): 3 * 3 * 3 * 2
            for candidate in expected_candidates
        }
    )
    outer_fold_ids = reference_fold_ids(
        np.asarray(inputs["target_values"]),
        np.asarray(inputs["block_ids"]),
        target_family="categorical",
        fold_count=3,
    )
    for outer_fold_id, inner in result.inner_split_plans.items():
        outer_train = np.flatnonzero(outer_fold_ids != outer_fold_id)
        local_reference = reference_fold_ids(
            np.asarray(inputs["target_values"])[outer_train],
            np.asarray(inputs["block_ids"])[outer_train],
            target_family="categorical",
            fold_count=3,
        )
        expected_inner_tests = {
            tuple(outer_train[np.flatnonzero(local_reference == fold_id)])
            for fold_id in range(3)
        }
        assert {tuple(split.test_indices) for split in inner.splits} == expected_inner_tests
    for record in result.fold_records:
        assert record.selected_parameters == expected_candidates[0]
        assert set(record.metrics) == {"balanced_accuracy", "auc"}


def test_tuned_mode_uses_only_outer_training_rows_for_inner_transforms_and_estimator_fits(
    monkeypatch,
):
    """Recorded raw tensors and unique targets must match exact outer/inner training sets."""
    inputs = make_numerical_inputs(n_blocks=9)
    inputs.update(
        {"outer_fold_count": 3, "inner_fold_count": 3, "regularization_mode": "tuned"}
    )
    values = np.asarray(inputs["target_values"])
    blocks = np.asarray(inputs["block_ids"])
    expected_numerical_candidates = tuple(
        {"alpha": alpha, "l1_ratio": ratio}
        for alpha in (0.001, 0.01, 0.1, 1.0, 10.0)
        for ratio in (0.1, 0.5, 0.9)
    )
    expected_outer = reference_fold_ids(values, blocks, target_family="numerical", fold_count=3)
    transform_rows = []
    fit_log = []
    fit_and_score_rows = []
    original_transform = modeling.fit_region_transform
    original_fit_and_score = modeling.fit_and_score

    def record_transform(training_hz, requested_pc_count):
        """Record sentinels carried in raw regional training tensors only."""
        tensor = np.asarray(training_hz)
        transform_rows.append(tuple(sorted(float(value) for value in tensor[:, 0, 0])))
        return original_transform(training_hz, requested_pc_count)

    def factory(*, target_family, parameters=None):
        """Build fast numerical estimators that retain unique target row identities."""
        return RecordingEstimator(target_family, parameters, fit_log)

    def record_fit_and_score(
        estimator,
        train_features,
        train_targets,
        held_out_features,
        held_out_targets,
        *,
        target_family,
    ):
        """Record exact training and held-out target identities before scoring."""
        fit_and_score_rows.append(
            (
                tuple(sorted(float(value) for value in train_targets)),
                tuple(sorted(float(value) for value in held_out_targets)),
            )
        )
        return original_fit_and_score(
            estimator,
            train_features,
            train_targets,
            held_out_features,
            held_out_targets,
            target_family=target_family,
        )

    monkeypatch.setattr(modeling, "fit_region_transform", record_transform)
    monkeypatch.setattr(modeling, "make_estimator", factory)
    monkeypatch.setattr(modeling, "fit_and_score", record_fit_and_score)
    result = modeling.decode_target(**inputs)
    assert modeling.tuning_candidates("numerical") == expected_numerical_candidates
    expected_train_rows = []
    for outer_fold_id in range(3):
        outer_train = np.flatnonzero(expected_outer != outer_fold_id)
        expected_train_rows.append(tuple(sorted(values[outer_train])))
        inner = modeling.make_inner_splits(
            values,
            blocks,
            outer_train,
            target_family="numerical",
            fold_count=3,
        )
        for split in inner.splits:
            expected_train_rows.append(tuple(sorted(values[split.train_indices])))

    assert set(result.inner_split_plans) == {0, 1, 2}
    assert Counter(transform_rows) == Counter(
        row_values for row_values in expected_train_rows for _region in range(2)
    )
    observed_fit_rows = Counter(entry["targets"] for entry in fit_log)
    expected_fit_counts = Counter()
    for outer_fold_id in range(3):
        outer_train = np.flatnonzero(expected_outer != outer_fold_id)
        expected_fit_counts[tuple(sorted(values[outer_train]))] += 3 * 2
        inner = result.inner_split_plans[outer_fold_id]
        for split in inner.splits:
            expected_fit_counts[tuple(sorted(values[split.train_indices]))] += 3 * 2 * 15
    assert observed_fit_rows == expected_fit_counts
    expected_fit_and_score_counts = Counter()
    for outer_fold_id, inner in result.inner_split_plans.items():
        outer_test = set(np.flatnonzero(result.outer_fold_ids == outer_fold_id))
        outer_train = np.flatnonzero(result.outer_fold_ids != outer_fold_id)
        expected_fit_and_score_counts[
            (tuple(sorted(values[outer_train])), tuple(sorted(values[list(outer_test)])))
        ] += 3 * 2
        local_reference = reference_fold_ids(
            values[outer_train],
            blocks[outer_train],
            target_family="numerical",
            fold_count=3,
        )
        expected_inner_tests = {
            tuple(outer_train[np.flatnonzero(local_reference == fold_id)])
            for fold_id in range(3)
        }
        actual_inner_tests = {tuple(split.test_indices) for split in inner.splits}
        assert actual_inner_tests == expected_inner_tests
        for split in inner.splits:
            split_target_pair = (
                tuple(sorted(values[split.train_indices])),
                tuple(sorted(values[split.test_indices])),
            )
            expected_fit_and_score_counts[
                split_target_pair
            ] += 3 * 2 * 15
            assert set(split.train_indices).isdisjoint(outer_test)
            assert set(split.test_indices).isdisjoint(outer_test)
            assert set(blocks[split.train_indices]).isdisjoint(blocks[split.test_indices])
    assert Counter(fit_and_score_rows) == expected_fit_and_score_counts


def test_invalid_inner_candidate_is_discarded_and_invalid_inner_fold_prevents_fallback(monkeypatch):
    """Any invalid candidate is rejected; impossible inner CV leaves tuned cells unavailable."""
    first_candidate = (("C", 0.01), ("l1_ratio", 0.1))
    second_candidate = {"C": 0.01, "l1_ratio": 0.5}
    fit_log = []
    first_candidate_score_calls = []
    invalid_score_calls = []

    class OneInvalidFoldEstimator(RecordingEstimator):
        """Return NaN exactly once for the first candidate, then finite scores."""

        def predict_proba(self, features):
            """Make one inner candidate fold invalid while leaving all later fits finite."""
            parameter_tuple = None
            if self.parameters is not None:
                parameter_tuple = tuple(sorted(self.parameters.items()))
            if parameter_tuple == first_candidate:
                first_candidate_score_calls.append(len(features))
                if not invalid_score_calls:
                    invalid_score_calls.append(len(features))
                    return np.full((len(features), 2), np.nan)
            return super().predict_proba(features)

    def factory(*, target_family, parameters=None):
        """Construct fakes with one non-finite first-candidate inner evaluation."""
        return OneInvalidFoldEstimator(
            target_family,
            parameters,
            fit_log,
        )

    monkeypatch.setattr(modeling, "make_estimator", factory)
    valid_inputs = make_classification_inputs(n_time_bins=1, n_blocks=9)
    valid_inputs.update(
        {"outer_fold_count": 3, "inner_fold_count": 3, "regularization_mode": "tuned"}
    )
    valid_result = modeling.decode_target(**valid_inputs)
    impossible_inputs = make_classification_inputs(n_time_bins=1, n_blocks=3)
    impossible_inputs.update(
        {"outer_fold_count": 3, "inner_fold_count": 3, "regularization_mode": "tuned"}
    )
    impossible_result = modeling.decode_target(**impossible_inputs)

    selected_second = [
        record
        for record in valid_result.fold_records
        if record.selected_parameters == second_candidate
    ]
    selected_first = [
        record
        for record in valid_result.fold_records
        if record.selected_parameters == dict(first_candidate)
    ]
    assert len(invalid_score_calls) == 1
    assert len(first_candidate_score_calls) > 3
    assert len(selected_second) == 1
    assert len(selected_first) + len(selected_second) == len(valid_result.fold_records)
    assert any(entry["parameters"] == dict(first_candidate) for entry in fit_log)
    assert len(impossible_result.inner_split_plans) == 3
    assert len(impossible_result.cell_summaries) == 3 * 2
    assert all(not plan.is_available for plan in impossible_result.inner_split_plans.values())
    for summary in impossible_result.cell_summaries.values():
        assert not summary.is_available
        assert "inner" in summary.unavailable_reason.lower()


def test_select_tuning_candidate_breaks_exact_valid_score_ties_by_declared_order():
    """The earliest valid candidate wins when every inner mean is exactly tied."""
    candidates = (
        {"C": 0.01, "l1_ratio": 0.1},
        {"C": 0.1, "l1_ratio": 0.5},
        {"C": 1.0, "l1_ratio": 0.9},
    )

    selected = modeling.select_tuning_candidate(
        candidates,
        candidate_scores=((0.75, 0.75, 0.75),) * 3,
        candidate_valid=(True, True, True),
    )

    assert selected == 0


def test_fixed_numerical_decode_uses_elasticnet_native_targets_and_complete_r2_aggregation():
    """The real fixed numerical path uses ElasticNet and retains finite native-target R2."""
    inputs = make_numerical_inputs(n_blocks=10)
    result = modeling.decode_target(**inputs)

    assert set(result.outer_fold_ids) == set(range(5))
    assert len(result.fold_records) == 5 * 1 * 3 * 2
    for record in result.fold_records:
        assert record.estimator_class == "ElasticNet"
        assert (
            record.coefficient_scale_label
            == "native target units per pooled training standard deviation"
        )
        assert np.isfinite(record.metrics["r2"])
        assert np.isfinite(record.coefficients).all()
    assert all(summary.is_available for summary in result.cell_summaries.values())
    assert all(np.isfinite(summary.primary_mean) for summary in result.cell_summaries.values())


def test_classification_metrics_use_one_positive_score_vector_for_threshold_and_auc():
    """Balanced accuracy thresholds at 0.5 and AUC uses those same positive scores."""
    target_values = np.array([0, 0, 1, 1])
    positive_scores = np.array([0.51, 0.90, 0.60, 0.10])

    metrics = modeling.classification_metrics(target_values, positive_scores, positive_class=1)

    assert metrics["balanced_accuracy"] == pytest.approx(0.25)
    assert metrics["auc"] == pytest.approx(0.25)


def test_fit_and_score_extracts_saved_positive_class_from_nonconstant_probability_columns():
    """Class order, not a fixed probability column, determines saved positive scores."""

    class ReversedProbabilityEstimator:
        """Estimator with nonconstant probabilities and deliberately reversed classes."""

        def fit(self, features, targets):
            """Expose a binary classifier whose positive class occupies column zero."""
            self.coef_ = np.ones((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([1, 0])
            return self

        def predict_proba(self, features):
            """Return nonconstant rows ordered as classes ``[1, 0]``."""
            assert len(features) == 4
            return np.array([[0.90, 0.10], [0.20, 0.80], [0.80, 0.20], [0.10, 0.90]])

    fit_result = modeling.fit_and_score(
        ReversedProbabilityEstimator(),
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 1, 0, 1]),
        np.array([[4.0], [5.0], [6.0], [7.0]]),
        np.array([1, 0, 1, 0]),
        target_family="categorical",
    )

    assert fit_result.is_valid
    assert fit_result.positive_class == 1
    np.testing.assert_allclose(fit_result.positive_scores, [0.90, 0.20, 0.80, 0.10])
    assert fit_result.metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert fit_result.metrics["auc"] == pytest.approx(1.0)


def test_regression_r2_preserves_native_values_and_rejects_constant_or_singleton_test_targets():
    """R2 is native-scale; constant/singleton held-out target vectors are unavailable."""
    assert modeling.regression_r2(
        np.array([10.0, 20.0, 30.0]),
        np.array([8.0, 21.0, 28.0]),
    ) == pytest.approx(0.955)
    assert modeling.regression_r2(np.array([2.0, 2.0]), np.array([1.0, 3.0])) is None
    assert modeling.regression_r2(np.array([2.0]), np.array([2.0])) is None


def test_below_chance_and_negative_scores_remain_valid_but_missing_or_nan_scores_do_not():
    """Below-reference scores stay valid while NaN prevents a primary mean."""
    classification = modeling.aggregate_complete_folds((0.25, 0.25, 0.25), expected_fold_count=3)
    regression = modeling.aggregate_complete_folds((-2.0, -0.5, -1.0), expected_fold_count=3)
    nan_summary = modeling.aggregate_complete_folds((0.7, np.nan, 0.9), expected_fold_count=3)
    incomplete = modeling.aggregate_complete_folds(
        (0.7, None, 0.9),
        expected_fold_count=3,
        invalid_reasons=(None, "convergence_warning", None),
    )

    assert classification.is_available
    assert classification.primary_mean == pytest.approx(0.25)
    assert regression.is_available
    assert regression.primary_mean == pytest.approx(-7.0 / 6.0)
    assert not nan_summary.is_available
    assert not incomplete.is_available
    assert incomplete.valid_fold_count == 2
    assert "convergence" in incomplete.unavailable_reason.lower()


@pytest.mark.parametrize(
    ("estimator", "family", "reason_fragment"),
    [
        ("nan_coefficients", "numerical", "coefficient"),
        ("nan_predictions", "numerical", "finite"),
        ("nan_probabilities", "categorical", "finite"),
    ],
)
def test_nonfinite_model_outputs_invalidate_fit_without_hiding_unexpected_errors(
    estimator,
    family,
    reason_fragment,
):
    """Non-finite artifacts are invalidity; programming failures must propagate."""

    class NonfiniteEstimator:
        """Minimal estimator exposing one chosen non-finite output location."""

        def fit(self, features, targets):
            """Expose coefficients/intercepts with an optional NaN coefficient."""
            coefficient = np.nan if estimator == "nan_coefficients" else 1.0
            self.coef_ = np.full((1, features.shape[1]), coefficient)
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict(self, features):
            """Return numerical predictions with an optional NaN."""
            value = np.nan if estimator == "nan_predictions" else 0.0
            return np.full(len(features), value)

        def predict_proba(self, features):
            """Return categorical probabilities with an optional NaN."""
            value = np.nan if estimator == "nan_probabilities" else 0.5
            return np.tile(np.array([[1.0 - value, value]]), (len(features), 1))

    result = modeling.fit_and_score(
        NonfiniteEstimator(),
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 1, 0, 1]) if family == "categorical" else np.arange(4.0),
        np.array([[0.5], [2.5]]),
        np.array([0, 1]) if family == "categorical" else np.array([0.5, 2.5]),
        target_family=family,
    )

    assert not result.is_valid
    assert reason_fragment in result.reason.lower()

    class BrokenEstimator:
        """Estimator whose unexpected exception must not become scientific missingness."""

        def fit(self, _features, _targets):
            """Raise a programming-like exception without a convergence warning."""
            raise RuntimeError("unexpected implementation failure")

    with pytest.raises(RuntimeError, match="unexpected"):
        modeling.fit_and_score(
            BrokenEstimator(),
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.5], [1.5]]),
            np.array([0.5, 1.5]),
            target_family="numerical",
        )


def test_convergence_warning_invalidates_but_converged_all_zero_solution_is_valid():
    """Warnings invalidate a fold; strong valid regularization may select zero."""

    class WarningEstimator:
        """Estimator that emits a scikit-learn convergence warning during fit."""

        def fit(self, features, targets):
            """Warn and expose otherwise valid categorical estimator attributes."""
            warnings.warn("synthetic non-convergence", ConvergenceWarning)
            self.coef_ = np.zeros((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return finite binary probabilities despite the warning."""
            return np.tile(np.array([[0.5, 0.5]]), (len(features), 1))

    warning_result = modeling.fit_and_score(
        WarningEstimator(),
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0, 1, 0, 1]),
        np.array([[0.5], [2.5]]),
        np.array([0, 1]),
        target_family="categorical",
    )
    zero_result = modeling.fit_and_score(
        ElasticNet(alpha=1e12, l1_ratio=0.5),
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([[0.5], [2.5]]),
        np.array([0.0, 1.0]),
        target_family="numerical",
    )

    assert not warning_result.is_valid
    assert "convergence" in warning_result.reason.lower()
    assert zero_result.is_valid
    np.testing.assert_allclose(zero_result.coefficients, 0.0, atol=1e-8)


def test_unavailable_pfc_only_invalidates_pfc_and_combined_cells_without_collapse(monkeypatch):
    """No usable PFC feature invalidates PFC/combined but leaves HPC standalone usable."""

    class FastClassifier:
        """Deterministic finite classifier for the feature-set availability boundary."""

        def fit(self, features, targets):
            """Expose valid classification attributes without iterative fitting."""
            self.coef_ = np.ones((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return finite positive-class scores."""
            return np.tile(np.array([[0.4, 0.6]]), (len(features), 1))

    monkeypatch.setattr(
        modeling,
        "make_estimator",
        lambda *, target_family, parameters=None: FastClassifier(),
    )
    inputs = make_classification_inputs()
    inputs["pfc_rate_tensor_hz"] = np.ones_like(inputs["pfc_rate_tensor_hz"])
    result = modeling.decode_target(**inputs)

    for time_bin in range(2):
        for representation in ("pca", "units"):
            assert not selected_summary(result, time_bin, "PFC", representation).is_available
            assert not selected_summary(result, time_bin, "PFC+HPC", representation).is_available
            assert selected_summary(result, time_bin, "HPC", representation).is_available


def patch_fast_classification_estimator(monkeypatch):
    """Replace iterative logistic fitting with a deterministic finite classifier.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Test-scoped patching utility used to replace the estimator factory.

    Returns
    -------
    None
        The modeling estimator factory is patched for categorical test calls.
    """

    class FastClassifier:
        """Finite one-fit classifier that retains the positive-class coefficient sign."""

        def fit(self, features, targets):
            """Expose positive standardized-feature coefficients and an intercept."""
            self.coef_ = np.ones((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return one reusable finite positive-class score vector."""
            return np.tile(np.array([[0.4, 0.6]]), (len(features), 1))

    monkeypatch.setattr(
        modeling,
        "make_estimator",
        lambda *, target_family, parameters=None: FastClassifier(),
    )


def test_direct_unit_records_and_summary_preserve_positive_orientation_and_all_metadata(
    monkeypatch,
):
    """Direct coefficients retain orientation, scale, status, and intercepts."""
    patch_fast_classification_estimator(monkeypatch)
    result = decode_target()
    pfc_unit_records = [
        record
        for record in result.fold_records
        if record.region_configuration == "PFC" and record.representation == "units"
    ]
    summary = modeling.summarize_direct_unit_coefficients(
        feature_ids=("probe-pfc:11", "probe-hpc:5"),
        regions=("PFC", "HPC"),
        coefficients=np.array([[1e-9, -2.0], [-3.0, -4.0], [np.nan, -6.0]]),
        intercepts=np.array([7.0, 8.0, 9.0]),
        feature_statuses=("constant_in_one_fold", "available"),
    )

    assert pfc_unit_records
    for record in pfc_unit_records:
        assert record.positive_class == 1
        assert (
            record.coefficient_scale_label
            == "log-odds change per pooled training standard deviation"
        )
        assert record.coefficients[0] > 0.0
        assert np.isfinite(record.intercept)
    assert summary.feature_ids == ("probe-hpc:5", "probe-pfc:11")
    assert summary.regions == ("HPC", "PFC")
    assert summary.feature_statuses == ("available", "constant_in_one_fold")
    np.testing.assert_allclose(summary.mean_absolute_coefficients, [4.0, 1.5])
    np.testing.assert_allclose(summary.signed_median_coefficients, [-4.0, -1.5])
    np.testing.assert_allclose(summary.signed_iqr_coefficients, [2.0, 1.5])
    np.testing.assert_allclose(summary.selection_frequencies, [1.0, 0.5])
    np.testing.assert_array_equal(summary.contributing_fold_counts, [3, 2])
    np.testing.assert_array_equal(summary.nonzero_counts_per_fit, [1, 2, 1])
    np.testing.assert_allclose(summary.nonzero_fractions_per_fit, [0.5, 1.0, 1.0])
    np.testing.assert_allclose(summary.intercepts, [7.0, 8.0, 9.0])


def test_fixed_seed_repeats_outer_assignments_scores_and_coefficients(monkeypatch):
    """Frozen seed zero must make repeated fixed-mode modeling deterministic."""
    patch_fast_classification_estimator(monkeypatch)
    first = decode_target()
    second = decode_target()

    np.testing.assert_array_equal(first.outer_fold_ids, second.outer_fold_ids)
    assert set(first.cell_summaries) == set(second.cell_summaries)
    for key in first.cell_summaries:
        assert first.cell_summaries[key].primary_mean == pytest.approx(
            second.cell_summaries[key].primary_mean
        )
    first_records = sorted(first.fold_records, key=lambda record: record.record_key)
    second_records = sorted(second.fold_records, key=lambda record: record.record_key)
    for first_record, second_record in zip(first_records, second_records, strict=True):
        assert first_record.record_key == second_record.record_key
        np.testing.assert_allclose(first_record.coefficients, second_record.coefficients)
        assert first_record.intercept == pytest.approx(second_record.intercept)
        assert first_record.metrics == pytest.approx(second_record.metrics)


def test_estimator_factory_uses_installed_sklearn_api_frozen_controls_and_omits_penalty(
    monkeypatch,
):
    """Logistic must omit deprecated ``penalty`` while PCA uses frozen controls."""
    controls = frozen_controls()["estimators"]
    logistic_signature = inspect.signature(LogisticRegression)
    assert logistic_signature.parameters["penalty"].default == "deprecated"
    assert "l1_ratio" in logistic_signature.parameters
    assert "solver" in logistic_signature.parameters
    assert "alpha" in inspect.signature(ElasticNet).parameters
    assert "whiten" in inspect.signature(PCA).parameters
    captured_kwargs = {}
    sentinel = object()

    def capture_logistic(**kwargs):
        """Capture constructor keyword arguments without invoking sklearn."""
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(modeling, "LogisticRegression", capture_logistic)
    assert modeling.make_estimator(target_family="categorical") is sentinel
    assert "penalty" not in captured_kwargs
    assert captured_kwargs == controls["LogisticRegression"]

    monkeypatch.setattr(modeling, "LogisticRegression", LogisticRegression)
    logistic = modeling.make_estimator(target_family="categorical")
    numerical = modeling.make_estimator(target_family="numerical")
    assert isinstance(logistic, LogisticRegression)
    assert isinstance(numerical, ElasticNet)
    for parameter_name, expected_value in controls["ElasticNet"].items():
        assert getattr(numerical, parameter_name) == expected_value


def test_outer_split_failure_returns_an_audit_ready_unavailable_target_result():
    """An invalid outer plan must remain explainable in the principal result."""
    inputs = make_classification_inputs(n_blocks=3)
    inputs["target_values"] = np.array([0, 0, 1, 1, 0, 0] * 2)
    inputs["block_ids"] = np.repeat(np.arange(3), 4)
    inputs["outer_fold_count"] = 3
    expected_plan = modeling.make_outer_splits(
        np.asarray(inputs["target_values"]),
        np.asarray(inputs["block_ids"]),
        target_family="categorical",
        fold_count=3,
    )

    result = modeling.decode_target(**inputs)

    assert not expected_plan.is_available
    assert not result.is_available
    assert result.status == "unavailable"
    assert result.unavailable_reason == expected_plan.unavailable_reason
    assert "class" in result.unavailable_reason.lower()
    assert np.all(result.outer_fold_ids == -1)


@pytest.mark.parametrize("fold_count", (3, 5))
def test_outer_splits_accept_only_the_configured_integer_fold_counts(fold_count):
    """Outer CV accepts the two approved scientific fold counts, not any integer."""
    inputs = make_classification_inputs(n_blocks=10)

    split_plan = modeling.make_outer_splits(
        np.asarray(inputs["target_values"]),
        np.asarray(inputs["block_ids"]),
        target_family="categorical",
        fold_count=fold_count,
    )

    assert split_plan.is_available
    assert len(split_plan.splits) == fold_count


@pytest.mark.parametrize("fold_count", (True, 3.0, "3", 2, 4, 6))
def test_outer_splits_reject_unapproved_fold_count_types_and_values(fold_count):
    """Invalid outer fold settings are programming/configuration errors, not results."""
    inputs = make_classification_inputs(n_blocks=10)

    with pytest.raises(ValueError):
        modeling.make_outer_splits(
            np.asarray(inputs["target_values"]),
            np.asarray(inputs["block_ids"]),
            target_family="categorical",
            fold_count=fold_count,
        )


@pytest.mark.parametrize("fold_count", (3, True, 3.0, "3", 2, 4, 5))
def test_inner_splits_accept_only_three_and_reject_other_fold_count_settings(fold_count):
    """Inner CV has one scientifically frozen three-fold contract."""
    inputs = make_classification_inputs(n_blocks=6)
    values = np.asarray(inputs["target_values"])
    blocks = np.asarray(inputs["block_ids"])
    indices = np.arange(values.size)

    if type(fold_count) is int and fold_count == 3:
        split_plan = modeling.make_inner_splits(
            values,
            blocks,
            indices,
            target_family="categorical",
            fold_count=fold_count,
        )
        assert split_plan.is_available
    else:
        with pytest.raises(ValueError):
            modeling.make_inner_splits(
                values,
                blocks,
                indices,
                target_family="categorical",
                fold_count=fold_count,
            )


@pytest.mark.parametrize(
    "outer_train_indices",
    (
        np.array([False, True]),
        np.array([0.1, 1.2, 2.4, 3.6, 4.8, 5.9]),
        np.array(["0", "1", "2", "3", "4", "5"]),
    ),
)
def test_inner_splits_reject_noninteger_indices_without_coercion(outer_train_indices):
    """Inner row identities must be supplied as native integer positions."""
    inputs = make_classification_inputs(n_blocks=6)

    with pytest.raises(ValueError):
        modeling.make_inner_splits(
            np.asarray(inputs["target_values"]),
            np.asarray(inputs["block_ids"]),
            outer_train_indices,
            target_family="categorical",
            fold_count=3,
        )


def test_fixed_results_and_records_retain_target_and_inactive_inner_configuration(monkeypatch):
    """Fixed audit records preserve target identity and inactive inner-CV settings."""
    patch_fast_classification_estimator(monkeypatch)
    inputs = make_classification_inputs()
    inputs["target_identifier"] = "persisted_synthetic_target"
    inputs["inner_fold_count"] = 3

    result = modeling.decode_target(**inputs)

    assert result.target_identifier == "persisted_synthetic_target"
    assert result.inner_fold_count == 3
    assert result.fold_records
    for record in result.fold_records:
        assert record.record_key == (
            record.outer_fold_id,
            record.time_bin_index,
            record.region_configuration,
            record.representation,
        )
        assert record.target_identifier == "persisted_synthetic_target"
        assert record.inner_fold_count == 3


def test_estimator_factory_allows_only_family_tuning_overrides_and_keeps_fixed_defaults():
    """Candidate overrides cannot alter frozen solver, weighting, or convergence controls."""
    controls = frozen_controls()["estimators"]
    categorical = modeling.make_estimator(
        target_family="categorical",
        parameters={"C": 0.25, "l1_ratio": 0.75},
    )
    numerical = modeling.make_estimator(
        target_family="numerical",
        parameters={"alpha": 0.25, "l1_ratio": 0.75},
    )

    assert categorical.C == pytest.approx(0.25)
    assert categorical.l1_ratio == pytest.approx(0.75)
    assert categorical.solver == controls["LogisticRegression"]["solver"]
    assert numerical.alpha == pytest.approx(0.25)
    assert numerical.l1_ratio == pytest.approx(0.75)
    assert numerical.max_iter == controls["ElasticNet"]["max_iter"]

    rejected_overrides = (
        ("categorical", {"solver": "lbfgs"}),
        ("categorical", {"class_weight": "balanced"}),
        ("categorical", {"random_state": 9}),
        ("categorical", {"penalty": "l2"}),
        ("categorical", {"max_iter": 9}),
        ("categorical", {"positive": True}),
        ("categorical", {"alpha": 0.1}),
        ("numerical", {"C": 0.1}),
        ("numerical", {"max_iter": 9}),
        ("numerical", {"selection": "random"}),
        ("numerical", {"positive": True}),
        ("numerical", {"random_state": 9}),
        ("numerical", {"tol": 1e-3}),
        ("numerical", {"unrecognized_control": 1.0}),
    )
    for target_family, parameters in rejected_overrides:
        with pytest.raises(ValueError):
            modeling.make_estimator(target_family=target_family, parameters=parameters)

    fixed_categorical = modeling.make_estimator(target_family="categorical")
    fixed_numerical = modeling.make_estimator(target_family="numerical")
    assert fixed_categorical.C == controls["LogisticRegression"]["C"]
    assert fixed_categorical.l1_ratio == controls["LogisticRegression"]["l1_ratio"]
    assert fixed_numerical.alpha == controls["ElasticNet"]["alpha"]
    assert fixed_numerical.l1_ratio == controls["ElasticNet"]["l1_ratio"]


@pytest.mark.parametrize(
    "target_values",
    (
        np.array([-1, 1, -1, 1, -1, 1]),
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([False, True, False, True, False, True]),
    ),
)
def test_categorical_splits_require_exact_nonboolean_numeric_zero_one_encoding(target_values):
    """Categorical split labels are the frozen nonboolean numeric encoding {0, 1}."""
    block_ids = np.repeat(np.arange(3), 2)

    with pytest.raises(ValueError):
        modeling.make_outer_splits(
            target_values,
            block_ids,
            target_family="categorical",
            fold_count=3,
        )


def test_categorical_splits_accept_float_coded_zero_one_labels():
    """Target construction's numeric float-coded binary labels remain valid for CV."""
    target_values = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    block_ids = np.repeat(np.arange(3), 2)

    split_plan = modeling.make_outer_splits(
        target_values,
        block_ids,
        target_family="categorical",
        fold_count=3,
    )

    assert split_plan.is_available


@pytest.mark.parametrize(
    ("region_name", "invalid_unit_ids"),
    (
        ("pfc_unit_ids", ("", "probe-pfc:19")),
        ("pfc_unit_ids", ("probe-pfc:11", "probe-pfc:11")),
        ("hpc_unit_ids", ("", "probe-hpc:17")),
        ("hpc_unit_ids", ("probe-hpc:5", "probe-hpc:5")),
    ),
)
def test_decode_target_rejects_empty_or_duplicate_unit_ids_before_fitting(
    monkeypatch,
    region_name,
    invalid_unit_ids,
):
    """Stable region IDs must be nonempty and unique before any fit can begin."""
    inputs = make_classification_inputs()
    inputs[region_name] = invalid_unit_ids

    def fail_transform(*_args, **_kwargs):
        """Fail if invalid stable IDs reach any training transform operation."""
        raise AssertionError("Unit ID validation must happen before transform fitting.")

    monkeypatch.setattr(modeling, "fit_region_transform", fail_transform)
    with pytest.raises(ValueError):
        modeling.decode_target(**inputs)


@pytest.mark.parametrize(
    "outer_train_indices",
    (
        np.array(0, dtype=int),
        np.array([[0, 1]], dtype=int),
    ),
)
def test_inner_splits_reject_scalar_or_matrix_index_arrays(outer_train_indices):
    """Inner split row positions must be a one-dimensional native integer array."""
    inputs = make_classification_inputs(n_blocks=6)

    with pytest.raises(ValueError):
        modeling.make_inner_splits(
            np.asarray(inputs["target_values"]),
            np.asarray(inputs["block_ids"]),
            outer_train_indices,
            target_family="categorical",
            fold_count=3,
        )


def test_inner_splits_keep_an_empty_native_integer_selection_as_unavailable():
    """A genuinely empty one-dimensional integer selection is a scientific state."""
    inputs = make_classification_inputs(n_blocks=6)

    split_plan = modeling.make_inner_splits(
        np.asarray(inputs["target_values"]),
        np.asarray(inputs["block_ids"]),
        np.array([], dtype=int),
        target_family="categorical",
        fold_count=3,
    )

    assert not split_plan.is_available
    assert "empty" in split_plan.unavailable_reason.lower()


@pytest.mark.parametrize(
    ("requested_count_name", "invalid_requested_count"),
    (
        ("pfc_requested_pc_count", True),
        ("pfc_requested_pc_count", 2.5),
        ("pfc_requested_pc_count", "2"),
        ("pfc_requested_pc_count", 0),
        ("pfc_requested_pc_count", -1),
        ("hpc_requested_pc_count", True),
        ("hpc_requested_pc_count", 2.5),
        ("hpc_requested_pc_count", "2"),
        ("hpc_requested_pc_count", 0),
        ("hpc_requested_pc_count", -1),
    ),
)
def test_decode_validates_requested_pc_counts_before_constructing_outer_splits(
    monkeypatch,
    requested_count_name,
    invalid_requested_count,
):
    """Invalid requested PCs are configuration errors before any split decision."""
    inputs = make_classification_inputs(n_blocks=3)
    inputs["target_values"] = np.array([0, 0, 1, 1, 0, 0] * 2)
    inputs["block_ids"] = np.repeat(np.arange(3), 4)
    inputs["outer_fold_count"] = 3
    inputs[requested_count_name] = invalid_requested_count

    def fail_outer_split(*_args, **_kwargs):
        """Fail if invalid requested PCs reach outer grouped-split construction."""
        raise AssertionError("Requested PC counts must be validated before outer splitting.")

    monkeypatch.setattr(modeling, "make_outer_splits", fail_outer_split)
    with pytest.raises(ValueError):
        modeling.decode_target(**inputs)


@pytest.mark.parametrize(
    ("region_name", "invalid_container"),
    (
        ("pfc_unit_ids", "ab"),
        ("pfc_unit_ids", b"ab"),
        ("hpc_unit_ids", "ab"),
        ("hpc_unit_ids", b"ab"),
    ),
)
def test_decode_rejects_scalar_string_or_bytes_unit_id_containers_before_fitting(
    monkeypatch,
    region_name,
    invalid_container,
):
    """Scalar text/bytes containers must not be interpreted as unit-ID sequences."""
    inputs = make_classification_inputs()
    inputs[region_name] = invalid_container

    def fail_transform(*_args, **_kwargs):
        """Fail if scalar unit-ID containers reach regional transform fitting."""
        raise AssertionError("Scalar unit-ID containers must fail before transform fitting.")

    monkeypatch.setattr(modeling, "fit_region_transform", fail_transform)
    with pytest.raises(ValueError):
        modeling.decode_target(**inputs)
