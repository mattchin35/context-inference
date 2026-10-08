"""Tests for inter-regional in-memory result and array contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    PCAConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    TemporalConfig,
)
from src.neural_analysis.interregional import records


EXPECTED_COLUMNS = {
    "fold_assignments": (
        "session_id", "trial_row", "original_index_repr", "block_value_json",
        "block_present", "fold_id", "status", "reason",
    ),
    "trial_membership": (
        "session_id", "trial_row", "alignment", "condition", "original_index_repr",
        "reward_status_valid", "alignment_valid", "choice_match", "context_match",
        "user_included", "block_present", "condition_match", "scientific_eligible",
        "condition_included", "cv_included", "scientific_exclusion_reasons_json",
    ),
    "fold_scores": (
        "session_id", "direction", "representation", "model_family", "condition", "window",
        "target_id", "fold_id", "evaluation_scope", "target_rank", "restricted_status",
        "restricted_reason", "full_status", "full_reason", "status", "reason",
        "n_train_trials", "n_test_trials", "n_train_rows", "n_test_rows",
        "train_row_set_sha256", "test_row_set_sha256", "restricted_feature_count",
        "full_feature_count", "restricted_rank", "full_rank", "restricted_df_resid",
        "full_df_resid", "restricted_converged", "full_converged",
        "restricted_iterations", "full_iterations", "r2_restricted", "r2_full", "delta_r2",
        "mse_restricted", "mse_full", "deviance_restricted", "deviance_full",
        "null_deviance", "deviance_explained_restricted", "deviance_explained_full",
        "delta_deviance_explained",
    ),
    "target_summaries": (
        "session_id", "evaluation_scope", "direction", "representation", "model_family",
        "condition", "window", "target_id", "metric_name", "target_rank", "status", "reason",
        "requested_folds", "valid_folds", "mean_value",
    ),
    "population_summaries": (
        "session_id", "evaluation_scope", "direction", "representation", "model_family",
        "condition", "window", "metric_name", "status", "reason", "n_targets", "q25",
        "median", "q75",
    ),
    "pca_fits": (
        "session_id", "scope", "fold_id", "region", "status", "reason",
        "requested_components", "actual_components", "n_training_trials",
        "n_training_observations", "retained_unit_ids_json", "omitted_unit_ids_json",
    ),
    "granger_scores": (
        "session_id", "direction", "representation", "model_family", "condition", "window",
        "target_id", "evaluation_scope", "target_rank", "restricted_status",
        "restricted_reason", "full_status", "full_reason", "status", "reason", "diagnostic",
        "n_trials", "n_rows", "restricted_feature_count", "full_feature_count",
        "restricted_rank", "full_rank", "restricted_df_resid", "full_df_resid",
        "restricted_converged", "full_converged", "restricted_iterations", "full_iterations",
        "sse_restricted", "sse_full", "linear_granger", "llf_restricted", "llf_full",
        "deviance_restricted", "deviance_full", "likelihood_ratio",
        "mean_deviance_improvement",
    ),
}


def _population(role: str, probe: str, units: tuple[int, ...]) -> ResolvedRegionalPopulation:
    """Construct one explicit resolved population for schema tests."""
    return ResolvedRegionalPopulation(
        role=role,
        probe_id=probe,
        channel_source="explicit",
        selected_channels=(0,),
        cluster_ids=units,
        unit_ids=tuple(f"{probe}:{unit}" for unit in units),
    )


def _config(**changes: object) -> InterregionalAnalysisConfig:
    """Construct a compact valid scientific configuration for record tests."""
    base = InterregionalAnalysisConfig(
        session_metadata_path="/data/session/neural_session.json",
        pfc_population=RegionalPopulationConfig(role="PFC", probe_id="pfc"),
        hpc_population=RegionalPopulationConfig(role="HPC", probe_id="hpc"),
        windows=AnalysisWindows(),
        prediction_windows=("before",),
        temporal=TemporalConfig(),
        pca=PCAConfig(pfc_components=1, hpc_components=1),
        filters=FilterConfig(conditions=("all",)),
    )
    return replace(base, **changes)


def test_all_empty_tables_have_the_frozen_columns_and_dtypes() -> None:
    """Every present and future stage starts from one exact typed schema."""
    tables = records.make_empty_result_tables()

    assert tuple(tables) == tuple(EXPECTED_COLUMNS)
    for name, expected_columns in EXPECTED_COLUMNS.items():
        table = tables[name]
        assert tuple(table.columns) == expected_columns
        assert table.empty
        assert table.dtypes.astype(str).to_dict() == records.RESULT_TABLE_DTYPES[name]
        records.validate_result_table(name, table)


def test_table_validation_rejects_schema_dtype_key_and_sorting_errors() -> None:
    """Saved tables cannot drift in columns, dtypes, uniqueness, or key order."""
    tables = records.make_empty_result_tables()
    with pytest.raises(ValueError, match="columns"):
        records.validate_result_table(
            "fold_assignments", tables["fold_assignments"].drop(columns="reason")
        )

    wrong_dtype = tables["fold_assignments"].copy()
    wrong_dtype["trial_row"] = pd.Series(dtype="Int64")
    with pytest.raises(ValueError, match="dtype"):
        records.validate_result_table("fold_assignments", wrong_dtype)

    rows = [
        {
            "session_id": "s",
            "trial_row": trial_row,
            "original_index_repr": str(trial_row),
            "block_value_json": "1",
            "block_present": True,
            "fold_id": 0,
            "status": "ok",
            "reason": "",
        }
        for trial_row in (1, 0)
    ]
    unsorted = records.result_table_from_rows("fold_assignments", rows, sort=False)
    with pytest.raises(ValueError, match="sorted"):
        records.validate_result_table("fold_assignments", unsorted)

    duplicate = records.result_table_from_rows("fold_assignments", [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        records.validate_result_table("fold_assignments", duplicate)


def test_status_reason_and_canonical_json_values_are_validated() -> None:
    """Only frozen status/reason pairs and compact canonical JSON are persisted."""
    valid = {
        "session_id": "s",
        "trial_row": 0,
        "original_index_repr": "0",
        "block_value_json": "1",
        "block_present": True,
        "fold_id": 0,
        "status": "ok",
        "reason": "",
    }
    records.validate_result_table(
        "fold_assignments", records.result_table_from_rows("fold_assignments", [valid])
    )

    with pytest.raises(ValueError, match="status"):
        records.result_table_from_rows("fold_assignments", [{**valid, "status": "made_up"}])
    with pytest.raises(ValueError, match="reason"):
        records.result_table_from_rows("fold_assignments", [{**valid, "reason": "made_up"}])
    with pytest.raises(ValueError, match="canonical JSON"):
        records.result_table_from_rows(
            "fold_assignments", [{**valid, "block_value_json": "[1, 2]"}]
        )


def test_trial_membership_enforces_exclusion_order_and_mask_relationships() -> None:
    """Membership provenance cannot contradict the scientific and CV masks."""
    valid = {
        "session_id": "s",
        "trial_row": 0,
        "alignment": "choice_time",
        "condition": "all",
        "original_index_repr": "0",
        "reward_status_valid": False,
        "alignment_valid": True,
        "choice_match": False,
        "context_match": True,
        "user_included": True,
        "block_present": True,
        "condition_match": True,
        "scientific_eligible": False,
        "condition_included": False,
        "cv_included": False,
        "scientific_exclusion_reasons_json": '["invalid_reward_status","choice_filter_mismatch"]',
    }
    records.validate_result_table(
        "trial_membership", records.result_table_from_rows("trial_membership", [valid])
    )

    with pytest.raises(ValueError, match="fixed order"):
        records.result_table_from_rows(
            "trial_membership",
            [
                {
                    **valid,
                    "scientific_exclusion_reasons_json": (
                        '["choice_filter_mismatch","invalid_reward_status"]'
                    ),
                }
            ],
        )
    with pytest.raises(ValueError, match="condition_included"):
        records.result_table_from_rows(
            "trial_membership", [{**valid, "condition_included": True}]
        )


def test_metric_specific_target_statuses_coexist_for_one_target() -> None:
    """Defined MSE is retained when the same target's R-squared is incomplete."""
    base = {
        "session_id": "s",
        "evaluation_scope": "held_out_cv",
        "direction": "HPC_to_PFC",
        "representation": "units",
        "model_family": "ols",
        "condition": "all",
        "window": "before",
        "target_id": "pfc:1",
        "target_rank": None,
        "requested_folds": 5,
        "valid_folds": 5,
    }
    table = records.result_table_from_rows(
        "target_summaries",
        [
            {
                **base,
                "metric_name": "mse_full",
                "status": "ok",
                "reason": "",
                "mean_value": 1.5,
            },
            {
                **base,
                "metric_name": "r2_full",
                "status": "incomplete_folds",
                "reason": "incomplete_requested_folds",
                "valid_folds": 4,
                "mean_value": None,
            },
        ],
    )

    records.validate_result_table("target_summaries", table)
    assert table.loc[table["metric_name"] == "mse_full", "mean_value"].iloc[0] == 1.5


def test_expected_cv_key_grid_includes_units_and_ols_pcs_but_not_poisson_pcs() -> None:
    """Five folds are materialized for every applicable requested target."""
    config = _config(
        representations=("units", "pcs"),
        analyses=("ols_cv", "poisson_cv"),
    )
    populations = (_population("PFC", "pfc", (1, 2)), _population("HPC", "hpc", (3,)))
    keys = records.expected_fold_score_keys("session", config, populations)

    assert len(keys) == 40
    assert not any(key[2:4] == ("pcs", "poisson") for key in keys)
    records.validate_fold_score_key_grid(
        pd.DataFrame(keys, columns=records.FOLD_SCORE_KEY),
        "session",
        config,
        populations,
    )

    with pytest.raises(ValueError, match="key grid"):
        records.validate_fold_score_key_grid(
            pd.DataFrame(keys[:-1], columns=records.FOLD_SCORE_KEY),
            "session",
            config,
            populations,
        )


def test_array_records_enforce_shapes_axes_and_physical_units() -> None:
    """Count and history arrays preserve their documented axes and integer counts."""
    tensor = records.RegionalCountTensor(
        counts=np.zeros((2, 3, 1), dtype=np.int64),
        trial_rows=np.array([1, 4], dtype=np.int64),
        original_index_labels=("a", "b"),
        bin_edges_s=np.array([-0.3, -0.2, -0.1, 0.0], dtype=np.float64),
        unit_ids=("pfc:1",),
    )
    assert tensor.counts.shape == (2, 3, 1)

    with pytest.raises(ValueError, match="shape"):
        records.RegionalCountTensor(
            counts=np.zeros((2, 2, 1), dtype=np.int64),
            trial_rows=np.array([1, 4], dtype=np.int64),
            original_index_labels=("a", "b"),
            bin_edges_s=np.array([-0.3, -0.2, -0.1, 0.0], dtype=np.float64),
            unit_ids=("pfc:1",),
        )

    history = records.HistoryMatrices(
        responses=np.zeros((4, 2)),
        target_history=np.zeros((4, 4)),
        source_history=np.zeros((4, 6)),
        row_trial=np.array([0, 0, 1, 1], dtype=np.int64),
        row_target_bin=np.array([1, 2, 1, 2], dtype=np.int64),
    )
    assert history.responses.shape == (4, 2)

    with pytest.raises(ValueError, match="positive integer"):
        records.RegionalPCATransform(
            retained_unit_ids=("pfc:1",),
            omitted_unit_ids=(),
            training_mean=np.array([0.0]),
            training_scale=np.array([1.0]),
            components=np.array([[1.0]]),
            explained_variance=np.array([1.0]),
            explained_variance_ratio=np.array([1.0]),
            n_training_observations=1.5,  # type: ignore[arg-type]
        )


def test_interregional_results_validate_exact_metadata_and_empty_future_schemas() -> None:
    """The pure result record contains canonical metadata and all seven tables."""
    config = _config()
    populations = (_population("PFC", "pfc", (1,)), _population("HPC", "hpc", (2,)))
    result = records.InterregionalResults(
        schema_version="1",
        analysis_version="interregional-regression-v1",
        coverage_assumption_version="implicit-complete-v1",
        configuration={"analysis_version": "interregional-regression-v1"},
        session_id="session",
        resolved_populations=populations,
        whole_bin_edges_s=np.linspace(-2.0, 2.0, 41, dtype=np.float64),
        units_and_axes=records.default_units_and_axes(),
        randomness_used=False,
        random_seed=None,
        **records.make_empty_result_tables(),
    )

    records.validate_interregional_results(result, config)
    assert result.fold_scores.empty
    with pytest.raises(FrozenInstanceError):
        result.session_id = "changed"  # type: ignore[misc]

    bad_edges = replace(result, whole_bin_edges_s=result.whole_bin_edges_s.astype(np.float32))
    with pytest.raises(ValueError, match="float64"):
        records.validate_interregional_results(bad_edges, config)

    bad_axes = replace(result, units_and_axes={"count_tensor": {}})
    with pytest.raises(ValueError, match="units_and_axes"):
        records.validate_interregional_results(bad_axes, config)
