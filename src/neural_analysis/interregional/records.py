"""Typed array and tidy-table contracts for inter-regional regression results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from numbers import Integral
import re
from typing import Any

import numpy as np
import pandas as pd

from .configuration import (
    ANALYSIS_VERSION,
    COVERAGE_ASSUMPTION_VERSION,
    N_CV_FOLDS,
    RESULT_SCHEMA_VERSION,
    InterregionalAnalysisConfig,
    ResolvedRegionalPopulation,
    validate_resolved_populations,
)


FIT_STATUS_VALUES = (
    "ok",
    "metric_unavailable",
    "fit_unavailable",
    "incomplete_folds",
    "not_applicable",
)
UNAVAILABILITY_REASONS = (
    "missing_block",
    "invalid_reward_status",
    "invalid_alignment",
    "choice_filter_mismatch",
    "context_filter_mismatch",
    "user_excluded",
    "no_eligible_trials",
    "no_train_trials",
    "no_test_trials",
    "no_train_rows",
    "no_test_rows",
    "history_exceeds_window",
    "no_units",
    "pca_no_variable_units",
    "pca_insufficient_components",
    "constant_training_target",
    "rank_deficient_restricted",
    "rank_deficient_full",
    "nonpositive_df_restricted",
    "nonpositive_df_full",
    "nonfinite_coefficients",
    "nonfinite_predictions",
    "constant_test_target",
    "zero_null_deviance",
    "poisson_nonconverged_restricted",
    "poisson_nonconverged_full",
    "poisson_fit_error_restricted",
    "poisson_fit_error_full",
    "nonpositive_poisson_mean",
    "zero_granger_residual",
    "nested_fit_inconsistency",
    "incomplete_requested_folds",
    "no_complete_targets",
)
GRANGER_DIAGNOSTICS = ("", "nested_roundoff")
SCIENTIFIC_EXCLUSION_REASON_ORDER = (
    "invalid_reward_status",
    "invalid_alignment",
    "choice_filter_mismatch",
    "context_filter_mismatch",
    "user_excluded",
)

FOLD_ASSIGNMENT_KEY = ("session_id", "trial_row")
TRIAL_MEMBERSHIP_KEY = ("session_id", "trial_row", "alignment", "condition")
FOLD_SCORE_KEY = (
    "session_id",
    "direction",
    "representation",
    "model_family",
    "condition",
    "window",
    "target_id",
    "fold_id",
)
TARGET_SUMMARY_KEY = (
    "session_id",
    "evaluation_scope",
    "direction",
    "representation",
    "model_family",
    "condition",
    "window",
    "target_id",
    "metric_name",
)
POPULATION_SUMMARY_KEY = (
    "session_id",
    "evaluation_scope",
    "direction",
    "representation",
    "model_family",
    "condition",
    "window",
    "metric_name",
)
PCA_FIT_KEY = ("session_id", "scope", "fold_id", "region")
GRANGER_SCORE_KEY = (
    "session_id",
    "direction",
    "representation",
    "model_family",
    "condition",
    "window",
    "target_id",
)

RESULT_TABLE_PRIMARY_KEYS = {
    "fold_assignments": FOLD_ASSIGNMENT_KEY,
    "trial_membership": TRIAL_MEMBERSHIP_KEY,
    "fold_scores": FOLD_SCORE_KEY,
    "target_summaries": TARGET_SUMMARY_KEY,
    "population_summaries": POPULATION_SUMMARY_KEY,
    "pca_fits": PCA_FIT_KEY,
    "granger_scores": GRANGER_SCORE_KEY,
}

_FOLD_ASSIGNMENT_DTYPES = {
    "session_id": "string",
    "trial_row": "int64",
    "original_index_repr": "string",
    "block_value_json": "string",
    "block_present": "boolean",
    "fold_id": "Int64",
    "status": "string",
    "reason": "string",
}
_TRIAL_MEMBERSHIP_DTYPES = {
    "session_id": "string",
    "trial_row": "int64",
    "alignment": "string",
    "condition": "string",
    "original_index_repr": "string",
    "reward_status_valid": "boolean",
    "alignment_valid": "boolean",
    "choice_match": "boolean",
    "context_match": "boolean",
    "user_included": "boolean",
    "block_present": "boolean",
    "condition_match": "boolean",
    "scientific_eligible": "boolean",
    "condition_included": "boolean",
    "cv_included": "boolean",
    "scientific_exclusion_reasons_json": "string",
}
_FOLD_SCORE_DTYPES = {
    "session_id": "string",
    "direction": "string",
    "representation": "string",
    "model_family": "string",
    "condition": "string",
    "window": "string",
    "target_id": "string",
    "fold_id": "int64",
    "evaluation_scope": "string",
    "target_rank": "Int64",
    "restricted_status": "string",
    "restricted_reason": "string",
    "full_status": "string",
    "full_reason": "string",
    "status": "string",
    "reason": "string",
    "n_train_trials": "int64",
    "n_test_trials": "int64",
    "n_train_rows": "int64",
    "n_test_rows": "int64",
    "train_row_set_sha256": "string",
    "test_row_set_sha256": "string",
    "restricted_feature_count": "int64",
    "full_feature_count": "int64",
    "restricted_rank": "Int64",
    "full_rank": "Int64",
    "restricted_df_resid": "Int64",
    "full_df_resid": "Int64",
    "restricted_converged": "boolean",
    "full_converged": "boolean",
    "restricted_iterations": "Int64",
    "full_iterations": "Int64",
    "r2_restricted": "Float64",
    "r2_full": "Float64",
    "delta_r2": "Float64",
    "mse_restricted": "Float64",
    "mse_full": "Float64",
    "deviance_restricted": "Float64",
    "deviance_full": "Float64",
    "null_deviance": "Float64",
    "deviance_explained_restricted": "Float64",
    "deviance_explained_full": "Float64",
    "delta_deviance_explained": "Float64",
}
_TARGET_SUMMARY_DTYPES = {
    "session_id": "string",
    "evaluation_scope": "string",
    "direction": "string",
    "representation": "string",
    "model_family": "string",
    "condition": "string",
    "window": "string",
    "target_id": "string",
    "metric_name": "string",
    "target_rank": "Int64",
    "status": "string",
    "reason": "string",
    "requested_folds": "int64",
    "valid_folds": "int64",
    "mean_value": "Float64",
}
_POPULATION_SUMMARY_DTYPES = {
    "session_id": "string",
    "evaluation_scope": "string",
    "direction": "string",
    "representation": "string",
    "model_family": "string",
    "condition": "string",
    "window": "string",
    "metric_name": "string",
    "status": "string",
    "reason": "string",
    "n_targets": "int64",
    "q25": "Float64",
    "median": "Float64",
    "q75": "Float64",
}
_PCA_FIT_DTYPES = {
    "session_id": "string",
    "scope": "string",
    "fold_id": "Int64",
    "region": "string",
    "status": "string",
    "reason": "string",
    "requested_components": "int64",
    "actual_components": "int64",
    "n_training_trials": "int64",
    "n_training_observations": "int64",
    "retained_unit_ids_json": "string",
    "omitted_unit_ids_json": "string",
}
_GRANGER_SCORE_DTYPES = {
    "session_id": "string",
    "direction": "string",
    "representation": "string",
    "model_family": "string",
    "condition": "string",
    "window": "string",
    "target_id": "string",
    "evaluation_scope": "string",
    "target_rank": "Int64",
    "restricted_status": "string",
    "restricted_reason": "string",
    "full_status": "string",
    "full_reason": "string",
    "status": "string",
    "reason": "string",
    "diagnostic": "string",
    "n_trials": "int64",
    "n_rows": "int64",
    "restricted_feature_count": "int64",
    "full_feature_count": "int64",
    "restricted_rank": "Int64",
    "full_rank": "Int64",
    "restricted_df_resid": "Int64",
    "full_df_resid": "Int64",
    "restricted_converged": "boolean",
    "full_converged": "boolean",
    "restricted_iterations": "Int64",
    "full_iterations": "Int64",
    "sse_restricted": "Float64",
    "sse_full": "Float64",
    "linear_granger": "Float64",
    "llf_restricted": "Float64",
    "llf_full": "Float64",
    "deviance_restricted": "Float64",
    "deviance_full": "Float64",
    "likelihood_ratio": "Float64",
    "mean_deviance_improvement": "Float64",
}

RESULT_TABLE_DTYPES = {
    "fold_assignments": _FOLD_ASSIGNMENT_DTYPES,
    "trial_membership": _TRIAL_MEMBERSHIP_DTYPES,
    "fold_scores": _FOLD_SCORE_DTYPES,
    "target_summaries": _TARGET_SUMMARY_DTYPES,
    "population_summaries": _POPULATION_SUMMARY_DTYPES,
    "pca_fits": _PCA_FIT_DTYPES,
    "granger_scores": _GRANGER_SCORE_DTYPES,
}

_JSON_COLUMNS = {
    "fold_assignments": ("block_value_json",),
    "trial_membership": ("scientific_exclusion_reasons_json",),
    "pca_fits": ("retained_unit_ids_json", "omitted_unit_ids_json"),
}
_DIRECTIONS = {"HPC_to_PFC", "PFC_to_HPC"}
_REPRESENTATIONS = {"units", "pcs"}
_MODEL_FAMILIES = {"ols", "poisson"}
_WINDOWS = {"before", "after", "whole"}
_EVALUATION_SCOPES = {"held_out_cv", "in_sample"}
_FIT_ONLY_STATUSES = {"ok", "fit_unavailable"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _readonly_array(value: object, dtype: np.dtype[Any], name: str) -> np.ndarray:
    """Return an owned read-only NumPy array with one exact dtype."""
    array = np.asarray(value)
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _validate_nonnegative_unique_rows(rows: np.ndarray, name: str) -> None:
    """Validate ascending unique zero-based row positions."""
    if rows.ndim != 1 or np.any(rows < 0):
        raise ValueError(f"{name} must be a one-dimensional nonnegative array.")
    if rows.size > 1 and np.any(np.diff(rows) <= 0):
        raise ValueError(f"{name} must be strictly ascending and unique.")


@dataclass(frozen=True)
class RegionalCountTensor:
    """Trial-aligned spike counts with axes ``(trial, time_bin, unit)``.

    ``counts`` contains integer spike counts per bin. ``bin_edges_s`` is in
    seconds relative to alignment and has one more entry than the time axis.
    """

    counts: np.ndarray
    trial_rows: np.ndarray
    original_index_labels: tuple[str, ...]
    bin_edges_s: np.ndarray
    unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate axis lengths, dtypes, identities, and nonnegative counts."""
        counts = _readonly_array(self.counts, np.dtype("int64"), "counts")
        trial_rows = _readonly_array(self.trial_rows, np.dtype("int64"), "trial_rows")
        edges = _readonly_array(self.bin_edges_s, np.dtype("float64"), "bin_edges_s")
        if counts.ndim != 3:
            raise ValueError("counts must have shape (trial, time_bin, unit).")
        _validate_nonnegative_unique_rows(trial_rows, "trial_rows")
        if np.any(counts < 0):
            raise ValueError("counts must contain nonnegative spike counts per bin.")
        if edges.ndim != 1 or edges.size != counts.shape[1] + 1:
            raise ValueError("bin_edges_s shape must be (time_bin + 1,).")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges_s must be finite and strictly increasing.")
        labels = tuple(str(value) for value in self.original_index_labels)
        units = tuple(str(value) for value in self.unit_ids)
        if len(labels) != counts.shape[0] or trial_rows.size != counts.shape[0]:
            raise ValueError("Trial metadata lengths must match the count trial axis shape.")
        if len(units) != counts.shape[2] or not units or len(set(units)) != len(units):
            raise ValueError("unit_ids must uniquely match the count unit axis shape.")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "trial_rows", trial_rows)
        object.__setattr__(self, "original_index_labels", labels)
        object.__setattr__(self, "bin_edges_s", edges)
        object.__setattr__(self, "unit_ids", units)


@dataclass(frozen=True)
class FoldAssignment:
    """Session-level fold assignment for zero-based trial rows."""

    trial_rows: np.ndarray
    original_index_labels: tuple[str, ...]
    block_values_json: tuple[str, ...]
    fold_ids: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Validate equal row-axis lengths and zero-based five-fold IDs."""
        rows = _readonly_array(self.trial_rows, np.dtype("int64"), "trial_rows")
        _validate_nonnegative_unique_rows(rows, "trial_rows")
        lengths = {
            rows.size,
            len(self.original_index_labels),
            len(self.block_values_json),
            len(self.fold_ids),
        }
        if len(lengths) != 1:
            raise ValueError("FoldAssignment fields must share one trial-row axis.")
        for value in self.block_values_json:
            _validate_canonical_json(value, "block_values_json")
        if any(
            fold_id is not None
            and (
                isinstance(fold_id, bool)
                or not isinstance(fold_id, Integral)
                or not 0 <= fold_id < N_CV_FOLDS
            )
            for fold_id in self.fold_ids
        ):
            raise ValueError("fold_ids must be None or zero-based fold integers 0-4.")
        object.__setattr__(self, "trial_rows", rows)


@dataclass(frozen=True)
class HistoryMatrices:
    """Aligned response/history matrices for within-trial regression.

    Activity matrices use axes ``(observation, feature)``. ``row_trial`` and
    ``row_target_bin`` are zero-based identities with shape ``(observation,)``.
    Activity values retain the caller's documented representation units.
    """

    responses: np.ndarray
    target_history: np.ndarray
    source_history: np.ndarray
    row_trial: np.ndarray
    row_target_bin: np.ndarray

    def __post_init__(self) -> None:
        """Validate finite two-dimensional arrays and aligned row identities."""
        matrices: list[np.ndarray] = []
        for name in ("responses", "target_history", "source_history"):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite two-dimensional array.")
            matrix = np.array(matrix, copy=True)
            matrix.setflags(write=False)
            matrices.append(matrix)
            object.__setattr__(self, name, matrix)
        row_trial = _readonly_array(self.row_trial, np.dtype("int64"), "row_trial")
        row_bin = _readonly_array(self.row_target_bin, np.dtype("int64"), "row_target_bin")
        n_rows = matrices[0].shape[0]
        if any(matrix.shape[0] != n_rows for matrix in matrices[1:]):
            raise ValueError("All history matrices must share the observation axis.")
        if row_trial.shape != (n_rows,) or row_bin.shape != (n_rows,):
            raise ValueError("Row identity arrays must have shape (observation,).")
        if np.any(row_trial < 0) or np.any(row_bin < 0):
            raise ValueError("Row identities must be nonnegative zero-based integers.")
        object.__setattr__(self, "row_trial", row_trial)
        object.__setattr__(self, "row_target_bin", row_bin)


@dataclass(frozen=True)
class RegionalPCATransform:
    """One fitted regional PCA transform in standardized activity space.

    Means and population standard deviations have shape ``(retained_unit,)``.
    Components have shape ``(component, retained_unit)`` and explained-variance
    arrays have shape ``(component,)``.
    """

    retained_unit_ids: tuple[str, ...]
    omitted_unit_ids: tuple[str, ...]
    training_mean: np.ndarray
    training_scale: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    n_training_observations: int

    def __post_init__(self) -> None:
        """Validate fitted PCA axes and finite standardization values."""
        retained = tuple(str(value) for value in self.retained_unit_ids)
        omitted = tuple(str(value) for value in self.omitted_unit_ids)
        if not retained or len(set(retained + omitted)) != len(retained) + len(omitted):
            raise ValueError("Retained and omitted unit identities must be nonempty/disjoint.")
        mean = np.asarray(self.training_mean, dtype=np.float64)
        scale = np.asarray(self.training_scale, dtype=np.float64)
        components = np.asarray(self.components, dtype=np.float64)
        variance = np.asarray(self.explained_variance, dtype=np.float64)
        ratio = np.asarray(self.explained_variance_ratio, dtype=np.float64)
        if mean.shape != (len(retained),) or scale.shape != mean.shape:
            raise ValueError("PCA mean/scale shape must match retained units.")
        if components.ndim != 2 or components.shape[1] != len(retained):
            raise ValueError("PCA components shape must be (component, retained_unit).")
        if variance.shape != (components.shape[0],) or ratio.shape != variance.shape:
            raise ValueError("Explained-variance shape must match the component axis.")
        transform_arrays = (mean, scale, components, variance, ratio)
        if not all(np.all(np.isfinite(value)) for value in transform_arrays):
            raise ValueError("PCA transform arrays must be finite.")
        if np.any(scale <= 0):
            raise ValueError("PCA training scales must be positive.")
        if (
            isinstance(self.n_training_observations, bool)
            or not isinstance(self.n_training_observations, Integral)
            or self.n_training_observations <= 0
        ):
            raise ValueError("n_training_observations must be a positive integer.")
        object.__setattr__(
            self, "n_training_observations", int(self.n_training_observations)
        )
        for name, value in (
            ("training_mean", mean),
            ("training_scale", scale),
            ("components", components),
            ("explained_variance", variance),
            ("explained_variance_ratio", ratio),
        ):
            copy = np.array(value, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        object.__setattr__(self, "retained_unit_ids", retained)
        object.__setattr__(self, "omitted_unit_ids", omitted)


def make_empty_result_tables() -> dict[str, pd.DataFrame]:
    """Construct all seven empty result tables with exact extension dtypes.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Tables in stable schema order. Every table has zero rows and its frozen
        columns/dtypes, including tables for analysis stages not yet run.
    """
    return {
        name: pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in dtypes.items()})
        for name, dtypes in RESULT_TABLE_DTYPES.items()
    }


def _validate_canonical_json(value: object, column: str) -> None:
    """Require compact sorted JSON text without Python object values."""
    if not isinstance(value, str):
        raise ValueError(f"{column} must contain canonical JSON strings.")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{column} must contain canonical JSON strings.") from error
    encoded = json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if value != encoded:
        raise ValueError(f"{column} must contain canonical JSON strings.")


def _validate_status_reason_columns(table: pd.DataFrame) -> None:
    """Validate generic and restricted/full status/reason pairs."""
    for prefix in ("", "restricted_", "full_"):
        status_column = f"{prefix}status" if prefix else "status"
        reason_column = f"{prefix}reason" if prefix else "reason"
        if status_column not in table or reason_column not in table:
            continue
        allowed_statuses = _FIT_ONLY_STATUSES if prefix else set(FIT_STATUS_VALUES)
        for status, reason in zip(table[status_column], table[reason_column], strict=True):
            if status not in allowed_statuses:
                raise ValueError(f"Unknown {status_column} value: {status!r}.")
            if reason not in ({""} | set(UNAVAILABILITY_REASONS)):
                raise ValueError(f"Unknown {reason_column} value: {reason!r}.")
            if status == "ok" and reason != "":
                raise ValueError(f"{reason_column} must be empty when {status_column} is ok.")
            if status != "ok" and reason == "":
                raise ValueError(f"{reason_column} must explain an unavailable status.")


def _validate_trial_membership(table: pd.DataFrame) -> None:
    """Require exclusion provenance and derived masks to agree row by row."""
    if table.empty:
        return
    source_by_reason = (
        ("invalid_reward_status", "reward_status_valid"),
        ("invalid_alignment", "alignment_valid"),
        ("choice_filter_mismatch", "choice_match"),
        ("context_filter_mismatch", "context_match"),
        ("user_excluded", "user_included"),
    )
    flag_columns = tuple(column for _, column in source_by_reason) + (
        "block_present",
        "condition_match",
        "scientific_eligible",
        "condition_included",
        "cv_included",
    )
    for row in table.itertuples(index=False):
        values = row._asdict()
        if any(pd.isna(values[column]) for column in flag_columns):
            raise ValueError("trial_membership Boolean provenance fields must not be null.")
        expected_reasons = [
            reason for reason, column in source_by_reason if not bool(values[column])
        ]
        actual_reasons = json.loads(values["scientific_exclusion_reasons_json"])
        if actual_reasons != expected_reasons:
            raise ValueError(
                "scientific_exclusion_reasons_json must match the fixed order and source flags."
            )
        scientific_eligible = not expected_reasons
        if bool(values["scientific_eligible"]) != scientific_eligible:
            raise ValueError(
                "scientific_eligible must equal the condition-independent source mask."
            )
        condition_included = scientific_eligible and bool(values["condition_match"])
        if bool(values["condition_included"]) != condition_included:
            raise ValueError(
                "condition_included must combine scientific eligibility and condition_match."
            )
        cv_included = condition_included and bool(values["block_present"])
        if bool(values["cv_included"]) != cv_included:
            raise ValueError("cv_included must combine condition inclusion and block_present.")


def _validate_value_vocabularies(name: str, table: pd.DataFrame) -> None:
    """Validate frozen categorical fields, statuses, hashes, and JSON columns."""
    _validate_status_reason_columns(table)
    allowed_by_column = {
        "direction": _DIRECTIONS,
        "representation": _REPRESENTATIONS,
        "model_family": _MODEL_FAMILIES,
        "window": _WINDOWS,
        "evaluation_scope": _EVALUATION_SCOPES,
        "region": {"PFC", "HPC"},
        "alignment": {"choice_time", "start_time"},
    }
    for column, allowed in allowed_by_column.items():
        if column in table and not table.empty:
            invalid = set(table[column].dropna().tolist()) - allowed
            if invalid:
                raise ValueError(f"{column} contains unsupported values: {sorted(invalid)}")
    if "diagnostic" in table:
        invalid = set(table["diagnostic"].dropna().tolist()) - set(GRANGER_DIAGNOSTICS)
        if invalid:
            raise ValueError(f"diagnostic contains unsupported values: {sorted(invalid)}")
    for column in _JSON_COLUMNS.get(name, ()):
        for value in table[column]:
            _validate_canonical_json(value, column)
    if name == "trial_membership":
        _validate_trial_membership(table)
    for column in ("train_row_set_sha256", "test_row_set_sha256"):
        if column in table:
            for value in table[column].dropna():
                if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                    raise ValueError(f"{column} must contain lowercase SHA-256 values or null.")
    if "fold_id" in table:
        fold_ids = table["fold_id"].dropna()
        if ((fold_ids < 0) | (fold_ids >= N_CV_FOLDS)).any():
            raise ValueError("fold_id must be in the zero-based range 0-4.")


def result_table_from_rows(
    name: str,
    rows: Sequence[Mapping[str, object]],
    *,
    sort: bool = True,
) -> pd.DataFrame:
    """Build one exactly typed result table from complete row mappings.

    Parameters
    ----------
    name : str
        One key from ``RESULT_TABLE_DTYPES``.
    rows : sequence of mappings
        Complete scalar row values in the table's documented units.
    sort : bool, default True
        Sort by the frozen primary key. ``False`` exists for validation tests;
        persisted tables must always be sorted.

    Returns
    -------
    pandas.DataFrame
        A new table with the exact column order and pandas dtypes.
    """
    if name not in RESULT_TABLE_DTYPES:
        raise ValueError(f"Unknown result table {name!r}.")
    expected_columns = tuple(RESULT_TABLE_DTYPES[name])
    for row in rows:
        if set(row) != set(expected_columns):
            raise ValueError(f"{name} row columns do not match the frozen schema.")
    table = pd.DataFrame(rows, columns=expected_columns)
    try:
        table = table.astype(RESULT_TABLE_DTYPES[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} row values do not match the frozen dtypes.") from error
    _validate_value_vocabularies(name, table)
    if sort and not table.empty:
        table = table.sort_values(
            list(RESULT_TABLE_PRIMARY_KEYS[name]), kind="stable", na_position="last"
        ).reset_index(drop=True)
    return table


def validate_result_table(name: str, table: pd.DataFrame) -> None:
    """Validate one result table's exact schema, values, key uniqueness, and order.

    Parameters
    ----------
    name : str
        One frozen result-table name.
    table : pandas.DataFrame
        Candidate table. Units and axes are column-specific as documented in
        the implementation plan.

    Returns
    -------
    None
        Returns after successful validation; the input is not modified.
    """
    if name not in RESULT_TABLE_DTYPES:
        raise ValueError(f"Unknown result table {name!r}.")
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    expected_dtypes = RESULT_TABLE_DTYPES[name]
    if tuple(table.columns) != tuple(expected_dtypes):
        raise ValueError(f"{name} columns do not match the frozen schema.")
    actual_dtypes = table.dtypes.astype(str).to_dict()
    if actual_dtypes != expected_dtypes:
        raise ValueError(f"{name} dtype mapping does not match the frozen schema.")
    _validate_value_vocabularies(name, table)
    primary_key = list(RESULT_TABLE_PRIMARY_KEYS[name])
    if table.duplicated(primary_key).any():
        raise ValueError(f"{name} contains a duplicate primary key.")
    if not table.empty:
        sorted_table = table.sort_values(
            primary_key, kind="stable", na_position="last"
        ).reset_index(drop=True)
        if not table.reset_index(drop=True).equals(sorted_table):
            raise ValueError(f"{name} must be sorted by its primary key.")


def expected_fold_score_keys(
    session_id: str,
    config: InterregionalAnalysisConfig,
    resolved_populations: Sequence[ResolvedRegionalPopulation],
) -> tuple[tuple[object, ...], ...]:
    """Return the complete applicable held-out score key grid.

    Parameters
    ----------
    session_id : str
        Stable session identifier.
    config : InterregionalAnalysisConfig
        Requested conditions, windows, representations, stages, and PCA ranks.
    resolved_populations : sequence of ResolvedRegionalPopulation
        Exactly one PFC and one HPC population with ordered target identities.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Sorted keys in ``FOLD_SCORE_KEY`` order. Fold IDs are zero-based.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a nonempty string.")
    populations = {population.role: population for population in resolved_populations}
    if set(populations) != {"PFC", "HPC"} or len(resolved_populations) != 2:
        raise ValueError(
            "resolved_populations must contain exactly one PFC and one HPC population."
        )
    validate_resolved_populations(populations["PFC"], populations["HPC"])
    model_families = []
    if "ols_cv" in config.analyses:
        model_families.append("ols")
    if "poisson_cv" in config.analyses:
        model_families.append("poisson")
    target_region_by_direction = {"HPC_to_PFC": "PFC", "PFC_to_HPC": "HPC"}
    requested_components = {
        "PFC": config.pca.pfc_components,
        "HPC": config.pca.hpc_components,
    }
    keys: list[tuple[object, ...]] = []
    for direction, target_region in target_region_by_direction.items():
        for representation in config.representations:
            if representation == "units":
                targets = populations[target_region].unit_ids
            else:
                targets = tuple(
                    f"{target_region}:PC{rank:02d}"
                    for rank in range(1, requested_components[target_region] + 1)
                )
            for model_family in model_families:
                if model_family == "poisson" and representation == "pcs":
                    continue
                for condition in config.filters.conditions:
                    for window in config.prediction_windows:
                        for target_id in targets:
                            for fold_id in range(N_CV_FOLDS):
                                keys.append(
                                    (
                                        session_id,
                                        direction,
                                        representation,
                                        model_family,
                                        condition,
                                        window,
                                        target_id,
                                        fold_id,
                                    )
                                )
    return tuple(sorted(keys))


def validate_fold_score_key_grid(
    table: pd.DataFrame,
    session_id: str,
    config: InterregionalAnalysisConfig,
    resolved_populations: Sequence[ResolvedRegionalPopulation],
) -> None:
    """Require exactly the complete applicable fold-score primary-key grid.

    ``table`` may be a full fold-score table or a key-only table. It must expose
    ``FOLD_SCORE_KEY`` columns; other columns are ignored by this grid check.
    """
    missing_columns = set(FOLD_SCORE_KEY) - set(table.columns)
    if missing_columns:
        raise ValueError(f"fold-score key grid is missing columns: {sorted(missing_columns)}")
    actual = [tuple(row) for row in table.loc[:, FOLD_SCORE_KEY].itertuples(index=False, name=None)]
    expected = expected_fold_score_keys(session_id, config, resolved_populations)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("fold-score key grid is incomplete, duplicated, or contains extra keys.")


def default_units_and_axes() -> dict[str, object]:
    """Return the exact JSON-compatible units and axis metadata mapping."""
    return {
        "count_tensor": {
            "axes": ["trial", "time_bin", "unit"],
            "value_unit": "spike_count_per_bin",
        },
        "whole_bin_edges_s": {
            "axis": ["time_bin_edge"],
            "unit": "seconds_relative_to_alignment",
        },
        "history_row_identity": {
            "axis": ["observation"],
            "fields": ["trial_row", "target_bin_position"],
        },
    }


@dataclass(frozen=True)
class InterregionalResults:
    """Pure computational result with seven typed tidy tables.

    ``whole_bin_edges_s`` has shape ``(n_whole_bins + 1,)`` in seconds
    relative to the selected alignment event. No fitted estimators or design
    matrices are stored.
    """

    schema_version: str
    analysis_version: str
    coverage_assumption_version: str
    configuration: Mapping[str, object]
    session_id: str
    resolved_populations: tuple[ResolvedRegionalPopulation, ResolvedRegionalPopulation]
    whole_bin_edges_s: np.ndarray
    units_and_axes: Mapping[str, object]
    randomness_used: bool
    random_seed: int | None
    fold_assignments: pd.DataFrame
    trial_membership: pd.DataFrame
    fold_scores: pd.DataFrame
    target_summaries: pd.DataFrame
    population_summaries: pd.DataFrame
    pca_fits: pd.DataFrame
    granger_scores: pd.DataFrame


def validate_interregional_results(
    result: InterregionalResults,
    config: InterregionalAnalysisConfig,
) -> None:
    """Validate top-level versions, metadata, axes, and all seven table schemas.

    Parameters
    ----------
    result : InterregionalResults
        Candidate pure in-memory result record.
    config : InterregionalAnalysisConfig
        Scientific configuration used to resolve the expected whole-window
        edge count in seconds.

    Returns
    -------
    None
        Returns after validation; neither argument is modified.
    """
    expected_versions = (
        ("schema_version", result.schema_version, RESULT_SCHEMA_VERSION),
        ("analysis_version", result.analysis_version, ANALYSIS_VERSION),
        (
            "coverage_assumption_version",
            result.coverage_assumption_version,
            COVERAGE_ASSUMPTION_VERSION,
        ),
    )
    for name, actual, expected in expected_versions:
        if actual != expected:
            raise ValueError(f"{name} must equal {expected!r}.")
    if not isinstance(result.configuration, Mapping):
        raise ValueError("configuration must be a canonical mapping.")
    if not isinstance(result.session_id, str) or not result.session_id:
        raise ValueError("session_id must be a nonempty string.")
    if len(result.resolved_populations) != 2:
        raise ValueError("resolved_populations must contain PFC and HPC records.")
    populations = {population.role: population for population in result.resolved_populations}
    if set(populations) != {"PFC", "HPC"}:
        raise ValueError("resolved_populations must contain PFC and HPC records.")
    validate_resolved_populations(populations["PFC"], populations["HPC"])
    edges = np.asarray(result.whole_bin_edges_s)
    if edges.dtype != np.dtype("float64"):
        raise ValueError("whole_bin_edges_s must have dtype float64.")
    expected_bin_count = round(
        (config.windows.whole_stop_s - config.windows.whole_start_s)
        / config.temporal.bin_size_s
    )
    if edges.shape != (expected_bin_count + 1,):
        raise ValueError("whole_bin_edges_s has the wrong time-bin-edge shape.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("whole_bin_edges_s must be finite and strictly increasing.")
    if edges[0] != config.windows.whole_start_s or edges[-1] != config.windows.whole_stop_s:
        raise ValueError("whole_bin_edges_s endpoints must match the configured window.")
    if result.units_and_axes != default_units_and_axes():
        raise ValueError("units_and_axes does not match the frozen mapping.")
    if result.randomness_used is not False or result.random_seed is not None:
        raise ValueError(
            "Deterministic results require randomness_used=false and random_seed=null."
        )
    for name in RESULT_TABLE_DTYPES:
        validate_result_table(name, getattr(result, name))
