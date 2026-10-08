"""Focused orchestration for inter-regional preparation and regression CV."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pynapple as nap

from .configuration import InterregionalAnalysisConfig, ResolvedRegionalPopulation
from .linear import (
    add_intercept,
    fit_ols_targets,
    predict_ols_targets,
    score_ols_predictions,
    summarize_complete_cv_targets,
)
from .pca import (
    RegionalPCATransforms,
    fit_fold_regional_pcas,
    transform_regional_activity,
)
from .poisson import (
    PoissonFit,
    PoissonFitUnavailable,
    fit_poisson_target,
    predict_poisson_mean,
    score_poisson_predictions,
)
from .preparation import (
    AnalysisTrialMasks,
    build_analysis_trial_masks,
    build_block_fold_assignment,
    build_history_matrices,
    build_regional_count_tensor,
    fingerprint_row_identities,
    select_window_bins,
    validate_regional_tensor_axes,
)
from .records import (
    FoldAssignment,
    RegionalCountTensor,
    result_table_from_rows,
)


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class PreparedInterregionalSession:
    """In-memory prepared counts, masks, folds, and population identities.

    Regional tensors use axes ``(trial, whole_window_bin, unit)`` and contain
    integer spike counts per bin. Mask arrays are indexed by zero-based rows in
    the complete source trial table.
    """

    session_id: str
    config: InterregionalAnalysisConfig
    pfc_population: ResolvedRegionalPopulation
    hpc_population: ResolvedRegionalPopulation
    pfc_counts: RegionalCountTensor
    hpc_counts: RegionalCountTensor
    trial_masks: AnalysisTrialMasks
    fold_assignment: FoldAssignment

    def __post_init__(self) -> None:
        """Validate roles and the shared prepared trial/time axes."""
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string.")
        if self.pfc_population.role != "PFC" or self.hpc_population.role != "HPC":
            raise ValueError("Prepared populations must have PFC and HPC roles.")
        validate_regional_tensor_axes(self.pfc_counts, self.hpc_counts)
        expected_rows = np.flatnonzero(self.trial_masks.scientific_eligible)
        if not np.array_equal(self.pfc_counts.trial_rows, expected_rows):
            raise ValueError(
                "Regional count trial rows must equal the scientific eligibility mask."
            )
        if len(self.fold_assignment.fold_ids) != len(
            self.trial_masks.scientific_eligible
        ):
            raise ValueError("Fold assignment must cover the complete trial table.")


@dataclass(frozen=True)
class _FoldModelData:
    """Shared train/test histories and designs for one CV fold cell.

    Response matrices have shape ``(observation, target)``. Design matrices
    have shape ``(observation, coefficient)`` and begin with one intercept.
    Unit responses are spike counts per bin; PC responses use PCA-score units.
    Trial counts, row counts, ranks, feature counts, and residual degrees of
    freedom are unitless.
    """

    train_responses: np.ndarray
    test_responses: np.ndarray
    restricted_train: np.ndarray
    full_train: np.ndarray
    restricted_test: np.ndarray
    full_test: np.ndarray
    restricted_info: tuple[int, int, int, str | None]
    full_info: tuple[int, int, int, str | None]
    n_train_trials: int
    n_test_trials: int
    n_train_rows: int
    n_test_rows: int
    train_hash: str | None
    test_hash: str | None


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **details: object,
) -> None:
    """Emit one execution-only progress dictionary when a callback is present."""
    if callback is not None:
        callback({"event": event, **details})


def _cell_progress_details(data: _FoldModelData) -> dict[str, object]:
    """Return exact shapes, byte sizes, and row counts for one model cell."""
    arrays = {
        "train_response": data.train_responses,
        "test_response": data.test_responses,
        "restricted_train": data.restricted_train,
        "full_train": data.full_train,
        "restricted_test": data.restricted_test,
        "full_test": data.full_test,
    }
    details: dict[str, object] = {
        "n_train_trials": data.n_train_trials,
        "n_test_trials": data.n_test_trials,
        "n_train_rows": data.n_train_rows,
        "n_test_rows": data.n_test_rows,
    }
    for name, array in arrays.items():
        details[f"{name}_shape"] = [int(value) for value in array.shape]
        details[f"{name}_bytes"] = int(array.nbytes)
    return details


def _target_progress_due(completed_targets: int, total_targets: int) -> bool:
    """Return whether first, every-25th, or final target progress is due."""
    return (
        completed_targets == 1
        or completed_targets % 25 == 0
        or completed_targets == total_targets
    )


def prepare_interregional_session(
    session_id: str,
    pfc_spikes: nap.TsGroup,
    hpc_spikes: nap.TsGroup,
    pfc_population: ResolvedRegionalPopulation,
    hpc_population: ResolvedRegionalPopulation,
    trial_df: pd.DataFrame,
    config: InterregionalAnalysisConfig,
) -> PreparedInterregionalSession:
    """Prepare one already-loaded session without reading or writing paths.

    Parameters
    ----------
    session_id : str
        Stable session identifier.
    pfc_spikes, hpc_spikes : pynapple.TsGroup
        Regional spike timestamps in seconds, keyed by integer cluster ID.
    pfc_population, hpc_population : ResolvedRegionalPopulation
        Explicit ordered unit selections.
    trial_df : pandas.DataFrame
        Complete trial table with one row per trial and alignment times in seconds.
    config : InterregionalAnalysisConfig
        Validated scientific settings.

    Returns
    -------
    PreparedInterregionalSession
        Integer count tensors on the scientific trial axis plus full-table masks
        and the deterministic five-fold mapping.
    """
    masks = build_analysis_trial_masks(trial_df, config.alignment, config.filters)
    scientific_rows = np.flatnonzero(masks.scientific_eligible).astype(np.int64)
    pfc_counts = build_regional_count_tensor(
        pfc_spikes,
        pfc_population,
        trial_df,
        scientific_rows,
        config.alignment,
        config.windows,
        config.temporal,
    )
    hpc_counts = build_regional_count_tensor(
        hpc_spikes,
        hpc_population,
        trial_df,
        scientific_rows,
        config.alignment,
        config.windows,
        config.temporal,
    )
    return PreparedInterregionalSession(
        session_id=session_id,
        config=config,
        pfc_population=pfc_population,
        hpc_population=hpc_population,
        pfc_counts=pfc_counts,
        hpc_counts=hpc_counts,
        trial_masks=masks,
        fold_assignment=build_block_fold_assignment(trial_df),
    )


def _design_diagnostics(design: np.ndarray) -> tuple[int, int, int, str | None]:
    """Return feature count, rank, residual df, and an availability reason."""
    feature_count = int(design.shape[1])
    rank = int(np.linalg.matrix_rank(design)) if design.shape[0] else 0
    df_resid = int(design.shape[0] - feature_count)
    if rank != feature_count:
        return feature_count, rank, df_resid, "rank_deficient"
    if df_resid <= 0:
        return feature_count, rank, df_resid, "nonpositive_df"
    return feature_count, rank, df_resid, None


def _build_fold_model_data(
    *,
    prepared: PreparedInterregionalSession,
    window: str,
    fold_id: int,
    target_activity: np.ndarray,
    source_activity: np.ndarray,
    condition_trials: np.ndarray,
    fold_ids: np.ndarray,
    tensor_rows: np.ndarray,
) -> _FoldModelData:
    """Build shared restricted/full train/test data for one fold cell.

    Activity arrays have shape ``(trial, whole_window_bin, feature)`` in spike
    counts per bin or PCA-score units. Trial selectors and ``fold_ids`` have
    shape ``(trial,)`` on the same tensor trial axis; ``tensor_rows`` contains
    the corresponding zero-based source-table rows. The returned response and
    design matrices use observation rows formed from trial/bin histories.
    """
    target_window, bin_positions = select_window_bins(
        target_activity,
        window,
        prepared.config.windows,
        prepared.config.temporal,
    )
    source_window, source_bin_positions = select_window_bins(
        source_activity,
        window,
        prepared.config.windows,
        prepared.config.temporal,
    )
    if not np.array_equal(bin_positions, source_bin_positions):
        raise ValueError("Target and source selected-window bin axes differ.")
    block_present = np.asarray([value is not None for value in fold_ids], dtype=bool)
    train_trials = condition_trials & block_present & (fold_ids != fold_id)
    test_trials = condition_trials & (fold_ids == fold_id)
    train_history = build_history_matrices(
        target_window[train_trials],
        source_window[train_trials],
        tensor_rows[train_trials],
        bin_positions,
        prepared.config.temporal.lag_bins,
        prepared.config.temporal.order_bins,
    )
    test_history = build_history_matrices(
        target_window[test_trials],
        source_window[test_trials],
        tensor_rows[test_trials],
        bin_positions,
        prepared.config.temporal.lag_bins,
        prepared.config.temporal.order_bins,
    )
    restricted_train = add_intercept(train_history.target_history)
    full_train = add_intercept(
        np.column_stack((train_history.target_history, train_history.source_history))
    )
    restricted_test = add_intercept(test_history.target_history)
    full_test = add_intercept(
        np.column_stack((test_history.target_history, test_history.source_history))
    )
    return _FoldModelData(
        train_responses=train_history.responses,
        test_responses=test_history.responses,
        restricted_train=restricted_train,
        full_train=full_train,
        restricted_test=restricted_test,
        full_test=full_test,
        restricted_info=_design_diagnostics(restricted_train),
        full_info=_design_diagnostics(full_train),
        n_train_trials=int(np.unique(train_history.row_trial).size),
        n_test_trials=int(np.unique(test_history.row_trial).size),
        n_train_rows=int(train_history.responses.shape[0]),
        n_test_rows=int(test_history.responses.shape[0]),
        train_hash=(
            fingerprint_row_identities(
                train_history.row_trial, train_history.row_target_bin
            )
            if train_history.responses.shape[0]
            else None
        ),
        test_hash=(
            fingerprint_row_identities(
                test_history.row_trial, test_history.row_target_bin
            )
            if test_history.responses.shape[0]
            else None
        ),
    )


def _empty_score_row(
    prepared: PreparedInterregionalSession,
    direction: str,
    condition: str,
    window: str,
    target_id: str,
    fold_id: int,
    restricted_reason: str | None,
    full_reason: str | None,
    n_train_trials: int,
    n_test_trials: int,
    n_train_rows: int,
    n_test_rows: int,
    train_hash: str | None,
    test_hash: str | None,
    restricted_diagnostics: tuple[int, int, int],
    full_diagnostics: tuple[int, int, int],
    representation: str = "units",
    target_rank: int | None = None,
    model_family: str = "ols",
) -> dict[str, object]:
    """Construct one unavailable fold row in the frozen result-table schema."""
    restricted_features, restricted_rank, restricted_df = restricted_diagnostics
    full_features, full_rank, full_df = full_diagnostics
    paired_reason = restricted_reason or full_reason
    if paired_reason is None:
        raise ValueError("An unavailable fold row requires at least one fit reason.")
    return {
        "session_id": prepared.session_id,
        "direction": direction,
        "representation": representation,
        "model_family": model_family,
        "condition": condition,
        "window": window,
        "target_id": target_id,
        "fold_id": fold_id,
        "evaluation_scope": "held_out_cv",
        "target_rank": target_rank,
        "restricted_status": "ok" if restricted_reason is None else "fit_unavailable",
        "restricted_reason": restricted_reason or "",
        "full_status": "ok" if full_reason is None else "fit_unavailable",
        "full_reason": full_reason or "",
        "status": "fit_unavailable",
        "reason": paired_reason,
        "n_train_trials": n_train_trials,
        "n_test_trials": n_test_trials,
        "n_train_rows": n_train_rows,
        "n_test_rows": n_test_rows,
        "train_row_set_sha256": train_hash,
        "test_row_set_sha256": test_hash,
        "restricted_feature_count": restricted_features,
        "full_feature_count": full_features,
        "restricted_rank": restricted_rank,
        "full_rank": full_rank,
        "restricted_df_resid": restricted_df,
        "full_df_resid": full_df,
        "restricted_converged": None,
        "full_converged": None,
        "restricted_iterations": None,
        "full_iterations": None,
        "r2_restricted": None,
        "r2_full": None,
        "delta_r2": None,
        "mse_restricted": None,
        "mse_full": None,
        "deviance_restricted": None,
        "deviance_full": None,
        "null_deviance": None,
        "deviance_explained_restricted": None,
        "deviance_explained_full": None,
        "delta_deviance_explained": None,
    }


def _fold_ids_on_tensor(prepared: PreparedInterregionalSession) -> np.ndarray:
    """Map nullable full-table fold IDs onto the scientific count-tensor axis."""
    full_fold_ids = np.asarray(prepared.fold_assignment.fold_ids, dtype=object)
    return full_fold_ids[prepared.pfc_counts.trial_rows]


def _score_one_ols_fold(
    *,
    prepared: PreparedInterregionalSession,
    direction: str,
    representation: str,
    condition: str,
    window: str,
    fold_id: int,
    target_activity: np.ndarray,
    source_activity: np.ndarray,
    target_ids: tuple[str, ...],
    target_ranks: tuple[int | None, ...],
    condition_trials: np.ndarray,
    fold_ids: np.ndarray,
    tensor_rows: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    cell_index: int = 1,
    total_cells: int = 1,
) -> list[dict[str, object]]:
    """Fit and score one representation/direction/condition/window/fold cell.

    Activity arrays have shape ``(trial, whole_window_bin, feature)``. Trial
    selectors index the first axis. Returned dictionaries follow the frozen
    ``fold_scores`` schema; linear PC scores have arbitrary squared PCA-score
    units while unit scores have squared spike-count-per-bin units.
    """
    cell_identity = {
        "model_family": "ols",
        "representation": representation,
        "direction": direction,
        "condition": condition,
        "window": window,
        "fold_id": fold_id,
        "cell_index": cell_index,
        "total_cells": total_cells,
        "total_targets": len(target_ids),
    }
    _emit_progress(progress_callback, "analysis_cell_prepare", **cell_identity)
    data = _build_fold_model_data(
        prepared=prepared,
        window=window,
        fold_id=fold_id,
        target_activity=target_activity,
        source_activity=source_activity,
        condition_trials=condition_trials,
        fold_ids=fold_ids,
        tensor_rows=tensor_rows,
    )
    _emit_progress(
        progress_callback,
        "analysis_cell_start",
        **cell_identity,
        **_cell_progress_details(data),
    )
    restricted_train = data.restricted_train
    full_train = data.full_train
    restricted_test = data.restricted_test
    full_test = data.full_test
    restricted_info = data.restricted_info
    full_info = data.full_info
    train_hash = data.train_hash
    test_hash = data.test_hash
    counts = (
        data.n_train_trials,
        data.n_test_trials,
        data.n_train_rows,
        data.n_test_rows,
    )
    rows: list[dict[str, object]] = []

    def finish_cell() -> list[dict[str, object]]:
        """Emit final unitless target counts and return accumulated score rows."""
        _emit_progress(
            progress_callback,
            "analysis_cell_end",
            **cell_identity,
            completed_targets=len(rows),
            unavailable_targets=sum(row["status"] != "ok" for row in rows),
        )
        return rows

    def append_unavailable(
        target_position: int,
        restricted_reason: str | None,
        full_reason: str | None,
    ) -> None:
        """Append one unavailable row while preserving shared diagnostics."""
        rows.append(
            _empty_score_row(
                prepared,
                direction,
                condition,
                window,
                target_ids[target_position],
                fold_id,
                restricted_reason,
                full_reason,
                *counts,
                train_hash,
                test_hash,
                restricted_info[:3],
                full_info[:3],
                representation=representation,
                target_rank=target_ranks[target_position],
            )
        )

    available_targets = data.train_responses.shape[1]
    if counts[2] == 0 or counts[3] == 0:
        row_reason = "no_train_rows" if counts[2] == 0 else "no_test_rows"
        for target_position in range(len(target_ids)):
            reason = (
                "pca_insufficient_components"
                if target_position >= available_targets
                else row_reason
            )
            append_unavailable(target_position, reason, reason)
        return finish_cell()

    restricted_reason = {
        "rank_deficient": "rank_deficient_restricted",
        "nonpositive_df": "nonpositive_df_restricted",
    }.get(restricted_info[3])
    full_reason = {
        "rank_deficient": "rank_deficient_full",
        "nonpositive_df": "nonpositive_df_full",
    }.get(full_info[3])
    restricted_fit = (
        fit_ols_targets(restricted_train, data.train_responses)
        if restricted_reason is None
        else None
    )
    full_fit = (
        fit_ols_targets(full_train, data.train_responses)
        if full_reason is None
        else None
    )
    if restricted_reason is not None or full_reason is not None:
        for target_position in range(len(target_ids)):
            if target_position >= available_targets:
                append_unavailable(
                    target_position,
                    "pca_insufficient_components",
                    "pca_insufficient_components",
                )
                continue
            target_restricted_reason = restricted_reason
            target_full_reason = full_reason
            if (
                restricted_fit is not None
                and restricted_fit.constant_targets[target_position]
            ):
                target_restricted_reason = "constant_training_target"
            if full_fit is not None and full_fit.constant_targets[target_position]:
                target_full_reason = "constant_training_target"
            append_unavailable(
                target_position, target_restricted_reason, target_full_reason
            )
        return finish_cell()

    assert restricted_fit is not None and full_fit is not None
    restricted_predictions = predict_ols_targets(
        restricted_test, restricted_fit.coefficients
    )
    full_predictions = predict_ols_targets(full_test, full_fit.coefficients)
    scores = score_ols_predictions(
        data.test_responses, restricted_predictions, full_predictions
    )
    for target_position, target_id in enumerate(target_ids):
        if target_position >= available_targets:
            append_unavailable(
                target_position,
                "pca_insufficient_components",
                "pca_insufficient_components",
            )
            continue
        if restricted_fit.constant_targets[target_position]:
            append_unavailable(
                target_position,
                "constant_training_target",
                "constant_training_target",
            )
            continue
        r2_available = bool(scores.r2_available[target_position])
        rows.append(
            {
                "session_id": prepared.session_id,
                "direction": direction,
                "representation": representation,
                "model_family": "ols",
                "condition": condition,
                "window": window,
                "target_id": target_id,
                "fold_id": fold_id,
                "evaluation_scope": "held_out_cv",
                "target_rank": target_ranks[target_position],
                "restricted_status": "ok",
                "restricted_reason": "",
                "full_status": "ok",
                "full_reason": "",
                "status": "ok" if r2_available else "metric_unavailable",
                "reason": "" if r2_available else "constant_test_target",
                "n_train_trials": counts[0],
                "n_test_trials": counts[1],
                "n_train_rows": counts[2],
                "n_test_rows": counts[3],
                "train_row_set_sha256": train_hash,
                "test_row_set_sha256": test_hash,
                "restricted_feature_count": restricted_fit.feature_count,
                "full_feature_count": full_fit.feature_count,
                "restricted_rank": restricted_fit.rank,
                "full_rank": full_fit.rank,
                "restricted_df_resid": restricted_fit.df_resid,
                "full_df_resid": full_fit.df_resid,
                "restricted_converged": None,
                "full_converged": None,
                "restricted_iterations": None,
                "full_iterations": None,
                "r2_restricted": (
                    float(scores.r2_restricted[target_position])
                    if r2_available
                    else None
                ),
                "r2_full": (
                    float(scores.r2_full[target_position]) if r2_available else None
                ),
                "delta_r2": (
                    float(scores.delta_r2[target_position]) if r2_available else None
                ),
                "mse_restricted": float(scores.mse_restricted[target_position]),
                "mse_full": float(scores.mse_full[target_position]),
                "deviance_restricted": None,
                "deviance_full": None,
                "null_deviance": None,
                "deviance_explained_restricted": None,
                "deviance_explained_full": None,
                "delta_deviance_explained": None,
            }
        )
    return finish_cell()


def _poisson_side_reason(reason: str, side: str) -> str:
    """Map one target-local fit reason to a frozen restricted/full reason."""
    if reason in {"poisson_nonconverged", "poisson_fit_error"}:
        return f"{reason}_{side}"
    return reason


def _score_one_poisson_fold(
    *,
    prepared: PreparedInterregionalSession,
    direction: str,
    condition: str,
    window: str,
    fold_id: int,
    target_activity: np.ndarray,
    source_activity: np.ndarray,
    target_ids: tuple[str, ...],
    condition_trials: np.ndarray,
    fold_ids: np.ndarray,
    tensor_rows: np.ndarray,
    progress_callback: ProgressCallback | None = None,
    cell_index: int = 1,
    total_cells: int = 1,
) -> list[dict[str, object]]:
    """Fit target-wise direct-unit Poisson models for one held-out fold cell.

    Activity arrays have shape ``(trial, whole_window_bin, unit)`` in spike
    counts per bin. Trial selectors index the first axis. Returned dictionaries
    follow the frozen ``fold_scores`` schema; deviance is dimensionless and MSE
    is in squared spike counts per bin.
    """
    cell_identity = {
        "model_family": "poisson",
        "representation": "units",
        "direction": direction,
        "condition": condition,
        "window": window,
        "fold_id": fold_id,
        "cell_index": cell_index,
        "total_cells": total_cells,
        "total_targets": len(target_ids),
    }
    _emit_progress(progress_callback, "analysis_cell_prepare", **cell_identity)
    data = _build_fold_model_data(
        prepared=prepared,
        window=window,
        fold_id=fold_id,
        target_activity=target_activity,
        source_activity=source_activity,
        condition_trials=condition_trials,
        fold_ids=fold_ids,
        tensor_rows=tensor_rows,
    )
    _emit_progress(
        progress_callback,
        "analysis_cell_start",
        **cell_identity,
        **_cell_progress_details(data),
    )
    counts = (
        data.n_train_trials,
        data.n_test_trials,
        data.n_train_rows,
        data.n_test_rows,
    )
    if data.n_train_rows == 0 or data.n_test_rows == 0:
        reason = "no_train_rows" if data.n_train_rows == 0 else "no_test_rows"
        unavailable_rows = [
            _empty_score_row(
                prepared,
                direction,
                condition,
                window,
                target_id,
                fold_id,
                reason,
                reason,
                *counts,
                data.train_hash,
                data.test_hash,
                data.restricted_info[:3],
                data.full_info[:3],
                model_family="poisson",
            )
            for target_id in target_ids
        ]
        _emit_progress(
            progress_callback,
            "analysis_cell_end",
            **cell_identity,
            completed_targets=len(unavailable_rows),
            unavailable_targets=len(unavailable_rows),
        )
        return unavailable_rows

    design_reasons = {
        "restricted": {
            "rank_deficient": "rank_deficient_restricted",
            "nonpositive_df": "nonpositive_df_restricted",
        }.get(data.restricted_info[3]),
        "full": {
            "rank_deficient": "rank_deficient_full",
            "nonpositive_df": "nonpositive_df_full",
        }.get(data.full_info[3]),
    }
    designs = {
        "restricted": (data.restricted_train, data.restricted_test),
        "full": (data.full_train, data.full_test),
    }
    rows: list[dict[str, object]] = []
    for target_position, target_id in enumerate(target_ids):
        training_counts = data.train_responses[:, target_position]
        test_counts = data.test_responses[:, target_position]
        fits: dict[str, PoissonFit | None] = {"restricted": None, "full": None}
        reasons = dict(design_reasons)
        side_scores: dict[str, object | None] = {"restricted": None, "full": None}
        for side in ("restricted", "full"):
            if reasons[side] is not None:
                continue
            train_design, test_design = designs[side]
            try:
                fitted = fit_poisson_target(train_design, training_counts)
                fits[side] = fitted
                expected_counts = predict_poisson_mean(test_design, fitted.parameters)
                side_scores[side] = score_poisson_predictions(
                    test_counts, expected_counts, expected_counts
                )
            except PoissonFitUnavailable as error:
                reasons[side] = _poisson_side_reason(error.reason, side)

        restricted_fit = fits["restricted"]
        full_fit = fits["full"]
        restricted_scores = side_scores["restricted"]
        full_scores = side_scores["full"]
        paired_reason = reasons["restricted"] or reasons["full"]
        normalized_available = (
            paired_reason is None
            and restricted_scores is not None
            and full_scores is not None
            and restricted_scores.normalized_available
            and full_scores.normalized_available
        )
        row_status = (
            "fit_unavailable"
            if paired_reason is not None
            else "ok" if normalized_available else "metric_unavailable"
        )
        row_reason = paired_reason or ("" if normalized_available else "zero_null_deviance")
        rows.append(
            {
                "session_id": prepared.session_id,
                "direction": direction,
                "representation": "units",
                "model_family": "poisson",
                "condition": condition,
                "window": window,
                "target_id": target_id,
                "fold_id": fold_id,
                "evaluation_scope": "held_out_cv",
                "target_rank": None,
                "restricted_status": (
                    "ok" if reasons["restricted"] is None else "fit_unavailable"
                ),
                "restricted_reason": reasons["restricted"] or "",
                "full_status": "ok" if reasons["full"] is None else "fit_unavailable",
                "full_reason": reasons["full"] or "",
                "status": row_status,
                "reason": row_reason,
                "n_train_trials": counts[0],
                "n_test_trials": counts[1],
                "n_train_rows": counts[2],
                "n_test_rows": counts[3],
                "train_row_set_sha256": data.train_hash,
                "test_row_set_sha256": data.test_hash,
                "restricted_feature_count": data.restricted_info[0],
                "full_feature_count": data.full_info[0],
                "restricted_rank": data.restricted_info[1],
                "full_rank": data.full_info[1],
                "restricted_df_resid": data.restricted_info[2],
                "full_df_resid": data.full_info[2],
                "restricted_converged": (
                    restricted_fit.converged if restricted_fit is not None else None
                ),
                "full_converged": full_fit.converged if full_fit is not None else None,
                "restricted_iterations": (
                    restricted_fit.iterations if restricted_fit is not None else None
                ),
                "full_iterations": (
                    full_fit.iterations if full_fit is not None else None
                ),
                "r2_restricted": None,
                "r2_full": None,
                "delta_r2": None,
                "mse_restricted": (
                    restricted_scores.mse_restricted
                    if restricted_scores is not None
                    else None
                ),
                "mse_full": (
                    full_scores.mse_restricted if full_scores is not None else None
                ),
                "deviance_restricted": (
                    restricted_scores.deviance_restricted
                    if restricted_scores is not None
                    else None
                ),
                "deviance_full": (
                    full_scores.deviance_restricted
                    if full_scores is not None
                    else None
                ),
                "null_deviance": (
                    restricted_scores.null_deviance
                    if restricted_scores is not None
                    else full_scores.null_deviance
                    if full_scores is not None
                    else None
                ),
                "deviance_explained_restricted": (
                    restricted_scores.deviance_explained_restricted
                    if restricted_scores is not None
                    and restricted_scores.normalized_available
                    else None
                ),
                "deviance_explained_full": (
                    full_scores.deviance_explained_restricted
                    if full_scores is not None and full_scores.normalized_available
                    else None
                ),
                "delta_deviance_explained": (
                    full_scores.deviance_explained_restricted
                    - restricted_scores.deviance_explained_restricted
                    if normalized_available
                    else None
                ),
            }
        )
        completed_targets = target_position + 1
        if _target_progress_due(completed_targets, len(target_ids)):
            _emit_progress(
                progress_callback,
                "poisson_target_progress",
                **cell_identity,
                completed_targets=completed_targets,
                unavailable_targets=sum(row["status"] != "ok" for row in rows),
                last_target_id=target_id,
            )
    _emit_progress(
        progress_callback,
        "analysis_cell_end",
        **cell_identity,
        completed_targets=len(rows),
        unavailable_targets=sum(row["status"] != "ok" for row in rows),
    )
    return rows


def _resolve_fold_pcas(
    prepared: PreparedInterregionalSession,
    fold_ids: np.ndarray,
    supplied: Mapping[int, RegionalPCATransforms] | None,
) -> dict[int, RegionalPCATransforms]:
    """Return exactly five validated training-only regional PCA pairs."""
    expected_folds = set(range(5))
    if supplied is not None:
        resolved = dict(supplied)
        if set(resolved) != expected_folds:
            raise ValueError("fold_pcas must contain exactly folds 0 through 4.")
        for fold_id, fitted in resolved.items():
            if fitted.scope != "fold" or fitted.fold_id != fold_id:
                raise ValueError("Cross-validation requires matching fold-scoped PCA fits.")
        return resolved
    tensor_rows = prepared.pfc_counts.trial_rows
    condition_masks = {
        condition: prepared.trial_masks.condition_masks[condition][tensor_rows]
        for condition in prepared.config.filters.conditions
    }
    block_present = np.asarray([value is not None for value in fold_ids], dtype=bool)
    requests = {
        "PFC": prepared.config.pca.pfc_components,
        "HPC": prepared.config.pca.hpc_components,
    }
    return {
        fold_id: fit_fold_regional_pcas(
            pfc_activity=prepared.pfc_counts.counts,
            hpc_activity=prepared.hpc_counts.counts,
            pfc_unit_ids=prepared.pfc_counts.unit_ids,
            hpc_unit_ids=prepared.hpc_counts.unit_ids,
            training_trials=block_present & (fold_ids != fold_id),
            condition_masks=condition_masks,
            requested_conditions=prepared.config.filters.conditions,
            requested_components=requests,
            fold_id=fold_id,
        )
        for fold_id in range(5)
    }


def fit_cross_validation_pcas(
    prepared: PreparedInterregionalSession,
) -> tuple[dict[int, RegionalPCATransforms], pd.DataFrame]:
    """Fit five shared regional PCA pairs and construct provenance rows.

    Parameters
    ----------
    prepared : PreparedInterregionalSession
        Regional count tensors with axes ``(trial, whole_window_bin, unit)`` in
        spike counts per bin, plus scientific condition and fold masks.

    Returns
    -------
    tuple[dict[int, RegionalPCATransforms], pandas.DataFrame]
        Fold-indexed in-memory transforms and the exact typed ``pca_fits``
        metadata table. Component counts and observation counts are unitless.
    """
    if "pcs" not in prepared.config.representations:
        raise ValueError("fit_cross_validation_pcas requires the pcs representation.")
    fold_ids = _fold_ids_on_tensor(prepared)
    fitted_by_fold = _resolve_fold_pcas(prepared, fold_ids, supplied=None)
    requested_by_region = {
        "PFC": prepared.config.pca.pfc_components,
        "HPC": prepared.config.pca.hpc_components,
    }
    rows: list[dict[str, object]] = []
    for fold_id, fitted_pair in fitted_by_fold.items():
        for region, fitted in (("PFC", fitted_pair.pfc), ("HPC", fitted_pair.hpc)):
            rows.append(
                {
                    "session_id": prepared.session_id,
                    "scope": "fold",
                    "fold_id": fold_id,
                    "region": region,
                    "status": "ok",
                    "reason": "",
                    "requested_components": requested_by_region[region],
                    "actual_components": fitted.actual_components,
                    "n_training_trials": fitted_pair.n_training_trials,
                    "n_training_observations": fitted.n_training_observations,
                    "retained_unit_ids_json": json.dumps(
                        list(fitted.retained_unit_ids),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                    "omitted_unit_ids_json": json.dumps(
                        list(fitted.omitted_unit_ids),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                }
            )
    return fitted_by_fold, result_table_from_rows("pca_fits", rows)


def run_linear_cross_validation(
    prepared: PreparedInterregionalSession,
    *,
    fold_pcas: Mapping[int, RegionalPCATransforms] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run requested bidirectional unit and/or fold-local-PC five-fold OLS.

    Parameters
    ----------
    prepared : PreparedInterregionalSession
        Shared activity with axes ``(trial, whole_window_bin, unit)``, masks,
        and grouped fold assignments. Counts are spikes per configured bin.
    fold_pcas : mapping[int, RegionalPCATransforms] or None, optional
        Optional reusable fold-local transforms keyed by zero-based fold. A
        descriptive all-data transform is rejected. When omitted, transforms
        are fit once per fold if the PC representation is requested.
    progress_callback : callable or None, optional
        Execution-only receiver for dictionaries containing unitless cell
        identities plus array shapes and byte sizes. It does not alter fits.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Exact typed fold scores, complete-five-fold target summaries, and
        population quartiles. R-squared is dimensionless. Unit MSE is squared
        spike-count-per-bin; PC MSE is squared PCA-score units.
    """
    if "ols_cv" not in prepared.config.analyses:
        raise ValueError("run_linear_cross_validation requires ols_cv in analyses.")
    fold_ids = _fold_ids_on_tensor(prepared)
    tensor_rows = prepared.pfc_counts.trial_rows
    if "pcs" in prepared.config.representations:
        resolved_pcas = (
            fit_cross_validation_pcas(prepared)[0]
            if fold_pcas is None
            else _resolve_fold_pcas(prepared, fold_ids, fold_pcas)
        )
    else:
        resolved_pcas = {}
    if fold_pcas is not None and "pcs" not in prepared.config.representations:
        raise ValueError("fold_pcas were supplied but PCs are not requested.")
    pc_activity: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_id, fitted in resolved_pcas.items():
        pc_activity[fold_id] = (
            transform_regional_activity(
                prepared.pfc_counts.counts,
                prepared.pfc_counts.unit_ids,
                fitted.pfc,
            ),
            transform_regional_activity(
                prepared.hpc_counts.counts,
                prepared.hpc_counts.unit_ids,
                fitted.hpc,
            ),
        )

    score_rows: list[dict[str, object]] = []
    direction_specs = (
        ("HPC_to_PFC", "PFC"),
        ("PFC_to_HPC", "HPC"),
    )
    unit_activity = {
        "PFC": prepared.pfc_counts.counts,
        "HPC": prepared.hpc_counts.counts,
    }
    unit_ids = {
        "PFC": prepared.pfc_counts.unit_ids,
        "HPC": prepared.hpc_counts.unit_ids,
    }
    requested_pc_counts = {
        "PFC": prepared.config.pca.pfc_components,
        "HPC": prepared.config.pca.hpc_components,
    }
    total_cells = (
        len(direction_specs)
        * len(prepared.config.representations)
        * len(prepared.config.filters.conditions)
        * len(prepared.config.prediction_windows)
        * 5
    )
    cell_index = 0
    for direction, target_region in direction_specs:
        source_region = "HPC" if target_region == "PFC" else "PFC"
        for representation in prepared.config.representations:
            for condition in prepared.config.filters.conditions:
                condition_trials = prepared.trial_masks.condition_masks[condition][
                    tensor_rows
                ]
                for window in prepared.config.prediction_windows:
                    for fold_id in range(5):
                        cell_index += 1
                        if representation == "units":
                            target_activity = unit_activity[target_region]
                            source_activity = unit_activity[source_region]
                            target_ids = unit_ids[target_region]
                            target_ranks = (None,) * len(target_ids)
                        else:
                            pfc_scores, hpc_scores = pc_activity[fold_id]
                            regional_scores = {"PFC": pfc_scores, "HPC": hpc_scores}
                            target_activity = regional_scores[target_region]
                            source_activity = regional_scores[source_region]
                            requested = requested_pc_counts[target_region]
                            target_ids = tuple(
                                f"{target_region}:PC{rank:02d}"
                                for rank in range(1, requested + 1)
                            )
                            target_ranks = tuple(range(1, requested + 1))
                        score_rows.extend(
                            _score_one_ols_fold(
                                prepared=prepared,
                                direction=direction,
                                representation=representation,
                                condition=condition,
                                window=window,
                                fold_id=fold_id,
                                target_activity=target_activity,
                                source_activity=source_activity,
                                target_ids=target_ids,
                                target_ranks=target_ranks,
                                condition_trials=condition_trials,
                                fold_ids=fold_ids,
                                tensor_rows=tensor_rows,
                                progress_callback=progress_callback,
                                cell_index=cell_index,
                                total_cells=total_cells,
                            )
                        )
    fold_scores = result_table_from_rows("fold_scores", score_rows)
    target_summaries, population_summaries = summarize_complete_cv_targets(fold_scores)
    return fold_scores, target_summaries, population_summaries


def run_poisson_cross_validation(
    prepared: PreparedInterregionalSession,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run bidirectional target-wise five-fold Poisson CV for direct units.

    Parameters
    ----------
    prepared : PreparedInterregionalSession
        Shared activity with axes ``(trial, whole_window_bin, unit)`` in spike
        counts per bin, scientific condition masks, and grouped fold IDs.
    progress_callback : callable or None, optional
        Execution-only receiver for cell boundaries and coarse target progress.
        Shape axes and byte fields describe the exact arrays used by each cell.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Exact typed Poisson fold scores, complete-five-fold target summaries,
        and population quartiles. Deviance metrics are dimensionless and MSE
        is in squared spike counts per bin. Raw deviance remains fold-only.
    """
    if "poisson_cv" not in prepared.config.analyses:
        raise ValueError("run_poisson_cross_validation requires poisson_cv in analyses.")
    if "units" not in prepared.config.representations:
        raise ValueError("Poisson cross-validation requires direct units.")

    fold_ids = _fold_ids_on_tensor(prepared)
    tensor_rows = prepared.pfc_counts.trial_rows
    unit_activity = {
        "PFC": prepared.pfc_counts.counts,
        "HPC": prepared.hpc_counts.counts,
    }
    unit_ids = {
        "PFC": prepared.pfc_counts.unit_ids,
        "HPC": prepared.hpc_counts.unit_ids,
    }
    score_rows: list[dict[str, object]] = []
    total_cells = (
        2
        * len(prepared.config.filters.conditions)
        * len(prepared.config.prediction_windows)
        * 5
    )
    cell_index = 0
    for direction, target_region in (
        ("HPC_to_PFC", "PFC"),
        ("PFC_to_HPC", "HPC"),
    ):
        source_region = "HPC" if target_region == "PFC" else "PFC"
        for condition in prepared.config.filters.conditions:
            condition_trials = prepared.trial_masks.condition_masks[condition][
                tensor_rows
            ]
            for window in prepared.config.prediction_windows:
                for fold_id in range(5):
                    cell_index += 1
                    score_rows.extend(
                        _score_one_poisson_fold(
                            prepared=prepared,
                            direction=direction,
                            condition=condition,
                            window=window,
                            fold_id=fold_id,
                            target_activity=unit_activity[target_region],
                            source_activity=unit_activity[source_region],
                            target_ids=unit_ids[target_region],
                            condition_trials=condition_trials,
                            fold_ids=fold_ids,
                            tensor_rows=tensor_rows,
                            progress_callback=progress_callback,
                            cell_index=cell_index,
                            total_cells=total_cells,
                        )
                    )
    fold_scores = result_table_from_rows("fold_scores", score_rows)
    target_summaries, population_summaries = summarize_complete_cv_targets(
        fold_scores,
        metric_names=(
            "deviance_explained_restricted",
            "deviance_explained_full",
            "delta_deviance_explained",
            "mse_restricted",
            "mse_full",
        ),
    )
    return fold_scores, target_summaries, population_summaries
