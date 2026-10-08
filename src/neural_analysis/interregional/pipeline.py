"""Focused orchestration for inter-regional preparation and direct-unit OLS."""

from __future__ import annotations

from dataclasses import dataclass

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
) -> dict[str, object]:
    """Construct one explicit unavailable direct-unit OLS fold row."""
    restricted_features, restricted_rank, restricted_df = restricted_diagnostics
    full_features, full_rank, full_df = full_diagnostics
    paired_reason = restricted_reason or full_reason
    if paired_reason is None:
        raise ValueError("An unavailable fold row requires at least one fit reason.")
    return {
        "session_id": prepared.session_id,
        "direction": direction,
        "representation": "units",
        "model_family": "ols",
        "condition": condition,
        "window": window,
        "target_id": target_id,
        "fold_id": fold_id,
        "evaluation_scope": "held_out_cv",
        "target_rank": None,
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


def run_linear_cross_validation(
    prepared: PreparedInterregionalSession,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run bidirectional direct-unit five-fold OLS on one prepared session.

    Parameters
    ----------
    prepared : PreparedInterregionalSession
        Shared integer count tensors, full-table eligibility masks, and grouped
        fold assignments. Count axes are ``(trial, whole_window_bin, unit)``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Exact typed fold scores, complete-five-fold target summaries, and
        population quartiles. R-squared is dimensionless and MSE is in squared
        spike-count-per-bin units.
    """
    if "ols_cv" not in prepared.config.analyses:
        raise ValueError("run_linear_cross_validation requires ols_cv in analyses.")
    fold_ids = _fold_ids_on_tensor(prepared)
    tensor_rows = prepared.pfc_counts.trial_rows
    directions = (
        (
            "HPC_to_PFC",
            prepared.pfc_counts,
            prepared.hpc_counts,
        ),
        (
            "PFC_to_HPC",
            prepared.hpc_counts,
            prepared.pfc_counts,
        ),
    )
    score_rows: list[dict[str, object]] = []
    for direction, target_tensor, source_tensor in directions:
        for condition in prepared.config.filters.conditions:
            condition_trials = prepared.trial_masks.condition_masks[condition][
                tensor_rows
            ]
            for window in prepared.config.prediction_windows:
                target_window, bin_positions = select_window_bins(
                    target_tensor.counts,
                    window,
                    prepared.config.windows,
                    prepared.config.temporal,
                )
                source_window, source_bin_positions = select_window_bins(
                    source_tensor.counts,
                    window,
                    prepared.config.windows,
                    prepared.config.temporal,
                )
                if not np.array_equal(bin_positions, source_bin_positions):
                    raise ValueError("Target and source selected-window bin axes differ.")
                for fold_id in range(5):
                    block_present = np.asarray(
                        [value is not None for value in fold_ids], dtype=bool
                    )
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
                        np.column_stack(
                            (train_history.target_history, train_history.source_history)
                        )
                    )
                    restricted_test = add_intercept(test_history.target_history)
                    full_test = add_intercept(
                        np.column_stack(
                            (test_history.target_history, test_history.source_history)
                        )
                    )
                    restricted_info = _design_diagnostics(restricted_train)
                    full_info = _design_diagnostics(full_train)
                    train_hash = (
                        fingerprint_row_identities(
                            train_history.row_trial, train_history.row_target_bin
                        )
                        if train_history.responses.shape[0]
                        else None
                    )
                    test_hash = (
                        fingerprint_row_identities(
                            test_history.row_trial, test_history.row_target_bin
                        )
                        if test_history.responses.shape[0]
                        else None
                    )
                    counts = (
                        int(np.unique(train_history.row_trial).size),
                        int(np.unique(test_history.row_trial).size),
                        int(train_history.responses.shape[0]),
                        int(test_history.responses.shape[0]),
                    )
                    if counts[2] == 0 or counts[3] == 0:
                        reason = "no_train_rows" if counts[2] == 0 else "no_test_rows"
                        for target_id in target_tensor.unit_ids:
                            score_rows.append(
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
                                    train_hash,
                                    test_hash,
                                    restricted_info[:3],
                                    full_info[:3],
                                )
                            )
                        continue
                    if restricted_info[3] is not None or full_info[3] is not None:
                        restricted_reason = {
                            "rank_deficient": "rank_deficient_restricted",
                            "nonpositive_df": "nonpositive_df_restricted",
                        }.get(restricted_info[3])
                        full_reason = {
                            "rank_deficient": "rank_deficient_full",
                            "nonpositive_df": "nonpositive_df_full",
                        }.get(full_info[3])
                        restricted_fit = (
                            fit_ols_targets(restricted_train, train_history.responses)
                            if restricted_reason is None
                            else None
                        )
                        full_fit = (
                            fit_ols_targets(full_train, train_history.responses)
                            if full_reason is None
                            else None
                        )
                        for target_position, target_id in enumerate(target_tensor.unit_ids):
                            target_restricted_reason = restricted_reason
                            target_full_reason = full_reason
                            if (
                                restricted_fit is not None
                                and restricted_fit.constant_targets[target_position]
                            ):
                                target_restricted_reason = "constant_training_target"
                            if (
                                full_fit is not None
                                and full_fit.constant_targets[target_position]
                            ):
                                target_full_reason = "constant_training_target"
                            score_rows.append(
                                _empty_score_row(
                                    prepared,
                                    direction,
                                    condition,
                                    window,
                                    target_id,
                                    fold_id,
                                    target_restricted_reason,
                                    target_full_reason,
                                    *counts,
                                    train_hash,
                                    test_hash,
                                    restricted_info[:3],
                                    full_info[:3],
                                )
                            )
                        continue

                    restricted_fit = fit_ols_targets(
                        restricted_train, train_history.responses
                    )
                    full_fit = fit_ols_targets(full_train, train_history.responses)
                    restricted_predictions = predict_ols_targets(
                        restricted_test, restricted_fit.coefficients
                    )
                    full_predictions = predict_ols_targets(
                        full_test, full_fit.coefficients
                    )
                    scores = score_ols_predictions(
                        test_history.responses,
                        restricted_predictions,
                        full_predictions,
                    )
                    for target_position, target_id in enumerate(target_tensor.unit_ids):
                        if restricted_fit.constant_targets[target_position]:
                            score_rows.append(
                                _empty_score_row(
                                    prepared,
                                    direction,
                                    condition,
                                    window,
                                    target_id,
                                    fold_id,
                                    "constant_training_target",
                                    "constant_training_target",
                                    *counts,
                                    train_hash,
                                    test_hash,
                                    restricted_info[:3],
                                    full_info[:3],
                                )
                            )
                            continue
                        r2_available = bool(scores.r2_available[target_position])
                        score_rows.append(
                            {
                                "session_id": prepared.session_id,
                                "direction": direction,
                                "representation": "units",
                                "model_family": "ols",
                                "condition": condition,
                                "window": window,
                                "target_id": target_id,
                                "fold_id": fold_id,
                                "evaluation_scope": "held_out_cv",
                                "target_rank": None,
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
                                    float(scores.r2_full[target_position])
                                    if r2_available
                                    else None
                                ),
                                "delta_r2": (
                                    float(scores.delta_r2[target_position])
                                    if r2_available
                                    else None
                                ),
                                "mse_restricted": float(
                                    scores.mse_restricted[target_position]
                                ),
                                "mse_full": float(scores.mse_full[target_position]),
                                "deviance_restricted": None,
                                "deviance_full": None,
                                "null_deviance": None,
                                "deviance_explained_restricted": None,
                                "deviance_explained_full": None,
                                "delta_deviance_explained": None,
                            }
                        )
    fold_scores = result_table_from_rows("fold_scores", score_rows)
    target_summaries, population_summaries = summarize_complete_cv_targets(
        fold_scores
    )
    return fold_scores, target_summaries, population_summaries
