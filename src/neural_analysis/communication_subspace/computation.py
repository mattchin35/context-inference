"""Pure nested validation across matched neural populations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .modeling import (
    RidgeDecomposition,
    TrainingTransform,
    fit_ridge_decomposition,
    fit_training_transform,
    inverse_transform_predictions,
    predict_rank,
    select_alpha,
    transform_counts,
)
from .preparation import PreparedInputs, PopulationDraw, ValidationSplit, build_validation_splits, draw_populations
from .scoring import score_target_predictions, summarize_draw_scores, summarize_fold_scores


ALPHA_GRID = 10.0 ** np.arange(-6, 7, dtype=np.float64)
DIRECTIONS = (("PFC->PFC", "PFC", "PFC"), ("PFC->HPC", "PFC", "HPC"),
              ("HPC->HPC", "HPC", "HPC"), ("HPC->PFC", "HPC", "PFC"))


@dataclass(frozen=True)
class FitRecord:
    """Selected outer fit factors for one draw/direction/fold/alpha.

    Indices and IDs address n selected source and target units in the original
    regional unit axes. Transforms have n entries in counts/bin; decomposition
    coefficients/basis use only variable units and dimensionless z-scores.
    Split indices are held separately in ComputationResult.splits.
    """

    draw: int
    direction: str
    fold: int
    alpha: float
    source_indices: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    source_unit_ids: tuple[str, ...]
    target_unit_ids: tuple[str, ...]
    source_transform: TrainingTransform
    target_transform: TrainingTransform
    decomposition: RidgeDecomposition


@dataclass(frozen=True)
class ComputationResult:
    """Per-target/fold/draw/curve scores and compact selected model factors.

    DataFrames store dimensionless R2 and alpha, counts/bin squared SSE/SST,
    nullable requested RRR rank, identities, validity, status and reasons.
    draws contain selected unit indices; splits contain common-choice indices.
    status is success, partial, or unavailable, based on complete rank-1 and
    unrestricted-ridge support for all directions/draws.
    """

    fold_scores: pd.DataFrame
    inner_scores: pd.DataFrame
    fit_records: tuple[FitRecord, ...]
    draw_scores: pd.DataFrame
    curve_scores: pd.DataFrame
    draws: tuple[PopulationDraw, ...]
    splits: tuple[ValidationSplit, ...]
    status: str


def compute_communication_subspace(
    prepared: PreparedInputs,
    *,
    seed: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> ComputationResult:
    """Fit nested-CV ridge/RRR to paired (choice, bin, unit) counts/bin.

    prepared has chronological common choices, two finite float64 regional
    count tensors (W,B,U), unit IDs, and selected trial positions. seed drives
    five unit draws; progress receives (completed, total, message) after each
    draw/fold combination.
    Return target-level held-out scores and inner candidate scores as DataFrames,
    compact selected outer fits, complete-support summaries, draws, splits, and
    a scientific completion status. Inputs are never modified; no I/O occurs.
    """
    _validate_prepared(prepared)
    choice_times = prepared.choices.iloc[prepared.choice_positions].choice_time.to_numpy(dtype=float)
    splits = build_validation_splits(choice_times)
    draws = draw_populations(prepared.unit_ids, seed=seed)
    requested_ranks = range(1, len(draws[0].source_indices["PFC"]) + 1)
    fold_rows: list[dict] = []
    inner_rows: list[dict] = []
    fit_records: list[FitRecord] = []
    total_tasks = len(draws) * len(splits)

    for draw_number, draw in enumerate(draws):
        for fold_number, split in enumerate(splits):
            for direction, source_region, target_region in DIRECTIONS:
                source_indices = draw.source_indices[source_region]
                target_indices = draw.target_indices[target_region]
                source_ids = tuple(prepared.unit_ids[source_region][index] for index in source_indices)
                target_ids = tuple(prepared.unit_ids[target_region][index] for index in target_indices)
                source_counts = prepared.counts[source_region][:, :, source_indices]
                target_counts = prepared.counts[target_region][:, :, target_indices]
                candidate_rows = _evaluate_inner_grid(
                    source_counts, target_counts, split, draw_number, direction, fold_number,
                    requested_ranks,
                )
                inner_rows.extend(candidate_rows)
                outer_rows, outer_fits = _evaluate_outer_fold(
                    source_counts, target_counts, source_indices, target_indices,
                    source_ids, target_ids, split, draw_number, direction, fold_number,
                    requested_ranks, candidate_rows,
                )
                fold_rows.extend(outer_rows)
                fit_records.extend(outer_fits)
            if progress is not None:
                completed = draw_number * len(splits) + fold_number + 1
                progress(completed, total_tasks,
                         f"Completed draw {draw_number + 1}/{len(draws)}, "
                         f"fold {fold_number + 1}/{len(splits)}")

    fold_scores = pd.DataFrame(fold_rows)
    inner_scores = pd.DataFrame(inner_rows)
    for table in (fold_scores, inner_scores):
        table["rank"] = table["rank"].astype("Int64")
    draw_scores = summarize_fold_scores(fold_scores, expected_folds=len(splits))
    curve_scores = summarize_draw_scores(draw_scores, expected_draws=len(draws))
    essential = curve_scores.loc[
        ((curve_scores["model"] == "rrr") & (curve_scores["rank"] == 1).fillna(False)) |
        (curve_scores["model"] == "ridge")
    ]
    success = len(essential) == len(DIRECTIONS) * 2 and essential.status.eq("ok").all()
    status = "success" if success else ("partial" if fold_scores.valid.any() else "unavailable")
    return ComputationResult(fold_scores, inner_scores, tuple(fit_records), draw_scores,
                             curve_scores, tuple(draws), tuple(splits), status)


# Helper functions
def _validate_prepared(prepared: PreparedInputs) -> None:
    """Require paired finite count tensors (choice, bin, unit) in spikes/bin."""
    positions = np.asarray(prepared.choice_positions)
    if (positions.ndim != 1 or positions.dtype.kind not in "iu" or
            np.any(positions < 0) or np.any(positions >= len(prepared.choices)) or
            np.any(np.diff(positions) <= 0)):
        raise ValueError("choice positions must be increasing valid integer indices")
    if len(prepared.selected_trials) != len(positions):
        raise ValueError("selected trial axis does not match choice positions")
    for region in ("PFC", "HPC"):
        if region not in prepared.counts or region not in prepared.unit_ids:
            raise ValueError(f"missing {region} counts or unit identities")
        counts = np.asarray(prepared.counts[region])
        if counts.ndim != 3 or not np.isfinite(counts).all():
            raise ValueError(f"{region} counts must be finite (choice, bin, unit) values")
        if np.any(counts < 0):
            raise ValueError(f"{region} spike counts must be nonnegative")
        if counts.shape[2] != len(prepared.unit_ids[region]):
            raise ValueError(f"{region} unit axis does not match unit IDs")
        if any(not isinstance(unit_id, str) or not unit_id.partition(":")[0] or
               not unit_id.partition(":")[2] for unit_id in prepared.unit_ids[region]):
            raise ValueError(f"{region} unit IDs must be probe-qualified strings")
    if prepared.counts["PFC"].shape[:2] != prepared.counts["HPC"].shape[:2]:
        raise ValueError("PFC/HPC choice and bin axes must match")
    if prepared.counts["PFC"].shape[0] != len(prepared.choice_positions):
        raise ValueError("selected choice axis does not match prepared counts")
    edges = np.asarray(prepared.bin_edges_s, dtype=float)
    if (edges.ndim != 1 or len(edges) != prepared.counts["PFC"].shape[1] + 1 or
            not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)):
        raise ValueError("bin edges must be finite, increasing, and match the bin axis")


def _flat_partition(counts: NDArray[np.float64], choice_indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """Flatten selected (choice, bin, unit) counts/bin to (observation, unit)."""
    selected = counts[choice_indices]
    return selected.reshape(-1, selected.shape[2])


def _model_keys(requested_ranks: range) -> tuple[tuple[str, int | None], ...]:
    """Return unrestricted ridge and each requested integer RRR rank."""
    return (("ridge", None),) + tuple(("rrr", rank) for rank in requested_ranks)


def _evaluate_inner_grid(
    source_counts: NDArray[np.float64], target_counts: NDArray[np.float64],
    split: ValidationSplit, draw: int, direction: str, fold: int, requested_ranks: range,
) -> list[dict]:
    """Score each alpha/model on inner-training/validation choice rows only.

    Counts are (choice, bin, selected unit) spikes/bin. Return candidate table
    dictionaries with dimensionless validation mean R2 and explicit status.
    """
    source_train = _flat_partition(source_counts, split.inner_train_indices)
    source_validation = _flat_partition(source_counts, split.inner_validation_indices)
    target_train = _flat_partition(target_counts, split.inner_train_indices)
    target_validation = _flat_partition(target_counts, split.inner_validation_indices)
    source_transform = fit_training_transform(source_train)
    target_transform = fit_training_transform(target_train)
    source_z = transform_counts(source_train, source_transform)
    target_z = transform_counts(target_train, target_transform)
    source_validation_z = transform_counts(source_validation, source_transform)
    rows = []
    for alpha in ALPHA_GRID:
        decomposition = None
        fit_reason = ""
        if source_z.shape[1] == 0 or target_z.shape[1] == 0:
            fit_reason = "no variable training source or target"
        else:
            try:
                decomposition = fit_ridge_decomposition(source_z, target_z, float(alpha))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                fit_reason = f"numerical fit failure: {error}"
        for model, rank in _model_keys(requested_ranks):
            base = dict(draw=draw, direction=direction, fold=fold, model=model,
                        rank=rank, alpha=float(alpha))
            if decomposition is None:
                rows.append(dict(base, mean_r2=float("nan"), status="unavailable", reason=fit_reason))
            elif rank is not None and rank > decomposition.max_rank:
                rows.append(dict(base, mean_r2=float("nan"), status="unsupported_rank",
                                 reason="rank exceeds inner training design cap"))
            else:
                try:
                    prediction_z = predict_rank(source_validation_z, decomposition, rank)
                    prediction = inverse_transform_predictions(prediction_z, target_transform)
                    scores = score_target_predictions(target_validation, prediction)
                    available = np.isfinite(scores.mean_r2)
                    rows.append(dict(base, mean_r2=scores.mean_r2,
                                     status="ok" if available else "undefined_score",
                                     reason="" if available else "all validation target R2 undefined"))
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                    rows.append(dict(base, mean_r2=float("nan"), status="numerical_failure",
                                     reason=str(error)))
    return rows


def _evaluate_outer_fold(
    source_counts: NDArray[np.float64], target_counts: NDArray[np.float64],
    source_indices: NDArray[np.int64], target_indices: NDArray[np.int64],
    source_ids: tuple[str, ...], target_ids: tuple[str, ...], split: ValidationSplit,
    draw: int, direction: str, fold: int, requested_ranks: range, inner_rows: list[dict],
) -> tuple[list[dict], list[FitRecord]]:
    """Refit selected alphas on outer training and score each held-out target.

    Counts have shape (choice, bin, selected unit), spikes/bin. Return target-row
    dictionaries with SSE/SST in (spikes/bin)^2 and compact outer fit records.
    """
    source_train = _flat_partition(source_counts, split.outer_train_indices)
    source_test = _flat_partition(source_counts, split.outer_test_indices)
    target_train = _flat_partition(target_counts, split.outer_train_indices)
    target_test = _flat_partition(target_counts, split.outer_test_indices)
    source_transform = fit_training_transform(source_train)
    target_transform = fit_training_transform(target_train)
    source_z = transform_counts(source_train, source_transform)
    target_z = transform_counts(target_train, target_transform)
    source_test_z = transform_counts(source_test, source_transform)
    fold_rows = []
    fit_records = []
    fits_by_alpha: dict[float, RidgeDecomposition | None] = {}
    fit_errors: dict[float, str] = {}

    for model, rank in _model_keys(requested_ranks):
        candidates = [row for row in inner_rows if row["model"] == model and row["rank"] == rank]
        selected_alpha = select_alpha(
            np.array([row["alpha"] for row in candidates]),
            np.array([row["mean_r2"] for row in candidates]),
        )
        status = "ok"
        reason = ""
        scores = None
        if selected_alpha is None:
            status = "unsupported_rank" if all(row["status"] == "unsupported_rank" for row in candidates) else "tuning_unavailable"
            reason = "rank exceeds inner training design cap" if status == "unsupported_rank" else "no valid inner R2 candidate"
        elif source_z.shape[1] == 0 or target_z.shape[1] == 0:
            status, reason = "unavailable", "no variable outer training source or target"
        else:
            if selected_alpha not in fits_by_alpha:
                try:
                    fits_by_alpha[selected_alpha] = fit_ridge_decomposition(
                        source_z, target_z, selected_alpha)
                    fit_records.append(FitRecord(
                        draw, direction, fold, selected_alpha,
                        np.asarray(source_indices, dtype=np.int64).copy(),
                        np.asarray(target_indices, dtype=np.int64).copy(),
                        source_ids, target_ids, source_transform, target_transform,
                        fits_by_alpha[selected_alpha],
                    ))
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                    fits_by_alpha[selected_alpha] = None
                    fit_errors[selected_alpha] = str(error)
            decomposition = fits_by_alpha[selected_alpha]
            if decomposition is None:
                status, reason = "numerical_failure", fit_errors[selected_alpha]
            elif rank is not None and rank > decomposition.max_rank:
                status, reason = "unsupported_rank", "rank exceeds outer training design cap"
            else:
                try:
                    predicted_z = predict_rank(source_test_z, decomposition, rank)
                    predicted = inverse_transform_predictions(predicted_z, target_transform)
                    scores = score_target_predictions(target_test, predicted)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                    status, reason = "numerical_failure", str(error)

        for target_number, target_id in enumerate(target_ids):
            base = dict(draw=draw, direction=direction, fold=fold, model=model,
                        rank=rank, target_id=target_id, alpha=selected_alpha)
            if scores is None:
                fold_rows.append(dict(base, sse=float("nan"), sst=float("nan"),
                                      r2=float("nan"), valid=False, status=status, reason=reason))
            else:
                valid = bool(scores.valid_mask[target_number])
                fold_rows.append(dict(base, sse=float(scores.sse[target_number]),
                                      sst=float(scores.sst[target_number]),
                                      r2=float(scores.r2[target_number]), valid=valid,
                                      status="ok" if valid else "undefined_score",
                                      reason="" if valid else "zero held-out target variation"))
    return fold_rows, fit_records
