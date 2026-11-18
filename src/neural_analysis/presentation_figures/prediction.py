"""Figure 5 fixed-penalty population prediction on purged task-block folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.neural_analysis.communication_subspace.computation import DIRECTIONS
from src.neural_analysis.communication_subspace.configuration import DRAW_COUNT, OUTER_FOLD_COUNT
from src.neural_analysis.communication_subspace.modeling import (
    fit_ridge_decomposition,
    fit_training_transform,
    predict_rank,
    transform_counts,
)
from src.neural_analysis.communication_subspace.preparation import PopulationDraw, draw_populations
from src.neural_analysis.population_decoding.features import fit_fold_pca, transform_pc_activity

from .preparation import BlockSplit, PAIRED_WINDOW_S, PreparedChoices, PreparedWindow


ALPHA = 0.1
TEN_PCS = 10
MODEL_KEYS = ["draw", "direction", "panel", "count", "model"]
FOLD_COLUMNS = MODEL_KEYS + ["fold", "score_index", "status", "reason"]
DRAW_COLUMNS = MODEL_KEYS + ["n_folds", "n_neurons", "mean_r2", "status", "reason"]
CURVE_COLUMNS = ["panel", "count", "direction", "source_region", "target_region", "model",
                 "source_pool_size", "n_draws", "draw_ids", "mean_r2", "min_r2", "max_r2",
                 "min_neurons", "max_neurons", "status", "reason"]


@dataclass(frozen=True)
class FoldPopulation:
    """Training/test (bin observation, variable unit) z-scores and raw errors.

    variable_mask indexes the original draw pool. Variable-unit test residuals
    are in spikes/bin relative to the fold's training mean. baseline_sse holds
    squared residual sums for the whole pool. scale is the training SD in
    spikes/bin for the variable units.
    """

    train_z: np.ndarray
    test_z: np.ndarray
    variable_mask: np.ndarray
    variable_test_residual: np.ndarray
    baseline_sse: np.ndarray
    scale: np.ndarray


def compute_prediction(prepared: PreparedChoices, seed: int = 0) -> dict:
    """Return Figure 5 numeric fields from prepared choice windows and an integer RNG seed.

    Uses the paired four-second window with 100-ms counts (choice, bin, unit)
    in spikes/bin. Returns named result tables and arrays. model_sse and
    baseline_sse are float (fold score row, target pool position) in squared
    spikes/bin. Join score rows via score_index and target IDs via populations.
    Indices into window choices are stored in folds, while row_positions links
    those choices to the original trial table. R2 and alpha are dimensionless.
    """
    window = prepared.windows[PAIRED_WINDOW_S, 100]
    unit_counts = {region: len(prepared.unit_ids.get(region, ())) for region in ("PFC", "HPC")}
    source_pool_size = min(unit_counts.values()) // 2
    fold_rows, model_errors, baseline_errors, population_rows = [], [], [], []
    reason = ""
    if source_pool_size < 2:
        reason = "At least four good units per region are needed for matched populations."
    elif not len(window.row_positions) or not all(region in window.counts for region in unit_counts):
        reason = "No choices with simultaneous four-second PFC and HPC counts."
    else:
        for draw_id, draw in enumerate(draw_populations(prepared.unit_ids, seed=seed)):
            population_rows.extend(_population_rows(prepared.unit_ids, draw, draw_id))
            rows, model_sse, baseline_sse = score_population_draw(window, draw, draw_id)
            for row in rows:
                row["score_index"] = len(fold_rows)
                fold_rows.append(row)
            model_errors.extend(model_sse)
            baseline_errors.extend(baseline_sse)

    fold_scores = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    model_sse = np.asarray(model_errors).reshape(len(fold_rows), source_pool_size)
    baseline_sse = np.asarray(baseline_errors).reshape(len(fold_rows), source_pool_size)
    expected_folds = tuple(split.fold_id for split in window.splits)
    draw_scores = _summarize_draw_scores(fold_scores, model_sse, baseline_sse, expected_folds)
    return {
        "alpha": ALPHA, "seed": seed, "source_pool_size": source_pool_size,
        "row_positions": window.row_positions, "bin_edges_s": window.bin_edges_s,
        "available_units": pd.DataFrame(unit_counts.items(), columns=["region", "n_units"]),
        "populations": pd.DataFrame(population_rows, columns=[
            "draw", "region", "role", "pool_position", "unit_index", "unit_id"]),
        "folds": pd.DataFrame([_fold_row(split, len(window.bin_edges_s) - 1)
                               for split in window.splits]),
        "fold_scores": fold_scores, "model_sse": model_sse, "baseline_sse": baseline_sse,
        "draw_scores": draw_scores, "curves": _summarize_curves(draw_scores, source_pool_size, reason),
    }


def score_population_draw(
    window: PreparedWindow, draw: PopulationDraw, draw_id: int,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """Evaluate a fixed integer draw ID on all directions and prepared block folds.

    Draw indices select (n,) disjoint pools on each unit axis. Return a list of
    fold/model records and aligned per-record (n,) model and baseline SSE
    vectors in squared spikes/bin. Undefined fits carry explicit reasons.
    """
    size = len(draw.source_indices["PFC"])
    unit_sizes, pc_sizes = _sizes(size)
    rows, model_errors, baselines = [], [], []
    for split in window.splits:
        has_partitions = bool(len(split.train_indices) and len(split.test_indices))
        sources, targets, pc_designs = {}, {}, {}
        if has_partitions:
            for region in draw.source_indices:
                counts = window.counts[region]
                sources[region] = _fold_population(counts, draw.source_indices[region], split)
                targets[region] = _fold_population(counts, draw.target_indices[region], split)
                pc_designs[region] = _pc_design(sources[region])
        for direction, source_region, target_region in DIRECTIONS:
            if has_partitions:
                target = targets[target_region]
                baseline = target.baseline_sse
                outcomes = _score_direction(sources[source_region], target, pc_designs[source_region],
                                            unit_sizes, pc_sizes)
            else:
                baseline = np.full(size, np.nan)
                outcomes = [(panel, count, model, None, "Empty training or held-out partition.")
                            for panel, count, model in _main_models(size)]
            for panel, count, model, errors, reason in outcomes:
                scored = errors is not None
                rows.append(dict(draw=draw_id, direction=direction, panel=panel, count=count,
                                 model=model, fold=split.fold_id,
                                 status="ok" if scored else "unavailable", reason=reason))
                model_errors.append(errors if scored else np.full(size, np.nan))
                baselines.append(baseline)
    return rows, model_errors, baselines


def _score_direction(
    source: FoldPopulation, target: FoldPopulation, pc_design: tuple[np.ndarray, np.ndarray] | str,
    unit_sizes: tuple[int, ...], pc_sizes: tuple[int, ...],
) -> list[tuple[str, int, str, np.ndarray | None, str]]:
    """Return fits for one fold/direction as (panel, count, model, target SSE, reason).

    Source and target have matched original pool lengths but independently
    filtered variable columns. PC designs are dimensionless train/test arrays
    with the learned PC axes, or a string describing why no design is available.
    """
    size = unit_sizes[-1]
    outcomes = []
    for count in unit_sizes:
        variable = np.count_nonzero(source.variable_mask[:count])
        ranks = range(1, size + 1) if count == size else ()
        ridge, rank_scores, reason = _score_design(source.train_z[:, :variable],
                                                   source.test_z[:, :variable], target, ranks)
        outcomes.append(("A", count, "ridge", ridge, reason if ridge is None else ""))
        outcomes.extend(("C", rank, "rrr", errors, rank_reason)
                        for rank, errors, rank_reason in rank_scores)
    pc_count = pc_design[0].shape[1] if not isinstance(pc_design, str) else 0
    for count in pc_sizes:
        if count > pc_count:
            reason = pc_design if isinstance(pc_design, str) else f"Only {pc_count} training PCs supported."
            outcomes.append(("B", count, "ridge", None, reason))
            continue
        train, test = pc_design
        ranks = range(1, TEN_PCS + 1) if count == TEN_PCS else ()
        ridge, rank_scores, reason = _score_design(train[:, :count], test[:, :count], target, ranks)
        outcomes.append(("B", count, "ridge", ridge, reason if ridge is None else ""))
        outcomes.extend(("D", rank, "rrr", errors, rank_reason)
                        for rank, errors, rank_reason in rank_scores)
    if pc_count < TEN_PCS:
        detail = pc_design if isinstance(pc_design, str) else f"Only {pc_count} training PCs supported."
        outcomes.extend(("D", rank, "rrr", None, f"Ten source PCs are required. {detail}")
                        for rank in range(1, TEN_PCS + 1))
    return outcomes


def _score_design(
    train_z: np.ndarray, test_z: np.ndarray, target: FoldPopulation, ranks: range | tuple[int, ...],
) -> tuple[np.ndarray | None, list[tuple[int, np.ndarray | None, str]], str]:
    """Score one (observation, feature) standardized source design against a target.

    Return target-order raw-scale SSE for ridge and each requested rank, with
    reasons for unavailable fits. Successive RRR predictions add one ordered
    target component at a time from the same ridge decomposition.
    """
    if not train_z.shape[1] or not target.train_z.shape[1]:
        reason = "No variable training source or target for reduced-rank regression."
        return target.baseline_sse, [(rank, None, reason) for rank in ranks], ""
    try:
        decomposition = fit_ridge_decomposition(train_z, target.train_z, ALPHA)
        ridge_z = predict_rank(test_z, decomposition, None)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        reason = f"Ridge fit failed: {error}"
        return None, [(rank, None, reason) for rank in ranks], reason

    # Invert the training scaling in residual space to avoid adding a large mean.
    ridge_sse = _raw_sse(target, ridge_z)
    by_rank = []
    if ranks:
        basis = decomposition.right_vectors[:decomposition.max_rank]
        coordinates = ridge_z @ basis.T
        prediction_z = np.zeros_like(ridge_z)
        for rank in ranks:
            if rank > decomposition.max_rank:
                reason = f"Rank {rank} exceeds training design cap {decomposition.max_rank}."
                by_rank.append((rank, None, reason))
                continue
            prediction_z += coordinates[:, rank - 1, None] * basis[rank - 1]
            by_rank.append((rank, _raw_sse(target, prediction_z), ""))
    return ridge_sse, by_rank, ""


def _raw_sse(target: FoldPopulation, predicted_z: np.ndarray) -> np.ndarray:
    """Convert (test bin, variable target) z deviations to per-target raw-scale SSE."""
    result = target.baseline_sse.copy()
    residual = target.variable_test_residual - predicted_z * target.scale
    result[target.variable_mask] = np.square(residual).sum(axis=0)
    return result


def _fold_population(counts: np.ndarray, indices: np.ndarray, split: BlockSplit) -> FoldPopulation:
    """Transform selected (choice, bin, unit) counts/bin on the split's training bins."""
    train = counts[np.ix_(split.train_indices, np.arange(counts.shape[1]), indices)]
    test = counts[np.ix_(split.test_indices, np.arange(counts.shape[1]), indices)]
    train = train.reshape(-1, len(indices))
    test = test.reshape(-1, len(indices))
    transform = fit_training_transform(train)
    test_residual = test - transform.means
    return FoldPopulation(transform_counts(train, transform), transform_counts(test, transform),
                          transform.variable_mask, test_residual[:, transform.variable_mask],
                          np.square(test_residual).sum(axis=0),
                          transform.scales[transform.variable_mask])


def _pc_design(source: FoldPopulation) -> tuple[np.ndarray, np.ndarray] | str:
    """Return dimensionless train/test scores for all supported PCs, or a fit reason."""
    try:
        pca = fit_fold_pca(source.train_z)
        return transform_pc_activity(source.train_z, pca), transform_pc_activity(source.test_z, pca)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return f"Training PCA unavailable: {error}"


def _summarize_draw_scores(
    fold_scores: pd.DataFrame, model_sse: np.ndarray, baseline_sse: np.ndarray,
    expected_folds: tuple[int, ...],
) -> pd.DataFrame:
    """Pool fold SSE arrays (score row, target) into one dimensionless mean R2 per draw/model."""
    rows = []
    for keys, group in fold_scores.groupby(MODEL_KEYS, sort=False):
        completed = group.loc[group.status.eq("ok")]
        fold_ids = set(completed.fold)
        complete = (len(expected_folds) == OUTER_FOLD_COUNT and
                    len(completed) == OUTER_FOLD_COUNT and fold_ids == set(expected_folds))
        count, score = 0, np.nan
        if complete:
            numerator = model_sse[completed.score_index].sum(axis=0)
            denominator = baseline_sse[completed.score_index].sum(axis=0)
            valid = denominator > 0
            count = int(valid.sum())
            if count:
                score = float(np.mean(1 - numerator[valid] / denominator[valid]))
        reasons = [reason for reason in group.reason.unique() if reason]
        if not complete and not reasons:
            reasons.append(f"Only {len(fold_ids)} of {OUTER_FOLD_COUNT} held-out folds completed.")
        elif complete and not count:
            reasons.append("All target-neuron baseline sums are zero.")
        rows.append(dict(zip(MODEL_KEYS, keys), n_folds=len(fold_ids), n_neurons=count,
                         mean_r2=score, status="ok" if np.isfinite(score) else "unavailable",
                         reason=" ".join(reasons)))
    return pd.DataFrame(rows, columns=DRAW_COLUMNS)


def _summarize_curves(
    draw_scores: pd.DataFrame, source_pool_size: int, setup_reason: str = "",
) -> pd.DataFrame:
    """Return rank/sweep scores and references on the exact jointly supported draw IDs.

    Input is a draw/model table of dimensionless mean R2. Output is one row
    per displayed direction/point/model, with draw range and neuron support.
    A target group's within- and between-region directions share the
    intersection across their main models and all references at that point.
    """
    draws_by_model = {}
    reasons_by_model = {}
    for key, draws in draw_scores.groupby(MODEL_KEYS[1:], sort=False):
        available = draws.loc[draws.status.eq("ok") & np.isfinite(draws.mean_r2)]
        draws_by_model[key] = available.set_index("draw")
        reasons_by_model[key] = [reason for reason in draws.reason.unique() if reason]

    output = []
    observed = set(draw_scores[["panel", "count", "model"]].itertuples(index=False, name=None))
    points = _main_models(source_pool_size)
    for panel, count, model in points:
        if observed and (panel, count, model) not in observed:
            continue
        models = [(model, panel, count, model)]
        if panel in ("C", "D"):
            models.append(("ridge_units", "A", source_pool_size, "ridge"))
        if panel == "D":
            models.append(("ridge_10pc", "B", TEN_PCS, "ridge"))
        for target in ("PFC", "HPC"):
            comparison = [
                (direction, source, name, (direction, fit_panel, fit_count, fit_model))
                for direction, source, target_region in DIRECTIONS if target_region == target
                for name, fit_panel, fit_count, fit_model in models
            ]
            common_draws = set(range(DRAW_COUNT))
            reasons = []
            for _, _, _, fit_key in comparison:
                support = draws_by_model.get(fit_key)
                common_draws.intersection_update(support.index if support is not None else ())
                reasons.extend(reasons_by_model.get(fit_key, ()))
            draw_ids = tuple(sorted(common_draws))
            if setup_reason:
                reason = setup_reason
            elif not draw_ids and panel == "D":
                reason = "No shared draws support ten source PCs and all displayed fits."
            elif not draw_ids:
                reason = "No shared draws support all displayed fits."
            else:
                reason = ""
            if len(draw_ids) < DRAW_COUNT and reasons:
                reason = " ".join(filter(None, (reason, " ".join(dict.fromkeys(reasons)))))
            for direction, source, name, fit_key in comparison:
                values = draws_by_model[fit_key].loc[list(draw_ids)] if draw_ids else None
                output.append(dict(
                    panel=panel, count=count, direction=direction, source_region=source,
                    target_region=target, model=name, source_pool_size=source_pool_size,
                    n_draws=len(draw_ids), draw_ids=draw_ids,
                    mean_r2=values.mean_r2.mean() if draw_ids else np.nan,
                    min_r2=values.mean_r2.min() if draw_ids else np.nan,
                    max_r2=values.mean_r2.max() if draw_ids else np.nan,
                    min_neurons=values.n_neurons.min() if draw_ids else 0,
                    max_neurons=values.n_neurons.max() if draw_ids else 0,
                    status=("ok" if len(draw_ids) == DRAW_COUNT else
                            "partial" if draw_ids else "unavailable"),
                    reason=reason,
                ))
    return pd.DataFrame(output, columns=CURVE_COLUMNS)


def _sizes(size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return nominal source-unit and source-PC sweep counts for matched pool size n."""
    unit_sizes = tuple(sorted({size} | {2 ** exponent for exponent in range(size.bit_length())}))
    pc_sizes = tuple(sorted(set(range(1, min(size, TEN_PCS) + 1)) | set(unit_sizes)))
    return unit_sizes, pc_sizes


def _main_models(size: int) -> tuple[tuple[str, int, str], ...]:
    """List requested (panel, integer size or rank, model) points including unavailable ones."""
    unit_sizes, pc_sizes = _sizes(max(size, 1))
    return (tuple(("A", count, "ridge") for count in unit_sizes) +
            tuple(("B", count, "ridge") for count in pc_sizes) +
            tuple(("C", rank, "rrr") for rank in range(1, max(size, 1) + 1)) +
            tuple(("D", rank, "rrr") for rank in range(1, TEN_PCS + 1)))


def _population_rows(
    unit_ids: dict[str, tuple[str, ...]], draw: PopulationDraw, draw_id: int,
) -> list[dict]:
    """Return one audit row per drawn regional source/target index and its unit ID."""
    return [dict(draw=draw_id, region=region, role=role, pool_position=position,
                 unit_index=int(index), unit_id=unit_ids[region][index])
            for role, groups in (("source", draw.source_indices), ("target", draw.target_indices))
            for region, indices in groups.items()
            for position, index in enumerate(indices)]


def _fold_row(split: BlockSplit, bin_count: int) -> dict:
    """Return integer split indices and counts on the choice and flattened bin axes."""
    return dict(fold=split.fold_id, train_indices=tuple(split.train_indices),
                test_indices=tuple(split.test_indices), purged_indices=tuple(split.purged_indices),
                n_train_choices=len(split.train_indices), n_test_choices=len(split.test_indices),
                n_purged_choices=len(split.purged_indices), n_train_bins=len(split.train_indices) * bin_count,
                n_test_bins=len(split.test_indices) * bin_count)
