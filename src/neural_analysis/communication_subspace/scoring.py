"""Held-out, per-neuron prediction scores for communication subspaces."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class TargetScores:
    """Per-target SSE/SST in (spikes/bin)^2 and dimensionless held-out R2.

    sse, sst, r2, and valid_mask each have shape (target neuron,). Undefined
    R2 entries are NaN. mean_r2 is the arithmetic mean over valid targets,
    or NaN when none are valid.
    """

    sse: NDArray[np.float64]
    sst: NDArray[np.float64]
    r2: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    mean_r2: float


def score_target_predictions(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> TargetScores:
    """Score held-out actual/predicted counts/bin, each (observation, target).

    Return per-target SSE, SST, R2, validity mask, and the valid-target mean.
    SST is centered on each held-out target's own mean. Exactly zero SST makes
    R2 undefined; nonfinite inputs raise instead of silently dropping targets.
    """
    actual_counts = np.asarray(actual, dtype=np.float64)
    predicted_counts = np.asarray(predicted, dtype=np.float64)
    if actual_counts.ndim != 2 or actual_counts.shape != predicted_counts.shape:
        raise ValueError("actual and predicted must have matching two-dimensional shapes")
    if actual_counts.shape[0] == 0 or actual_counts.shape[1] == 0:
        raise ValueError("actual and predicted must be nonempty")
    if not np.isfinite(actual_counts).all() or not np.isfinite(predicted_counts).all():
        raise ValueError("actual and predicted must contain only finite values")
    sse = np.sum((actual_counts - predicted_counts) ** 2, axis=0)
    centered_actual = actual_counts - actual_counts.mean(axis=0)
    sst = np.sum(centered_actual ** 2, axis=0)
    if not np.isfinite(sse).all() or not np.isfinite(sst).all():
        raise ValueError("score sums must contain only finite values")
    valid_mask = sst > 0
    r2 = np.full(actual_counts.shape[1], np.nan)
    r2[valid_mask] = 1 - sse[valid_mask] / sst[valid_mask]
    mean_r2 = float(np.mean(r2[valid_mask])) if valid_mask.any() else float("nan")
    return TargetScores(sse, sst, r2, valid_mask, mean_r2)


def summarize_fold_scores(fold_scores: pd.DataFrame, expected_folds: int = 5) -> pd.DataFrame:
    """Aggregate target rows into draw-level mean held-out R2.

    fold_scores is a DataFrame with one row per target and columns draw,
    direction, fold, model, rank, r2, and valid. R2 is dimensionless; rank is
    nullable for unrestricted ridge. Return one row per (draw, direction,
    model, rank), with n_folds valid fold means, mean_r2 only if all expected
    folds are valid, and status ok/incomplete. Folds have equal weight.
    """
    if expected_folds < 1:
        raise ValueError("expected_folds must be positive")
    group_columns = ["draw", "direction", "model", "rank"]
    rows = []
    for keys, model_rows in fold_scores.groupby(group_columns, dropna=False, sort=False):
        fold_means = []
        for _, target_rows in model_rows.groupby("fold", sort=False):
            valid_rows = target_rows.loc[target_rows["valid"].astype(bool)]
            if not valid_rows.empty:
                fold_means.append(float(valid_rows["r2"].mean()))
        complete = len(fold_means) == expected_folds
        rows.append(dict(zip(group_columns, keys), n_folds=len(fold_means),
                         mean_r2=float(np.mean(fold_means)) if complete else float("nan"),
                         status="ok" if complete else "incomplete"))
    result = pd.DataFrame(rows, columns=group_columns + ["n_folds", "mean_r2", "status"])
    if not result.empty:
        result["rank"] = result["rank"].astype("Int64")
    return result


def summarize_draw_scores(draw_scores: pd.DataFrame, expected_draws: int = 5) -> pd.DataFrame:
    """Aggregate complete draw rows into direction/model/rank curves.

    draw_scores has one row per draw, direction, model, and nullable rank, with
    dimensionless mean_r2 and status ok/incomplete. Return one row per curve,
    including exact complete draw IDs, number of contributing draws, arithmetic
    mean R2 across those draws, and status ok only when all expected draws
    contribute. No uncertainty interval is inferred from repeated draws.
    """
    if expected_draws < 1:
        raise ValueError("expected_draws must be positive")
    group_columns = ["direction", "model", "rank"]
    rows = []
    for keys, model_rows in draw_scores.groupby(group_columns, dropna=False, sort=False):
        complete = model_rows.loc[(model_rows["status"] == "ok") &
                                  np.isfinite(model_rows["mean_r2"])]
        draw_ids = tuple(sorted(int(value) for value in complete["draw"]))
        rows.append(dict(zip(group_columns, keys), n_draws=len(draw_ids), draw_ids=draw_ids,
                         mean_r2=float(complete["mean_r2"].mean()) if draw_ids else float("nan"),
                         status="ok" if len(draw_ids) == expected_draws else "incomplete"))
    result = pd.DataFrame(rows, columns=group_columns +
                          ["n_draws", "draw_ids", "mean_r2", "status"])
    if not result.empty:
        result["rank"] = result["rank"].astype("Int64")
    return result
