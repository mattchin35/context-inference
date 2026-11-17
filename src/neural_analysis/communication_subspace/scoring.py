"""Held-out, per-neuron prediction scores for communication subspaces."""

from dataclasses import dataclass

import numpy as np
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
