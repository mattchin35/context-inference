"""Within-choice participation of fixed session PCA axes; no fitting or I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.neural_analysis.population.session_timeline import (
    RegionalTimelinePCA,
    SessionTimelineResult,
)

PARAMETERS = {
    "variance_ddof": 1,
    "threshold": 0.95,
    "conservation_tolerance": 1e-10,
    "threshold_tolerance": 1e-12,
}


@dataclass(frozen=True)
class WindowVarianceSummary:
    """Regional fixed-axis contributions for W choice windows, B bins, K PCs.

    choice_positions (W,) indexes the session choice table; bin_edges_s (B+1,)
    is relative to choice onset, seconds. unit_ids identify standardized units.
    pc_variance (W,K) is sample variance in ORIGINAL session PC order (squared
    standardized activity). pc_ids (W,K) contains ONE-based original PC IDs,
    sorted by decreasing local variance, stable for ties. fractions/cumulative
    (W,K) use that sorted order and the saved direct unit-space total_variance
    (W,), never a truncated score sum. counts95 (W,) is float with NaN when
    unavailable; status (W,) explains availability. A restricted count is not
    an exact all-axis count. basis_complete compares K with full_pc_count.
    """

    region: str
    probe_id: str
    choice_positions: np.ndarray
    bin_edges_s: np.ndarray
    unit_ids: tuple[str, ...]
    pc_variance: np.ndarray
    pc_ids: np.ndarray
    fractions: np.ndarray
    cumulative: np.ndarray
    total_variance: np.ndarray
    counts95: np.ndarray
    status: np.ndarray
    basis_complete: bool
    full_pc_count: int


@dataclass(frozen=True)
class SessionWindowVariance:
    """Session summary without duplicated scores or loadings.

    choices preserves the Stage 1 table (UTC choice_time and elapsed_s seconds).
    regions maps PFC/HPC to summaries; unavailable preserves missing-region
    reasons. meta records exact source identity, version, units, and parameters.
    """

    session_id: str
    time_origin_s: float
    choices: pd.DataFrame
    regions: dict[str, WindowVarianceSummary]
    unavailable: dict[str, str]
    meta: dict


def compute_window_variance(regional: RegionalTimelinePCA) -> WindowVarianceSummary:
    """Reduce original float64 scores (W,B,K) to fixed-axis window contributions.

    Denominator is regional.window_total_variance (W,), the saved sum of unit
    sample variances after session standardization. Return sorted contributions
    and the smallest count reaching 95%; never refit, cap by B-1, renormalize,
    or modify inputs. Nonfinite, zero, insufficient, and inconsistent windows
    have NaN counts. Incomplete bases retain explicitly restricted results.
    """
    scores = np.asarray(regional.scores, dtype=np.float64)
    windows, bins, pcs = scores.shape
    totals = np.asarray(regional.window_total_variance, dtype=np.float64)
    if totals.shape != (windows,):
        raise ValueError("Window totals must align with the score window axis.")
    full_count = min(
        regional.transform.n_training_observations,
        len(regional.transform.retained_unit_ids),
    )
    complete = pcs == full_count
    if pcs < 1 or pcs > full_count:
        raise ValueError("Retained PC count must be within the saved full-basis size.")
    variance = np.full((windows, pcs), np.nan)
    if bins >= 2:
        variance = np.var(scores, axis=1, ddof=PARAMETERS["variance_ddof"])
        # Constant decimal values can otherwise acquire mean-roundoff variance.
        variance[np.max(scores, axis=1) == np.min(scores, axis=1)] = 0
    order = np.argsort(-variance, axis=1, kind="stable")
    fractions = np.full_like(variance, np.nan)
    finite = np.isfinite(scores).all(axis=(1, 2)) & np.isfinite(totals)
    positive = finite & (totals > 0)
    fractions[positive] = (
        np.take_along_axis(variance, order, axis=1)[positive] / totals[positive, None]
    )
    cumulative = np.cumsum(fractions, axis=1)
    status = np.full(windows, "nonfinite", dtype="U24")
    status[finite & (totals == 0)] = "zero_variance"
    status[finite & (totals < 0)] = "invalid_total"
    coverage = np.sum(fractions, axis=1)
    tolerance = PARAMETERS["conservation_tolerance"]
    consistent = positive & (
        (np.abs(coverage - 1) <= tolerance) if complete else (coverage <= 1 + tolerance)
    )
    status[positive & ~consistent] = "variance_mismatch"
    reached = cumulative >= PARAMETERS["threshold"] - PARAMETERS["threshold_tolerance"]
    successful = consistent & reached.any(axis=1)
    status[consistent] = "not_reached"
    status[successful] = "ok" if complete else "restricted"
    counts = np.full(windows, np.nan)
    counts[successful] = np.argmax(reached[successful], axis=1) + 1
    if bins < 2:
        status[:] = "insufficient_bins"
        counts[:] = np.nan
    return WindowVarianceSummary(
        regional.region,
        regional.probe_id,
        regional.choice_positions.copy(),
        regional.bin_edges_s.copy(),
        regional.transform.retained_unit_ids,
        variance,
        order + 1,
        fractions,
        cumulative,
        totals.copy(),
        counts,
        status,
        complete,
        full_count,
    )


def compute_session_window_variance(
    result: SessionTimelineResult, meta: dict
) -> SessionWindowVariance:
    """Reduce each available region of a saved session; return summaries with meta.

    result contains unchanged Stage 1 scores/choice identity; meta is the Stage 2
    source/configuration dictionary. No random operations or source loading occur.
    """
    return SessionWindowVariance(
        result.session_id,
        result.time_origin_s,
        result.choices.copy(),
        {
            role: compute_window_variance(region)
            for role, region in result.regions.items()
        },
        dict(result.unavailable),
        dict(meta),
    )
