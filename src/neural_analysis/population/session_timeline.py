"""Choice-window population activity on one fixed elapsed-session timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd
import pynapple as nap

from src.behavior_analysis.project_utils import (
    get_experimenter_reward_flags,
    is_present_value,
    is_zero_flag,
)
from src.neural_analysis.interregional.pca import (
    fit_regional_pca,
    transform_regional_activity,
)
from src.neural_analysis.interregional.records import RegionalPCATransform
from src.neural_analysis.population.pca import build_trial_unit_rate_tensor_pynapple


ANALYSIS_VERSION = "session-timeline-pca-v1"
BIN_WIDTHS_S = (0.5, 0.1, 0.05, 0.02)
WINDOW_S = (-2.0, 2.0)


@dataclass(frozen=True)
class RegionalTimelinePCA:
    """One fixed regional basis and its choice-window scores.

    ``choice_positions`` indexes the shared choice table, shape (W,).
    ``scores`` is float64 (W, B, K) in unwhitened standardized-unit score
    units. ``bin_edges_s`` is (B+1,) in choice-relative seconds. The transform
    retains count means/SDs and (K, retained unit) loadings. Sample variance
    totals have shape scalar (session) and (W,) (within-window). ``color_limit``
    is the fixed maximum absolute score. Timing values are elapsed seconds.
    """

    region: str
    probe_id: str
    choice_positions: np.ndarray
    bin_edges_s: np.ndarray
    scores: np.ndarray
    transform: RegionalPCATransform
    total_session_variance: float
    window_total_variance: np.ndarray
    color_limit: float
    timings_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionTimelineResult:
    """Prepared session data, with no UI or file-management behavior.

    ``time_origin_s`` is first-trial UTC Unix seconds. ``choices`` has one row
    per completed ordinary choice: original row_position/trial_id, action,
    choice_time (UTC seconds), elapsed_s, and reward (1/0/NaN for rewarded,
    unrewarded, or unknown). Older saved choices may omit reward. ``events``
    retains source columns plus elapsed_s. ``regions`` maps PFC/HPC to
    independent regional results;
    ``unavailable`` maps unavailable region names to reasons. ``meta`` stores
    analysis parameters, source identities, and units/axis descriptions.
    """

    session_id: str
    time_origin_s: float
    choices: pd.DataFrame
    events: pd.DataFrame
    regions: dict[str, RegionalTimelinePCA]
    unavailable: dict[str, str]
    meta: dict


def prepare_timeline_choices(trial_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Return first-trial origin and completed ordinary choices from a trial table.

    Input rows are chronological processed trials, with start_time/choice_time in UTC seconds,
    action 0=right/1=left, and cur_trial or trial_id. Existing manual-reward
    flag aliases are accepted. Return (origin UTC seconds, choice DataFrame);
    row_position always means original zero-based position, never index label.
    Optional reward uses the existing 1=rewarded/0=unrewarded convention;
    missing or other values become NaN (unknown), never inferred from correct.
    The input is not modified; raw event choices are never reconstructed.
    """
    required = {"start_time", "choice_time", "action"}
    if not required.issubset(trial_df.columns) or trial_df.empty:
        raise ValueError(
            "Trial table needs start_time, choice_time, action and at least one row."
        )
    id_column = "cur_trial" if "cur_trial" in trial_df else "trial_id"
    if id_column not in trial_df:
        raise ValueError("Trial table needs cur_trial or trial_id identities.")
    origin = float(pd.to_numeric(trial_df["start_time"], errors="coerce").iloc[0])
    if not np.isfinite(origin):
        raise ValueError("First trial start must be finite UTC seconds.")
    times = pd.to_numeric(trial_df["choice_time"], errors="coerce").to_numpy(
        dtype=float
    )
    actions = pd.to_numeric(trial_df["action"], errors="coerce").to_numpy(dtype=float)
    ordinary = is_zero_flag(get_experimenter_reward_flags(trial_df)).to_numpy(
        dtype=bool
    )
    positions = np.flatnonzero(np.isfinite(times) & np.isin(actions, [0, 1]) & ordinary)
    if np.any(np.diff(times[positions]) < 0):
        raise ValueError("Completed choices must be in chronological source order.")
    rewards = np.full(len(trial_df), np.nan)
    if "reward" in trial_df:
        values = pd.to_numeric(trial_df["reward"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        rewards = np.where(np.isin(values, [0, 1]), values, np.nan)
    choices = pd.DataFrame(
        {
            "row_position": positions,
            "trial_id": trial_df[id_column].iloc[positions].to_numpy(),
            "choice_time": times[positions],
            "action": actions[positions].astype(int),
            "elapsed_s": times[positions] - origin,
            "reward": rewards[positions],
        }
    )
    return origin, choices


def trial_identities_match(
    first: pd.Series | np.ndarray, second: pd.Series | np.ndarray,
) -> bool:
    """Compare aligned original trial IDs without changing choice eligibility.

    Parameters
    ----------
    first, second : pd.Series or np.ndarray
        One-dimensional identity vectors (N,), with no physical units. Series
        index labels are ignored. Numeric IDs and numeric strings compare by
        value. Real missing values and recognized project string sentinels
        compare as missing. Other present tokens compare by stripped text.

    Returns
    -------
    bool
        True if shapes and every aligned identity agree semantically. Different
        malformed tokens never become equal merely because numeric coercion
        yields NaN. Paired missing IDs preserve an ordinary neural reference
        window, while motif selection separately requires consecutive IDs.
        Different shapes or non-vector inputs return False; inputs are unchanged.
    """
    left = np.asarray(first, dtype=object)
    right = np.asarray(second, dtype=object)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        return False
    left = pd.Series(left)
    right = pd.Series(right)
    left_present = is_present_value(left).to_numpy()
    right_present = is_present_value(right).to_numpy()
    if not np.array_equal(left_present, right_present):
        return False
    left = left[left_present]
    right = right[right_present]
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    numeric = left_numeric.notna().to_numpy()
    if not np.array_equal(numeric, right_numeric.notna().to_numpy()):
        return False
    if not np.array_equal(left_numeric.to_numpy()[numeric], right_numeric.to_numpy()[numeric]):
        return False
    left_text = left.astype(str).str.strip().to_numpy()[~numeric]
    right_text = right.astype(str).str.strip().to_numpy()[~numeric]
    return bool(np.array_equal(left_text, right_text))


def select_complete_choice_windows(
    choice_times_s: np.ndarray,
    coverage_s: tuple[float, float],
) -> np.ndarray:
    """Return positions of (N,) UTC-second choices with full [-2,+2) coverage.

    coverage_s is the probe's trusted (start, end) UTC-second pair. Return an
    integer (W,) index array; silence and elapsed-session zero are irrelevant.
    """
    times = np.asarray(choice_times_s, dtype=float)
    start, end = coverage_s
    if times.ndim != 1 or not np.isfinite([start, end]).all() or start > end:
        raise ValueError(
            "Expected one-dimensional times and finite ordered coverage bounds."
        )
    return np.flatnonzero(
        np.isfinite(times) & (times - 2 >= start) & (times + 2 <= end)
    )


def bin_choice_counts(
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    choice_times_s: np.ndarray,
    bin_width_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a probe's spikes into unsmoothed choice-centered counts.

    spike_group is keyed by cluster ID with UTC-second timestamps; cluster_ids
    is (U,) in output unit order and choice_times_s is (W,) UTC seconds.
    Return float64 counts (W,B,U) and relative-second edges (B+1,). The
    existing Hz builder is converted back to counts using the fixed bin width.
    """
    edges = choice_bin_edges(bin_width_s)
    choices = pd.DataFrame({"choice_time": np.asarray(choice_times_s, dtype=float)})
    rates, _ = build_trial_unit_rate_tensor_pynapple(
        spike_group,
        cluster_ids,
        choices,
        np.arange(len(choices)),
        "choice_time",
        WINDOW_S,
        bin_width_s,
    )
    return rates * bin_width_s, edges


def fit_timeline_activity(
    counts: np.ndarray,
    unit_ids: tuple[str, ...],
    choice_positions: np.ndarray,
    region: str,
    probe_id: str,
    bin_width_s: float,
) -> RegionalTimelinePCA:
    """Fit one full, unwhitened session PCA to count activity (W,B,U).

    unit_ids labels the final axis; choice_positions (W,) maps to the shared
    choice table. bin_width_s selects B=8/40/80/200. Return the fixed basis,
    float64 scores (W,B,K), and sample-variance totals in standardized space.
    All input activity must be finite; constant units are excluded globally.
    """
    edges = choice_bin_edges(bin_width_s)
    values = np.asarray(counts, dtype=np.float64)
    positions = np.asarray(choice_positions, dtype=int)
    if values.ndim != 3 or values.shape[:2] != (positions.size, edges.size - 1):
        raise ValueError(
            "Activity must have shape (choices, bins, units) for this bin width."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "Activity must contain only finite counts in complete windows."
        )
    start = perf_counter()
    fitted = fit_regional_pca(
        values,
        unit_ids,
        requested_components=min(values.shape[0] * values.shape[1], values.shape[2]),
    )
    fit_elapsed = perf_counter() - start
    start = perf_counter()
    scores = transform_regional_activity(values, unit_ids, fitted)
    # Reuse exactly the fitted unit scaling for both session and window totals.
    retained = [unit_ids.index(unit) for unit in fitted.retained_unit_ids]
    standardized = (
        values[:, :, retained] - fitted.training_mean
    ) / fitted.training_scale
    total = float(standardized.reshape(-1, len(retained)).var(axis=0, ddof=1).sum())
    window_totals = standardized.var(axis=1, ddof=1).sum(axis=1)
    return RegionalTimelinePCA(
        region,
        probe_id,
        positions,
        edges,
        scores,
        fitted,
        total,
        window_totals,
        float(np.abs(scores).max()),
        {
            "standardization_and_pca": fit_elapsed,
            "projection_and_totals": perf_counter() - start,
        },
    )


def compute_timeline_pca(
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    choices: pd.DataFrame,
    choice_positions: np.ndarray,
    region: str,
    probe_id: str,
    bin_width_s: float = 0.5,
) -> RegionalTimelinePCA:
    """Bin and fit one population, keeping explicit choice and unit axis identities.

    choices is the shared completed-choice table with UTC-second choice_time;
    choice_positions (W,) selects full regional windows. cluster_ids (U,)
    determines count-axis order. Return RegionalTimelinePCA with (W,B,K) scores.
    """
    start = perf_counter()
    counts, _ = bin_choice_counts(
        spike_group,
        cluster_ids,
        choices.iloc[choice_positions].choice_time.to_numpy(),
        bin_width_s,
    )
    binning_elapsed = perf_counter() - start
    ids = tuple(f"{probe_id}:{cluster}" for cluster in cluster_ids)
    result = fit_timeline_activity(
        counts, ids, choice_positions, region, probe_id, bin_width_s
    )
    result.timings_s["binning"] = binning_elapsed
    return result


def summarize_session_variance(
    explained_variance: np.ndarray,
    total_variance: float,
) -> dict[float, int | None]:
    """Return leading-PC counts for 80/90/95 percent of total sample variance.

    explained_variance is an ordered (K,) sample-variance spectrum; total_variance
    is from all standardized units, including variance outside retained PCs.
    None means a threshold was not reached. No subset renormalization occurs.
    """
    cumulative = np.cumsum(explained_variance) / total_variance
    counts = {}
    for threshold in (0.8, 0.9, 0.95):
        reached = np.flatnonzero(cumulative >= threshold - 1e-12)
        counts[threshold] = int(reached[0] + 1) if reached.size else None
    return counts


def choice_bin_edges(bin_width_s: float) -> np.ndarray:
    """Return (B+1,) half-open-window edges in seconds for a supported width.

    bin_width_s must be one of BIN_WIDTHS_S; the final edge is exactly +2 s.
    """
    if bin_width_s not in BIN_WIDTHS_S:
        raise ValueError(f"bin width must be one of {BIN_WIDTHS_S} seconds.")
    return np.linspace(-2.0, 2.0, int(round(4 / bin_width_s)) + 1)
