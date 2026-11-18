"""Time-weighted speed and whole-block, held-out paired canonical scores."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.neural_analysis.communication_subspace.preparation import _overlapping_indices
from src.neural_analysis.population.cca import fit_paired_cca, transform_cca_activity

from .preparation import PAIRED_WINDOW_S, REGIONS, PreparedChoices


METRICS = ("pfc_hpc", "pfc_speed", "hpc_speed")
SCORE_COLUMNS = (
    "block_id", "metric", "correlation", "n_pairs", "reason", "train_anchor",
    "joint_sign", "n_train_choices", "n_test_choices", "n_purged_choices",
    "n_train_speed_bins", "n_test_speed_bins",
)
TRACK_COLUMNS = (
    "row_position", "trial_id", "block_id", "start_s", "stop_s",
    "pfc_score", "hpc_score", "speed_mm_s",
)
EVENT_COLUMNS = ("row_position", "trial_id", "event", "time_s")


def bin_speed_magnitude(
    speed_utc_s: np.ndarray, speed_mm_s: np.ndarray,
    starts_s: np.ndarray, stops_s: np.ndarray, *,
    invalid_speed_intervals_s: np.ndarray | tuple = (),
) -> np.ndarray:
    """Time-average absolute logged velocity in complete (choice, bin) intervals.

    Inputs are matching (sample,) nondecreasing UTC Unix seconds and signed mm/s,
    and matching 2-D UTC half-open start/stop arrays. Each logged value holds
    until the following update. Duplicate timestamps use the last update.
    Nonfinite speeds and supplied (interval, 2) UTC gaps are unsupported.
    Return (choice, bin) mm/s with NaN outside complete recorded coverage.
    """
    times = np.asarray(speed_utc_s, dtype=float)
    speeds = np.abs(np.asarray(speed_mm_s, dtype=float))
    starts = np.asarray(starts_s, dtype=float)
    stops = np.asarray(stops_s, dtype=float)
    gaps = np.asarray(invalid_speed_intervals_s, dtype=float)
    if (times.ndim != 1 or times.shape != speeds.shape or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0)):
        raise ValueError("Treadmill updates need matching speed and nondecreasing finite UTC times.")
    if (starts.ndim != 2 or starts.shape != stops.shape or not np.isfinite(starts).all()
            or not np.isfinite(stops).all() or np.any(stops <= starts)):
        raise ValueError("Speed bins need matching finite UTC start and stop matrices with positive durations.")
    if gaps.size and (gaps.ndim != 2 or gaps.shape[1] != 2 or not np.isfinite(gaps).all()
                      or np.any(gaps[:, 1] < gaps[:, 0])):
        raise ValueError("Invalid speed intervals need finite, ordered UTC start and stop pairs.")
    result = np.full(starts.shape, np.nan)
    if len(times) < 2:
        return result

    values = speeds[:-1]
    invalid = ~np.isfinite(values)
    missing_updates = np.column_stack((times[:-1][invalid], times[1:][invalid]))
    gaps = np.vstack((missing_updates, gaps.reshape(-1, 2)))
    complete = (starts >= times[0]) & (stops <= times[-1])
    complete &= ~_crosses_gap(starts, stops, gaps)
    left, right = starts[complete], stops[complete]
    start_update = np.searchsorted(times, left, side="right") - 1
    stop_update = np.searchsorted(times, right, side="left") - 1
    # Integrate complete interior updates plus the two partial update intervals.
    result_values = values[start_update].copy()
    spanning = start_update != stop_update
    if np.any(spanning):
        safe_values = np.where(invalid, 0., values)
        cumulative = np.r_[0., np.cumsum(safe_values * np.diff(times))]
        first, last = start_update[spanning], stop_update[spanning]
        first_area = (times[first + 1] - left[spanning]) * safe_values[first]
        last_area = (right[spanning] - times[last]) * safe_values[last]
        interior_area = cumulative[last] - cumulative[first + 1]
        result_values[spanning] = (first_area + interior_area + last_area) / (
            right[spanning] - left[spanning]
        )
    result[complete] = result_values
    return result


def compute_movement(
    prepared: PreparedChoices, speed_utc_s: np.ndarray, speed_mm_s: np.ndarray, *,
    speed_reason: str = "", invalid_speed_intervals_s: np.ndarray | tuple = (),
) -> dict:
    """Produce Figure 6 score/replay tables from prepared paired spike counts.

    Speed inputs are (sample,) UTC seconds and signed mm/s. Empty inputs are
    accepted. Optional intervals are (interval, 2) UTC bounds. Return named
    tables with r per original block/metric, replay bin rows (UTC seconds,
    canonical SD, speed magnitude mm/s) and replay event rows (UTC seconds),
    along with availability reasons, support counts and speed units.
    """
    window = prepared.windows[PAIRED_WINDOW_S, 100]
    rows = window.row_positions
    selected_trials = prepared.trials.iloc[rows]
    edges = selected_trials.choice_time.to_numpy(dtype=float)[:, None] + window.bin_edges_s
    try:
        speeds = bin_speed_magnitude(
            speed_utc_s, speed_mm_s, edges[:, :-1], edges[:, 1:],
            invalid_speed_intervals_s=invalid_speed_intervals_s,
        )
    except ValueError as error:
        speeds = np.full(edges[:, :-1].shape, np.nan)
        speed_reason = str(error)
    n_speed = np.count_nonzero(np.isfinite(speeds))
    if not n_speed and not speed_reason:
        speed_reason = "No bins have complete recorded treadmill-speed coverage."

    scores, projections = score_cca_blocks(prepared, speeds, speed_reason=speed_reason)
    replay_tracks = pd.DataFrame(columns=TRACK_COLUMNS)
    replay_events = pd.DataFrame(columns=EVENT_COLUMNS)
    increase = np.nan
    if not window.counts or scores.empty:
        replay_reason = prepared.unavailable.get("paired_window", "No held-out task blocks with paired counts.")
    else:
        replay_reason = speed_reason or "No four consecutive covered choices had a positive speed increase."
        positions, increase = _select_replay(prepared, rows, selected_trials, speeds)
        if positions.size:
            if not all(np.isfinite(projections[region][positions]).all() for region in REGIONS):
                replay_reason = "The selected block has no speed-oriented held-out canonical pair."
            else:
                replay_tracks, replay_events = _replay_tables(
                    prepared, rows[positions], edges[positions], speeds[positions],
                    projections["PFC"][positions], projections["HPC"][positions],
                )
                replay_reason = ""

    return {
        "block_scores": scores, "replay_tracks": replay_tracks, "replay_events": replay_events,
        "replay_reason": replay_reason, "speed_reason": speed_reason, "speed_units": "mm/s",
        "replay_speed_increase_mm_s": increase,
        "n_complete_speed_bins": int(n_speed), "n_paired_bins": int(speeds.size),
    }


def score_cca_blocks(
    prepared: PreparedChoices, binned_speed_mm_s: np.ndarray, *, speed_reason: str = "",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit one CCA per held-out block and score its first canonical pair.

    prepared supplies paired float counts (choice, bin, unit), original rows
    and unit identities. binned_speed_mm_s is (choice, bin) magnitude with NaN
    for incomplete coverage. Return a (block x metric, column) support table
    and (choice, bin) held-out canonical SD projections per region. Projection
    arrays are NaN for blocks without a defined training speed orientation.
    """
    window = prepared.windows[PAIRED_WINDOW_S, 100]
    trials = prepared.trials.iloc[window.row_positions]
    times = trials.choice_time.to_numpy(dtype=float)
    speed = np.asarray(binned_speed_mm_s, dtype=float)
    expected = (len(trials), len(window.bin_edges_s) - 1)
    if speed.shape != expected:
        raise ValueError("Binned speed must match the paired choice and time-bin axes.")
    projections = {region: np.full(expected, np.nan) for region in REGIONS}
    result_rows = []
    if not window.counts:
        return pd.DataFrame(columns=SCORE_COLUMNS), projections

    block_codes, block_ids = pd.factorize(trials.cur_block, sort=False)
    for code, block in enumerate(block_ids):
        test = np.flatnonzero(block_codes == code)
        candidates = np.flatnonzero((block_codes >= 0) & (block_codes != code))
        purged = _overlapping_indices(candidates, test, times, PAIRED_WINDOW_S)
        train = np.setdiff1d(candidates, purged, assume_unique=True)
        block_rows, held_out = _score_block(
            block, window.counts, prepared.unit_ids, speed, train, test, len(purged), speed_reason,
        )
        result_rows.extend(block_rows)
        for region, values in held_out.items():
            projections[region][test] = values
    return pd.DataFrame(result_rows, columns=SCORE_COLUMNS), projections


def _crosses_gap(starts: np.ndarray, stops: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Find (choice, bin) half-open bins intersecting any (interval, 2) UTC gap."""
    gaps = gaps[gaps[:, 0] < gaps[:, 1]]
    if not len(gaps):
        return np.zeros(starts.shape, dtype=bool)
    ordered = gaps[np.argsort(gaps[:, 0], kind="stable")]
    last_prior = np.searchsorted(ordered[:, 0], stops, side="left") - 1
    latest_stop = np.maximum.accumulate(ordered[:, 1])
    return (last_prior >= 0) & (latest_stop[last_prior] > starts)


def _pearson(first: np.ndarray, second: np.ndarray) -> tuple[float, int, str]:
    """Return pairwise-finite Pearson r, observation count and status for matching arrays."""
    valid = np.isfinite(first) & np.isfinite(second)
    x, y = first[valid], second[valid]
    n = len(x)
    if n < 2:
        return np.nan, n, "Fewer than two valid paired bins."
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, n, "At least one series is constant."
    x, y = x - x.mean(), y - y.mean()
    correlation = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(correlation, -1., 1.)), n, ""


def _score_block(
    block_id: object, counts: dict[str, np.ndarray], unit_ids: dict[str, tuple[str, ...]],
    speed: np.ndarray, train: np.ndarray, test: np.ndarray, n_purged: int, speed_reason: str,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit from purged (choice,) train indices and score the fixed (test, bin) projections."""
    info = {
        "block_id": block_id, "train_anchor": np.nan, "joint_sign": 0,
        "n_train_choices": len(train), "n_test_choices": len(test), "n_purged_choices": n_purged,
        "n_train_speed_bins": int(np.count_nonzero(np.isfinite(speed[train]))),
        "n_test_speed_bins": int(np.count_nonzero(np.isfinite(speed[test]))),
    }
    try:
        if not len(train):
            raise ValueError("No training choices remain after holding out and purging this block.")
        fitted = fit_paired_cca(
            counts["PFC"][train], counts["HPC"][train], unit_ids["PFC"], unit_ids["HPC"],
        )
        held_out = {}
        for region in REGIONS:
            transform = fitted.transforms[region]
            first_pair = replace(
                transform, coefficients=transform.coefficients[:, :1], score_offset=transform.score_offset[:1],
            )
            held_out[region] = transform_cca_activity(counts[region][test], unit_ids[region], first_pair)[..., 0]
    except (ValueError, np.linalg.LinAlgError) as error:
        return [dict(info, metric=metric, correlation=np.nan, n_pairs=0, reason=str(error))
                for metric in METRICS], {}

    anchor, _, anchor_reason = _pearson(fitted.scores["PFC"][..., 0], speed[train])
    if np.isfinite(anchor) and abs(anchor) <= 8 * np.finfo(float).eps:
        anchor = 0.
    info["train_anchor"] = anchor
    neural_score = _pearson(held_out["PFC"], held_out["HPC"])
    comparisons = {"pfc_hpc": neural_score}
    if np.isfinite(anchor) and anchor != 0.:
        direction = -1 if anchor < 0. else 1
        info["joint_sign"] = direction
        held_out = {region: scores * direction for region, scores in held_out.items()}
        for region in REGIONS:
            comparisons[f"{region.lower()}_speed"] = _pearson(held_out[region], speed[test])
    else:
        reason = speed_reason or anchor_reason or "Training PFC-speed correlation is zero."
        for region in REGIONS:
            comparisons[f"{region.lower()}_speed"] = (np.nan, info["n_test_speed_bins"], reason)
        held_out = {}
    return [dict(info, metric=metric, correlation=r, n_pairs=n, reason=reason)
            for metric, (r, n, reason) in comparisons.items()], held_out


def _select_replay(
    prepared: PreparedChoices, rows: np.ndarray, trials: pd.DataFrame, speed: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Select earliest maximal positive speed increase; return four choice indices and mm/s."""
    chronology = pd.Index(prepared.choices.row_position).get_indexer(rows)
    blocks, _ = pd.factorize(trials.cur_block, sort=False)
    valid_speed = np.isfinite(speed).all(axis=1)
    means = speed.mean(axis=1)
    best = np.empty(0, dtype=int)
    best_increase = 0.
    for first in range(len(rows) - 3):
        indices = np.arange(first, first + 4)
        if not (blocks[first] >= 0 and np.all(blocks[indices] == blocks[first])
                and chronology[first] >= 0 and np.all(np.diff(chronology[indices]) == 1)
                and valid_speed[indices].all()):
            continue
        early, late = means[indices[:2]].mean(), means[indices[2:]].mean()
        increase = late - early
        # Equal duration-weighted increases can differ by floating-point roundoff.
        tolerance = 32 * np.finfo(float).eps * max(1., abs(early), abs(late))
        if increase > best_increase + tolerance:
            best, best_increase = indices, float(increase)
    return best, best_increase if best.size else np.nan


def _replay_tables(
    prepared: PreparedChoices, rows: np.ndarray, edges: np.ndarray, speed: np.ndarray,
    pfc: np.ndarray, hpc: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ordered (choice x bin) tracks and (event,) UTC choice/delivery rows."""
    from .replays import recorded_deliveries

    trials = prepared.trials.iloc[rows]
    n_bins = speed.shape[1]
    tracks = pd.DataFrame({
        "row_position": np.repeat(rows, n_bins),
        "trial_id": np.repeat(trials.cur_trial.to_numpy(), n_bins),
        "block_id": np.repeat(trials.cur_block.to_numpy(), n_bins),
        "start_s": edges[:, :-1].ravel(), "stop_s": edges[:, 1:].ravel(),
        "pfc_score": pfc.ravel(), "hpc_score": hpc.ravel(), "speed_mm_s": speed.ravel(),
    })
    deliveries = recorded_deliveries(prepared, rows)
    events = pd.DataFrame({
        "row_position": np.tile(rows, 2), "trial_id": np.tile(trials.cur_trial.to_numpy(), 2),
        "event": np.repeat(("choice", "reward_delivery"), len(rows)),
        "time_s": np.r_[trials.choice_time.to_numpy(dtype=float), deliveries],
    })
    return tracks, events.loc[np.isfinite(events.time_s)].sort_values("time_s", ignore_index=True)
