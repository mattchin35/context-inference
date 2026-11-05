"""Single-source choice-window activity and session-wide heatmap scaling.

Behavioral selection uses full-table positions. Neural coverage selects source
windows independently; motif assembly joins those positions without padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.population import session_timeline as timeline
from .selection import _ordinary_choices, prepare_sequence_trials


@dataclass(frozen=True)
class SourceActivity:
    """One selected source over all its eligible complete session windows.

    values is float (W,B): unsmoothed Hz for source_kind='unit', or signed
    unwhitened standardized-unit scores for 'pc'. row_positions is unique
    nonnegative integer (W,), indexing the full original trial table; neither
    trial IDs nor DataFrame labels belong on this axis. bin_edges_s is float
    (B+1,), choice-relative seconds covering [-2,+2) at a supported width.
    source_label is a nonempty probe-qualified display identity. Arrays are
    copied and made read-only; no source input array is modified.
    """

    values: np.ndarray
    row_positions: np.ndarray
    bin_edges_s: np.ndarray
    source_kind: str
    source_label: str

    def __post_init__(self) -> None:
        """Validate this record's documented shapes/units; copy arrays or raise ValueError."""
        values = np.asarray(self.values, dtype=float)
        positions = np.asarray(self.row_positions)
        edges = np.asarray(self.bin_edges_s, dtype=float)
        if self.source_kind not in ("unit", "pc"):
            raise ValueError("source_kind must be unit or pc.")
        if not isinstance(self.source_label, str) or not self.source_label.strip():
            raise ValueError("source_label must be a nonempty source identity.")
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Source values must be finite with shape (windows, bins).")
        _validate_positions(positions)
        if len(positions) != values.shape[0]:
            raise ValueError("row_positions must match the source window axis.")
        if self.source_kind == "unit" and np.any(values < 0):
            raise ValueError("Unit activity must be nonnegative Hz.")
        bins = values.shape[1]
        if bins not in (8, 40, 80, 200) or edges.shape != (bins + 1,):
            raise ValueError("Source bin edges must match a supported choice-window bin count.")
        if not np.array_equal(edges, timeline.choice_bin_edges(4.0 / bins)):
            raise ValueError("Source bin edges must span [-2,+2) with the supported uniform geometry.")
        for name, array in (("values", values), ("row_positions", positions), ("bin_edges_s", edges)):
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def extract_unit_activity(
    unit_spike_times_s: np.ndarray, full_trials: pd.DataFrame,
    coverage_s: tuple[float, float], bin_width_s: float = 0.5, *, source_label: str,
) -> SourceActivity:
    """Bin only the requested unit over all its ordinary covered choice windows.

    Parameters
    ----------
    unit_spike_times_s : np.ndarray
        Finite (S,) selected-unit timestamps in UTC Unix seconds, including
        empty or single-timestamp trains. Input ordering is not changed.
    full_trials : pd.DataFrame
        Full chronological (N,C) processed trial table, with field conventions
        documented by prepare_sequence_trials. Ordinary choices require finite
        choice_time, action 0/1 and known zero manual flag; outcome is unrestricted.
    coverage_s : tuple[float, float]
        Trusted ordered finite recording/synchronization bounds in UTC seconds.
        Only full [-2,+2) windows are eligible; spike extrema never define coverage.
    bin_width_s : float
        Supported bin width in seconds (.5/.1/.05/.02).
    source_label : str, keyword-only
        Probe-qualified selected-unit identity, for example A:7.

    Returns
    -------
    SourceActivity
        Unsmoothed (W,B) Hz (spike count divided by bin_width_s), original (W,)
        row positions and (B+1,) relative-second edges. Covered silence stays
        zero. No full windows returns (0,B), with no invented observations.
    """
    spikes = np.asarray(unit_spike_times_s, dtype=float)
    if spikes.ndim != 1 or not np.isfinite(spikes).all():
        raise ValueError("Unit spike times must be a finite one-dimensional array of UTC seconds.")
    bounds = _coverage_bounds(coverage_s)
    edges = timeline.choice_bin_edges(bin_width_s)
    trials = prepare_sequence_trials(full_trials)
    ordinary_positions = np.flatnonzero(_ordinary_choices(trials))
    choice_times = trials["choice_time"].to_numpy()[ordinary_positions]
    covered = timeline.select_complete_choice_windows(choice_times, bounds)
    positions = ordinary_positions[covered]
    if not len(positions):
        return SourceActivity(np.empty((0, len(edges) - 1)), positions, edges, "unit", source_label)

    # Explicit support keeps empty/degenerate trains valid for Pynapple. Expand
    # it to preserve spikes outside trusted bounds; eligibility still uses bounds.
    support_start, support_end = bounds
    if spikes.size:
        support_start = min(support_start, float(spikes.min()))
        support_end = max(support_end, float(spikes.max()))
    support = nap.IntervalSet(start=support_start, end=support_end, time_units="s")
    group = nap.TsGroup(
        {0: nap.Ts(t=np.sort(spikes), time_units="s", time_support=support)},
        time_support=support,
    )
    counts, edges = timeline.bin_choice_counts(
        group, np.array([0]), choice_times[covered], bin_width_s,
    )
    return SourceActivity(counts[:, :, 0] / bin_width_s, positions, edges, "unit", source_label)


def select_pc_activity(
    full_trials: pd.DataFrame, shared_choices: pd.DataFrame,
    regional: timeline.RegionalTimelinePCA, pc_rank: int,
) -> SourceActivity:
    """Select one saved PC and verify its original behavioral row identities.

    Parameters
    ----------
    full_trials : pd.DataFrame
        Full chronological (N,C) processed table using prepare_sequence_trials
        conventions. IDs, choice times (UTC seconds), actions and ordinary
        validity must agree with the selected shared-choice rows.
    shared_choices : pd.DataFrame
        Saved timeline (C_all,columns) table with integer row_position, original
        trial_id, UTC choice_time and action 0/1. DataFrame labels are ignored.
    regional : RegionalTimelinePCA
        Fixed saved regional basis with scores (W,B,K), choice_positions (W,)
        indexing shared_choices and bin_edges_s (B+1,) in relative seconds.
        Coverage eligibility has already been established for these windows.
    pc_rank : int
        One-based PC rank from the existing fixed basis. No fitting or scaling.

    Returns
    -------
    SourceActivity
        Signed unwhitened (W,B) values for exactly that PC, original (W,) source
        positions, and probe-qualified PC label. Paired recognized missing trial
        IDs retain otherwise eligible windows. Mismatched identities, duplicate
        positions, malformed axes and invalid ranks raise ValueError.
    """
    scores = np.asarray(regional.scores)
    if scores.ndim != 3:
        raise ValueError("Regional scores must have shape (windows, bins, PCs).")
    if isinstance(pc_rank, bool) or not isinstance(pc_rank, Integral) or not 1 <= pc_rank <= scores.shape[2]:
        raise ValueError("PC rank must be a valid one-based integer.")
    mapping = np.asarray(regional.choice_positions)
    _validate_positions(mapping)
    if len(mapping) != scores.shape[0] or np.any(mapping >= len(shared_choices)):
        raise ValueError("Regional choice_positions must index shared choice rows and match score windows.")
    if scores.shape[2] != regional.transform.actual_components:
        raise ValueError("Regional score PC axis does not match the saved fixed basis.")
    required = {"row_position", "trial_id", "choice_time", "action"}
    if not required.issubset(shared_choices.columns):
        raise ValueError("Shared choices need row_position, trial_id, choice_time and action identities.")
    selected_choices = shared_choices.iloc[mapping]
    positions = selected_choices["row_position"].to_numpy()
    _validate_positions(positions)
    trials = prepare_sequence_trials(full_trials)
    if np.any(positions >= len(trials)):
        raise ValueError("Shared choice row_position exceeds the full original trial table.")
    original_choices = trials.iloc[positions]
    if not timeline.trial_identities_match(selected_choices["trial_id"], original_choices["trial_id"]):
        raise ValueError("Shared choice trial_id identity differs from the original trial row.")
    for column in ("choice_time", "action"):
        previous = pd.to_numeric(selected_choices[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        original = pd.to_numeric(original_choices[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(previous).all() or not np.array_equal(previous, original):
            raise ValueError(f"Shared choice {column} identity differs from the original trial row.")
    if not _ordinary_choices(original_choices).all():
        raise ValueError("Shared choice identity refers to a nonordinary original trial.")
    label = f"{regional.region}/{regional.probe_id} PC{pc_rank}"
    return SourceActivity(scores[:, :, pc_rank - 1], positions, regional.bin_edges_s, "pc", label)


def scale_activity(source: SourceActivity, *, zscore: bool = False) -> dict:
    """Prepare selected-source display values using all unique eligible session bins.

    Parameters
    ----------
    source : SourceActivity
        Valid selected source: values (W,B), unique original row positions (W,)
        and relative-second edges (B+1,). No motif subset is accepted here.
    zscore : bool, keyword-only
        If True, apply session mean/population SD (ddof=0) to all source bins.

    Returns
    -------
    dict
        values (W,B), or None if display scaling is unavailable; copied
        row_positions/bin_edges_s; source_kind/source_label; value_units;
        session_mean/session_sd; color_limits (lower,upper) or None;
        zscore_available, constant_activity and unavailable_reason. Raw unit
        values are Hz with limits (0,session max); raw PC limits are symmetric.
        Z-scored values are source SD units with symmetric limits. Zero SD
        leaves raw values valid but disables z-scoring. W=0 has no reference,
        values or limits. No source arrays/basis/ranks are changed.
    """
    if not isinstance(zscore, bool):
        raise ValueError("zscore must be a boolean display setting.")
    result = {
        "values": None, "row_positions": source.row_positions.copy(),
        "bin_edges_s": source.bin_edges_s.copy(), "source_kind": source.source_kind,
        "source_label": source.source_label,
        "value_units": "source SD units" if zscore else (
            "Hz" if source.source_kind == "unit" else "unwhitened standardized-unit scores"
        ),
        "session_mean": None, "session_sd": None, "color_limits": None,
        "zscore_available": False, "constant_activity": False, "unavailable_reason": "",
    }
    if not source.values.size:
        result["unavailable_reason"] = "No eligible complete choice windows; session reference unavailable."
        return result
    constant = bool(source.values.min() == source.values.max())
    mean = float(source.values.flat[0]) if constant else float(source.values.mean())
    sd = 0.0 if constant else float(source.values.std(ddof=0))
    result.update(session_mean=mean, session_sd=sd, zscore_available=sd > 0, constant_activity=constant)
    if zscore and sd == 0:
        result["unavailable_reason"] = "Session SD is zero; z-scoring is unavailable for this constant source."
        return result
    values = (source.values - mean) / sd if zscore else source.values.copy()
    maximum = float(np.abs(values).max()) or 1.0
    symmetric = zscore or source.source_kind == "pc"
    result.update(values=values, color_limits=(-maximum if symmetric else 0.0, maximum))
    return result


def assemble_heatmap(scaled_activity: dict, match_positions: np.ndarray) -> dict:
    """Join behavior matches to covered source rows without changing the reference.

    Parameters
    ----------
    scaled_activity : dict
        Full-session output of scale_activity. Its row_positions (W,) determine
        coverage independently of whether display values (W,B) are available.
    match_positions : np.ndarray
        Integer original positions: (M,) central trials or (E,S) episodes with
        S displayed trial segments. Repeated windows across episodes remain
        repeated display segments; they never reweight session scaling.

    Returns
    -------
    dict
        values (M_covered,B) or (E_covered,S,B); row_positions with matching
        central/episode axes; behavioral_count, excluded_count (incomplete
        coverage), n (displayed rows) and unavailable_reason. If any episode
        segment is absent, the whole episode is excluded. Available n=0 values
        retain empty logical shapes without zero padding or separator columns.
        Unavailable scaling returns values=None/n=0 but still reports true
        coverage exclusions from source positions. Units remain those selected
        by scale_activity; the input reference and matches are not modified.
    """
    matches = np.asarray(match_positions)
    if matches.ndim not in (1, 2) or not np.issubdtype(matches.dtype, np.integer) or np.any(matches < 0):
        raise ValueError("Match positions must be nonnegative integer central (M,) or episode (E,S) arrays.")
    if matches.ndim == 2 and matches.shape[1] == 0:
        raise ValueError("Episodes need at least one displayed trial segment.")
    positions = np.asarray(scaled_activity["row_positions"])
    _validate_positions(positions)
    lookup = {int(position): index for index, position in enumerate(positions)}
    indices = np.fromiter(
        (lookup.get(int(position), -1) for position in matches.flat),
        dtype=int, count=matches.size,
    ).reshape(matches.shape)
    covered = indices >= 0 if matches.ndim == 1 else (indices >= 0).all(axis=1)
    excluded = int(np.count_nonzero(~covered))
    values = scaled_activity["values"]
    reason = scaled_activity["unavailable_reason"]
    if values is None:
        return {
            "values": None, "row_positions": matches[:0].copy(),
            "behavioral_count": len(matches), "excluded_count": excluded,
            "n": 0, "unavailable_reason": reason,
        }
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != len(positions):
        raise ValueError("Scaled values must match the covered source window axis.")
    return {
        "values": values[indices[covered]], "row_positions": matches[covered].copy(),
        "behavioral_count": len(matches), "excluded_count": excluded,
        "n": int(np.count_nonzero(covered)), "unavailable_reason": reason,
    }


# Small validation helpers; no metadata loading, fitting or behavioral matching.
def _validate_positions(positions: np.ndarray) -> None:
    """Input: unitless source/mapping positions. Return None; require unique integer (W,) >=0."""
    if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions < 0) or len(np.unique(positions)) != len(positions)):
        raise ValueError("row_position/choice mapping must contain unique nonnegative integers with shape (W,).")


def _coverage_bounds(coverage_s: tuple[float, float]) -> tuple[float, float]:
    """Input: trusted (start,end) UTC-second pair. Return finite ordered float bounds or raise ValueError."""
    try:
        bounds = np.asarray(coverage_s, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Trusted coverage must contain two finite ordered UTC-second bounds.") from error
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("Trusted coverage must contain two finite ordered UTC-second bounds.")
    return float(bounds[0]), float(bounds[1])
