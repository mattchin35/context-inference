"""Compose existing spike binning and full PCA with strict context-analysis pools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.behavior_analysis.project_utils import is_present_value
from src.neural_analysis.encoding.configuration import EncodingConfig, TIMING_MODES
from src.neural_analysis.encoding.predictors import (
    build_context_values,
    diagnose_context_events,
    select_baseline_trials,
    validate_trial_tables,
)
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import (
    ResolvedSession,
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
)
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage


@dataclass(frozen=True)
class PreparedRegion:
    """One immutable descriptive basis plus its regression row metadata.

    timeline retains the complete RegionalTimelinePCA, including original PC
    signs, full spectrum, unit means/scales and scores (W,B,K_full). pc_ranks
    is (K_fit,), one-based original ranks passing the numerical singular-value
    threshold; indexing scores uses pc_ranks-1 without renumbering the basis.
    bin_rows has W*B rows in trial-major/bin-minor order. context_values and
    context_valid map each timing identifier to (W,B) dimensionless float64
    states and Boolean masks. window_fold_ids is (W,), zero-based folds or -1
    when CV is unavailable. cv_unavailable_reason is empty for valid splits.
    reused identifies an exact saved basis; meta describes scientific inputs.
    """

    timeline: timeline.RegionalTimelinePCA
    pc_ranks: np.ndarray
    bin_rows: pd.DataFrame
    context_values: dict[str, np.ndarray]
    context_valid: dict[str, np.ndarray]
    window_fold_ids: np.ndarray
    cv_unavailable_reason: str
    reused: bool
    meta: dict


# Public loading and computations.
def load_behavior_inputs(
    config: EncodingConfig,
) -> tuple[ResolvedSession, pd.DataFrame, timeline.SessionTimelineResult]:
    """Load small inputs, validate complete correspondence, and retain strict choices.

    Parameters
    ----------
    config : EncodingConfig
        Validated paths to metadata, augmented trials and feature parameters.

    Returns
    -------
    tuple[ResolvedSession, pd.DataFrame, SessionTimelineResult]
        Resolved metadata, complete augmented table (T,C), and a behavior-only
        timeline with no regional activity. Source times are UTC seconds; its
        origin is the first unfiltered trial start. Choices retain original
        row_position/trial_id/reward and elapsed seconds after baseline
        exclusion. Events retain original UTC times plus elapsed_s. Advisory
        state-order messages live in meta['context_event_diagnostics'].
        Conflicting complete trial tables raise ValueError before spike loading.
    """
    session = resolve_session_metadata(load_session_metadata(config.session_metadata), config.session_metadata)
    if session.behavior.trial_table_file is None:
        raise ValueError("Session metadata must declare a source trial table")
    source = pd.read_csv(session.behavior.trial_table_file)
    trials = pd.read_csv(config.augmented_trials)
    validate_trial_tables(trials, source)
    baseline = select_baseline_trials(trials)
    origin, choices = timeline.prepare_timeline_choices(trials)
    valid_positions = baseline.loc[baseline.baseline_valid, "row_position"]
    choices = choices.loc[choices.row_position.isin(valid_positions)].reset_index(drop=True)
    events = pd.DataFrame(columns=["Time", "Event", "elapsed_s"])
    path = session.behavior.event_table_file
    if path is not None and path.is_file():
        events = pd.read_csv(path)
        if not {"Time", "Event"}.issubset(events.columns):
            raise ValueError("Source event table must contain Time and Event columns")
        events["Time"] = pd.to_numeric(events["Time"], errors="coerce")
        if not np.isfinite(events["Time"]).all():
            raise ValueError("Source event times must be finite UTC seconds")
        events["elapsed_s"] = events["Time"] - origin
    behavior = timeline.SessionTimelineResult(
        session.session_id, origin, choices, events, {}, {},
        {"context_event_diagnostics": diagnose_context_events(trials, events)},
    )
    return session, trials, behavior


def prepare_regional_activity(
    config: EncodingConfig,
    session: ResolvedSession,
    trials: pd.DataFrame,
    behavior: timeline.SessionTimelineResult,
) -> tuple[dict[str, PreparedRegion], dict[str, str]]:
    """Prepare independent fully covered regional PC pools before model masking.

    Parameters
    ----------
    config : EncodingConfig
        Native bin width and PFC/HPC probe assignments. Estimator, penalty,
        fit-window and predictor selections do not alter PCA preparation.
    session : ResolvedSession
        Resolved source paths corresponding to the validated behavior inputs.
    trials : pd.DataFrame
        Complete chronological (T,C) augmented table; UTC-second timestamps.
    behavior : SessionTimelineResult
        Strict baseline choices (W_shared,C), full unfiltered origin/events.

    Returns
    -------
    tuple[dict[str, PreparedRegion], dict[str, str]]
        Available PFC/HPC results and unavailable-region reasons. Each result
        uses one fixed, full unwhitened PCA on its regional full-coverage pool;
        regression uses only numerical nonzero original PCs. Five grouped
        window folds are assigned before predictor/window masks. Invalid CV
        preserves activity. Missing IRIG/manual alignment, no covered choices,
        and no variable activity make only that region unavailable. No files
        are written and candidate saved results are never changed.
    """
    width_s = config.bin_width_ms / 1000.0
    edges = timeline.choice_bin_edges(width_s)
    current = timeline_io.timeline_configuration(session, dict(config.probe_roles), width_s)
    candidate, candidate_reason = _load_candidate(config)
    # Build both variants on all original rows before choosing regional windows.
    full_context = {mode: build_context_values(trials, edges, mode) for mode in TIMING_MODES}
    regions = {}
    unavailable = {role: "No probe role configured" for role in ("PFC", "HPC") if role not in config.probe_roles}
    for role, probe in config.probe_roles.items():
        try:
            coverage = _irig_coverage(session, probe)
            positions = timeline.select_complete_choice_windows(
                behavior.choices.choice_time.to_numpy(dtype=float), coverage,
            )
            if not len(positions):
                raise ValueError("No baseline choice windows have complete regional coverage")
            selection = current["selection"][role]
            if "unavailable" in selection:
                raise ValueError(selection["unavailable"])
            saved, reuse_reason = _compatible_region(
                candidate, current, behavior, positions, role, probe, edges,
            )
            reused = saved is not None
            if saved is None:
                population = timeline_io.load_timeline_population(session, probe)
                saved = timeline.compute_timeline_pca(
                    population.spike_group, population.cluster_ids, behavior.choices,
                    positions, role, probe, width_s,
                )
            ranks = _nonzero_pc_ranks(saved)
            if not len(ranks):
                raise ValueError("Regional PCA has no numerically nonzero PCs")
            original_positions = behavior.choices.iloc[positions].row_position.to_numpy(dtype=int)
            values = {mode: arrays[0][original_positions] for mode, arrays in full_context.items()}
            validity = {mode: arrays[1][original_positions] for mode, arrays in full_context.items()}
            folds, cv_reason = _window_folds(trials.iloc[original_positions]["cur_block"])
            rows = build_bin_rows(trials, behavior.choices, saved, behavior.time_origin_s, folds)
            regions[role] = PreparedRegion(
                saved, ranks, rows, values, validity, folds, cv_reason, reused,
                {"source_configuration": current,
                 "reuse_reason": "exact compatible basis" if reused else candidate_reason or reuse_reason,
                 "pca_scope": "fixed descriptive session axes; includes regression test blocks",
                 "pc_rank_rule": "s > max(M,U)*eps(float64)*max(s)",
                 "randomness_used": False},
            )
        except (OSError, ValueError, KeyError) as error:
            unavailable[role] = str(error)
    return regions, unavailable


def build_bin_rows(
    trials: pd.DataFrame,
    choices: pd.DataFrame,
    regional: timeline.RegionalTimelinePCA,
    time_origin_s: float,
    window_fold_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Flatten regional windows to explicit observation identities and time edges.

    Parameters
    ----------
    trials : pd.DataFrame
        Complete (T,C) original augmented table with cur_block.
    choices : pd.DataFrame
        Shared baseline (W_shared,C) table with original row_position/trial_id
        and choice_time in UTC seconds.
    regional : RegionalTimelinePCA
        Regional (W,B,K) scores with choice_positions selecting shared choices
        and choice-relative bin_edges_s (B+1,).
    time_origin_s : float
        First unfiltered trial-start UTC seconds.
    window_fold_ids : np.ndarray or None
        Optional (W,) integer fold IDs 0..4 or -1 for unavailable CV. Passing
        the precomputed regional map avoids splitting twice. None computes
        the same deterministic five-fold map from these regional windows.

    Returns
    -------
    pd.DataFrame
        W*B rows ordered trial-major/bin-minor with original identity/block,
        shared choice_position, regional window_index, bin_index, absolute and
        elapsed/relative start/end seconds. before/after are disjoint Boolean
        fit-window masks; post includes bins starting exactly at zero. All rows
        have baseline_valid/coverage_valid=True; these are extracted full windows.
        fold_id repeats the regional window's fold unchanged across all bins.
    """
    selected = choices.iloc[regional.choice_positions]
    windows, bins = regional.scores.shape[:2]
    edges = regional.bin_edges_s
    if windows != len(selected) or len(edges) != bins + 1:
        raise ValueError("Regional scores, choices, and bin edges disagree")
    positions = selected.row_position.to_numpy(dtype=int)
    if window_fold_ids is None:
        window_fold_ids, _ = _window_folds(trials.iloc[positions].cur_block)
    folds = np.asarray(window_fold_ids)
    if (folds.shape != (windows,) or not np.issubdtype(folds.dtype, np.integer)
            or not np.isin(folds, [-1, 0, 1, 2, 3, 4]).all()):
        raise ValueError("window_fold_ids must contain one integer fold -1..4 per window")
    starts = np.tile(edges[:-1], windows)
    ends = np.tile(edges[1:], windows)
    choice_times = np.repeat(selected.choice_time.to_numpy(dtype=float), bins)
    absolute_starts = choice_times + starts
    absolute_ends = choice_times + ends
    return pd.DataFrame({
        "window_index": np.repeat(np.arange(windows), bins),
        "choice_position": np.repeat(regional.choice_positions, bins),
        "row_position": np.repeat(positions, bins),
        "trial_id": np.repeat(selected.trial_id.to_numpy(), bins),
        "block_id": np.repeat(trials.iloc[positions].cur_block.to_numpy(), bins),
        "bin_index": np.tile(np.arange(bins), windows),
        "choice_time": choice_times,
        "relative_start_s": starts, "relative_end_s": ends,
        "absolute_start_s": absolute_starts, "absolute_end_s": absolute_ends,
        "elapsed_start_s": absolute_starts - time_origin_s,
        "elapsed_end_s": absolute_ends - time_origin_s,
        "before": ends <= 0, "after": starts >= 0,
        "baseline_valid": True, "baseline_invalid_reason": "", "coverage_valid": True,
        "fold_id": np.repeat(folds, bins),
    })


# Source-boundary and numeric helpers.
def _irig_coverage(session: ResolvedSession, probe: str) -> tuple[float, float]:
    """Input resolved session/probe ID; return trusted start/end UTC seconds, no spike read."""
    source = resolve_probe_sources(session, probe)
    if source.alignment_file is None:
        raise ValueError("Missing aligned spike archive; trusted IRIG coverage required")
    with np.load(source.alignment_file, allow_pickle=False) as archive:
        if "irig_utc_unix" not in archive.files:
            raise ValueError("Manual alignment is unsupported: trusted IRIG coverage is required")
    coverage = inspect_probe_coverage(source)
    return coverage.start_s, coverage.end_s


def _load_candidate(config: EncodingConfig) -> tuple[timeline.SessionTimelineResult | None, str]:
    """Input config; return trusted saved timeline or None with diagnostic string, no unit changes."""
    if config.timeline_result is None:
        return None, "No saved timeline requested"
    try:
        return timeline_io.load_timeline_result(config.timeline_result), ""
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        return None, f"Saved timeline cannot be reused: {error}"


def _compatible_region(
    candidate: timeline.SessionTimelineResult | None,
    current: dict,
    behavior: timeline.SessionTimelineResult,
    positions: np.ndarray,
    role: str,
    probe: str,
    edges: np.ndarray,
) -> tuple[timeline.RegionalTimelinePCA | None, str]:
    """Check saved/current records and regional (W,) choice IDs; return remapped PCA or reason.

    All timestamps and edges are seconds, scores are fixed (W,B,K), unit order
    and scaling stay native. This function loads no spikes and never refits.
    """
    if candidate is None:
        return None, "No readable saved timeline"
    if any(candidate.meta.get(key) != value for key, value in current.items()):
        return None, "Saved timeline scientific source settings differ"
    if candidate.session_id != behavior.session_id or candidate.time_origin_s != behavior.time_origin_s:
        return None, "Saved timeline session identity or origin differs"
    saved = candidate.regions.get(role)
    if saved is None or saved.probe_id != probe:
        return None, "Saved timeline lacks the requested regional probe"
    if not np.array_equal(saved.bin_edges_s, edges):
        return None, "Saved bin geometry differs"
    old_positions = saved.choice_positions
    if (old_positions.ndim != 1 or not np.issubdtype(old_positions.dtype, np.integer)
            or np.any(old_positions < 0) or np.any(old_positions >= len(candidate.choices))):
        return None, "Saved regional choice positions are invalid"
    previous = candidate.choices.iloc[old_positions]
    expected = behavior.choices.iloc[positions]
    for column in ("row_position", "trial_id", "choice_time"):
        if column not in previous or not np.array_equal(previous[column].to_numpy(), expected[column].to_numpy()):
            return None, f"Saved regional {column} pool differs"
    fitted = saved.transform
    unit_ids = tuple(current["selection"][role]["unit_ids"])
    observations = len(positions) * (len(edges) - 1)
    components = min(observations, len(fitted.retained_unit_ids))
    if (fitted.input_unit_ids != unit_ids or fitted.n_training_observations != observations
            or fitted.actual_components != components or fitted.whiten
            or saved.scores.shape != (len(positions), len(edges) - 1, components)
            or not np.isfinite(saved.scores).all() or not np.all(fitted.training_scale > 0)):
        return None, "Saved PCA unit order, full basis, or scaling differs"
    if candidate.meta.get("units", {}).get("scores") != "unwhitened standardized-unit scores":
        return None, "Saved PCA score scaling is unsupported"
    return replace(saved, choice_positions=positions.copy()), ""


def _nonzero_pc_ranks(regional: timeline.RegionalTimelinePCA) -> np.ndarray:
    """Input fixed (W,B,K) PCA; return (K_fit,) one-based original numerical nonzero ranks."""
    observations = int(np.prod(regional.scores.shape[:2]))
    units = len(regional.transform.retained_unit_ids)
    singular = np.sqrt(np.maximum(regional.transform.explained_variance, 0) * (observations - 1))
    tolerance = max(observations, units) * np.finfo(np.float64).eps * singular.max(initial=0)
    return np.flatnonzero(singular > tolerance) + 1


def _window_folds(blocks: pd.Series) -> tuple[np.ndarray, str]:
    """Input (W,) native numeric block IDs; return (W,) folds 0..4 or -1 plus reason, no units."""
    groups = pd.to_numeric(blocks.where(is_present_value(blocks)), errors="coerce").to_numpy(dtype=float)
    missing = np.full(len(groups), -1, dtype=int)
    if not np.isfinite(groups).all():
        return missing, "Missing or invalid cur_block IDs in the regional pool"
    if len(np.unique(groups)) < 5:
        return missing, "Five grouped folds require at least 5 usable regional blocks"
    folds = missing.copy()
    splitter = GroupKFold(n_splits=5, shuffle=False)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros((len(groups), 1)), groups=groups)):
        folds[held_out] = fold
    return folds, ""
