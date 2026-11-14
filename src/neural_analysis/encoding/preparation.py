"""Shared two-second unit activity with independent optional descriptive PCA."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.behavior_analysis.project_utils import is_present_value
from .configuration import EncodingConfig
from .predictors import (
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
    """Activity-first data record; UTC/relative seconds and native unit identities.

    unit_counts(W,B,U) owns raw unsmoothed counts, unit_ids(U,) its last axis.
    choice_positions(W,) indexes baseline choices; bin_edges_s(B+1,) spans
    [-1,+1). bin_rows has W*B trial-major rows and window_fold_ids(W,) keeps
    each trial in one block fold, or -1. pca optionally stores full fixed
    scores(W,B,K_full); pc_ranks(K,) retains original numerical nonzero ranks.
    Missing PCA never invalidates unit activity. No arrays are modified here.
    """

    region: str
    probe_id: str
    unit_counts: np.ndarray
    unit_ids: tuple[str, ...]
    choice_positions: np.ndarray
    bin_edges_s: np.ndarray
    bin_rows: pd.DataFrame
    window_fold_ids: np.ndarray
    cv_unavailable_reason: str
    pca: timeline.RegionalTimelinePCA | None
    pc_ranks: np.ndarray
    pca_unavailable_reason: str
    meta: dict


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
    session = resolve_session_metadata(
        load_session_metadata(config.session_metadata), config.session_metadata
    )
    if session.behavior.trial_table_file is None:
        raise ValueError("Session metadata must declare a source trial table")
    source = pd.read_csv(session.behavior.trial_table_file)
    trials = pd.read_csv(config.augmented_trials)
    validate_trial_tables(trials, source)
    baseline = select_baseline_trials(trials)
    origin, choices = timeline.prepare_timeline_choices(trials)
    valid_positions = baseline.loc[baseline.baseline_valid, "row_position"]
    choices = choices.loc[choices.row_position.isin(valid_positions)].reset_index(
        drop=True
    )
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
        session.session_id,
        origin,
        choices,
        events,
        {},
        {},
        {"context_event_diagnostics": diagnose_context_events(trials, events)},
    )
    return session, trials, behavior


def prepare_regional_activity(
    config: EncodingConfig,
    session: ResolvedSession,
    trials: pd.DataFrame,
    behavior: timeline.SessionTimelineResult,
) -> tuple[dict, dict]:
    """Prepare regional counts(W,B,U), explicit rows/folds, and optional PC scores.

    Inputs are validated config/source session, complete UTC trial table and
    baseline behavioral timeline. Require IRIG coverage of [-1,+1) seconds,
    bin selected good/mua units once and assign block folds before outcome
    masking. Return region records and unavailable-region reason mappings.
    PCA uses only these same windows; no files or input arrays are changed.
    """
    width = config.bin_width_ms / 1000
    edges = timeline.choice_bin_edges(width, window_s=config.window_s)
    source_settings = timeline_io.timeline_configuration(
        session, dict(config.probe_roles), width, window_s=config.window_s
    )
    regions = {}
    unavailable = {
        role: "No probe role configured"
        for role in ("PFC", "HPC")
        if role not in config.probe_roles
    }
    for role, probe in config.probe_roles.items():
        try:
            coverage = _irig_coverage(session, probe)
            positions = timeline.select_complete_choice_windows(
                behavior.choices.choice_time.to_numpy(),
                coverage,
                window_s=config.window_s,
            )
            if not len(positions):
                raise ValueError(
                    "No baseline choice windows have complete regional coverage"
                )
            population = timeline_io.load_timeline_population(session, probe)
            counts, _ = timeline.bin_choice_counts(
                population.spike_group,
                population.cluster_ids,
                behavior.choices.iloc[positions].choice_time.to_numpy(),
                width,
                window_s=config.window_s,
            )
            original = behavior.choices.iloc[positions].row_position.to_numpy(dtype=int)
            folds, cv_reason = _window_folds(trials.iloc[original].cur_block)
            rows = build_bin_rows(
                trials,
                behavior.choices,
                positions,
                edges,
                behavior.time_origin_s,
                folds,
            )
            pca = None
            ranks = np.array([], dtype=int)
            pca_reason = "PCs were not requested"
            if "pcs" in config.response_representations:
                try:
                    pca = timeline.fit_timeline_activity(
                        counts,
                        population.unit_ids,
                        positions,
                        role,
                        probe,
                        width,
                        window_s=config.window_s,
                    )
                    ranks = _nonzero_pc_ranks(pca)
                    if not len(ranks):
                        raise ValueError("PCA has no numerically nonzero PCs")
                    pca_reason = ""
                except (ValueError, np.linalg.LinAlgError) as error:
                    pca = None
                    pca_reason = f"PCA unavailable: {error}"
            regions[role] = PreparedRegion(
                role,
                probe,
                counts,
                population.unit_ids,
                positions,
                edges,
                rows,
                folds,
                cv_reason,
                pca,
                ranks,
                pca_reason,
                {
                    "source_configuration": source_settings,
                    "window_s": list(config.window_s),
                    "pca_scope": "fixed descriptive session PCA; includes test-block activity",
                    "randomness_used": False,
                },
            )
        except (OSError, ValueError, KeyError) as error:
            unavailable[role] = str(error)
    return regions, unavailable


def build_bin_rows(
    trials: pd.DataFrame,
    choices: pd.DataFrame,
    choice_positions: np.ndarray,
    bin_edges_s: np.ndarray,
    time_origin_s: float,
    window_fold_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build trial-major bin rows from common geometry, independently of any PCA.

    trials/choices retain original row_position/trial_id/block and UTC choice
    times; positions(W,) indexes choices, edges(B+1,) are relative seconds,
    origin is first-trial UTC seconds and optional folds(W,) are -1 or 0..4.
    Return DataFrame(W*B,C) with original identities, UTC/elapsed/relative
    half-open edges, before/after masks and one fold per trial. Inputs unchanged.
    """
    positions = np.asarray(choice_positions)
    edges = np.asarray(bin_edges_s, dtype=float)
    if (
        positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or np.any(positions < 0)
        or np.any(positions >= len(choices))
        or edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Invalid choice positions or bin edges")
    selected = choices.iloc[positions]
    windows, bins = len(positions), len(edges) - 1
    original = selected.row_position.to_numpy(dtype=int)
    if window_fold_ids is None:
        window_fold_ids, _ = _window_folds(trials.iloc[original].cur_block)
    folds = np.asarray(window_fold_ids)
    if (
        folds.shape != (windows,)
        or not np.issubdtype(folds.dtype, np.integer)
        or not np.isin(folds, [-1, 0, 1, 2, 3, 4]).all()
    ):
        raise ValueError(
            "window_fold_ids must contain one integer fold -1..4 per window"
        )
    starts, ends = np.tile(edges[:-1], windows), np.tile(edges[1:], windows)
    times = np.repeat(selected.choice_time.to_numpy(dtype=float), bins)
    return pd.DataFrame(
        {
            "window_index": np.repeat(np.arange(windows), bins),
            "choice_position": np.repeat(positions, bins),
            "row_position": np.repeat(original, bins),
            "trial_id": np.repeat(selected.trial_id.to_numpy(), bins),
            "block_id": np.repeat(trials.iloc[original].cur_block.to_numpy(), bins),
            "bin_index": np.tile(np.arange(bins), windows),
            "choice_time": times,
            "relative_start_s": starts,
            "relative_end_s": ends,
            "absolute_start_s": times + starts,
            "absolute_end_s": times + ends,
            "elapsed_start_s": times + starts - time_origin_s,
            "elapsed_end_s": times + ends - time_origin_s,
            "before": ends <= 0,
            "after": starts >= 0,
            "baseline_valid": True,
            "coverage_valid": True,
            "fold_id": np.repeat(folds, bins),
        }
    )


def response_activity(
    region: PreparedRegion, representation: str
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """Return activity(W,B,R), original response IDs(R,), and physical-unit label.

    PreparedRegion owns count geometry. Units divide counts by bin width in
    seconds to obtain unsmoothed Hz; PCs select fixed original nonzero ranks.
    An absent PCA raises ValueError only for the PC representation. No fitting,
    file reads or input mutation occurs.
    """
    if representation == "units":
        return (
            region.unit_counts / (region.bin_edges_s[1] - region.bin_edges_s[0]),
            region.unit_ids,
            "Hz",
        )
    if representation != "pcs":
        raise ValueError("response representation must be units or pcs")
    if region.pca is None:
        raise ValueError(region.pca_unavailable_reason or "PCA unavailable")
    return (
        region.pca.scores[:, :, region.pc_ranks - 1],
        tuple(f"PC{rank}" for rank in region.pc_ranks),
        "PC score",
    )


# Small source and numeric helpers.
def _irig_coverage(session: ResolvedSession, probe: str) -> tuple[float, float]:
    """Input resolved session/probe ID; return trusted start/end UTC seconds, no spike read."""
    source = resolve_probe_sources(session, probe)
    if source.alignment_file is None:
        raise ValueError(
            "Missing aligned spike archive; trusted IRIG coverage required"
        )
    with np.load(source.alignment_file, allow_pickle=False) as archive:
        if "irig_utc_unix" not in archive.files:
            raise ValueError(
                "Manual alignment is unsupported: trusted IRIG coverage is required"
            )
    coverage = inspect_probe_coverage(source)
    return coverage.start_s, coverage.end_s


def _nonzero_pc_ranks(regional: timeline.RegionalTimelinePCA) -> np.ndarray:
    """Input fixed (W,B,K) PCA; return (K_fit,) one-based original numerical nonzero ranks."""
    observations = int(np.prod(regional.scores.shape[:2]))
    units = len(regional.transform.retained_unit_ids)
    singular = np.sqrt(
        np.maximum(regional.transform.explained_variance, 0) * (observations - 1)
    )
    tolerance = (
        max(observations, units) * np.finfo(np.float64).eps * singular.max(initial=0)
    )
    return np.flatnonzero(singular > tolerance) + 1


def _window_folds(blocks: pd.Series) -> tuple[np.ndarray, str]:
    """Input (W,) native numeric block IDs; return (W,) folds 0..4 or -1 plus reason, no units."""
    groups = pd.to_numeric(
        blocks.where(is_present_value(blocks)), errors="coerce"
    ).to_numpy(dtype=float)
    missing = np.full(len(groups), -1, dtype=int)
    if not np.isfinite(groups).all():
        return missing, "Missing or invalid cur_block IDs in the regional pool"
    if len(np.unique(groups)) < 5:
        return missing, "Five grouped folds require at least 5 usable regional blocks"
    folds = missing.copy()
    splitter = GroupKFold(n_splits=5, shuffle=False)
    for fold, (_, held_out) in enumerate(
        splitter.split(np.zeros((len(groups), 1)), groups=groups)
    ):
        folds[held_out] = fold
    return folds, ""
