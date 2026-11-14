"""Source loading and shared regional rates; PCA belongs only inside CV folds."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import normalize_experimenter_reward_column
from src.neural_analysis.encoding.predictors import select_baseline_trials
from src.neural_analysis.encoding.preparation import build_bin_rows
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata, resolve_probe_sources
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage

from .configuration import PopulationDecodingConfig
from .targets import PopulationTargets, assign_block_folds, build_population_targets, validate_trial_tables


@dataclass(frozen=True)
class PreparedPopulationRegion:
    """Native rates_hz(W,B,U), IDs(U), original row_positions(W), edges(B+1) seconds.

    bin_rows(W*B,C) retains UTC/elapsed/choice-relative half-open intervals and
    the shared fold map. meta records source settings and coverage; no PCA.
    """

    region: str
    probe_id: str
    rates_hz: np.ndarray
    unit_ids: tuple[str, ...]
    row_positions: np.ndarray
    bin_edges_s: np.ndarray
    bin_rows: pd.DataFrame
    meta: dict


@dataclass(frozen=True)
class PreparedPopulation:
    """Self-contained chronological session inputs; no fitted estimators.

    trials(N,C), baseline(N,) bool, targets(N,5), fold_ids(N,) -1..4 retain full
    source row order. behavior holds UTC origin, baseline choices and events;
    regions hold independent WBU Hz arrays. unavailable maps region to reason.
    """

    session_id: str
    trials: pd.DataFrame
    behavior: timeline.SessionTimelineResult
    baseline: np.ndarray
    targets: PopulationTargets
    fold_ids: np.ndarray
    regions: dict[str, PreparedPopulationRegion]
    unavailable: dict[str, str]
    meta: dict


def load_behavior_inputs(config: PopulationDecodingConfig) -> tuple:
    """Load small source tables/config; return session, trials, behavior, masks/labels/folds.

    Tables remain complete/chronological with UTC-second times. Behavior choices
    are baseline-filtered, with original row_position/trial_id. No spikes are
    read, no PCA fitted, and no files written. Missing correctness is accepted.
    """
    session = resolve_session_metadata(load_session_metadata(config.session_metadata), config.session_metadata)
    if session.behavior.trial_table_file is None:
        raise ValueError("Session metadata must declare the source trial table")
    trials = normalize_experimenter_reward_column(pd.read_csv(config.augmented_trials))
    validate_trial_tables(trials, pd.read_csv(session.behavior.trial_table_file))
    baseline = select_baseline_trials(trials).baseline_valid.to_numpy(dtype=bool)
    targets = build_population_targets(trials.state_int, trials.action, trials.reward,
                                       trials.experimenter_reward_given)
    folds, reason = assign_block_folds(trials.cur_block, baseline)
    origin, choices = timeline.prepare_timeline_choices(trials)
    choices = choices.loc[baseline[choices.row_position.to_numpy(dtype=int)]].reset_index(drop=True)
    events = pd.DataFrame(columns=["Time", "Event", "elapsed_s"])
    path = session.behavior.event_table_file
    if path is not None and path.is_file():
        events = pd.read_csv(path)
        if not {"Time", "Event"} <= set(events):
            raise ValueError("Event table needs Time/Event columns")
        events["Time"] = pd.to_numeric(events.Time, errors="coerce")
        if not np.isfinite(events.Time).all():
            raise ValueError("Event times must be finite UTC seconds")
        events["elapsed_s"] = events.Time - origin
    behavior = timeline.SessionTimelineResult(session.session_id, origin, choices, events, {}, {}, {"fold_reason": reason})
    return session, trials, behavior, baseline, targets, folds


def inspect_sources(config: PopulationDecodingConfig) -> dict[str, str]:
    """Inspect config/source paths and NPZ headers; return region blockers, no spike reads.

    IRIG member presence is required; source paths carry no units. Used by
    dry-run and preparation. Coverage values are inspected only in preparation.
    """
    session = resolve_session_metadata(load_session_metadata(config.session_metadata), config.session_metadata)
    blockers = {}
    for role, probe in config.probe_roles.items():
        try:
            source = resolve_probe_sources(session, probe)
            for field in ("alignment_file", "cluster_info_file", "channel_quality_file", "spike_clusters_file"):
                path = getattr(source, field)
                if path is None or not path.is_file():
                    raise ValueError(f"Missing {field}")
            with np.load(source.alignment_file, allow_pickle=False) as archive:
                if not {"irig_utc_unix", "spike_utc_unix"} <= set(archive.files):
                    raise ValueError("IRIG-backed alignment required; manual alignment is deferred")
        except (OSError, ValueError, KeyError) as error:
            blockers[role] = str(error)
    return blockers


def prepare_session(config: PopulationDecodingConfig) -> PreparedPopulation:
    """Bin independent regional windows from config; return shared native Hz preparation.

    Counts(W,B,U) become rates by division by bin-width seconds. Only complete
    IRIG-covered [-1,+1) windows survive. Global baseline folds precede regional
    coverage and target masking. No PCA, estimators, or output writes occur.
    """
    session, trials, behavior, baseline, targets, folds = load_behavior_inputs(config)
    unavailable = inspect_sources(config)
    unavailable.update({role: "No probe role configured" for role in ("PFC", "HPC") if role not in config.probe_roles})
    regions = {}
    width = config.bin_width_ms / 1000
    edges = timeline.choice_bin_edges(width, window_s=config.window_s)
    settings = timeline_io.timeline_configuration(session, dict(config.probe_roles), width, window_s=config.window_s)
    for role, probe in config.probe_roles.items():
        if role in unavailable:
            continue
        try:
            coverage = inspect_probe_coverage(resolve_probe_sources(session, probe))
            positions = timeline.select_complete_choice_windows(behavior.choices.choice_time.to_numpy(),
                (coverage.start_s, coverage.end_s), window_s=config.window_s)
            if not len(positions):
                raise ValueError("No complete baseline windows in trusted regional coverage")
            population = timeline_io.load_timeline_population(session, probe)
            counts, _ = timeline.bin_choice_counts(population.spike_group, population.cluster_ids,
                behavior.choices.iloc[positions].choice_time.to_numpy(), width, window_s=config.window_s)
            original = behavior.choices.iloc[positions].row_position.to_numpy(dtype=int)
            rows = build_bin_rows(trials, behavior.choices, positions, edges, behavior.time_origin_s, folds[original])
            regions[role] = PreparedPopulationRegion(role, probe, counts / width, tuple(population.unit_ids),
                original, edges, rows, {"coverage_utc_s": [coverage.start_s, coverage.end_s], "source_configuration": settings})
        except (OSError, ValueError, KeyError) as error:
            unavailable[role] = str(error)
    return PreparedPopulation(session.session_id, trials, behavior, baseline, targets, folds, regions,
                              unavailable, {"bin_width_ms": config.bin_width_ms, "window_s": list(config.window_s)})
