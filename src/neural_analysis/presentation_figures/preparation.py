"""Prepare aligned, good-unit PFC/HPC counts and shared task-block folds."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import normalize_experimenter_reward_column
from src.neural_analysis.communication_subspace.preparation import _overlapping_indices
from src.neural_analysis.population import session_timeline, session_timeline_io
from src.neural_analysis.population.run_session_timeline import resolve_timeline_probe_roles
from src.neural_analysis.population_decoding.targets import (
    PopulationTargets,
    assign_block_folds,
    build_population_targets,
    numeric_column,
    validate_trial_tables,
)
from src.neural_analysis.session_metadata import (
    ResolvedSession,
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
)
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage
from src.neural_analysis.task_decoding.conditions import build_condition_masks


REGIONS = ("PFC", "HPC")
REGIONAL_WINDOW_S = (-1., 1.)
PAIRED_WINDOW_S = (-2., 2.)
WINDOW_KEYS = ((REGIONAL_WINDOW_S, 100), (REGIONAL_WINDOW_S, 500), (PAIRED_WINDOW_S, 100))
CONDITIONS = ("correct_rewarded", "omission", "incorrect")
POPULATION_POLICY = {"population_quality": "good", "missing_label_policy": "exclude"}


@dataclass(frozen=True)
class BlockSplit:
    """Fold identity and 1D integer indices on the corresponding choice axis.

    train_indices, test_indices and purged_indices are sorted and dimensionless.
    Purged candidates intersect the test choice windows in physical time.
    """

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray


@dataclass(frozen=True)
class PreparedWindow:
    """A shared (choice, bin, unit) spike-count tensor per region, in spikes/bin.

    row_positions (choice,) gives original table row positions. bin_edges_s
    (bin+1,) is choice-relative seconds and defines half-open bins. Splits
    index the choice axis. counts is empty during preflight or missing coverage.
    """

    counts: dict[str, np.ndarray]
    row_positions: np.ndarray
    bin_edges_s: np.ndarray
    splits: tuple[BlockSplit, ...]


@dataclass(frozen=True)
class PreparedChoices:
    """Full chronological trial/event tables and shared regional inputs.

    baseline and condition_masks are Boolean (source trial,), fold_ids is
    integer (source trial,) with -1 for missing assignment, and targets
    retains the same trial axis. choices is the ordinary completed subset
    linked by row_position. Clock timestamps and coverage bounds are UTC Unix
    seconds, elapsed timestamps are seconds from time_origin_s. windows keys
    are (relative-second bounds, integer width in ms). unit_ids is ordered
    along each regional unit axis. unavailable gives source or analysis
    readiness reasons, and meta records source identities and trial audit.
    """

    session_id: str
    session: ResolvedSession
    time_origin_s: float
    trials: pd.DataFrame
    choices: pd.DataFrame
    events: pd.DataFrame
    baseline: np.ndarray
    targets: PopulationTargets
    condition_masks: dict[str, np.ndarray]
    fold_ids: np.ndarray
    unit_ids: dict[str, tuple[str, ...]]
    coverage_s: dict[str, tuple[float, float]]
    probe_roles: dict[str, str]
    windows: dict[tuple[tuple[float, float], int], PreparedWindow]
    unavailable: dict[str, str]
    meta: dict


def prepare_choices(
    metadata_path: Path | str, *, probe_roles: dict[str, str] | None = None,
) -> PreparedChoices:
    """Load a live metadata path and return joint choice counts at fixed widths.

    Optional roles map PFC/HPC to distinct metadata probe keys. Each raw
    population loads at most once. See PreparedChoices for axes and units.
    """
    prepared = preflight(metadata_path, probe_roles=probe_roles)
    if not any(len(window.row_positions) for window in prepared.windows.values()):
        return prepared

    unavailable = prepared.unavailable.copy()
    populations = {}
    for region, probe in prepared.probe_roles.items():
        try:
            population = session_timeline_io.load_timeline_population(
                prepared.session, probe, **POPULATION_POLICY,
            )
            if (population.unit_ids != prepared.unit_ids[region]
                    or population.coverage_s != prepared.coverage_s[region]):
                raise ValueError("Selected unit IDs or recording bounds changed during loading.")
            populations[region] = population
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            unavailable[region] = str(error)
    if set(populations) != set(REGIONS):
        return replace(prepared, unavailable=unavailable)

    windows = prepared.windows.copy()
    for (bounds, width_ms), window in prepared.windows.items():
        if not len(window.row_positions):
            continue
        times_s = numeric_column(prepared.trials.choice_time.iloc[window.row_positions])
        count_by_region = {}
        for region, population in populations.items():
            try:
                count_by_region[region] = _bin_region(population, times_s, width_ms / 1000, bounds)
            except (ValueError, RuntimeError) as error:
                unavailable[f"{region}_{bounds[1]:g}s_{width_ms}ms"] = str(error)
        if set(count_by_region) == set(REGIONS):
            windows[bounds, width_ms] = replace(window, counts=count_by_region)
    return replace(prepared, windows=windows, unavailable=unavailable)


def make_block_splits(
    choice_times_s: np.ndarray, fold_ids: np.ndarray, *, window_s: tuple[float, float],
) -> tuple[BlockSplit, ...]:
    """Build five purged splits from already assigned fold IDs on the choice axis.

    Both arrays are (choice,). choice_times_s is finite chronological UTC
    seconds. Integer fold IDs are 0..4 or -1 (unassigned). window_s contains
    finite ascending relative-second bounds. Return five index records, with
    empty train/test and purge arrays if a test fold has no eligible choices.
    """
    times = np.asarray(choice_times_s, dtype=float)
    folds = np.asarray(fold_ids)
    if (times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0)
            or times.shape != folds.shape or not np.issubdtype(folds.dtype, np.integer)
            or np.any((folds < -1) | (folds >= 5))):
        raise ValueError("Expected chronological choice times and aligned integer fold IDs -1..4.")
    if (np.asarray(window_s).shape != (2,) or not np.isfinite(window_s).all()
            or window_s[0] >= window_s[1]):
        raise ValueError("Choice window must have finite ascending bounds in seconds.")
    splits = []
    for fold_id in range(5):
        test = np.flatnonzero(folds == fold_id)
        candidates = np.flatnonzero((folds >= 0) & (folds != fold_id)) if test.size else test
        purged = _overlapping_indices(candidates, test, times, window_s)
        train = np.setdiff1d(candidates, purged, assume_unique=True)
        splits.append(BlockSplit(fold_id, train, test, purged))
    return tuple(splits)


def preflight(
    metadata_path: Path | str, *, probe_roles: dict[str, str] | None = None,
) -> PreparedChoices:
    """Inspect actual source files and readiness without opening raw spike arrays.

    metadata_path is a live session JSON path. Optional roles map PFC/HPC to
    distinct probes. Return PreparedChoices with original axes, UTC seconds,
    joint coverage, folds and empty count dictionaries. Reads only source
    tables, alignment headers and IRIG bounds. Writes no files.
    """
    path = Path(metadata_path).expanduser().resolve()
    session = resolve_session_metadata(load_session_metadata(path), path)
    roles = resolve_timeline_probe_roles(session, probe_roles)
    if set(roles) != set(REGIONS) or len(set(roles.values())) != len(REGIONS):
        raise ValueError("PFC and HPC require distinct configured probe roles.")

    trial_path = session.behavior.trial_table_file
    if trial_path is None or not trial_path.is_file():
        raise ValueError("Missing required trial table source.")
    trials = normalize_experimenter_reward_column(pd.read_csv(trial_path))
    validate_trial_tables(trials, trials)
    trials["row_position"] = np.arange(len(trials))
    time_origin_s, choices = session_timeline.prepare_timeline_choices(trials)
    if "correct" not in trials:
        trials["correct"] = np.nan
    baseline = trials.row_position.isin(choices.row_position).to_numpy()
    targets = build_population_targets(
        trials.state_int, trials.action, trials.reward, trials.experimenter_reward_given,
    )
    condition_masks, trial_audit = _trial_groups(trials, baseline, targets)
    # Assign the task blocks on the complete chronology before any readout filter.
    fold_ids, fold_reason = assign_block_folds(trials.cur_block, np.ones(len(trials), dtype=bool))

    paths = [path, trial_path, session.behavior.event_table_file, session.behavior.treadmill_file]
    unit_ids, coverage_s, unavailable = {}, {}, {}
    for region, probe in roles.items():
        try:
            source = resolve_probe_sources(session, probe)
            required_sources = {
                field: getattr(source, field) for field in (
                    "alignment_file", "spike_clusters_file", "cluster_info_file", "channel_quality_file",
                )
            }
            paths.extend(required_sources.values())
            for field, source_path in required_sources.items():
                if source_path is None or not source_path.is_file():
                    raise ValueError(f"Missing {field}: {source_path}")
            _, _, selected = session_timeline_io.select_timeline_population(
                session, probe, **POPULATION_POLICY,
            )
            coverage = inspect_probe_coverage(source)
            if coverage.source != "irig_utc_unix" or coverage.start_s >= coverage.end_s:
                raise ValueError("Positive-duration IRIG recording coverage is required.")
            unit_ids[region] = tuple(f"{probe}:{unit}" for unit in selected.cluster_id)
            coverage_s[region] = (coverage.start_s, coverage.end_s)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            unavailable[region] = str(error)

    windows = _shared_windows(choices, coverage_s, fold_ids)
    if fold_reason:
        unavailable["folds"] = fold_reason
    for name, bounds in (("regional_window", REGIONAL_WINDOW_S), ("paired_window", PAIRED_WINDOW_S)):
        if not len(windows[bounds, 100].row_positions):
            unavailable[name] = (
                f"No ordinary choices with full joint coverage for [{bounds[0]:g}, {bounds[1]:g}) s."
            )
    if len(unit_ids) == len(REGIONS) and min(map(len, unit_ids.values())) < 4:
        unavailable["prediction"] = "At least four good units per region are required for matched draws."
    events, event_reason = _read_events(session.behavior.event_table_file, time_origin_s)
    meta = {
        "trial_audit": trial_audit,
        "event_reason": event_reason,
        "sources": [session_timeline_io._file_identity(source) for source in paths if source is not None],
        **POPULATION_POLICY,
    }
    return PreparedChoices(
        session.session_id, session, time_origin_s, trials, choices, events, baseline, targets,
        condition_masks, fold_ids, unit_ids, coverage_s, roles, windows, unavailable, meta,
    )


def _trial_groups(
    trials: pd.DataFrame, baseline: np.ndarray, targets: PopulationTargets,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Return (N,) behavior masks and unitless counts on the full (N, C) trial table."""
    columns = ("correct", "reward", "action", "experimenter_reward_given")
    numeric_trials = pd.DataFrame({name: numeric_column(trials[name]) for name in columns})
    masks = build_condition_masks(numeric_trials, CONDITIONS)
    groups = {"overall": baseline.copy(), **{name: mask & baseline for name, mask in masks.items()}}
    audit = {
        "manual_rewards": int((numeric_trials.experimenter_reward_given.notna()
                               & numeric_trials.experimenter_reward_given.ne(0)).sum()),
        "unknown_manual_flag": int(numeric_trials.experimenter_reward_given.isna().sum()),
        "unknown_context": int(np.count_nonzero(baseline & ~targets.valid[:, 0])),
        "unknown_action": int(np.count_nonzero(~np.isin(numeric_trials.action, [0, 1]))),
        "unknown_reward": int(np.count_nonzero(baseline & ~np.isin(numeric_trials.reward, [0, 1]))),
        "unknown_correct": int(np.count_nonzero(baseline & ~np.isin(numeric_trials.correct, [0, 1]))),
        "incorrect_rewarded": int(np.count_nonzero(groups["incorrect"] & numeric_trials.reward.eq(1))),
    }
    return groups, audit


def _read_events(path: Path | None, origin_s: float) -> tuple[pd.DataFrame, str]:
    """Read optional event CSV into (event, column) UTC/elapsed seconds and a status string."""
    empty = pd.DataFrame(columns=["Time", "Event", "Note", "elapsed_s"])
    if path is None or not path.is_file():
        return empty, "Missing event table."
    try:
        events = pd.read_csv(path)
        if not {"Time", "Event"} <= set(events):
            raise ValueError("Event table needs Time and Event columns.")
        events["Time"] = numeric_column(events.Time)
        if not np.isfinite(events.Time).all():
            raise ValueError("Event times must be finite UTC seconds.")
    except (OSError, ValueError) as error:
        return empty, str(error)
    events = events.sort_values("Time", kind="stable", ignore_index=True)
    events["elapsed_s"] = events.Time - origin_s
    return events, ""


def _shared_windows(
    choices: pd.DataFrame, coverage_s: dict[str, tuple[float, float]], fold_ids: np.ndarray,
) -> dict[tuple[tuple[float, float], int], PreparedWindow]:
    """Build coverage and splits for each width from (ordinary choice, column) UTC inputs."""
    windows = {}
    rows = choices.row_position.to_numpy(dtype=int)
    times_s = choices.choice_time.to_numpy(dtype=float)
    supporting_rows = {}
    for bounds, width_ms in WINDOW_KEYS:
        if bounds not in supporting_rows:
            # A choice admitted to a regional readout can lack the wider paired window.
            indices = np.empty(0, dtype=int)
            if len(coverage_s) == len(REGIONS):
                start_s = max(start for start, _ in coverage_s.values())
                end_s = min(end for _, end in coverage_s.values())
                if start_s < end_s:
                    indices = session_timeline.select_complete_choice_windows(
                        times_s, (start_s, end_s), window_s=bounds,
                    )
            positions = rows[indices]
            splits = make_block_splits(times_s[indices], fold_ids[positions], window_s=bounds)
            supporting_rows[bounds] = positions, splits
        positions, splits = supporting_rows[bounds]
        edges = session_timeline.choice_bin_edges(width_ms / 1000, window_s=bounds)
        windows[bounds, width_ms] = PreparedWindow({}, positions, edges, splits)
    return windows


def _bin_region(
    population: session_timeline_io.TimelinePopulation, times_s: np.ndarray,
    width_s: float, bounds_s: tuple[float, float],
) -> np.ndarray:
    """Bin (choice,) UTC times and a loaded population into (choice, bin, unit) spikes/bin."""
    edges = session_timeline.choice_bin_edges(width_s, window_s=bounds_s)
    counts = np.empty((len(times_s), len(edges) - 1, len(population.unit_ids)))
    if not len(times_s):
        return counts
    # Pynapple intervals in one batch must be disjoint, including at endpoints.
    touching = np.diff(times_s) <= bounds_s[1] - bounds_s[0]
    single = np.r_[touching, False] | np.r_[False, touching]
    batches = [np.flatnonzero(~single)] if np.any(~single) else []
    batches.extend(np.array([index]) for index in np.flatnonzero(single))
    for indices in batches:
        values, _ = session_timeline.bin_choice_counts(
            population.spike_group, population.cluster_ids, times_s[indices], width_s,
            window_s=bounds_s,
        )
        if (values.shape != (len(indices), *counts.shape[1:]) or not np.isfinite(values).all()
                or np.any(values < 0)):
            raise ValueError("Expected finite nonnegative choice/bin/unit spike counts.")
        counts[indices] = values
    return counts
