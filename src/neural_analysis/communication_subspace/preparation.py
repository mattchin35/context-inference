"""Good-unit choice counts, matched draws, and purged whole-choice splits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.run_session_timeline import resolve_timeline_probe_roles
from src.neural_analysis.session_metadata import (
    ResolvedSession,
    load_session_metadata,
    resolve_session_metadata,
)
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage

from .configuration import (
    ANALYSIS_VERSION,
    DRAW_COUNT,
    OUTER_FOLD_COUNT,
    WINDOW_S,
    CommunicationConfig,
    validate_configuration,
)


@dataclass(frozen=True)
class SessionInspection:
    """Metadata-only paired selection, before loading raw spike arrays.

    choices contains all ordinary completed choices in chronological source
    order and UTC-second choice_time. selected_trials retains original CSV
    columns for common covered choices. choice_positions (W,) indexes choices,
    while unit_ids maps PFC/HPC to ordered probe-qualified strings (U_region,).
    coverage_s maps regions to trusted UTC-second bounds. configuration is a
    JSON-compatible identity including source paths and selected unit IDs.
    session and cluster_ids are retained for the subsequent raw loader.
    """

    session_id: str
    time_origin_s: float
    probe_roles: dict[str, str]
    choices: pd.DataFrame
    selected_trials: pd.DataFrame
    choice_positions: np.ndarray
    unit_ids: dict[str, tuple[str, ...]]
    cluster_ids: dict[str, np.ndarray]
    coverage_s: dict[str, tuple[float, float]]
    configuration: dict
    session: ResolvedSession


@dataclass(frozen=True)
class PreparedInputs:
    """Paired unsmoothed choice counts and complete source identities.

    counts maps PFC/HPC to float64 (W,B,U_region) spikes/bin. Both arrays
    share choice_positions (W,) into choices, selected_trials (W, source
    columns), and bin_edges_s (B+1,) relative seconds. choice_time and
    coverage_s are UTC seconds; unit_ids are probe-qualified and ordered.
    No train-derived filtering, scaling, or model fit has occurred.
    """

    session_id: str
    time_origin_s: float
    probe_roles: dict[str, str]
    choices: pd.DataFrame
    selected_trials: pd.DataFrame
    choice_positions: np.ndarray
    counts: dict[str, np.ndarray]
    unit_ids: dict[str, tuple[str, ...]]
    bin_edges_s: np.ndarray
    coverage_s: dict[str, tuple[float, float]]
    configuration: dict


@dataclass(frozen=True)
class PopulationDraw:
    """One seeded count-matched draw of disjoint regional source/target units.

    source_indices and target_indices each map PFC/HPC to integer (n,) indices
    into the corresponding original regional unit axis. Indices have no units.
    A draw's four arrays define stable populations for all comparisons/folds.
    """

    source_indices: dict[str, np.ndarray]
    target_indices: dict[str, np.ndarray]


@dataclass(frozen=True)
class ValidationSplit:
    """One five-fold outer/inner split on the common whole-choice axis.

    All arrays contain integer positions into selected common choices (W,).
    outer_purged_indices are non-test choices intersecting test intervals;
    inner_purged_indices are outer-training choices intersecting the inner
    validation interval. Intervals are UTC-second half-open [-2,+2) windows.
    """

    outer_train_indices: np.ndarray
    outer_test_indices: np.ndarray
    inner_train_indices: np.ndarray
    inner_validation_indices: np.ndarray
    outer_purged_indices: np.ndarray
    inner_purged_indices: np.ndarray


def inspect_session_inputs(config: CommunicationConfig) -> SessionInspection:
    """Inspect metadata, trial rows, good-unit labels, and trusted coverage.

    config supplies a metadata Path, PFC/HPC roles, width in milliseconds and
    seed. Return a SessionInspection with source-order choice indices, UTC
    times, selected source rows, and qualified unit IDs. Alignment headers and
    small metadata tables are read, but no spike timestamps or assignments.
    Missing required spike files, roles, labels, or common windows raise
    ValueError; optional events are never required.
    """
    validate_configuration(config)
    metadata_path = Path(config.session_metadata).expanduser().resolve()
    session = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    roles = resolve_timeline_probe_roles(session, dict(config.probe_roles) if config.probe_roles is not None else None)
    if set(roles) != {"PFC", "HPC"} or len(set(roles.values())) != 2:
        raise ValueError("PFC and HPC require distinct configured probe roles.")
    trial_path = session.behavior.trial_table_file
    if trial_path is None or not trial_path.is_file():
        raise ValueError("Missing required trial table source.")
    trials = pd.read_csv(trial_path)
    origin, choices = timeline.prepare_timeline_choices(trials)
    if choices.empty:
        raise ValueError("No completed ordinary choices in the trial table.")

    unit_ids = {}
    cluster_ids = {}
    channels = {}
    coverage_s = {}
    source_paths = [metadata_path, trial_path]
    for region in ("PFC", "HPC"):
        probe = roles[region]
        source, selected_channels, clusters = timeline_io.select_timeline_population(
            session, probe, population_quality="good", missing_label_policy="exclude",
        )
        for path in (source.alignment_file, source.spike_clusters_file,
                     source.cluster_info_file, source.channel_quality_file):
            if path is None or not path.is_file():
                raise ValueError(f"{region}: missing required spike or metadata source: {path}")
            source_paths.append(path)
        coverage = inspect_probe_coverage(source)
        if coverage.source != "irig_utc_unix":
            raise ValueError(f"{region}: IRIG-backed trusted recording coverage required.")
        coverage_s[region] = (coverage.start_s, coverage.end_s)
        selected_ids = clusters.cluster_id.to_numpy(dtype=int)
        cluster_ids[region] = selected_ids
        channels[region] = [int(channel) for channel in selected_channels]
        unit_ids[region] = tuple(f"{probe}:{unit}" for unit in selected_ids)
    common = np.arange(len(choices), dtype=int)
    for bounds in coverage_s.values():
        eligible = timeline.select_complete_choice_windows(
            choices.choice_time.to_numpy(dtype=float), bounds, window_s=WINDOW_S,
        )
        common = np.intersect1d(common, eligible, assume_unique=True)
    if not len(common):
        raise ValueError("No choice windows have complete common PFC/HPC coverage.")
    selected_trials = trials.iloc[choices.iloc[common].row_position.to_numpy(dtype=int)].copy().reset_index(drop=True)
    source_identity = [_file_identity(path) for path in source_paths]
    settings = {
        "analysis_version": ANALYSIS_VERSION,
        "session_id": session.session_id,
        "probe_roles": dict(roles),
        "bin_width_ms": config.bin_width_ms,
        "window_s": list(WINDOW_S),
        "seed": config.seed,
        "n_draws": DRAW_COUNT,
        "n_outer_folds": OUTER_FOLD_COUNT,
        "population_quality": "good",
        "missing_label_policy": "exclude",
        "unit_ids": {region: list(ids) for region, ids in unit_ids.items()},
        "channels": channels,
        "coverage_s": {region: list(bounds) for region, bounds in coverage_s.items()},
        "sources": source_identity,
    }
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("ascii")).hexdigest()[:16]
    settings["configuration_id"] = identity
    return SessionInspection(session.session_id, origin, roles, choices, selected_trials,
                             common, unit_ids, cluster_ids, coverage_s, settings, session)


def load_session_inputs(config: CommunicationConfig) -> PreparedInputs:
    """Load and bin both selected populations on the common choice axis.

    config supplies a metadata Path, PFC/HPC roles, width in milliseconds and
    seed. Return float64 counts (W,B,U_region) in spikes/bin with shared
    choice-relative edges (B+1,) seconds. Each region loads once. Overlapping
    or touching choice windows are binned separately because Pynapple's
    IntervalSet requires disjoint intervals; otherwise the existing batch
    binner is used. No activity-based unit selection occurs.
    """
    inspection = inspect_session_inputs(config)
    choices = inspection.choices.iloc[inspection.choice_positions]
    times = choices.choice_time.to_numpy(dtype=float)
    edges = timeline.choice_bin_edges(config.bin_width_s, window_s=WINDOW_S)
    counts = {}
    for region in ("PFC", "HPC"):
        population = timeline_io.load_timeline_population(
            inspection.session, inspection.probe_roles[region],
            population_quality="good", missing_label_policy="exclude",
        )
        if population.unit_ids != inspection.unit_ids[region]:
            raise ValueError(f"{region}: selected unit IDs changed during loading.")
        if population.coverage_s != inspection.coverage_s[region]:
            raise ValueError(f"{region}: trusted recording coverage changed during loading.")
        if np.any(np.diff(times) <= WINDOW_S[1] - WINDOW_S[0]):
            per_choice = [timeline.bin_choice_counts(
                population.spike_group, population.cluster_ids,
                np.asarray([time]), config.bin_width_s, window_s=WINDOW_S,
            )[0] for time in times]
            values = np.concatenate(per_choice, axis=0)
        else:
            values, _ = timeline.bin_choice_counts(
                population.spike_group, population.cluster_ids,
                times, config.bin_width_s, window_s=WINDOW_S,
            )
        if values.shape != (len(times), len(edges) - 1, len(population.unit_ids)):
            raise ValueError(f"{region}: binned counts changed choice/bin/unit axes.")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{region}: binned spike counts must be finite and nonnegative.")
        counts[region] = np.asarray(values, dtype=np.float64)
    return PreparedInputs(
        inspection.session_id, inspection.time_origin_s, inspection.probe_roles,
        inspection.choices, inspection.selected_trials, inspection.choice_positions,
        counts, inspection.unit_ids, edges, inspection.coverage_s,
        inspection.configuration,
    )


def draw_populations(
    unit_ids: Mapping[str, tuple[str, ...]], *, seed: int = 0,
    n_draws: int = DRAW_COUNT,
) -> tuple[PopulationDraw, ...]:
    """Draw fixed disjoint source/target unit indices for both regions.

    unit_ids maps PFC/HPC to ordered qualified identity tuples (U_region,).
    seed is a nonnegative integer; n_draws is a positive integer. Return a
    tuple of draws with independent regional integer arrays (n,), where
    n=floor(min(U_PFC,U_HPC)/2) >= 2. No raw activity is inspected or modified.
    """
    if (set(unit_ids) != {"PFC", "HPC"} or type(seed) is not int or seed < 0
            or type(n_draws) is not int or n_draws < 1):
        raise ValueError("Population draws need PFC/HPC unit IDs and valid seed/draw count.")
    for region, ids in unit_ids.items():
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{region}: unit IDs must be nonempty and unique.")
    n = min(len(unit_ids["PFC"]), len(unit_ids["HPC"])) // 2
    if n < 2:
        raise ValueError("At least four good units per region are needed for matched populations.")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        source = {}
        target = {}
        for region in ("PFC", "HPC"):
            order = rng.permutation(len(unit_ids[region]))
            source[region] = order[:n]
            target[region] = order[n:2 * n]
        draws.append(PopulationDraw(source, target))
    return tuple(draws)


def build_validation_splits(
    choice_times_s: np.ndarray, *, window_s: tuple[float, float] = WINDOW_S,
    n_outer_folds: int = OUTER_FOLD_COUNT,
) -> tuple[ValidationSplit, ...]:
    """Partition chronological common choices and purge intersecting windows.

    choice_times_s is a finite nondecreasing (W,) UTC-second array. window_s
    contains choice-relative half-open bounds in seconds; n_outer_folds is the
    fixed five-fold count. Return five records of original common-choice-axis
    integer indices. Latest ceil(20% of purged outer train) is inner validation;
    train/test/validation all require at least two whole choices. No bins are
    flattened and no test choice enters tuning or fitting.
    """
    times = np.asarray(choice_times_s, dtype=float)
    if (times.ndim != 1 or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0)):
        raise ValueError("Choice times must be finite chronological UTC seconds.")
    if (n_outer_folds != OUTER_FOLD_COUNT or np.asarray(window_s).shape != (2,)
            or not np.isfinite(window_s).all() or window_s[0] >= window_s[1]):
        raise ValueError("Expected five folds and finite ordered window bounds.")
    if len(times) < 2 * n_outer_folds:
        raise ValueError("Insufficient choices for five two-choice outer folds.")
    blocks = np.array_split(np.arange(len(times), dtype=int), n_outer_folds)
    splits = []
    for test in blocks:
        candidate = np.setdiff1d(np.arange(len(times), dtype=int), test, assume_unique=True)
        outer_purged = _overlapping_indices(candidate, test, times, window_s)
        outer_train = np.setdiff1d(candidate, outer_purged, assume_unique=True)
        validation_count = ceil(0.2 * len(outer_train))
        validation = outer_train[-validation_count:]
        inner_candidate = outer_train[:-validation_count]
        inner_purged = _overlapping_indices(inner_candidate, validation, times, window_s)
        inner_train = np.setdiff1d(inner_candidate, inner_purged, assume_unique=True)
        if min(len(test), len(validation), len(inner_train)) < 2:
            raise ValueError("Overlapping choice windows leave insufficient train/validation/test choices.")
        splits.append(ValidationSplit(outer_train, test, inner_train, validation,
                                      outer_purged, inner_purged))
    return tuple(splits)


# Helpers retain source identity and test physical intervals without fitting.
def _file_identity(path: Path) -> dict:
    """Return JSON fields path, byte size, and nanosecond mtime for a source Path."""
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _overlapping_indices(
    candidates: np.ndarray, held_out: np.ndarray,
    times_s: np.ndarray, window_s: tuple[float, float],
) -> np.ndarray:
    """Return candidate choice-axis indices intersecting held-out UTC windows.

    candidates and held_out are sorted integer (N,)/(M,) indices into finite
    chronological times_s (W,) UTC seconds. window_s is a two-scalar relative
    second interval; touching half-open endpoints do not count as overlap.
    """
    if len(candidates) == 0 or len(held_out) == 0:
        return np.empty(0, dtype=int)
    held_times = times_s[held_out]
    candidate_times = times_s[candidates]
    span = window_s[1] - window_s[0]
    insertion = np.searchsorted(held_times, candidate_times)
    overlap = np.zeros(len(candidates), dtype=bool)
    left = insertion > 0
    overlap[left] |= candidate_times[left] - held_times[insertion[left] - 1] < span
    right = insertion < len(held_times)
    overlap[right] |= held_times[insertion[right]] - candidate_times[right] < span
    return candidates[overlap]
