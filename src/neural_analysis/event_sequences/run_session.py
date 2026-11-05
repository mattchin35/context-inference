"""Explicit single-session event-sequence preparation for the app and Python batches.

The canonical configuration identifies current scientific sources/parameters.
Exact PCA archive identity is bound separately and preserved on normal reuse.
Recompute makes a fresh event run; an already compatible fixed PCA may be reused.
"""

from __future__ import annotations

from dataclasses import dataclass, KW_ONLY
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.neural_analysis.phase_synchrony.results import parse_unit_id
from src.neural_analysis.population import run_session_timeline as timeline_runner
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import (
    ResolvedSession, load_session_metadata, resolve_probe_sources, resolve_session_metadata,
)
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage
from . import io
from .activity import extract_unit_activity, select_pc_activity
from .selection import (
    TRANSITION_CONDITIONS, prepare_sequence_trials, select_central_sequences, select_ideal_transitions,
)


@dataclass(frozen=True)
class EventSequenceRequest:
    """Small explicit per-session preparation request; no display state or arrays.

    metadata_path is an existing neural_session.json Path/string; source_kind
    is unit/pc. Unit requests require probe-qualified unit_id and resolve quality
    None to good. PC requests require region PFC/HPC, allow probe_id override,
    and resolve quality None to good_mua. Explicit good/good_mua PC populations
    both exclude missing manual labels. bin_width_s is .5/.1/.05/.02 seconds.
    Region/probe override are PC-only; unit ID is unit-only. Rank, motif and
    z-scoring are live display settings, absent from this immutable record.
    """

    metadata_path: Path | str
    source_kind: str
    _: KW_ONLY
    unit_id: str | None = None
    region: str | None = None
    probe_id: str | None = None
    bin_width_s: float = 0.5
    population_quality: str | None = None

    def __post_init__(self) -> None:
        """Validate scalar request fields/units; canonicalize metadata Path and actual quality or raise ValueError."""
        if self.source_kind not in ("unit", "pc"):
            raise ValueError("source_kind must be unit or pc.")
        timeline.choice_bin_edges(self.bin_width_s)
        quality = self.population_quality if self.population_quality is not None else ("good" if self.source_kind == "unit" else "good_mua")
        if quality not in ("good", "good_mua"):
            raise ValueError("population_quality must be good or good_mua.")
        if self.source_kind == "unit":
            parse_unit_id(self.unit_id)
            if self.region is not None or self.probe_id is not None or quality != "good":
                raise ValueError("Unit requests require manual good quality; region/probe override are PC-only.")
        elif self.unit_id is not None or self.region not in ("PFC", "HPC"):
            raise ValueError("PC requests require region PFC/HPC and cannot specify a unit_id.")
        if self.probe_id is not None and (not isinstance(self.probe_id, str) or not self.probe_id):
            raise ValueError("PC probe_id must be a nonempty metadata probe name.")
        object.__setattr__(self, "metadata_path", Path(self.metadata_path).resolve())
        object.__setattr__(self, "population_quality", quality)


def prepare_event_sequences(
    request: EventSequenceRequest, *, rerun: bool = False, dry_run: bool = False,
) -> dict:
    """Prepare/reuse one explicit unit or fixed-population PC product.

    Parameters
    ----------
    request : EventSequenceRequest
        One session/source configuration; widths are seconds and units are
        explicitly probe-qualified. No experimental session batch is inferred.
    rerun : bool, keyword-only
        Force a fresh event run directory. A valid compatible upstream PCA is
        still reused; this flag does not request an upstream PCA refit.
    dry_run : bool, keyword-only
        Inspect small current metadata/behavior/quality inputs, returning a
        plan without raw spike reads, fits, outputs or worker startup.

    Returns
    -------
    dict
        status prepared/reused/dry-run, result_path Path (None for dry-run),
        configuration_id, elapsed_s and timings_s (measured wall seconds).
        Publication time includes final validation/hash and NPZ serialization;
        it is reported/logged after publication without rewriting the NPZ.
        Published outputs are
        timestamped session-local analysis_runs with final results.npz, saved
        labels/maps/matches, log, summary and exact main/batch script snapshots.
        Existing results remain immutable. Canonical source identity excludes
        display settings and unrelated Git commits; exact PCA path/config/bytes
        are separately bound in meta.pca_reference and checked before reuse.
    """
    started = perf_counter()
    session = resolve_session_metadata(load_session_metadata(request.metadata_path), request.metadata_path)
    parameters = _resolve_parameters(session, request)
    configuration = _source_configuration(session, parameters)
    trials = prepare_sequence_trials(pd.read_csv(session.behavior.trial_table_file))
    central = select_central_sequences(trials)
    absent = sorted({"state_int", "cur_block"} - set(trials.columns))
    reason = "Part II unavailable; missing required processed trial fields: " + ", ".join(absent) if absent else ""
    transitions = (
        {key: np.empty((0, key[1] + 2), dtype=int) for key in TRANSITION_CONDITIONS}
        if absent else select_ideal_transitions(trials)
    )
    timings = {"small_input_selection_s": perf_counter() - started, "reuse_validation_s": 0.0}
    identity = configuration["configuration_id"]
    if dry_run:
        return _report("dry-run", None, identity, started, timings)
    if not rerun:
        reuse_started = perf_counter()
        for product in io.discover_event_sequences(session.session_root):
            if product["meta"]["configuration_id"] != identity:
                continue
            try:
                io.load_event_sequences(product["result_path"])
            except (OSError, ValueError):
                continue
            timings["reuse_validation_s"] = perf_counter() - reuse_started
            return _report("reused", product["result_path"], identity, started, timings)
        timings["reuse_validation_s"] = perf_counter() - reuse_started

    repository = Path(__file__).resolve().parents[3]
    if session.session_root.resolve().is_relative_to(repository):
        raise ValueError("Experimental analysis outputs must be outside the source repository.")
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = session.session_root / "analysis_runs" / f"event_sequence_heatmaps_{stamp}_{identity}"
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("event_sequence_heatmaps")
    handler = logging.FileHandler(directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Session %s; configuration %s", session.session_id, json.dumps(configuration, sort_keys=True))
        for filename in ("run_session.py", "run_batch.py"):
            shutil.copyfile(Path(__file__).with_name(filename), directory / filename)
        packet = _prepare_source(session, trials, parameters)
        rows = packet["activity_row_positions"]
        inverse = np.full(len(trials), -1, dtype=int)
        inverse[rows] = np.arange(len(rows))
        packet.update(
            trials=trials, central_matches=central, transition_matches=transitions,
            full_to_activity=inverse,
            central_summary=io.summarize_match_coverage(central, rows),
            transition_summary=io.summarize_match_coverage(transitions, rows, transitions=True),
        )
        packet["meta"].update(_provenance(session, parameters, configuration, reason))
        timings.update(packet["meta"]["timings_s"])
        packet["meta"]["timings_s"] = dict(timings)
        summary = [
            "# Event sequence heatmaps", "", f"Session: {session.session_id}",
            f"Source: {packet['meta']['source_label']}", "",
            "Goal: inspect single-source activity in behavioral central-trial and ideal-transition motifs.",
            "Scripts: run_session.py; run_batch.py.",
            f"Bin width: {parameters['bin_width_s']} s; eligible windows: {len(rows)}.",
            "Canonical configuration identifies sources/parameters; meta.pca_reference binds the exact fixed PCA archive.",
            "Recompute creates a fresh event run and may reuse a compatible fixed PCA.", reason, "",
        ]
        (directory / "summary.md").write_text("\n".join(summary), encoding="ascii")
        publication_started = perf_counter()
        path = io.save_event_sequences(directory / "results.npz", packet)
        # Final publication timing belongs in the report/log: the immutable
        # final NPZ is not rewritten to add a measurement made after its save.
        timings["publication_s"] = perf_counter() - publication_started
        logger.info("Stage timings seconds %s", json.dumps(timings, sort_keys=True))
        logger.info("Prepared %s; elapsed execution seconds %.6f", path, perf_counter() - started)
    except Exception:
        logger.exception("Event sequence preparation failed; elapsed seconds %.6f", perf_counter() - started)
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return _report("prepared", path, identity, started, timings)


# Small current-input and orchestration helpers; saved display never calls these.
def _resolve_parameters(session: ResolvedSession, request: EventSequenceRequest) -> dict:
    """Input: session/request. Return resolved scalar source parameters (seconds/unitless IDs) or ValueError."""
    if request.source_kind == "unit":
        probe, _ = parse_unit_id(request.unit_id)
    elif request.probe_id is not None:
        probe = request.probe_id
    elif session.analysis is not None:
        probe = getattr(session.analysis.probe_roles, request.region.lower())
    else:
        raise ValueError("PC preparation requires metadata probe roles or an explicit probe_id override.")
    try:
        resolve_probe_sources(session, probe)
    except KeyError as error:
        raise ValueError(f"Unknown metadata probe selection: {probe}") from error
    return {
        "source_kind": request.source_kind, "unit_id": request.unit_id,
        "region": request.region, "probe_id": probe, "bin_width_s": request.bin_width_s,
        "population_quality": request.population_quality, "missing_label_policy": "exclude",
    }


def _source_configuration(session: ResolvedSession, parameters: dict) -> dict:
    """Input: resolved session/scalar parameters. Return JSON source/cache identity, without neural array reads.

    Relevant paths/size/mtime_ns, acquisition and allowed good inside-brain
    channels participate. Current optional events/unrelated probes/Git revision
    do not determine this source configuration. Width is seconds; IDs unitless.
    """
    source, channels, rows = timeline_io.select_timeline_population(
        session, parameters["probe_id"], population_quality=parameters["population_quality"],
        missing_label_policy="exclude",
    )
    if parameters["source_kind"] == "unit":
        _, cluster = parse_unit_id(parameters["unit_id"])
        if cluster not in rows.cluster_id.to_numpy():
            raise ValueError("Selected unit must have explicit manual good label on selected good inside-brain channels.")
    paths = (
        session.behavior.trial_table_file, source.alignment_file, source.spike_clusters_file,
        source.cluster_info_file, source.channel_quality_file,
    )
    configuration = {
        "analysis_version": io.ANALYSIS_VERSION, "session_id": session.session_id,
        "parameters": parameters, "acquisition_family": source.acquisition_family,
        "channels": channels.tolist(),
        "unit_ids": [f"{parameters['probe_id']}:{cluster}" for cluster in rows.cluster_id],
        "sources": [io.source_file_identity(path) for path in paths if path is not None],
    }
    configuration["configuration_id"] = io.configuration_digest(configuration)
    return configuration


def _prepare_source(session: ResolvedSession, trials: pd.DataFrame, parameters: dict) -> dict:
    """Input: session/(N,C) trials/scalar source settings. Return activity packet core with UTC coverage.

    Units contain Hz (W,B). PCs contain shared/regional maps plus a relative
    reference and runtime-only fixed timeline; no PC scores enter event NPZs.
    """
    probe = parameters["probe_id"]
    source = resolve_probe_sources(session, probe)
    stage_started = perf_counter()
    coverage = inspect_probe_coverage(source)
    bounds = (coverage.start_s, coverage.end_s)
    packet = {
        "session_root": session.session_root, "timeline_result": None,
        "shared_choices": pd.DataFrame(), "regional_choice_positions": np.empty(0, dtype=int),
        "unit_rates_hz": None,
        "meta": {
            "pca_reference": None, "coverage_s": list(bounds),
            "timings_s": {"coverage_inspection_s": perf_counter() - stage_started},
        },
    }
    timings = packet["meta"]["timings_s"]
    if parameters["source_kind"] == "unit":
        stage_started = perf_counter()
        times, assignments = io.load_probe_spike_arrays(session, probe)
        timings["unit_raw_loading_s"] = perf_counter() - stage_started
        _, cluster = parse_unit_id(parameters["unit_id"])
        stage_started = perf_counter()
        activity = extract_unit_activity(
            times[assignments == cluster], trials, bounds, parameters["bin_width_s"],
            source_label=parameters["unit_id"],
        )
        timings["unit_binning_s"] = perf_counter() - stage_started
        packet["unit_rates_hz"] = activity.values
        packet["meta"]["source_label"] = parameters["unit_id"]
    else:
        role = parameters["region"]
        stage_started = perf_counter()
        path = timeline_io.find_compatible_regional_timeline(
            session, role, probe, parameters["bin_width_s"],
            population_quality=parameters["population_quality"], missing_label_policy="exclude",
        )
        timings["pca_lookup_s"] = perf_counter() - stage_started
        timings["pca_preparation_s"] = 0.0
        if path is None:
            # No validated compatible basis exists. Force a new upstream result
            # rather than trusting the older runner's metadata-only cache lookup.
            stage_started = perf_counter()
            report = timeline_runner.run_session_timeline(
                session.metadata_path, {role: probe}, parameters["bin_width_s"], rerun=True,
                population_quality=parameters["population_quality"], missing_label_policy="exclude",
            )
            timings["pca_preparation_s"] = perf_counter() - stage_started
            path = report.result_path
        stage_started = perf_counter()
        result = timeline_io.load_timeline_result(path)
        timings["pca_loading_s"] = perf_counter() - stage_started
        if role not in result.regions:
            raise ValueError(f"Requested PCA source unavailable: {result.unavailable.get(role, role)}")
        packet["timeline_result"] = result
        packet["shared_choices"] = result.choices.copy(deep=True)
        packet["regional_choice_positions"] = result.regions[role].choice_positions.copy()
        activity = select_pc_activity(trials, result.choices, result.regions[role], 1)
        packet["meta"]["source_label"] = f"{role}/{probe} PCs ({parameters['population_quality']})"
        stage_started = perf_counter()
        packet["meta"]["pca_reference"] = io.make_pca_reference(
            session.session_root, path, role, probe, result.meta["configuration_id"],
        )
        timings["pca_hash_s"] = perf_counter() - stage_started
    packet.update(activity_row_positions=activity.row_positions.copy(), bin_edges_s=activity.bin_edges_s.copy())
    return packet


def _provenance(session: ResolvedSession, parameters: dict, configuration: dict, transition_reason: str) -> dict:
    """Input: session/settings/config/reason. Return saved provenance dict, with explicit units/axes and no randomness."""
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=False)
    return {
        "analysis_version": io.ANALYSIS_VERSION, "schema_version": io.SCHEMA_VERSION,
        "generator": "src.neural_analysis.event_sequences.run_session.prepare_event_sequences",
        "session_id": session.session_id, "source_kind": parameters["source_kind"],
        "parameters": parameters, "configuration": configuration,
        "configuration_id": configuration["configuration_id"], "random_seed": None,
        "randomness_used": False, "python": platform.python_version(),
        "code_revision": process.stdout.strip() if process.returncode == 0 else "unknown",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "machine": platform.platform(),
        "library_versions": {name: version(name) for name in ("numpy", "pandas", "pynapple", "scikit-learn")},
        "metadata_source": io.source_file_identity(session.metadata_path),
        "transition_unavailable_reason": transition_reason,
        "units": {"absolute_time": "UTC Unix seconds", "relative_time": "seconds", "unit_rates_hz": "Hz", "scores": "unwhitened standardized-unit scores"},
        "axes": {"unit_rates_hz": ["choice_window", "time_bin"], "episode": ["episode", "trial_segment", "time_bin"]},
    }


def _report(status: str, path: Path | None, identity: str, started_s: float, timings_s: dict) -> dict:
    """Input: status/Path/config ID/start clock/stage seconds. Return small elapsed/stage wall-time report."""
    return {
        "status": status, "result_path": path, "configuration_id": identity,
        "elapsed_s": perf_counter() - started_s, "timings_s": dict(timings_s),
    }
